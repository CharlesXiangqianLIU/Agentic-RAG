# agent/nodes/answer.py
import logging
import os

from agent.state import AgentState
from agent.context_utils import build_context, count_tokens, truncate_text
from agent.safety import check_answer
from domain.cache import get_domain_pack
from llm import get_llm_provider
from config import (
    MAX_CONTEXT_TOKENS,
    ANSWER_HISTORY_TURNS,
    ANSWER_HISTORY_MAX_TOKENS,
    ANSWER_REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = """You are an expert knowledge assistant.
Answer the question using ONLY the provided source passages.
After each factual claim, cite the source using this exact format: [Source: <filename> | Page <n> | Section: <section>]
Never invent facts, numbers, or quotes not present in the sources."""

_DEFAULT_COMPARISON_SYSTEM = """You are an expert knowledge assistant.
Answer the comparison question using ONLY the provided source passages.
Present your answer as a Markdown table; pick column headers that fit the question (e.g. Document, Date, Value, Notes).
After the table, add a brief 1-2 sentence summary of the key difference.
Cite each row with the source: [Source: <filename> | Page <n> | Section: <section>]
Never invent data not present in the sources."""

_DEFAULT_TREND_SYSTEM = """You are an expert knowledge assistant.
Answer the trend question using ONLY the provided source passages.
Present your answer as:
1. A numbered list of data points (e.g. "1. At <x>: <metric> = <value> [Source: ...]")
2. A concluding sentence describing the overall trend direction (increasing / decreasing / stable).
Never invent data not present in the sources."""


def _prompt(slot: str, default: str) -> str:
    """Return the prompt for `slot`, allowing a domain pack to override it."""
    return get_domain_pack().prompt_overrides.get(slot, default)

def _build_context(chunks: list[dict], max_tokens: int = MAX_CONTEXT_TOKENS) -> tuple[str, int]:
    """Alias for agent.context_utils.build_context — kept for backward compatibility."""
    return build_context(chunks, max_tokens=max_tokens)


def _max_tokens_for(question_type: str) -> int:
    """Return max_tokens budget for the LLM based on question type."""
    return 4096 if question_type in ("comparison", "trend") else 2048


def _select_system_prompt(question_type: str) -> str:
    """Return the appropriate system prompt for the given question type.

    A domain pack may override any of the three slots via
    ``prompt_overrides[answer_system | answer_comparison_system | answer_trend_system]``.
    """
    if question_type == "comparison":
        return _prompt("answer_comparison_system", _DEFAULT_COMPARISON_SYSTEM)
    if question_type == "trend":
        return _prompt("answer_trend_system", _DEFAULT_TREND_SYSTEM)
    return _prompt("answer_system", _DEFAULT_SYSTEM)


_DEFAULT_HISTORY_SUMMARY_SYSTEM = """You condense a multi-turn user/assistant conversation transcript into a brief running summary.
Output ONE short paragraph (≤ 5 sentences) preserving:
- The user's underlying goal across all turns.
- Any concrete entities or constraints mentioned (names, dates, filenames, numbers, codes).
- The latest assistant claim that subsequent turns build on.
Drop pleasantries, restated questions, and any safety markers like [UNSUPPORTED: ...].
Do NOT invent facts. Output the paragraph and nothing else."""


def _format_turn(turn: dict, remaining_tokens: int) -> tuple[str, int]:
    """Render a single Q/A turn as a token-bounded block. Returns (block, tokens)."""
    question = truncate_text(turn.get("question", ""), max_tokens=max(remaining_tokens // 3, 32))
    answer_budget = max(remaining_tokens - count_tokens(f"Q: {question}\nA: "), 32)
    answer = truncate_text(turn.get("answer", ""), max_tokens=answer_budget)
    block = f"Q: {question}\nA: {answer}".strip()
    return block, count_tokens(block)


def _build_history_prefix(history: list[dict]) -> str:
    """Format conversation history as a context prefix.

    The most recent ``ANSWER_HISTORY_TURNS`` turns are included verbatim
    (token-truncated as needed). When ``HISTORY_SUMMARY=1`` and the
    history exceeds ``ANSWER_HISTORY_TURNS`` turns, the older turns are
    summarised via a quick LLM call into a single paragraph that lives
    above the verbatim turns. The summary is best-effort: failures
    fall back to the verbatim-only behaviour silently.
    """
    if not history:
        return ""

    recent = history[-ANSWER_HISTORY_TURNS:]
    older = history[: -ANSWER_HISTORY_TURNS] if len(history) > ANSWER_HISTORY_TURNS else []

    summary_block = ""
    summary_tokens = 0
    if older and _history_summary_enabled():
        summary_text = _summarise_older_turns(older)
        if summary_text:
            summary_block = f"Earlier conversation summary:\n{summary_text}"
            summary_tokens = count_tokens(summary_block)

    blocks: list[str] = []
    remaining = max(ANSWER_HISTORY_MAX_TOKENS - summary_tokens, 0)
    for turn in reversed(recent):
        if remaining <= 0:
            break
        block, block_tokens = _format_turn(turn, remaining)
        if not block or block_tokens <= 0:
            continue
        if block_tokens > remaining and blocks:
            continue
        if block_tokens > remaining:
            block = truncate_text(block, max_tokens=remaining)
            block_tokens = count_tokens(block)
        blocks.insert(0, block)
        remaining -= block_tokens

    if not blocks and not summary_block:
        return ""

    parts: list[str] = []
    if summary_block:
        parts.append(summary_block)
    if blocks:
        parts.append("Prior conversation context:\n" + "\n".join(blocks))
    return "\n\n".join(parts) + "\n\n"


def _history_summary_enabled() -> bool:
    return os.getenv("HISTORY_SUMMARY", "0") == "1"


def _history_summary_max_tokens() -> int:
    return int(os.getenv("HISTORY_SUMMARY_MAX_TOKENS", "500"))


def _summarise_older_turns(turns: list[dict]) -> str:
    """Compress a list of older Q/A turns into a ≤ N-token paragraph.

    Returns an empty string on any failure — summarisation is an
    enhancement, never load-bearing.
    """
    try:
        transcript = "\n".join(
            f"User: {t.get('question', '')}\nAssistant: {t.get('answer', '')}"
            for t in turns
        )
        transcript = truncate_text(transcript, max_tokens=4_000)
        llm = get_llm_provider()
        summary = llm.complete(
            [{"role": "user", "content": transcript}],
            system=_prompt("history_summary_system", _DEFAULT_HISTORY_SUMMARY_SYSTEM),
            max_tokens=_history_summary_max_tokens(),
            timeout=20,
        )
        return truncate_text(summary.strip(), max_tokens=_history_summary_max_tokens())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ANSWER] history summarisation failed (%s); using verbatim history only", exc)
        return ""


def answer_node(state: AgentState) -> dict:
    chunks = list(state["evidence_map"].values())
    if not chunks:
        logger.warning("[ANSWER] evidence_map is empty — no source passages available")
        msg = "No relevant passages were found in the knowledge base for this question. Please rephrase your question or check that the documents have been ingested correctly."
        return {"draft_answer": msg, "final_answer": msg, "reflection_passed": True}
    context, n_included = build_context(chunks)
    if n_included < len(chunks):
        logger.info("[CONTEXT] Truncated to %d/%d chunks (%d token limit)", n_included, len(chunks), MAX_CONTEXT_TOKENS)

    question_type = state.get("question_type", "")
    system_prompt = _select_system_prompt(question_type)

    # Append filter scope note if active filters are set
    filters = state.get("metadata_filters") or {}
    if filters:
        scope_parts = [f"{k}={v}" for k, v in filters.items()]
        scope_note = f"\nNote: Search results are scoped to {', '.join(scope_parts)}. Reflect this scope in your answer."
        system_prompt = system_prompt + scope_note

    # Inject prior conversation turns for referential questions ("same project as before")
    history_prefix = _build_history_prefix(state.get("conversation_history") or [])
    history_tokens = count_tokens(history_prefix)
    if history_tokens > 0:
        context, n_included = build_context(chunks, max_tokens=MAX_CONTEXT_TOKENS - history_tokens)
        if n_included < len(chunks):
            logger.info("[CONTEXT] Truncated to %d/%d chunks after reserving %d tokens for history", n_included, len(chunks), history_tokens)

    llm = get_llm_provider()
    # Optional runtime side-channel: if the caller injected a "_token_sink"
    # callable into state, each streamed token is forwarded as it arrives.
    # This lets the Streamlit UI render tokens incrementally while the node
    # still returns the full answer at completion.
    token_sink = state.get("_token_sink") if isinstance(state, dict) else None
    tokens = []
    for token in llm.stream(
        [{"role": "user", "content": f"{history_prefix}Sources:\n{context}\n\nQuestion: {state['question']}"}],
        system=system_prompt,
        max_tokens=_max_tokens_for(question_type),
        timeout=ANSWER_REQUEST_TIMEOUT_SECONDS,
    ):
        tokens.append(token)
        if token_sink is not None:
            try:
                token_sink(token)
            except Exception:  # noqa: BLE001 — side-channel must never break the node
                logger.exception("[ANSWER] token_sink raised; continuing without UI streaming")
                token_sink = None
    draft = "".join(tokens)
    final = check_answer(draft, chunks)
    return {"draft_answer": draft, "final_answer": final}
