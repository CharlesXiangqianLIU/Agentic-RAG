# knowledge-rag/agent/nodes/critic.py
import json
import logging
import re
from agent.state import AgentState, CriticIssue
from agent.context_utils import build_context as _build_context
from domain.cache import get_domain_pack
from llm import get_llm_provider
from config import CRITIC_MAX_ROUNDS, CRITIC_REQUEST_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = """You are a critical fact-checker for knowledge-base answers.
For every sentence in the answer that contains a specific number, named entity, or factual claim, check if it is directly supported by the provided source passages.
Output JSON only:
{
  "overall": "PASS" or "FAIL",
  "issues": [
    {"claim": "<the problematic sentence>", "issue_type": "unsupported|missing_context|contradictory", "retry_query": "<suggested search to find supporting evidence>"}
  ]
}
If all claims are supported, set overall to PASS and issues to [].
Do NOT output anything outside the JSON object."""


def _prompt(slot: str, default: str) -> str:
    return get_domain_pack().prompt_overrides.get(slot, default)

_MAX_ROUNDS = CRITIC_MAX_ROUNDS


def _extract_json(raw: str) -> str:
    """Strip thinking tags and markdown fences so json.loads() can parse the output
    of models like gemma4 / qwen3 that wrap JSON in extra formatting."""
    # Remove <think>...</think> blocks (gemma4, qwen3 reasoning mode)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    # Extract content from ```json ... ``` or ``` ... ``` code blocks
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if m:
        return m.group(1).strip()
    # Fall back: find the first {...} object in free text
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        return m.group()
    return raw


_RED_FLAG_PHRASES = [
    "i cannot find",
    "no data",
    "not mentioned",
    "not in the sources",
    "not found in",
    "unclear",
    "not available",
    "cannot determine",
    "insufficient information",
    "not provided",
]


def _verification_warning(answer: str, issues: list[CriticIssue], round_n: int) -> str:
    lines = [
        f"Warning: This answer could not be fully verified against the retrieved sources after {round_n} review rounds.",
    ]
    if issues:
        issue_summaries = []
        for issue in issues[:3]:
            claim = issue.get("claim", "").strip()
            if claim:
                issue_summaries.append(f"- Unverified claim: {claim}")
        lines.extend(issue_summaries)
    lines.append("")
    lines.append(answer)
    return "\n".join(lines).strip()


def critic_node(state: AgentState) -> dict:
    round_n = state.get("critic_round", 0) + 1

    chunks = list(state["evidence_map"].values())
    context, n_included = _build_context(chunks)
    if n_included < len(chunks):
        logger.info(
            "[CRITIC] Truncated context from %d to %d chunks", len(chunks), n_included
        )
    llm = get_llm_provider()
    raw = llm.complete(
        [{"role": "user", "content": f"Answer:\n{state['draft_answer']}\n\nSources:\n{context}"}],
        system=_prompt("critic_system", _DEFAULT_SYSTEM),
        max_tokens=512,
        timeout=CRITIC_REQUEST_TIMEOUT_SECONDS,
    ).strip()

    try:
        parsed = json.loads(_extract_json(raw))
        issues: list[CriticIssue] = parsed.get("issues", [])
        passed = parsed.get("overall") == "PASS"
    except Exception:
        draft_lower = state["draft_answer"].lower()
        issue_type = "missing_context" if any(phrase in draft_lower for phrase in _RED_FLAG_PHRASES) else "unsupported"
        issues = [{
            "claim": state["draft_answer"][:200],
            "issue_type": issue_type,
            "retry_query": state["question"],
        }]
        passed = False

    updates: dict = {
        "critic_issues": issues,
        "critic_round": round_n,
        "reflection_passed": passed,
    }

    if round_n >= _MAX_ROUNDS and not passed:
        updates["final_answer"] = _verification_warning(state["final_answer"], issues, round_n)

    return updates
