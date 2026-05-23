# knowledge-rag/agent/nodes/orchestrate.py
import json
import logging
import os
import re

from langgraph.types import Send

from agent.state import AgentState
from config import ORCHESTRATE_CLASSIFY_TIMEOUT_SECONDS, ORCHESTRATE_PLAN_TIMEOUT_SECONDS
from domain.cache import get_domain_pack
from llm import get_llm_provider

_logger = logging.getLogger(__name__)


_DEFAULT_REWRITE_SYSTEM = """You expand a single search query into multiple semantically equivalent paraphrases for retrieval recall.
Return a JSON array of strings — each string is a distinct rewrite of the input query.
- Keep every paraphrase short and information-dense; aim for keyword-style phrasing.
- Do NOT change the intent or scope of the query.
- Do NOT repeat the input verbatim — every output string must be a different surface form.
- Do NOT add explanations, markdown, or code fences. Output ONLY the JSON array."""

def _query_rewrite_enabled() -> bool:
    return os.getenv("QUERY_REWRITE", "0") == "1"


def _query_rewrite_timeout() -> float:
    return float(os.getenv("QUERY_REWRITE_TIMEOUT_SECONDS", "10"))


def _query_rewrite_max_variants() -> int:
    return int(os.getenv("QUERY_REWRITE_MAX_VARIANTS", "3"))


# Cap total fan-out (sub_task × variants) so a 4-way comparison doesn't
# explode into 16 parallel worker calls.
def _query_rewrite_max_subtasks() -> int:
    return int(os.getenv("QUERY_REWRITE_MAX_SUBTASKS", "8"))

_DEFAULT_CLASSIFY_SYSTEM = """You classify research questions into exactly one of these four types:
- lookup: asking for a single specific value or attribute
- comparison: asking to compare data across multiple documents or entries
- trend: asking about patterns or trends across many data points over time or conditions
- reasoning: asking why something happened, causal analysis, or explanatory questions

Respond with ONLY the type word (lookup, comparison, trend, or reasoning). Nothing else."""

_DEFAULT_PLAN_SYSTEM = """You are a research planning assistant. Given a question and its type, output a JSON array of sub-tasks.

Each item MUST follow this exact schema:
{"task": "<specific search query string>", "agent_type": "<lookup|comparison|trend|reasoning>"}

Rules:
- lookup: exactly 1 task with agent_type="lookup"
- comparison: 2-4 tasks, each with agent_type="comparison"
- trend: 2-3 tasks, each with agent_type="trend"
- reasoning: 2-4 tasks, mix of agent_type="lookup" and agent_type="reasoning"

Examples:

Question: What is the maximum number of vacation days allowed per year?
Type: lookup
Output: [{"task": "vacation days policy annual limit", "agent_type": "lookup"}]

Question: Compare the parental leave policies in the 2023 and 2024 employee handbooks.
Type: comparison
Output: [{"task": "parental leave policy 2023 handbook", "agent_type": "comparison"}, {"task": "parental leave policy 2024 handbook", "agent_type": "comparison"}, {"task": "parental leave eligibility duration comparison", "agent_type": "comparison"}]

Question: What is the trend in quarterly revenue from Q1 2023 through Q4 2024?
Type: trend
Output: [{"task": "quarterly revenue 2023 Q1 Q2 Q3 Q4", "agent_type": "trend"}, {"task": "quarterly revenue 2024 Q1 Q2 Q3 Q4", "agent_type": "trend"}, {"task": "revenue growth trajectory quarterly", "agent_type": "trend"}]

Question: Why was the data retention policy shortened from 7 years to 3 years?
Type: reasoning
Output: [{"task": "data retention policy 7 years prior version", "agent_type": "lookup"}, {"task": "data retention policy 3 years current version", "agent_type": "lookup"}, {"task": "rationale data retention policy change compliance risk", "agent_type": "reasoning"}]

Output ONLY a valid JSON array. No explanation, no markdown, no code fences."""

_VALID_TYPES = {"lookup", "comparison", "trend", "reasoning"}


def _prompt(slot: str, default: str) -> str:
    return get_domain_pack().prompt_overrides.get(slot, default)


def orchestrate_node(state: AgentState) -> dict:
    llm = get_llm_provider()

    # Classify
    raw_type = llm.complete(
        [{"role": "user", "content": state["question"]}],
        system=_prompt("classify_system", _DEFAULT_CLASSIFY_SYSTEM),
        max_tokens=10,
        timeout=ORCHESTRATE_CLASSIFY_TIMEOUT_SECONDS,
    ).strip().lower()
    question_type = raw_type if raw_type in _VALID_TYPES else "lookup"

    # Plan
    response = llm.complete(
        [{"role": "user", "content": f"Question: {state['question']}\nType: {question_type}"}],
        system=_prompt("plan_system", _DEFAULT_PLAN_SYSTEM),
        max_tokens=256,
        timeout=ORCHESTRATE_PLAN_TIMEOUT_SECONDS,
    )
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        sub_tasks = json.loads(match.group()) if match else None
        if not isinstance(sub_tasks, list) or not sub_tasks:
            raise ValueError("empty or invalid")
        for t in sub_tasks:
            if "task" not in t or "agent_type" not in t:
                raise ValueError("missing fields")
            if t["agent_type"] not in _VALID_TYPES:
                t["agent_type"] = "lookup"  # coerce invalid types to lookup instead of raising
    except Exception:
        sub_tasks = [{"task": state["question"], "agent_type": question_type}]

    # Optional query rewriting: expand each sub_task into N paraphrases to
    # raise recall when surface forms in the corpus differ from the user's
    # phrasing. Disabled by default to avoid the extra LLM call.
    if _query_rewrite_enabled():
        sub_tasks = _expand_with_rewrites(llm, sub_tasks)

    return {"question_type": question_type, "sub_tasks": sub_tasks}


def _expand_with_rewrites(llm, sub_tasks: list[dict]) -> list[dict]:
    """Replace each sub_task with up to ``_QUERY_REWRITE_MAX_VARIANTS`` rewrites.

    Original sub_task is always kept as the first variant. Failures fall
    back to the original list — rewriting is an enhancement, not a hard
    dependency.
    """
    max_variants = _query_rewrite_max_variants()
    max_subtasks = _query_rewrite_max_subtasks()
    expanded: list[dict] = []
    for sub_task in sub_tasks:
        rewrites = _generate_rewrites(llm, sub_task["task"])
        # Deduplicate while preserving order; original task first.
        seen: set[str] = set()
        for candidate in [sub_task["task"], *rewrites]:
            cleaned = candidate.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                expanded.append({"task": cleaned, "agent_type": sub_task["agent_type"]})
            if len(seen) >= max_variants:
                break
        if len(expanded) >= max_subtasks:
            _logger.info(
                "[ORCHESTRATE] capping sub_tasks at %d after query rewriting",
                max_subtasks,
            )
            return expanded[:max_subtasks]
    return expanded


def _generate_rewrites(llm, query: str) -> list[str]:
    """Ask the LLM for paraphrases of ``query``. Empty list on any failure."""
    try:
        response = llm.complete(
            [{"role": "user", "content": query}],
            system=_prompt("rewrite_system", _DEFAULT_REWRITE_SYSTEM),
            max_tokens=200,
            timeout=_query_rewrite_timeout(),
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[ORCHESTRATE] query rewrite call failed (%s); skipping", exc)
        return []

    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if not match:
            return []
        items = json.loads(match.group())
        return [str(item) for item in items if isinstance(item, str)]
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[ORCHESTRATE] failed to parse rewrites (%s); skipping", exc)
        return []


def dispatch_workers(state: AgentState) -> list[Send]:
    """Conditional edge function: fan-out sub-tasks to worker nodes via Send API.

    Spreads the full state so each worker copy has all context (question, filters, etc.).
    """
    tasks = state.get("sub_tasks") or [
        {"task": state["question"], "agent_type": state.get("question_type", "lookup")}
    ]
    return [
        Send("worker", {
            **state,
            "current_sub_task": t["task"],
            "current_agent_type": t["agent_type"],
        })
        for t in tasks
    ]
