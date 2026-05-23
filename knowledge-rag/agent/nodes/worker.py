# knowledge-rag/agent/nodes/worker.py
import concurrent.futures
import logging
import threading
from agent.state import AgentState, WorkerResult
from agent.tools import search_reports, compare_across_reports
from agent.analytics_tools import extract_structured_data, statistical_summary, multi_hop_search
from config import WORKER_TIMEOUT_SECONDS
from domain.cache import get_domain_pack

logger = logging.getLogger(__name__)


# Shared executor for timed worker calls. Reusing threads across calls
# bounds resource consumption when a fn times out: the previous bare-thread
# implementation spawned one daemon thread per call and leaked it on
# timeout. Here, a timed-out future leaves its thread in the pool, which
# the pool will reclaim once fn finally returns — leaks are capped at
# max_workers, not unbounded.
_executor: concurrent.futures.ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
_EXECUTOR_MAX_WORKERS = 8


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is None:
            _executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="worker",
            )
        return _executor


def _run_with_timeout(fn, timeout: int = WORKER_TIMEOUT_SECONDS) -> list[dict]:
    """Run fn in a pooled thread with a wall-clock timeout.

    Returns ``[]`` on timeout and logs an error. Exceptions raised by
    ``fn`` propagate to the caller (matching the previous implementation).
    """
    future = _get_executor().submit(fn)
    try:
        return future.result(timeout=max(timeout, 0))
    except concurrent.futures.TimeoutError:
        # cancel() only succeeds if the future hasn't started yet; if it's
        # already running, the thread keeps running and the pool will
        # reclaim it on completion.
        future.cancel()
        logger.error(
            "[WORKER] Timeout after %ss — returning empty result. "
            "Consider increasing WORKER_TIMEOUT_SECONDS.",
            timeout,
        )
        return []

def _domain_fields() -> list[str]:
    """Field names declared by the active domain pack (empty by default)."""
    return get_domain_pack().fields


def _fields_in_text(text: str) -> list[str]:
    lower = text.lower()
    hits: list[tuple[int, str]] = []
    for field in _domain_fields():
        pos = lower.find(field.lower())
        if pos >= 0:
            hits.append((pos, field))
    hits.sort()
    return [field for _, field in hits]


def _extract_field_hint(sub_task: str) -> str:
    """Extract the most relevant domain field name from a sub_task string.

    Returns an empty string when no domain pack is configured.
    """
    fields = _fields_in_text(sub_task)
    return fields[0] if fields else ""


def _extract_trend_axes(question: str, sub_task: str) -> tuple[str, str]:
    """Infer metric and independent variable for trend questions."""
    lower_question = question.lower()
    question_fields = _fields_in_text(question)
    task_fields = _fields_in_text(sub_task)

    metric = ""
    for field in question_fields:
        if f"trend in {field}" in lower_question or f"change in {field}" in lower_question:
            metric = field
            break
    if not metric:
        metric = question_fields[0] if question_fields else (task_fields[0] if task_fields else "")

    independent = ""
    for field in question_fields:
        if field == metric:
            continue
        if any(
            phrase in lower_question
            for phrase in (
                f"as {field}",
                f"with {field}",
                f"versus {field}",
                f"vs {field}",
                f"by {field}",
                f"against {field}",
            )
        ):
            independent = field
            break
    if not independent:
        for field in question_fields + task_fields:
            if field != metric:
                independent = field
                break

    return metric, independent


def worker_node(state: AgentState) -> dict:
    """Execute specialized worker strategies based on agent_type.

    Args:
        state: AgentState with current_sub_task and current_agent_type populated

    Returns:
        Dict with "worker_results" key containing list of WorkerResult items
    """
    sub_task = state["current_sub_task"]
    agent_type = state["current_agent_type"]
    question = state["question"]
    filters = state.get("metadata_filters") or None  # None means no filter

    if agent_type == "lookup":
        chunks = _run_with_timeout(lambda: _lookup_strategy(sub_task, filters))
    elif agent_type == "comparison":
        chunks = _run_with_timeout(lambda: _comparison_strategy(sub_task, filters))
    elif agent_type == "trend":
        chunks = _run_with_timeout(lambda: _trend_strategy(sub_task, question, filters))
    elif agent_type == "reasoning":
        chunks = _run_with_timeout(lambda: _reasoning_strategy(sub_task, question, filters))
    else:
        chunks = _run_with_timeout(lambda: _lookup_strategy(sub_task, filters))

    return {"worker_results": [WorkerResult(sub_task=sub_task, agent_type=agent_type, chunks=chunks)]}


def _invoke_args(base: dict, filters: dict | None) -> dict:
    """Build invoke args dict, omitting 'filters' key entirely when filters is None/empty."""
    if filters:
        return {**base, "filters": filters}
    return base


def _lookup_strategy(sub_task: str, filters: dict | None = None) -> list[dict]:
    """Direct search for relevant chunks."""
    return search_reports.invoke(_invoke_args({"query": sub_task}, filters))


def _comparison_strategy(sub_task: str, filters: dict | None = None) -> list[dict]:
    """Compare across multiple reports and extract structured fields."""
    field_hint = _extract_field_hint(sub_task)
    groups = compare_across_reports.invoke(_invoke_args({"query": sub_task, "field": field_hint}, filters))
    chunks: list[dict] = []
    for group in groups:
        chunks.extend(group.get("entries", []))

    extraction_fields = _domain_fields()[:4]
    structured = extract_structured_data(chunks, extraction_fields) if extraction_fields else []
    if structured:
        summary_text = "; ".join(
            f"{e['field']}={e['value']}{e['unit']} ({e['attribution'][:40]})"
            for e in structured
        )
        chunks.append({
            "text": summary_text,
            "attribution": "structured_extraction",
            "score": 1.0,
            "payload": {},
        })
    return chunks


def _trend_strategy(sub_task: str, question: str, filters: dict | None = None) -> list[dict]:
    """Search and compute statistical summary."""
    chunks = list(search_reports.invoke(_invoke_args({"query": sub_task}, filters)))
    metric, independent_var = _extract_trend_axes(question, sub_task)
    summary = statistical_summary(chunks, metric, independent_var)
    if summary:
        chunks.append({
            "text": summary,
            "attribution": "statistical_summary",
            "score": 1.0,
            "payload": {},
        })
    return chunks


def _reasoning_strategy(sub_task: str, question: str, filters: dict | None = None) -> list[dict]:
    """Multi-hop search for causal/reasoning questions."""
    return multi_hop_search(sub_task, [question], filters=filters)
