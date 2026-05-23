# knowledge-rag/agent/nodes/retry_search.py
from agent.state import AgentState, WorkerResult
from agent.tools import search_reports


def retry_search_node(state: AgentState) -> dict:
    """Search for additional evidence based on Critic's structured retry queries."""
    new_results: list[WorkerResult] = []
    filters = state.get("metadata_filters") or None
    seen_queries: set[str] = set()
    for issue in state.get("critic_issues", []):
        retry_query = issue.get("retry_query", "").strip()
        if not retry_query or retry_query in seen_queries:
            continue
        seen_queries.add(retry_query)
        invoke_args = {"query": retry_query}
        if filters:
            invoke_args["filters"] = filters
        chunks = search_reports.invoke(invoke_args)
        new_results.append(
            WorkerResult(sub_task=retry_query, agent_type="retry", chunks=chunks)
        )
    return {"worker_results": new_results}
