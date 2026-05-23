# knowledge-rag/agent/graph.py
import os
from langgraph.graph import StateGraph, END
import config as _cfg
from agent.state import AgentState
from agent.nodes.orchestrate import orchestrate_node, dispatch_workers
from agent.nodes.worker import worker_node
from agent.nodes.synthesis import synthesis_node
from agent.nodes.answer import answer_node
from agent.nodes.critic import critic_node
from agent.nodes.retry_search import retry_search_node
from agent.observability import timed_node


def _after_critic(state: AgentState) -> str:
    if state.get("reflection_passed") or state.get("critic_round", 0) >= _cfg.CRITIC_MAX_ROUNDS:
        return "end"
    return "retry"


def build_graph():
    # Activate LangSmith tracing if configured
    if _cfg.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGSMITH_API_KEY", _cfg.LANGSMITH_API_KEY)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", _cfg.LANGCHAIN_TRACING_V2)
        os.environ.setdefault("LANGCHAIN_PROJECT", _cfg.LANGCHAIN_PROJECT)

    g = StateGraph(AgentState)

    g.add_node("orchestrate", timed_node("orchestrate", orchestrate_node))
    g.add_node("worker", timed_node("worker", worker_node))
    g.add_node("synthesis", timed_node("synthesis", synthesis_node))
    g.add_node("answer", timed_node("answer", answer_node))
    g.add_node("critic", timed_node("critic", critic_node))
    g.add_node("retry_search", timed_node("retry_search", retry_search_node))

    g.set_entry_point("orchestrate")
    g.add_conditional_edges("orchestrate", dispatch_workers)
    g.add_edge("worker", "synthesis")
    g.add_edge("synthesis", "answer")
    g.add_edge("answer", "critic")
    g.add_conditional_edges(
        "critic",
        _after_critic,
        {"end": END, "retry": "retry_search"},
    )
    g.add_edge("retry_search", "synthesis")

    return g.compile()
