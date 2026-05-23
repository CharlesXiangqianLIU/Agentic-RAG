# agent/state.py
import operator
from typing import TypedDict, Annotated, NotRequired
from langgraph.graph.message import add_messages


class WorkerResult(TypedDict):
    sub_task: str
    agent_type: str   # "lookup" | "comparison" | "trend" | "reasoning" | "retry"
    chunks: list[dict]


class CriticIssue(TypedDict):
    claim: str
    issue_type: str   # "unsupported" | "missing_context" | "contradictory"
    retry_query: str


class AgentState(TypedDict):
    question: str
    question_type: str
    sub_tasks: list[dict]          # [{"task": str, "agent_type": str}]

    # Populated by Send payload for each worker invocation
    current_sub_task: str
    current_agent_type: str

    # fan-in: operator.add reducer concatenates results from parallel workers
    worker_results: Annotated[list[WorkerResult], operator.add]

    evidence_map: dict             # chunk_key -> chunk dict
    draft_answer: str
    final_answer: str

    critic_issues: list[CriticIssue]
    critic_round: int
    reflection_passed: bool

    messages: Annotated[list, add_messages]
    metadata_filters: NotRequired[dict]   # optional; absent means no filter
    conversation_history: NotRequired[list[dict]]  # [{question, answer}] last N turns for context
