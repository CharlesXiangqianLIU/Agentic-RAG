# frontend/app.py
"""
Knowledge Base -- Streamlit Chat Interface.

Run with: streamlit run frontend/app.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from datetime import datetime
from agent.graph import build_graph
from agent.state import AgentState
from domain.cache import get_domain_pack
from frontend.export import chunks_to_excel
from frontend.graph_runner import run_graph_streaming
from frontend.persistence import load_history, save_message, clear_history, write_audit
from config import GRAPH_TIMEOUT_SECONDS, HISTORY_PAGE_SIZE, LLM_PROVIDER, QDRANT_TIMEOUT_SECONDS


# Sidebar filter inputs are driven by the active domain pack. With an empty
# pack we fall back to two generic free-text filters mapped to `category`
# and `type` metadata keys (matching the example sidecar JSON schema).
# Up to ``_PRIMARY_FILTER_COUNT`` filters are shown inline; any extras
# from the domain pack live inside an Advanced Filters expander so the
# sidebar doesn't grow unbounded.
_DEFAULT_FILTER_FIELDS = [("Category", "category"), ("Type", "type")]
_PRIMARY_FILTER_COUNT = 2


def _filter_fields() -> list[tuple[str, str]]:
    """Return every (label, metadata_key) pair to surface in the sidebar.

    Pack fields take precedence; generic Category / Type are appended only
    when the pack supplies fewer than two fields, so the empty-pack baseline
    still has something usable.
    """
    pack_fields = get_domain_pack().fields
    out: list[tuple[str, str]] = [
        (f.title(), f.lower().replace(" ", "_")) for f in pack_fields
    ]
    for label, key in _DEFAULT_FILTER_FIELDS:
        if len(out) >= 2:
            break
        out.append((label, key))
    return out

def _check_auth() -> bool:
    """Return True if APP_PASSWORD is not set (open access) or user has authenticated."""
    import os
    # Determine required password
    try:
        required = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        required = ""
    if not required:
        required = os.getenv("APP_PASSWORD", "")

    # No password configured → open access
    if not required:
        return True

    # Already authenticated this session
    if st.session_state.get("authenticated"):
        return True

    # Show password form
    st.title("Knowledge Base")
    st.warning("This application is password-protected.")
    pwd = st.text_input("Password", type="password", key="auth_input")
    if st.button("Login"):
        if pwd == required:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


_NODE_LABELS = {
    "orchestrate": "Classifying and planning sub-tasks...",
    "worker": "Executing specialist searches...",
    "synthesis": "Merging evidence...",
    "answer": "Generating answer...",
    "critic": "Fact-checking answer...",
    "retry_search": "Re-searching for missing evidence...",
}


@st.cache_resource
def load_graph():
    # Pre-warm the embedding and reranker models before the graph runs any queries.
    # This avoids the PyTorch meta-tensor error caused by accelerate's lazy loading
    # when models are first initialised inside a LangGraph worker thread.
    from retrieval.embedder import embed_texts
    from retrieval.reranker import _get_reranker
    embed_texts(["warmup"])
    _get_reranker()
    return build_graph()


@st.cache_data(ttl=10, show_spinner=False)
def _check_qdrant_health() -> bool:
    """Ping Qdrant on startup. Returns True if reachable."""
    try:
        from qdrant_client import QdrantClient
        import os
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url, timeout=QDRANT_TIMEOUT_SECONDS)
        client.get_collections()
        return True
    except Exception:
        return False


@st.cache_data(ttl=10, show_spinner=False)
def _get_collection_stats() -> dict | None:
    """Return Qdrant collection stats (count). Returns None if unreachable."""
    try:
        from qdrant_client import QdrantClient
        import os
        from config import QDRANT_COLLECTION
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url, timeout=QDRANT_TIMEOUT_SECONDS)
        count = client.count(collection_name=QDRANT_COLLECTION)
        return {"count": count.count}
    except Exception:
        return None


def classify_error(e: Exception) -> str:
    """Map exceptions to user-friendly messages."""
    msg = str(e).lower()
    if "timeout" in msg or "timed out" in msg:
        return "The request timed out. Try a simpler question, or check your model and Qdrant service responsiveness."
    if "connection" in msg or "refused" in msg or "qdrant" in msg:
        return "Could not reach the knowledge base. Is Qdrant running? (`docker-compose up qdrant`)"
    if "rate limit" in msg or "429" in msg:
        return "API rate limit reached. Please wait a moment and try again."
    if "api key" in msg or "unauthorized" in msg or "authentication" in msg:
        if LLM_PROVIDER == "anthropic":
            return "API authentication failed. Check your ANTHROPIC_API_KEY in .env."
        return "API authentication failed. Check your configured provider credentials and endpoint settings."
    return f"An unexpected error occurred: {e}"


_HISTORY_CONTEXT_TURNS = 2


def make_initial_state(
    question: str,
    metadata_filters: dict | None = None,
    session_history: list | None = None,
) -> AgentState:
    # Extract last N assistant turns as conversation context
    conversation_history = []
    if session_history:
        assistant_turns = [m for m in session_history if m.get("role") == "assistant"]
        for turn in assistant_turns[-_HISTORY_CONTEXT_TURNS:]:
            if turn.get("question") and turn.get("content"):
                conversation_history.append({"question": turn["question"], "answer": turn["content"]})

    return AgentState(
        question=question,
        question_type="",
        sub_tasks=[],
        current_sub_task="",
        current_agent_type="",
        worker_results=[],
        evidence_map={},
        draft_answer="",
        final_answer="",
        critic_issues=[],
        critic_round=0,
        reflection_passed=False,
        messages=[],
        metadata_filters=metadata_filters or {},
        conversation_history=conversation_history,
    )


st.set_page_config(
    page_title="Knowledge Base",
    page_icon="📚",
    layout="wide",
)

if not _check_auth():
    st.stop()

# Session state — load from SQLite on first run before sidebar reads history
if "history" not in st.session_state:
    st.session_state.history = load_history()

if "history_offset" not in st.session_state:
    st.session_state.history_offset = 0

with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Clear History", help="Delete all conversation history"):
        clear_history()
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("Search Filters")
    st.caption("Leave blank to search all documents.")
    _all_filters = _filter_fields()
    _filter_widgets: list[tuple[str, str]] = []  # collected (key, value) pairs

    # Show the first N filters inline; tuck any extras inside an expander
    # so domains with many fields don't crowd the sidebar.
    _primary = _all_filters[:_PRIMARY_FILTER_COUNT]
    _extras = _all_filters[_PRIMARY_FILTER_COUNT:]

    for _label, _key in _primary:
        _value = st.text_input(_label, placeholder=f"e.g. {_label.lower()}", key=f"filter_{_key}")
        _filter_widgets.append((_key, _value))

    if _extras:
        with st.expander(f"Advanced Filters ({len(_extras)} more)"):
            for _label, _key in _extras:
                _value = st.text_input(
                    _label,
                    placeholder=f"e.g. {_label.lower()}",
                    key=f"filter_{_key}",
                )
                _filter_widgets.append((_key, _value))

    st.divider()
    # Qdrant status indicator
    if not _check_qdrant_health():
        st.sidebar.error(
            "⚠ Qdrant unreachable.\n\nStart it with:\n```\ndocker-compose up qdrant\n```"
        )
    else:
        stats = _get_collection_stats()
        if stats:
            st.sidebar.success(f"✓ Qdrant connected — {stats['count']:,} chunks indexed", icon="🟢")
        else:
            st.sidebar.success("✓ Qdrant connected", icon="🟢")

    st.divider()
    if st.session_state.get("history"):
        from frontend.export_md import conversation_to_markdown
        md_bytes = conversation_to_markdown(st.session_state.history)
        st.download_button(
            label="📄 Export Conversation",
            data=md_bytes,
            file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            help="Download full conversation as Markdown",
        )

st.title("Knowledge Base")
st.caption("Ask questions about the documents in your knowledge base. Answers include source citations.")

# Chat history display — show most recent HISTORY_PAGE_SIZE messages
history = st.session_state.history
total = len(history)
offset = st.session_state.history_offset
# Compute window: show from (total - HISTORY_PAGE_SIZE - offset) to end
start = max(0, total - HISTORY_PAGE_SIZE - offset)
visible = history[start:]

if start > 0:
    if st.button(f"⬆ Load {min(HISTORY_PAGE_SIZE, start)} earlier messages"):
        st.session_state.history_offset += HISTORY_PAGE_SIZE
        st.rerun()

for msg in visible:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("question_type"):
            st.caption(f"Query type: {msg['question_type']}")
        if msg.get("chunks"):
            with st.expander("View sources"):
                for chunk in msg["chunks"]:
                    st.markdown(f"**{chunk.get('attribution', '')}**")
                    st.text(chunk.get("text", ""))
            excel_bytes = chunks_to_excel(msg.get("question", ""), msg["chunks"])
            st.download_button(
                label="Export sources to Excel",
                data=excel_bytes,
                file_name="rag_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_{msg.get('id', id(msg))}",
            )

# Input
query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.history_offset = 0
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    save_message(role="user", content=query, question=query)

    metadata_filters = {}
    for _key, _value in _filter_widgets:
        if _value and _value.strip():
            metadata_filters[_key] = _value.strip()

    with st.chat_message("assistant"):
        answer = ""
        chunks = []
        q_type = ""
        # Streaming output lands here; on exception we write the error message instead.
        answer_placeholder = st.empty()

        try:
            graph = load_graph()
            running_text = ""
            result: dict = {}
            timed_out = False

            with st.status("Processing your question...", expanded=True) as status:
                def _on_event(node_name: str, node_output: dict) -> None:
                    del node_output
                    label = _NODE_LABELS.get(node_name, f"Running {node_name}...")
                    status.update(label=label)

                for record in run_graph_streaming(
                    graph=graph,
                    initial_state=make_initial_state(query, metadata_filters, st.session_state.history),
                    timeout_seconds=GRAPH_TIMEOUT_SECONDS,
                    on_event=_on_event,
                ):
                    kind = record[0]
                    if kind == "token":
                        running_text += record[1]
                        answer_placeholder.markdown(running_text)
                    elif kind == "done":
                        _, result, timed_out = record

                if timed_out:
                    status.update(label="Timed out.", state="error", expanded=False)
                else:
                    status.update(label="Done.", state="complete", expanded=False)

            if timed_out:
                answer = f"The query timed out after {GRAPH_TIMEOUT_SECONDS}s. Try a simpler question or check if Qdrant is running."
            else:
                # Prefer the safety-post-processed final_answer over the raw streamed text
                # (final_answer carries [UNSUPPORTED: ...] tags applied by check_answer).
                answer = result.get("final_answer") or running_text
            if not answer and not timed_out:
                answer = "No relevant information was found. Try rephrasing your question."

            # Overwrite the placeholder with the final (possibly post-processed) text.
            answer_placeholder.markdown(answer)

            chunks = list(result.get("evidence_map", {}).values())
            q_type = result.get("question_type", "")
        except Exception as e:
            answer = classify_error(e)
            chunks = []
            q_type = ""
            answer_placeholder.markdown(answer)

        if q_type:
            st.caption(f"Query type: {q_type}")

        if chunks:
            with st.expander(f"View {len(chunks)} source(s)"):
                for chunk in chunks:
                    st.markdown(f"**{chunk.get('attribution', '')}**")
                    st.text(chunk.get("text", ""))

            excel_bytes = chunks_to_excel(query, chunks)
            st.download_button(
                label="Export sources to Excel",
                data=excel_bytes,
                file_name="rag_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_current",
            )

    db_id = save_message(
        role="assistant",
        content=answer,
        question=query,
        chunks=chunks,
        question_type=q_type,
    )
    # Audit-log every turn (separate table; survives history clears).
    try:
        write_audit(
            question=query,
            answer=answer,
            question_type=q_type,
            evidence=chunks,
            metadata_filters=metadata_filters,
        )
    except Exception:  # noqa: BLE001 — audit logging must never break the UI
        pass
    msg_entry = {
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "question": query,
        "question_type": q_type,
        "id": db_id,
    }
    st.session_state.history.append(msg_entry)
