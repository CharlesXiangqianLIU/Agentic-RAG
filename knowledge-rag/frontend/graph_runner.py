"""Timeout-aware graph execution helpers for the Streamlit frontend.

Two surfaces:

* ``run_graph_with_timeout`` — blocking call, returns ``(merged_state,
  timed_out)``. Kept for tests and any caller that doesn't need
  token-level streaming.
* ``run_graph_streaming`` — generator that yields live events as the
  graph runs:
      ("token", str)                       — single LLM token from answer_node
      ("event", node_name, node_output)    — a graph node has completed
      ("done", merged_state, timed_out)    — run finished or timed out
  This is what the Streamlit UI consumes to render answers
  token-by-token while still surfacing per-node status updates.
"""
from queue import Empty, Queue
import threading
import time
from typing import Callable, Iterator


# Channel sentinel used inside the merged event queue.
_KIND_EVENT = "event"
_KIND_TOKEN = "token"
_KIND_DONE = "done"
_KIND_ERROR = "error"


def _spawn_graph_worker(
    graph,
    initial_state: dict,
    event_queue: "Queue[tuple[str, object]]",
    stop_event: threading.Event,
) -> threading.Thread:
    """Start a background thread that drives ``graph.stream``.

    The thread injects a ``_token_sink`` callable into ``initial_state``
    that pushes tokens into ``event_queue`` as they arrive from the LLM.
    Node-completion events from LangGraph are forwarded to the same
    queue, so the consumer sees a single ordered stream of tokens and
    events.
    """

    def _token_sink(token: str) -> None:
        if stop_event.is_set():
            return
        event_queue.put((_KIND_TOKEN, token))

    enriched_state = {**initial_state, "_token_sink": _token_sink}

    def _worker() -> None:
        try:
            for event in graph.stream(enriched_state, stream_mode="updates"):
                if stop_event.is_set():
                    break
                event_queue.put((_KIND_EVENT, event))
            event_queue.put((_KIND_DONE, None))
        except Exception as exc:  # noqa: BLE001 — propagate to consumer
            event_queue.put((_KIND_ERROR, exc))

    thread = threading.Thread(target=_worker, daemon=True, name="graph-runner")
    thread.start()
    return thread


def _merge_node_output(result: dict, node_output: dict) -> None:
    """Apply one node's state delta to the running merged result."""
    for key, value in node_output.items():
        if isinstance(value, list) and isinstance(result.get(key), list):
            result[key] = result[key] + value
        else:
            result[key] = value


def run_graph_streaming(
    graph,
    initial_state: dict,
    timeout_seconds: int,
    on_event: Callable[[str, dict], None] | None = None,
) -> Iterator[tuple]:
    """Generator yielding live ``("token"|"event"|"done", ...)`` records.

    Always ends with exactly one ``("done", merged_state, timed_out)``
    item, even on timeout, so consumers can finalise UI state in a single
    place. Propagates exceptions raised by the graph thread.
    """
    result: dict = {}
    event_queue: Queue[tuple[str, object]] = Queue()
    stop_event = threading.Event()

    _spawn_graph_worker(graph, initial_state, event_queue, stop_event)

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop_event.set()
            yield (_KIND_DONE, result, True)
            return

        try:
            kind, payload = event_queue.get(timeout=remaining)
        except Empty:
            stop_event.set()
            yield (_KIND_DONE, result, True)
            return

        if kind == _KIND_TOKEN:
            yield (_KIND_TOKEN, payload)
            continue

        if kind == _KIND_DONE:
            yield (_KIND_DONE, result, False)
            return

        if kind == _KIND_ERROR:
            raise payload  # type: ignore[misc]

        # kind == _KIND_EVENT
        event = payload
        for node_name, node_output in event.items():  # type: ignore[union-attr]
            if on_event is not None:
                on_event(node_name, node_output)
            _merge_node_output(result, node_output)
            yield (_KIND_EVENT, node_name, node_output)


def run_graph_with_timeout(
    graph,
    initial_state: dict,
    timeout_seconds: int,
    on_event: Callable[[str, dict], None] | None = None,
) -> tuple[dict, bool]:
    """Blocking convenience wrapper around ``run_graph_streaming``.

    Drains the generator and returns ``(merged_state, timed_out)``.
    Tokens are discarded by this surface — callers that need them should
    consume ``run_graph_streaming`` directly.
    """
    result: dict = {}
    timed_out = False
    for record in run_graph_streaming(graph, initial_state, timeout_seconds, on_event):
        if record[0] == _KIND_DONE:
            _, result, timed_out = record
            break
    return result, timed_out
