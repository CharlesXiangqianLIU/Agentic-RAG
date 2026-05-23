# LangSmith tracing

Knowledge RAG can publish every graph run to LangSmith. With tracing
on, each Q&A turn appears as a single trace with one span per node
(`orchestrate → worker → synthesis → answer → critic → retry?`) and
nested LLM / tool calls hanging off each span. This is the cheapest
way to debug "the answer is wrong" — open the trace, find the node
that returned bad data, and zoom in.

Tracing is **off by default**. Turning it on is a one-line `.env`
change and a 2 % latency hit on the first call (warm-up of the
LangSmith SDK background uploader).

## Enabling

1. Create a project + API key at <https://smith.langchain.com>.
2. Set three env vars in `knowledge-rag/.env`:

```dotenv
LANGSMITH_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=knowledge-rag      # name as it'll show in the LangSmith UI
```

3. Restart the app (`make ui` or `docker compose up --build`).

That's it — `agent/graph.py:build_graph()` reads these on startup and
propagates them into the LangChain global environment that the
LangGraph runtime consults. There is no code change needed to enable
or disable.

To disable, set `LANGCHAIN_TRACING_V2=false` (or remove `LANGSMITH_API_KEY`)
and restart. There is no soft-reload path inside Streamlit — the env
is captured on process start.

## What you'll see in a trace

For each user question, LangSmith shows a single trace with:

```
└─ build_graph                         (root span — total wall time)
   ├─ orchestrate                      (classify + plan LLM calls)
   │  ├─ AnthropicProvider.complete    (classify_system prompt)
   │  └─ AnthropicProvider.complete    (plan_system prompt)
   ├─ worker (×N, parallel via Send)
   │  └─ search_reports                (tool call)
   │     ├─ hybrid_search              (qdrant query)
   │     └─ rerank                     (BGE-Reranker-v2-m3)
   ├─ synthesis                        (dedup + semantic merge)
   ├─ answer                           (streamed LLM call)
   │  └─ AnthropicProvider.stream
   ├─ critic                           (fact-check LLM call)
   │  └─ AnthropicProvider.complete
   └─ retry_search? (if critic flagged unsupported claims)
```

Each span carries:

- **Inputs** — the state slice the node received (question, sub_task,
  filters, etc.).
- **Outputs** — the state delta the node returned.
- **Run time** — wall clock and (for LLM calls) token counts.
- **Errors** — any exception, with the stack the node raised.

## Common debugging recipes

| Symptom                                              | Where to look                                                                                                                  |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Answer hallucinates a fact                           | `answer` span → output `draft_answer` vs `final_answer`. If they differ, the safety pass tagged something `[UNSUPPORTED: ...]`. |
| Critic loops 3× then warns "could not be verified"   | `retry_search` spans — open each `critic_issues` payload to see what claims the LLM repeatedly couldn't ground.                |
| Worker returns empty chunks                          | `worker` span → `search_reports` tool call inputs. Compare the rewritten query against your Qdrant collection.                  |
| Question gets classified wrong                       | `orchestrate` → first LLM call → the raw classification token. Tune the `classify_system` slot in your domain pack.             |
| One sub-task takes 30 s when others take 2 s         | `worker` parallel spans — the slow one is usually a `multi_hop_search` reasoning call.                                          |
| `[UNSUPPORTED]` appears on a number that IS in the source | `answer` span → check whether the source uses an alternate format (`1,234` vs `1234`). The safety check now normalises both, but a stale trace from before that fix is informative. |

## Tracing in production

LangSmith uploads happen on a background thread; they don't block the
graph. But the SDK does buffer in memory — if your app produces tens
of thousands of traces per hour, set
`LANGCHAIN_CALLBACKS_BACKGROUND=false` to flush synchronously
(slower but bounded memory). For most knowledge-base workloads (a few
hundred questions per day) the default is fine.

## Sampling

To trace only some traffic, set `LANGCHAIN_TRACING_V2=true` along with
`LANGSMITH_SAMPLING_RATE=0.1` to capture 10 % of runs at random. The
sampling decision is per-trace, not per-span, so an entire Q&A turn is
either captured or skipped — the trace tree is never partial.

## Privacy considerations

LangSmith uploads:

- The user's question.
- Every retrieved chunk's text and attribution.
- The LLM's draft and final answers.

If any of those carry sensitive data (PII, customer records), enable a
private LangSmith deployment instead of the SaaS, or set
`LANGSMITH_ENDPOINT` to your self-hosted instance. The system never
uploads source file binaries — only the chunked text — but that's
still enough to leak content. Treat LangSmith as a downstream
processor and apply the same access controls.

## Disabling per node

To exclude a specific node from tracing (e.g. the embedder, which
LangSmith doesn't show usefully), wrap the call in
`langsmith.run_helpers.traceable(run_type="chain", name=...)` only on
nodes you want — and leave the others untraced. The current code base
relies on LangGraph's built-in tracing of every node; finer-grained
control is a future enhancement, not a feature today.
