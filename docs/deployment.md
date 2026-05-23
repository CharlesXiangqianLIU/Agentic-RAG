# Deployment guide

Knowledge RAG ships with `docker-compose.yml` that runs the Streamlit
app and a local Qdrant. That's enough for a desktop pilot. This guide
covers what you need for a real deployment: secrets, TLS, backups,
monitoring, and multi-tenant separation.

The goal is a deployment that survives the boring parts — restarts,
disk replacements, credential rotations — without losing data or
crashing the chat UI.

---

## 1. Architecture at a glance

A production deployment has four moving pieces:

```
┌──────────────────┐    ┌──────────────────┐
│  Reverse proxy   │ →  │ knowledge-rag    │ ── streamlit (port 8501)
│  (Caddy / nginx) │    │ container        │ ── audit/history sqlite/postgres
│  TLS termination │    └─────────┬────────┘
└──────────────────┘              │ HTTP 6333
                                  ↓
                         ┌──────────────────┐
                         │  Qdrant          │ ── /qdrant/storage (PV)
                         │  container       │
                         └──────────────────┘
                                  │
                          ┌───────────────┐
                          │  Object store │   ← nightly Qdrant snapshots
                          │  S3 / MinIO   │
                          └───────────────┘
```

`knowledge-rag` itself is **stateless** beyond `~/.knowledge_rag_history.db`
(SQLite chat history + audit log). Mount that as a persistent volume —
or swap to Postgres if you have multiple replicas (see `DATABASE_URL`
in `persistence.py`).

---

## 2. Secrets

Don't bake `ANTHROPIC_API_KEY` into the image. Pick one:

**Docker Compose v2 secrets**

```yaml
services:
  app:
    secrets:
      - anthropic_api_key
    environment:
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_api_key
secrets:
  anthropic_api_key:
    file: ./secrets/anthropic_api_key  # plaintext on the host, mode 0400
```

Then in the container entrypoint:
```sh
export ANTHROPIC_API_KEY="$(cat $ANTHROPIC_API_KEY_FILE)"
exec streamlit run frontend/app.py
```

**Kubernetes**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: knowledge-rag-secrets
type: Opaque
stringData:
  ANTHROPIC_API_KEY: sk-ant-...
  APP_PASSWORD: rotated-monthly
```

Reference via `envFrom: [secretRef: {name: knowledge-rag-secrets}]`.

**Cloud secret managers** (AWS Secrets Manager, GCP Secret Manager, Vault)

Wrap the container entrypoint with the cloud's secret-fetch CLI; never
put the key into `.env` on a shared host.

---

## 3. TLS

Streamlit speaks plain HTTP. Put it behind Caddy, nginx, or Cloudflare.
Minimum responsibilities of the reverse proxy:

- Terminate TLS (Let's Encrypt or your corporate CA).
- Force HTTPS — Streamlit's WebSocket upgrade fails if mixed.
- Set sane timeouts. The graph can take 30–60 s for retrieval +
  generation; default nginx 60 s `proxy_read_timeout` is borderline.
  Bump to 180 s.
- Forward `X-Forwarded-For` so audit logs can record the originating IP
  if you later add a `user_label` derivation.

Example Caddyfile:

```caddy
rag.example.com {
    reverse_proxy 127.0.0.1:8501 {
        flush_interval -1                  # stream tokens immediately
        transport http {
            read_timeout 180s
            write_timeout 180s
        }
    }
    encode zstd gzip
}
```

`flush_interval -1` is **critical** for the streaming UI — without it,
Caddy buffers chunks and you lose the token-by-token effect.

---

## 4. Persistence

### Qdrant

The vector index lives in `./data/qdrant_storage` (host) →
`/qdrant/storage` (container). It is the **only** stateful part of the
RAG pipeline that's expensive to rebuild — a re-ingest of all source
docs takes hours on CPU.

**Backup strategy**

Qdrant snapshots are the safe path:

```sh
# create a snapshot of one collection
curl -X POST http://qdrant:6333/collections/knowledge_rag/snapshots

# list snapshots
curl http://qdrant:6333/collections/knowledge_rag/snapshots
```

Snapshots are tar files under `./data/qdrant_storage/collections/knowledge_rag/snapshots/`.
Rsync those to S3/MinIO nightly:

```sh
aws s3 sync ./data/qdrant_storage/collections/knowledge_rag/snapshots/ \
            s3://my-bucket/knowledge-rag/$(date +%F)/
```

**Restore**

```sh
curl -X POST http://qdrant:6333/collections/knowledge_rag/snapshots/recover \
     -H 'Content-Type: application/json' \
     -d '{"location": "file:///qdrant/storage/snapshots/<snapshot>.tar"}'
```

Test the restore quarterly — backups you never restore are theatre.

### Chat history + audit log

`~/.knowledge_rag_history.db` is SQLite by default. Mount the host dir
to keep it across container restarts:

```yaml
services:
  app:
    volumes:
      - ./data/state:/root
```

For multi-replica deploys (HA, blue/green), set `DATABASE_URL` to a
Postgres connection string — `persistence.py` detects the scheme and
switches backends (see "Postgres" section below).

---

## 5. Health checks

Add a sidecar liveness/readiness probe pair. Streamlit itself doesn't
expose `/health`, but two lightweight probes work:

| Probe        | What it does                                          |
|--------------|--------------------------------------------------------|
| `livenessProbe` | TCP socket on 8501 — restarts container if hung.      |
| `readinessProbe` | HTTP GET `/` on 8501 — only mark Ready when Streamlit's static landing page responds. |

For Qdrant:

```yaml
livenessProbe:
  httpGet:
    path: /readyz
    port: 6333
  initialDelaySeconds: 30
```

---

## 6. Monitoring

The system already logs to stdout + `~/.knowledge_rag.log`. In a
container, redirect both to stdout/stderr and let your log driver
(loki, fluent-bit, Datadog) collect them. Key log lines to alert on:

| Pattern                                  | What it means                          |
|------------------------------------------|----------------------------------------|
| `[WORKER] Timeout`                       | The graph hit `WORKER_TIMEOUT_SECONDS`. Raise the timeout or shrink retrieval `top_k`. |
| `[RETRY]` repeated 3× same request       | LLM API in a bad state — page someone. |
| `Qdrant unreachable`                     | Vector DB is gone — the chat will fail open with a friendly message. |
| `[ANSWER] evidence_map is empty`         | Question got no hits. Often misuse, but watch the rate; spikes = ingestion problem. |
| `[SYNTHESIS] semantic dedup unavailable` | `bge-m3` failed to load. Means Anthropic-only path still works; ingestion is broken. |

LangSmith (optional) gives you per-node tracing if you set
`LANGSMITH_API_KEY` + `LANGCHAIN_TRACING_V2=true`.

---

## 7. Resource sizing

Rough numbers, no GPU:

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| Streamlit + graph | 1 core | 2 GB | 500 MB (history db) |
| bge-m3 embedder | 2 cores | 3 GB (model + KV cache) | 2 GB (model files) |
| BGE-Reranker-v2-m3 | 1 core | 2 GB | 600 MB |
| Qdrant (100 k chunks) | 1 core | 1 GB | 500 MB |
| Qdrant (1 M chunks)   | 2 cores | 8 GB | 5 GB |

GPU drops embedder/reranker latency 10–20×; see
`docs/embedder-upgrade.md` (or the embedder section in the README) for
how to enable CUDA / fp16.

---

## 8. Multi-tenant separation

Two ways to host multiple tenants:

**Collection-per-tenant** (cheapest, single Qdrant)

Set `QDRANT_COLLECTION=acme_inc` in tenant Acme's `.env`, `globex_corp`
in Globex's. One Qdrant, many collections. Cross-tenant leakage is
prevented at the collection boundary. Easy ops.

**Process-per-tenant** (strictest)

Run a separate `knowledge-rag` container + Qdrant per tenant. Useful
if your compliance posture forbids any shared infra. Costs roughly 2×
RAM per tenant because each holds its own embedder.

In both cases, set distinct `APP_PASSWORD` (or front with corporate
SSO via the reverse proxy).

---

## 9. Pre-flight checklist

Run through this before opening to users:

- [ ] All env vars in `.env` are real values, not placeholders.
- [ ] `APP_PASSWORD` is set (or you've wired SSO at the proxy).
- [ ] `make test` passes locally on the deployment commit.
- [ ] Qdrant snapshot has been taken AND restored once.
- [ ] `~/.knowledge_rag_history.db` is on a mounted volume.
- [ ] Reverse proxy has `flush_interval -1` (or equivalent) for streaming.
- [ ] `ingestion.run_ingestion` has been run on at least 3 real docs.
- [ ] A trial query returns answers with `[Source: ... | Page N]` citations.
- [ ] At least one `[UNSUPPORTED: ...]` case has been observed (proves safety pass is firing).
- [ ] LangSmith / log aggregator is receiving lines.
- [ ] If using a domain pack, `pytest tests/test_domain_pack.py` passes against it.
