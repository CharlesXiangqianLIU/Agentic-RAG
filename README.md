# Knowledge RAG

[![tests](https://github.com/CharlesXiangqianLIU/Agentic-RAG/actions/workflows/test.yml/badge.svg)](https://github.com/CharlesXiangqianLIU/Agentic-RAG/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A fully private **Agentic RAG** system for querying a private document corpus.
Works out of the box as a general-purpose knowledge base; can be specialised
to any domain (chemistry, legal, HR, finance, …) by pointing at a single YAML
**domain pack**.

---

## What it does

- Parses **`.docx`, `.md`, `.txt`, `.pdf`, `.html`** natively, preserving
  paragraph hierarchy, tables, and (where available) page breaks
- Hybrid retrieval: **dense (bge-m3) + sparse + metadata filter** in Qdrant,
  reranked with BGE-Reranker-v2-m3
- Agentic workflow on LangGraph: `orchestrate → worker → synthesis → answer →
  critic → (retry?)`, with up to 3 self-correcting retry rounds
- Tools: `search_reports` (general retrieval), `compare_across_reports`
  (cross-document comparison), `multi_hop_search`, `statistical_summary`
- Safety pass flags numeric claims (and, if a domain pack defines an entity
  vocabulary, named entities) not supported by retrieved chunks with
  `[UNSUPPORTED: ...]`
- Streamlit chat UI with source attribution (file • page • section) and
  Excel export of supporting chunks
- Pluggable LLM providers: **Anthropic** (default) / **DeepSeek (vLLM)** /
  **Ollama** — swap via one `.env` flag, zero code changes
- **Domain pack** (optional): point `DOMAIN_PACK_PATH` at a YAML file to
  inject domain abbreviations, synonym groups, unit-normalisation regex,
  filterable fields, and prompt overrides — without touching code

---

## Architecture

```
knowledge-rag/
├── ingestion/    # multi-format parsing → normalisation → hierarchical chunking → Qdrant indexing
│   └── parsers/  # markdown / text / pdf / html backends behind parse_document()
├── retrieval/    # embedder (bge-m3) + hybrid searcher + reranker
├── llm/          # LLMProvider ABC; Anthropic / DeepSeek / Ollama implementations
├── agent/        # LangGraph nodes (orchestrate/worker/synthesis/answer/critic/retry) + tools + safety
├── domain/       # domain pack loader + cache + example packs (e.g. chemistry.yaml)
├── frontend/     # Streamlit app + Excel / Markdown export
└── evaluation/   # Ragas runner + gold Q&A set (qa_gold.json)
```

Key invariants:
- **Table rows are atomic** — never split; each data row gets the header row prepended
- **Domain knowledge lives in YAML, not code** — empty pack = fully generic RAG;
  populated pack adds abbreviations, synonyms, fields, unit patterns, and prompt
  overrides at runtime
- **Synonym expansion at index time**, not query time — both source token and
  canonical form land in the same chunk
- **LLM swap = one `.env` change** (`LLM_PROVIDER=deepseek|anthropic|ollama`)
- **One parse dispatcher** — `ingestion.parser.parse_document(path)` routes by
  file extension; all backends return the same `ParsedDocument` shape

---

## Quick start

### 1. Install

```bash
cd knowledge-rag
pip install -e ".[dev]"
```

Requires Python ≥ 3.11.

### 2. Configure

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (or switch LLM_PROVIDER=deepseek / ollama)
```

### 3. Start Qdrant

```bash
docker compose up -d qdrant
```

### 4. Ingest documents

Put any mix of `.docx`, `.md`, `.txt`, `.pdf`, `.html` files in
`knowledge-rag/data/reports/` (optionally with per-file `<name>.json` sidecars
for arbitrary metadata, e.g. `{"category": "policy", "doc_type": "handbook"}`),
then:

```bash
python -m ingestion.run_ingestion --docs-dir ./data/reports
```

### 5. Launch the UI

```bash
streamlit run frontend/app.py
```

Or run everything via Docker:

```bash
docker compose up --build
```

---

## Domain pack (optional)

A domain pack is a YAML file that lets you specialise the generic pipeline
for a specific subject area without forking code. It can declare:

| Section            | Effect at runtime                                            |
|--------------------|--------------------------------------------------------------|
| `abbreviations`    | Expanded at index time (`DCM → Dichloromethane`)              |
| `synonym_groups`   | Attached to chunk metadata so queries hit any group member    |
| `fields`           | Used by `worker` for comparison/trend field hints + UI filters |
| `unit_patterns`    | Regex replacements run by `normalize_text` (`80 deg C → 80 °C`) |
| `prompt_overrides` | Replace any of `answer_system`, `answer_comparison_system`, `answer_trend_system`, `classify_system`, `plan_system`, `critic_system` |

Enable a pack by setting one env var:

```dotenv
DOMAIN_PACK_PATH=domain/examples/chemistry.yaml
```

A worked example is shipped at
`knowledge-rag/domain/examples/chemistry.yaml` — it restores the original
synthetic-chemistry behaviour (DCM/THF/Pd(OAc)₂ vocabulary, °C/°F unit
normalisation, comparison/trend field hints, and chemistry-flavoured prompts).
Use it as a template for your own domain.

With no `DOMAIN_PACK_PATH` (or an empty path) the system is fully generic:
no abbreviation expansion, no synonym groups, no field hints, generic
prompts.

---

## Testing

```bash
# All tests (no external services required — everything is mocked)
pytest tests/ -q

# A single file / case
pytest tests/test_chunker.py -v
pytest tests/test_chunker.py::test_table_rows_are_individual_chunks -v
```

## Evaluation

Ragas runner over `evaluation/eval_set/qa_gold.json` (replace the placeholder
entries with domain expert annotations of `expected_answer`, `source_file`,
`source_page`, `source_section`):

```bash
python -m evaluation.ragas_runner
```

Target scores: **faithfulness > 0.85**, **answer_relevancy > 0.80**,
**context_recall > 0.80**, **context_precision > 0.75**.

---

## LLM provider swap

Change one line in `.env`:

```dotenv
LLM_PROVIDER=anthropic   # claude-sonnet-4-6 via Anthropic API (default)
# LLM_PROVIDER=deepseek  # local vLLM at VLLM_BASE_URL
# LLM_PROVIDER=ollama    # local Ollama for fully offline runs
```

No code changes — the factory in `llm/__init__.py` instantiates the matching
provider.

---

## Status

General-purpose foundation in place. To use it on a real corpus you typically
need to (a) drop documents into `data/reports/`, (b) optionally write a domain
pack for your area, (c) annotate enough items in `qa_gold.json` (target
50–100) for Ragas scores to be meaningful.
