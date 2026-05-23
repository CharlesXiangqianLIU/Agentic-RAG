# Changelog

All notable changes to **Knowledge RAG** are recorded here. Dates are
in ISO 8601. Format loosely follows [Keep a Changelog](https://keepachangelog.com).
We do not yet cut formal version numbers, so entries are grouped under
dated headings rather than versions.

## 2026-05-23

### Added
- HR example domain pack at `knowledge-rag/domain/examples/hr.yaml`,
  covering acronyms, synonym groups, fields, date / currency unit
  patterns, entity patterns (employee IDs, policy refs, US Code
  citations), and a full set of prompt overrides. Pinned by
  `tests/test_hr_pack.py`.
- Demo source documents in `knowledge-rag/data/reports/` —
  `employee_handbook.md`, `release_notes_v2.4.txt`,
  `product_announcement.html`, and the existing neutral `.docx` —
  so new users can run a meaningful ingestion / Q&A pass out of the
  box. Each carries a sidecar JSON for metadata filtering.
- `docs/langsmith-tracing.md` — how to enable LangSmith, what each
  span shows, six common debugging recipes, sampling and privacy
  considerations.

### Changed
- `.gitignore` no longer blanket-ignores `data/reports/*.docx` and
  `*.json`; the demo files now ship with the repo. Add your own
  user-corpus rules locally if you place private files alongside.

## 2026-05-22

### Added
- `docs/deployment.md` — production deployment guide: secrets,
  reverse proxy + TLS, Qdrant snapshot backups, health checks,
  monitoring patterns, resource sizing, multi-tenant separation.
- `docs/writing-a-domain-pack.md` — step-by-step guide to writing a
  domain pack, with an HR policy worked example.
- Optional Postgres backend in `frontend/persistence.py`. Set
  `DATABASE_URL=postgresql://...` to switch from local SQLite to
  Postgres (`pip install -e ".[postgres]"`). API surface unchanged.
- GPU autodetection for the bge-m3 embedder. CUDA → MPS → CPU
  fallback, with fp16 enabled automatically on accelerators.
  `EMBEDDING_DEVICE` and `EMBEDDING_USE_FP16` env vars can force a
  specific configuration.
- Conversation history summarisation in `agent/nodes/answer.py`. When
  `HISTORY_SUMMARY=1` and the chat exceeds `ANSWER_HISTORY_TURNS`,
  earlier turns are compressed into a single ≤ 500-token paragraph;
  the most recent turns stay verbatim. Domain packs can override the
  summariser prompt via the `history_summary_system` slot.
- End-to-end test harness in `tests/e2e/`. Skipped automatically when
  the test Qdrant (port 6334) isn't reachable. `Makefile` gained
  `make e2e`, `make e2e-up`, `make e2e-down`.
- `docker-compose.test.yml` for spinning up the e2e Qdrant.

## 2026-05-21

### Added
- Multi-query expansion in `agent/nodes/orchestrate.py`. When
  `QUERY_REWRITE=1`, each planned sub-task is replaced by 2–3
  semantically equivalent paraphrases for higher retrieval recall.
  Capped at `QUERY_REWRITE_MAX_SUBTASKS` (default 8) and falls back
  to the unrewritten plan on any LLM failure.
- Semantic deduplication in `agent/nodes/synthesis.py`. Embeds the
  top-K chunks with bge-m3 and merges anything above the cosine
  threshold (default 0.92). Disabled in unit tests; enable with
  `SEMANTIC_DEDUP=1`.
- Audit log table in `frontend/persistence.py`. Every Q&A turn is
  recorded (question, evidence, answer, has-unsupported flag,
  metadata filters, timestamp). Survives "Clear History" because
  it's intended for compliance use.
- Advanced filter expander in the Streamlit sidebar. All
  domain-pack `fields` become text inputs; the first two are
  inline and the rest live under an expander.
- `entity_patterns` field on domain packs. Regexes feed into
  `analytics_tools._extract_entities` for second-hop search seeding.
  Patterns are validated at load time.
- `llm/retry.py` — shared `with_retry` helper backing all three
  LLM providers. Eliminates the previously duplicated try/except
  ladders.
- Dynamic `MAX_CONTEXT_TOKENS` defaulting in `config.py`. The
  context budget now scales with the active LLM's window
  (Claude 200 k / GPT-4o 128 k / DeepSeek 64 k / Gemma 8 k / …);
  explicit env values still win.
- PDF text/table deduplication: `parsers/pdf.py` crops table
  bounding boxes out of the prose extract via
  `Page.outside_bbox()`, so the same content no longer enters the
  index twice as both linearised text and `ParsedTable` rows.
- Real-time streaming UI. `agent/nodes/answer.py` forwards every
  LLM token to a `_token_sink` callback in state; the Streamlit
  chat uses an `st.empty()` placeholder to render incrementally,
  then overwrites with the safety-post-processed `final_answer`.
- PDF table extraction via pdfplumber. Falls back to pypdf
  text-only with a warning when pdfplumber isn't installed.
- Robustness: `searcher.get_client()` is now double-checked-lock
  protected; `agent/nodes/worker.py:_run_with_timeout` uses a
  bounded `ThreadPoolExecutor` (no more leaked daemon threads on
  worker timeout); `domain.loader` compiles every regex at load
  time; `safety.py` recognises thousands separators, scientific
  notation, unicode minus, and percentages with whitespace.
- CI workflow at `.github/workflows/test.yml`, `Makefile` (with
  `make help`), top-level `.gitignore`, `requirements.txt` mirror of
  `pyproject.toml`, and `.python-version`.

### Changed
- `agent/tools_new.py` → `agent/analytics_tools.py`; the matching
  test module followed the rename.
- Removed the long-standing backwards-compat alias
  `_extract_chemical_entities = _extract_entities`.

## 2026-05-20

### Added
- Qdrant collection dimension self-check on first `ensure_collection`
  call. Raises a helpful `RuntimeError` instead of a cryptic Qdrant
  error when `EMBEDDING_MODEL` was swapped without recreating the
  collection.

### Fixed
- `frontend/export.py` rewritten to read from `chunk["payload"]` with
  attribution fallback. The previous implementation read fields that
  the indexer never populated, so the Excel export was always blank
  in production.
- `tests/test_worker_node.py` and `tests/test_retry_search_node.py`
  assertions updated to match the actual `_invoke_args` behaviour
  ("omit `filters` key when None") instead of expecting an explicit
  `filters=None` arg.
- `tests/test_run_ingestion.py` `main()` tests now mock
  `retrieval.embedder.embed_texts` and `ingestion.indexer.ensure_collection`
  so they don't try to spin up the real embedder, which fails on
  older torch versions due to the `torch.load` safety check.

## 2026-05-19 to 2026-05-20 — chemistry → general refactor

The project started life as a synthetic-chemistry-only RAG. This
window of work made it domain-agnostic.

### Renamed
- Source directory `chemical-rag/` → `knowledge-rag/`. Qdrant
  collection default, log path, LangSmith project name, and SQLite
  history filename all rebased onto the new name.

### Added
- `knowledge-rag/domain/` package: `DomainPack` dataclass, YAML
  loader, `@lru_cache` accessor. Five configurable axes
  (abbreviations, synonym groups, fields, unit patterns, prompt
  overrides). Empty / missing path returns an empty pack; malformed
  YAML raises `ValueError`.
- `knowledge-rag/domain/examples/chemistry.yaml` — the original
  chemistry behaviour, now opt-in. Activate by setting
  `DOMAIN_PACK_PATH=domain/examples/chemistry.yaml`.
- Multi-format ingestion: `ingestion/parser.parse_document(path)`
  dispatches by extension to per-format backends in
  `ingestion/parsers/` — `markdown.py` (markdown-it-py), `text.py`,
  `pdf.py`, `html.py` (BeautifulSoup with chrome stripping). All
  backends emit the same `ParsedDocument` shape.

### Changed
- `ingestion/normalizer.py`: hard-coded `ABBREVIATION_MAP`,
  `SYNONYM_GROUPS`, and unit-pattern constants removed. Both
  `normalize_text` and `expand_synonyms` consult the active domain
  pack at call time; empty pack = passthrough.
- `agent/safety.py`: chemistry entity set replaced with a
  `_known_entities()` function reading from the pack. Empty pack
  disables the entity check; numeric claim check remains
  domain-agnostic.
- `agent/nodes/worker.py`: `_CHEMISTRY_FIELDS` constant replaced
  with `_domain_fields()` from the pack.
- All six LangGraph prompts (`answer_system`,
  `answer_comparison_system`, `answer_trend_system`,
  `classify_system`, `plan_system`, `critic_system`) neutralised
  and made overridable via `pack.prompt_overrides`.
- `agent/tools.py` docstrings and filter examples neutralised.
- `frontend/app.py`: UI titles, captions, placeholders, page icon
  (🧪 → 📚) neutralised. Sidebar filter inputs driven by
  `pack.fields` with a generic Category / Type fallback.
- `llm/anthropic_provider.py`: removed the "synthetic chemistry
  assistant" wording from the default system prompt.

### Removed
- `chemical-rag-system-en.md` and `chemical-rag-system-zh.md` design
  docs.
- Chemistry questions from `evaluation/eval_set/qa_gold.json`;
  replaced with two schema placeholder entries that domain experts
  fill in.

### Tests
- `tests/fixtures/create_fixture.py` rewritten to produce a
  neutral business `sample.docx`.
- `tests/test_chunker.py` and `tests/test_parser.py` rewritten
  against the current API (`ParsedRow` rather than the long-since-
  renamed `TableRow`; `ParsedParagraph` no longer carries a `style`
  field). Eliminated two long-standing collection errors.
- `tests/conftest.py` default question made neutral.
