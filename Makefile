PKG := knowledge-rag
PY  := python -m

.PHONY: help install test test-fast lint ui ingest eval clean e2e e2e-up e2e-down

help:  ## Show this help.
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install knowledge-rag in editable mode with dev extras.
	cd $(PKG) && pip install -e ".[dev]"

test:  ## Run the full unit/mocked pytest suite.
	cd $(PKG) && $(PY) pytest tests/ -q --ignore=tests/e2e

test-fast:  ## Run pytest but skip the slow integration tests.
	cd $(PKG) && $(PY) pytest tests/ -q --ignore=tests/e2e -m "not slow"

e2e:  ## Run the end-to-end suite (requires a live Qdrant; see make e2e-up).
	cd $(PKG) && $(PY) pytest tests/e2e -v

e2e-up:  ## Start the e2e Qdrant in the background on port 6334.
	cd $(PKG) && docker compose -f docker-compose.test.yml up -d qdrant

e2e-down:  ## Tear down the e2e Qdrant + its volume.
	cd $(PKG) && docker compose -f docker-compose.test.yml down -v

lint:  ## Sanity-grep for residual chemistry hard-coding outside the example pack.
	@! grep -rln 'chemical_rag\|chemical-rag' $(PKG) --include='*.py' --include='*.toml' --include='*.yml'

ui:  ## Launch the Streamlit chat interface.
	cd $(PKG) && streamlit run frontend/app.py

ingest:  ## Ingest all documents under knowledge-rag/data/reports/.
	cd $(PKG) && $(PY) ingestion.run_ingestion --docs-dir ./data/reports

eval:  ## Run the Ragas evaluation over evaluation/eval_set/qa_gold.json.
	cd $(PKG) && $(PY) evaluation.ragas_runner

clean:  ## Remove caches and build artifacts.
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
