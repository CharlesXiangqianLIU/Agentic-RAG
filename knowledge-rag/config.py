import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler as _RotatingFileHandler

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_rag")
QDRANT_TIMEOUT_SECONDS = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "10"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_TIMEOUT_SECONDS = float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "60"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-32B")
VLLM_TIMEOUT_SECONDS = float(os.getenv("VLLM_TIMEOUT_SECONDS", "60"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:27b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
DOCS_DIR = os.getenv("DOCS_DIR", str(Path(__file__).parent / "data" / "reports"))
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", str(Path.home() / ".knowledge_rag.log"))
DOMAIN_PACK_PATH = os.getenv("DOMAIN_PACK_PATH", "")

CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "300"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "600"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
CRITIC_MAX_ROUNDS = int(os.getenv("CRITIC_MAX_ROUNDS", "3"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.3"))
SYNTHESIS_TOP_K = int(os.getenv("SYNTHESIS_TOP_K", "30"))


def _model_context_window(model_id: str) -> int:
    """Best-effort lookup of an LLM's input context window in tokens.

    Returns 0 when the model is unknown — the caller treats that as a
    signal to fall back to the legacy 8 000-token default.
    """
    name = (model_id or "").lower()
    if "claude" in name:
        return 200_000          # Anthropic Claude 4.x family
    if "gpt-4o" in name or "gpt-4-turbo" in name:
        return 128_000
    if "gpt-4" in name:
        return 8_192
    if "deepseek" in name:
        return 64_000           # DeepSeek-R1 family
    if "gemma" in name:
        return 8_192
    if "llama" in name or "qwen" in name:
        return 32_000
    return 0


def _default_context_budget() -> int:
    """Pick a sensible MAX_CONTEXT_TOKENS based on the active LLM_PROVIDER.

    Uses ~40 % of the model's window so the remainder can hold the
    system prompt, the question, conversation history, and the LLM's
    own output without overflow.
    """
    if LLM_PROVIDER == "anthropic":
        model = ANTHROPIC_MODEL
    elif LLM_PROVIDER == "deepseek":
        model = VLLM_MODEL
    elif LLM_PROVIDER == "ollama":
        model = OLLAMA_MODEL
    else:
        model = ""
    window = _model_context_window(model)
    if window <= 0:
        return 8_000
    return max(8_000, int(window * 0.4))


# Explicit env var wins; otherwise scale to the active LLM's window so
# claude-sonnet-4-6 (200k) doesn't keep behaving like an 8k-token model.
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", str(_default_context_budget())))
ANSWER_HISTORY_TURNS = int(os.getenv("ANSWER_HISTORY_TURNS", "2"))
ANSWER_HISTORY_MAX_TOKENS = int(os.getenv("ANSWER_HISTORY_MAX_TOKENS", "1200"))
INGESTION_BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", "32"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
GRAPH_TIMEOUT_SECONDS = int(os.getenv("GRAPH_TIMEOUT_SECONDS", "120"))
WORKER_TIMEOUT_SECONDS = int(os.getenv("WORKER_TIMEOUT_SECONDS", "30"))
HISTORY_PAGE_SIZE = int(os.getenv("HISTORY_PAGE_SIZE", "20"))
ORCHESTRATE_CLASSIFY_TIMEOUT_SECONDS = float(os.getenv("ORCHESTRATE_CLASSIFY_TIMEOUT_SECONDS", "15"))
ORCHESTRATE_PLAN_TIMEOUT_SECONDS = float(os.getenv("ORCHESTRATE_PLAN_TIMEOUT_SECONDS", "20"))
ANSWER_REQUEST_TIMEOUT_SECONDS = float(os.getenv("ANSWER_REQUEST_TIMEOUT_SECONDS", "45"))
CRITIC_REQUEST_TIMEOUT_SECONDS = float(os.getenv("CRITIC_REQUEST_TIMEOUT_SECONDS", "20"))

# LangSmith / LangChain tracing (optional — set env vars to enable)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "knowledge-rag")

import sys as _sys


class _StdoutHandler(logging.StreamHandler):
    """StreamHandler that resolves sys.stdout lazily so pytest capsys works."""

    def emit(self, record):
        self.stream = _sys.stdout
        super().emit(record)


_handler = _StdoutHandler()
_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
))
logging.root.setLevel(logging.INFO)
# Add our stdout handler if not already present (idempotent on re-import)
if not any(isinstance(h, _StdoutHandler) for h in logging.root.handlers):
    logging.root.addHandler(_handler)

# Add rotating file handler (idempotent on re-import; silently skipped if path is unwritable)
try:
    _file_handler = _RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    if not any(isinstance(h, _RotatingFileHandler) for h in logging.root.handlers):
        logging.root.addHandler(_file_handler)
except Exception:
    pass
