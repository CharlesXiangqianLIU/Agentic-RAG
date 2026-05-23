"""
Ingest documents from a directory into Qdrant.

Usage:
    python -m ingestion.run_ingestion --docs-dir ./data/reports

Supported formats: ``.docx``, ``.md`` / ``.markdown``, ``.txt``, ``.pdf``,
``.html`` / ``.htm`` — all routed through ``ingestion.parser.parse_document``.

Optional: place a sidecar JSON file alongside each source document to inject
arbitrary metadata that the agent can later filter on. Example:

    handbook_2024.pdf
    handbook_2024.json  ←  {"category": "policy", "doc_type": "handbook", "year": 2024}
"""
import argparse
import concurrent.futures
import json
import logging
import threading
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

from ingestion.parser import parse_document, SUPPORTED_EXTENSIONS
from ingestion.chunker import chunk_document
from ingestion.indexer import index_chunks
from ingestion.state_tracker import load_state, save_state, is_changed, compute_md5

# Thread-safe lock for updating shared state dict
_state_lock = threading.Lock()


def ingest_file(path: Path) -> int:
    """Parse, chunk, and index a single source document. Returns chunk count."""
    doc = parse_document(path)
    chunks = chunk_document(doc)

    # Load sidecar metadata if present
    sidecar = path.with_suffix(".json")
    metadata: dict = {}
    if sidecar.exists():
        with open(sidecar) as f:
            metadata = json.load(f)

    for chunk in chunks:
        chunk.metadata.setdefault("project_id", path.stem)
        chunk.metadata.update(metadata)

    index_chunks(chunks)
    return len(chunks)


def _process_file(path: Path, i: int, total: int, force: bool, state: dict) -> tuple[str, int]:
    """Process one file; return (status_line, chunk_count). Thread-safe."""
    if not force and not is_changed(path, state):
        return (f"  [{i}/{total}] {path.name} → [SKIP] unchanged", 0)

    label = "[NEW]" if path.name not in state else "[UPDATED]"
    try:
        n = ingest_file(path)
        md5 = compute_md5(path)
        with _state_lock:
            state[path.name] = md5
        return (f"  [{i}/{total}] {path.name} → {label} {n} chunks", n)
    except Exception as e:
        return (f"  [{i}/{total}] {path.name} → ERROR: {e}", 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents from a directory into Qdrant"
    )
    parser.add_argument(
        "--docs-dir",
        required=True,
        help="Directory containing source documents (searched recursively). "
             f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all files even if unchanged",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel ingestion workers",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        logger.error("Error: directory not found: %s", docs_dir)
        raise SystemExit(1)

    source_files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        source_files.extend(docs_dir.rglob(f"*{ext}"))
    source_files = sorted(set(source_files))
    logger.info("Found %d source file(s) in %s", len(source_files), docs_dir)

    # Pre-warm the embedding model in the main thread before spawning workers.
    # FlagEmbedding initialises PyTorch weights on first call; doing this once
    # here avoids "meta tensor" errors when multiple threads try to load the
    # model simultaneously.
    logger.info("Loading embedding model (first run downloads ~1.7 GB)...")
    from retrieval.embedder import embed_texts as _warmup_embed
    _warmup_embed(["warmup"])
    logger.info("Embedding model ready.")

    # Pre-create the Qdrant collection before parallel workers start.
    from ingestion.indexer import ensure_collection
    ensure_collection()

    state = {} if args.force else load_state(docs_dir)

    total_chunks = 0
    skipped = 0
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_file, path, i, len(source_files), args.force, state): path
                   for i, path in enumerate(source_files, 1)}
        with tqdm(total=len(futures), desc="Ingesting", unit="file") as pbar:
            for future in concurrent.futures.as_completed(futures):
                line, n = future.result()
                logger.info(line)
                if "[SKIP]" in line:
                    skipped += 1
                elif "ERROR" in line:
                    errors += 1
                else:
                    total_chunks += n
                pbar.update(1)

    save_state(docs_dir, state)
    logger.info("Done. Processed: %d chunks, Skipped: %d file(s), Errors: %d", total_chunks, skipped, errors)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
