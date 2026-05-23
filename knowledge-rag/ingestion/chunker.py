# ingestion/chunker.py
from dataclasses import dataclass, field
import tiktoken
from ingestion.parser import ParsedDocument, ParsedTable
from ingestion.normalizer import normalize_text, expand_synonyms
from config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS

_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class Chunk:
    text: str
    source_file: str
    page_number: int
    section: str
    chunk_type: str             # "paragraph" | "table_row"
    metadata: dict = field(default_factory=dict)
    synonyms: list[str] = field(default_factory=list)


def _table_to_chunks(table: ParsedTable, source_file: str) -> list[Chunk]:
    """Each data row (after header) becomes its own chunk, with header prepended for context."""
    if not table.rows:
        return []
    header_row = table.rows[0]
    header_cells = header_row.cells
    header = header_row.to_text()
    chunks = []
    for row in table.rows[1:]:
        text = normalize_text(f"{header}\n{row.to_text()}")
        syns: list[str] = []
        for cell in row.cells:
            syns.extend(expand_synonyms(cell.strip()))

        # Extract structured fields by zipping header cells with data row cells
        structured_fields: dict[str, str] = {}
        for header_cell, data_cell in zip(header_cells, row.cells):
            header_str = header_cell.strip()
            data_str = normalize_text(data_cell.strip())
            # Skip cells where header or value is empty string
            if header_str and data_str:
                structured_fields[header_str] = data_str

        chunks.append(Chunk(
            text=text,
            source_file=source_file,
            page_number=row.page_number,
            section=table.section,
            chunk_type="table_row",
            synonyms=list(set(syns)),
            metadata={"structured_fields": structured_fields},
        ))
    return chunks


def _paragraphs_to_chunks(doc: ParsedDocument) -> list[Chunk]:
    """Merge paragraphs into token-bounded chunks, flushing at section boundaries.

    When a chunk overflows, the last CHUNK_OVERLAP_TOKENS worth of text from the
    previous chunk are carried over as a sliding-window overlap so cross-boundary
    context is preserved for retrieval.
    """
    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_tokens = 0
    current_section = "Introduction"
    current_page = 1
    _overlap_tail: list[str] = []  # texts carried over from the previous chunk

    def flush() -> None:
        nonlocal current_texts, current_tokens, _overlap_tail
        if current_texts:
            chunks.append(Chunk(
                text=normalize_text(" ".join(current_texts)),
                source_file=doc.filename,
                page_number=current_page,
                section=current_section,
                chunk_type="paragraph",
            ))
            # Compute overlap tail: walk back through current_texts until we have
            # CHUNK_OVERLAP_TOKENS tokens (or exhaust all texts).
            if CHUNK_OVERLAP_TOKENS > 0:
                tail: list[str] = []
                tail_tokens = 0
                for t in reversed(current_texts):
                    t_tok = _token_count(t)
                    if tail_tokens + t_tok > CHUNK_OVERLAP_TOKENS:
                        break
                    tail.insert(0, t)
                    tail_tokens += t_tok
                _overlap_tail = tail
            else:
                _overlap_tail = []
        current_texts.clear()
        current_tokens = 0

    for para in doc.paragraphs:
        if para.is_header:
            flush()
            _overlap_tail = []  # section boundary — no overlap across sections
            current_section = para.text
            current_page = para.page_number
            continue

        tokens = _token_count(para.text)
        if current_tokens + tokens > CHUNK_MAX_TOKENS:
            flush()
            current_page = para.page_number
            # Seed new chunk with overlap from previous chunk
            if _overlap_tail:
                current_texts.extend(_overlap_tail)
                current_tokens = sum(_token_count(t) for t in _overlap_tail)

        current_texts.append(para.text)
        current_tokens += tokens

    flush()
    return chunks


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunks.extend(_paragraphs_to_chunks(doc))
    for table in doc.tables:
        chunks.extend(_table_to_chunks(table, doc.filename))
    return chunks
