# agent/safety.py
"""Post-process LLM answers: flag claims not supported by source chunks.

Two checks run on each sentence:
  1. Numeric claims — any number in the sentence must appear in the sources.
     Recognises plain integers, decimals, thousands grouping (``1,234.56``),
     scientific notation (``1.5e3``), negatives (``-15``, unicode minus
     ``−15``) and percentages with optional whitespace (``87 %``). The
     comparison strips thousand separators and normalises unicode minus
     on both sides so number formatting differences don't cause false
     positives. This check is domain-agnostic and always on.
  2. Domain-entity claims — abbreviations declared in the active domain
     pack (e.g. "DCM" in a chemistry pack) must appear in the sources.
     With an empty pack, this check is silently skipped.
"""
import re

from domain.cache import get_domain_pack


# A number token: optional sign (ascii ``+/-`` or unicode minus), digits with
# optional thousands grouping, optional decimal, optional scientific notation,
# optional whitespace + ``%``. Stays a single capture so ``findall`` returns
# the whole matched string per occurrence.
_NUMBER_RE = re.compile(
    r"[+\-−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+\-]?\d+)?\s*%?"
)


def _known_entities() -> frozenset[str]:
    return frozenset(get_domain_pack().abbreviations.keys())


def _check_entities(sentence: str, source_text: str) -> bool:
    """Return True if sentence contains a known domain entity absent from source_text."""
    for entity in _known_entities():
        # Word-boundary match (case-sensitive — abbreviations are usually case-bearing)
        if re.search(r'\b' + re.escape(entity) + r'\b', sentence):
            if not re.search(r'\b' + re.escape(entity) + r'\b', source_text, re.IGNORECASE):
                return True
    return False


def _normalize_number(s: str) -> str:
    """Strip whitespace, lowercase, drop thousand separators, normalise minus."""
    return s.replace(",", "").replace("−", "-").strip().lower()


def _number_appears_in(number: str, source_lower: str, source_no_commas: str) -> bool:
    """Return True if a number token from the answer appears in the source text.

    Tries the raw form first, then a form with thousand separators stripped,
    so ``1,234`` in the answer matches either ``1,234`` or ``1234`` in the
    source (and vice versa).
    """
    raw = number.strip().lower()
    if raw and raw in source_lower:
        return True
    normalised = _normalize_number(number)
    return bool(normalised) and normalised in source_no_commas


def check_answer(answer: str, chunks: list[dict]) -> str:
    source_lower = " ".join(c.get("text", "") for c in chunks).lower()
    source_no_commas = source_lower.replace(",", "").replace("−", "-")
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    result = []
    for sentence in sentences:
        numbers = _NUMBER_RE.findall(sentence)
        numbers_unsupported = numbers and not any(
            _number_appears_in(n, source_lower, source_no_commas) for n in numbers
        )
        entities_unsupported = _check_entities(sentence, source_lower)
        if numbers_unsupported or entities_unsupported:
            result.append(f"[UNSUPPORTED: {sentence.strip()}]")
        else:
            result.append(sentence)
    return " ".join(result)
