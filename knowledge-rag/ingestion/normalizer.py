"""Domain-aware text normalization and synonym expansion.

`normalize_text` and `expand_synonyms` both consult the active
`DomainPack` (loaded once per process via `domain.cache.get_domain_pack`).
With the default (empty) pack, `normalize_text` returns the input
unchanged and `expand_synonyms` returns just the input term — making
the knowledge-rag pipeline domain-agnostic out of the box.

To inject domain knowledge (abbreviations, synonym groups, unit
patterns), point `DOMAIN_PACK_PATH` at a YAML file. See
`domain/examples/chemistry.yaml` for a worked example.
"""
from __future__ import annotations

import re

from domain.cache import get_domain_pack


def normalize_text(text: str) -> str:
    """Normalize units and expand abbreviations using the active domain pack."""
    pack = get_domain_pack()
    for pattern, replacement in pack.unit_patterns:
        text = re.sub(pattern, replacement, text)
    for abbr, full in pack.abbreviations.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return text


def expand_synonyms(term: str) -> list[str]:
    """Return all known synonyms for a term, including the term itself.

    Empty pack -> always returns ``[term]``.
    """
    for group in get_domain_pack().synonym_groups:
        if term in group:
            return list(group)
    return [term]
