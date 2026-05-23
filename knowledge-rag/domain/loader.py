"""Domain pack loader.

A domain pack is a YAML file with this shape (all keys optional):

    abbreviations:
      DCM: Dichloromethane
    synonym_groups:
      - [DCM, Dichloromethane, methylene chloride]
    fields:
      - yield
      - temperature
    unit_patterns:
      - ["(\\d+)\\s*deg\\s*[Cc]", "\\1 \\u00b0C"]
    prompt_overrides:
      answer_system: "You are an expert in <my domain>. ..."
      classify_system: "..."

If the path is empty, missing, or unreadable, an empty DomainPack is
returned. A malformed YAML file raises a ValueError so the operator
notices the misconfiguration.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class DomainPack:
    abbreviations: dict[str, str] = field(default_factory=dict)
    synonym_groups: list[frozenset[str]] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    unit_patterns: list[tuple[str, str]] = field(default_factory=list)
    # Regex patterns used by analytics_tools._extract_entities for second-hop
    # query construction. Each pattern's matches are appended to the entity
    # list as-is. Empty list = entity extraction relies only on bare uppercase
    # codes (e.g. PRJ-031) which are domain-agnostic and always on.
    entity_patterns: list[str] = field(default_factory=list)
    prompt_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (
            self.abbreviations
            or self.synonym_groups
            or self.fields
            or self.unit_patterns
            or self.entity_patterns
            or self.prompt_overrides
        )


def load_domain_pack(path: str | Path | None) -> DomainPack:
    """Load a domain pack from a YAML file.

    Empty/None path, missing file, or unreadable file -> empty DomainPack.
    Malformed YAML (parser error or wrong shape) -> ValueError.
    """
    if not path:
        return DomainPack()
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        log.warning("Domain pack path %s does not exist; using empty pack.", resolved)
        return DomainPack()

    import yaml  # local import: only required when a pack is configured

    try:
        with resolved.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse domain pack {resolved}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Domain pack {resolved} must be a YAML mapping at the top level.")

    return DomainPack(
        abbreviations=_parse_abbreviations(data.get("abbreviations") or {}),
        synonym_groups=_parse_synonym_groups(data.get("synonym_groups") or []),
        fields=_parse_fields(data.get("fields") or []),
        unit_patterns=_parse_unit_patterns(data.get("unit_patterns") or []),
        entity_patterns=_parse_entity_patterns(data.get("entity_patterns") or []),
        prompt_overrides=_parse_prompt_overrides(data.get("prompt_overrides") or {}),
    )


def _parse_abbreviations(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("`abbreviations` must be a mapping of abbreviation -> full form.")
    return {str(k): str(v) for k, v in raw.items()}


def _parse_synonym_groups(raw: object) -> list[frozenset[str]]:
    if not isinstance(raw, list):
        raise ValueError("`synonym_groups` must be a list of lists of strings.")
    groups: list[frozenset[str]] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple, set)):
            raise ValueError(f"synonym_group entry must be a list of strings, got {type(entry).__name__}")
        groups.append(frozenset(str(item) for item in entry))
    return groups


def _parse_fields(raw: object) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError("`fields` must be a list of strings.")
    return [str(item) for item in raw]


def _parse_unit_patterns(raw: object) -> list[tuple[str, str]]:
    if not isinstance(raw, list):
        raise ValueError("`unit_patterns` must be a list of [pattern, replacement] pairs.")
    pairs: list[tuple[str, str]] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"unit_patterns entry must be a [pattern, replacement] pair, got {entry!r}")
        pattern, replacement = str(entry[0]), str(entry[1])
        # Compile at load time so configuration errors surface immediately
        # rather than the first time normalize_text() is called on real data.
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"unit_patterns entry has an invalid regex {pattern!r}: {exc}"
            ) from exc
        pairs.append((pattern, replacement))
    return pairs


def _parse_entity_patterns(raw: object) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError("`entity_patterns` must be a list of regex strings.")
    patterns: list[str] = []
    for entry in raw:
        pattern = str(entry)
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"entity_patterns has an invalid regex {pattern!r}: {exc}"
            ) from exc
        patterns.append(pattern)
    return patterns


def _parse_prompt_overrides(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("`prompt_overrides` must be a mapping of slot_name -> prompt text.")
    return {str(k): str(v) for k, v in raw.items()}
