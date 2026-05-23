# knowledge-rag/agent/analytics_tools.py
import re
from retrieval.searcher import hybrid_search
from retrieval.reranker import rerank
from domain.cache import get_domain_pack


def _known_abbrevs() -> frozenset[str]:
    """Domain-pack abbreviations whose key is uppercase and at least 2 chars long."""
    return frozenset(
        k for k in get_domain_pack().abbreviations if k == k.upper() and len(k) >= 2
    )


# Always-on uppercase-code pattern (e.g. PRJ-031, LOT-A2). Captures generic
# project / batch / SKU identifiers in any domain. Domain packs add
# additional patterns via ``entity_patterns`` (e.g. ICD codes, statute refs,
# °C readings — anything the domain wants to use for second-hop queries).
_GENERIC_CODE_PATTERN = r'\b[A-Z]{2,}-?\d+\b'


def _extract_entities(text: str) -> list[str]:
    """Extract domain entities + uppercase codes from text for second-hop queries.

    Picks up, in order: (1) known abbreviations declared in the active domain
    pack, (2) custom regex patterns from ``domain_pack.entity_patterns``
    (e.g. ``\\d+\\s*°C`` for temperatures), (3) the always-on uppercase code
    pattern. With an empty pack, only (3) fires.
    """
    pack = get_domain_pack()
    found: list[str] = []

    for abbr in _known_abbrevs():
        if re.search(rf'\b{re.escape(abbr)}\b', text):
            found.append(abbr)

    for pattern in pack.entity_patterns:
        for m in re.finditer(pattern, text):
            found.append(m.group())

    for m in re.finditer(_GENERIC_CODE_PATTERN, text):
        found.append(m.group())

    return list(dict.fromkeys(found))  # deduplicate preserving order


def extract_structured_data(chunks: list[dict], fields: list[str]) -> list[dict]:
    """Extract structured field=value pairs from chunks.

    Prefers structured_fields stored in chunk payload (exact values from table cells).
    Falls back to regex on chunk text when payload field is absent.
    """
    extracted = []
    for chunk in chunks:
        attribution = chunk.get("attribution", "")
        structured_fields = chunk.get("payload", {}).get("structured_fields", {})
        text = chunk.get("text", "")

        for field in fields:
            # Prefer exact payload value — case-insensitive key match
            payload_value = None
            for k, v in structured_fields.items():
                if k.lower() == field.lower():
                    payload_value = v
                    break

            if payload_value is not None:
                # Parse numeric value + unit from the stored string (e.g. "87%", "80 °C")
                m = re.match(r'(-?\d+\.?\d*)\s*(%|°C|mol%|eq\.?|mL|g|L)?', payload_value.strip())
                if m:
                    extracted.append({
                        "field": field,
                        "value": m.group(1),
                        "unit": (m.group(2) or "").strip(),
                        "attribution": attribution,
                    })
                else:
                    # Non-numeric value (e.g. catalyst name) — store as-is
                    extracted.append({
                        "field": field,
                        "value": payload_value,
                        "unit": "",
                        "attribution": attribution,
                    })
            else:
                # Fallback: regex on text
                pattern = rf'\b{re.escape(field)}\s*[=:]\s*(-?\d+\.?\d*)\s*(%|°C|mol%|eq\.?|mL|g|L)?'
                for val, unit in re.findall(pattern, text, re.IGNORECASE):
                    extracted.append({
                        "field": field,
                        "value": val,
                        "unit": unit.strip() if unit else "",
                        "attribution": attribution,
                    })
    return extracted


def _find_structured_value(structured_fields: dict, field: str) -> str | None:
    for key, value in structured_fields.items():
        if key.lower() == field.lower():
            return str(value)
    return None


def _parse_numeric_value(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r'-?\d+\.?\d*', str(value).strip())
    if not match:
        return None
    return float(match.group())


def _series_pairs(chunks: list[dict], metric: str, independent_var: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    if not independent_var:
        return pairs

    for chunk in chunks:
        structured_fields = chunk.get("payload", {}).get("structured_fields", {})
        if not structured_fields:
            continue
        x_raw = _find_structured_value(structured_fields, independent_var)
        y_raw = _find_structured_value(structured_fields, metric)
        x_val = _parse_numeric_value(x_raw)
        y_val = _parse_numeric_value(y_raw)
        if x_val is None or y_val is None:
            continue
        pairs.append((x_val, y_val))
    return pairs


def _trend_from_series(pairs: list[tuple[float, float]]) -> str:
    if len(pairs) < 2:
        return "undetermined"

    ordered = sorted(pairs, key=lambda item: item[0])
    first_y = ordered[0][1]
    last_y = ordered[-1][1]
    baseline = max(abs(first_y), 1e-6)
    delta_ratio = (last_y - first_y) / baseline
    if delta_ratio > 0.05:
        return "increasing"
    if delta_ratio < -0.05:
        return "decreasing"
    return "stable"


def statistical_summary(chunks: list[dict], metric: str, independent_var: str = "") -> str:
    """Compute min/max/mean and a trend label for a metric across chunks."""
    if not metric:
        return ""

    numbers = []
    for chunk in chunks:
        structured_fields = chunk.get("payload", {}).get("structured_fields", {})
        text = chunk.get("text", "")

        # Prefer exact payload value — case-insensitive key match
        payload_value = _find_structured_value(structured_fields, metric)

        if payload_value is not None:
            numeric_value = _parse_numeric_value(payload_value)
            if numeric_value is not None:
                numbers.append(numeric_value)
        else:
            # Fallback: regex on text
            pattern = rf'\b{re.escape(metric)}\s*[=:]\s*(\d+\.?\d*)'
            numbers.extend(float(m) for m in re.findall(pattern, text, re.IGNORECASE))

    if not numbers:
        return ""

    mn, mx, avg = min(numbers), max(numbers), sum(numbers) / len(numbers)
    trend = _trend_from_series(_series_pairs(chunks, metric, independent_var))
    trend_suffix = f", trend_vs_{independent_var}={trend}" if independent_var else f", trend={trend}"

    return (
        f"Statistical summary for '{metric}': "
        f"n={len(numbers)}, min={mn:.4g}, max={mx:.4g}, mean={avg:.4g}{trend_suffix}"
    )


def multi_hop_search(initial_query: str, follow_up_hints: list[str], filters: dict | None = None) -> list[dict]:
    """Two-hop retrieval: initial search, extract entities, then targeted follow-up."""
    first_results = hybrid_search(initial_query, enable_rerank=False, filters=filters)
    first_ranked = rerank(initial_query, first_results)

    entity_text = " ".join(r.text for r in first_ranked[:3])
    entities = _extract_entities(entity_text)

    second_query_parts = (follow_up_hints or [])[:2] + entities[:3]
    if second_query_parts:
        second_query = " ".join(second_query_parts)
        second_results = hybrid_search(second_query, enable_rerank=False, filters=filters)
        second_ranked = rerank(second_query, second_results)
        all_results = list(first_ranked) + list(second_ranked)
    else:
        all_results = list(first_ranked)

    seen, merged = set(), []
    for r in all_results:
        key = r.attribution
        if key not in seen:
            seen.add(key)
            merged.append({
                "text": r.text,
                "attribution": r.attribution,
                "score": r.score,
                "payload": r.payload,
            })
    return merged
