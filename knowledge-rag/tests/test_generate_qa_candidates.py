import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluation.generate_qa_candidates import (
    _extract_source_file,
    _generate_qa,
    _sample_chunks,
    generate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_result(attribution: str, text: str = "Some passage text."):
    r = MagicMock()
    r.attribution = attribution
    r.text = text
    r.payload = {}
    return r


def _make_llm(response: str):
    llm = MagicMock()
    llm.complete.return_value = response
    return llm


_VALID_JSON_RESPONSE = json.dumps(
    {
        "question": "What is the optimal temperature for this reaction?",
        "suggested_answer": "The optimal temperature is 80 °C.",
        "type": "parameter_lookup",
    }
)

_ATTRIBUTION = "[Source: report_001.docx | Page 3 | Section: Results]"


# ---------------------------------------------------------------------------
# Test 1: hybrid_search is called and output list has 1 item
# ---------------------------------------------------------------------------

def test_generate_candidates_calls_hybrid_search(tmp_path):
    result = _make_search_result(_ATTRIBUTION)

    with (
        patch("evaluation.generate_qa_candidates.hybrid_search", return_value=[result]) as mock_hs,
        patch("evaluation.generate_qa_candidates.get_llm_provider", return_value=_make_llm(_VALID_JSON_RESPONSE)),
    ):
        output_file = tmp_path / "qa_candidates.json"
        items = generate(n=1, output_path=output_file)

    assert mock_hs.called
    assert len(items) == 1


# ---------------------------------------------------------------------------
# Test 2: invalid JSON response → item is skipped, output list is empty
# ---------------------------------------------------------------------------

def test_generate_candidates_skips_invalid_json(tmp_path):
    result = _make_search_result(_ATTRIBUTION)

    with (
        patch("evaluation.generate_qa_candidates.hybrid_search", return_value=[result]),
        patch("evaluation.generate_qa_candidates.get_llm_provider", return_value=_make_llm("This is not JSON at all.")),
    ):
        output_file = tmp_path / "qa_candidates.json"
        items = generate(n=1, output_path=output_file)

    assert items == []


# ---------------------------------------------------------------------------
# Test 3: duplicate attributions → only 1 chunk is processed
# ---------------------------------------------------------------------------

def test_generate_candidates_deduplicates_chunks(tmp_path):
    r1 = _make_search_result(_ATTRIBUTION, text="First passage.")
    r2 = _make_search_result(_ATTRIBUTION, text="Duplicate passage.")

    with (
        patch("evaluation.generate_qa_candidates.hybrid_search", return_value=[r1, r2]),
        patch("evaluation.generate_qa_candidates.get_llm_provider", return_value=_make_llm(_VALID_JSON_RESPONSE)),
    ):
        output_file = tmp_path / "qa_candidates.json"
        items = generate(n=10, output_path=output_file)

    # Only 1 unique attribution → only 1 item processed
    assert len(items) == 1


# ---------------------------------------------------------------------------
# Test 4: output dict has the correct required fields
# ---------------------------------------------------------------------------

def test_generate_candidates_output_has_correct_fields(tmp_path):
    result = _make_search_result(_ATTRIBUTION)

    with (
        patch("evaluation.generate_qa_candidates.hybrid_search", return_value=[result]),
        patch("evaluation.generate_qa_candidates.get_llm_provider", return_value=_make_llm(_VALID_JSON_RESPONSE)),
    ):
        output_file = tmp_path / "qa_candidates.json"
        items = generate(n=1, output_path=output_file)

    assert len(items) == 1
    entry = items[0]
    for field in ("id", "type", "question", "suggested_answer", "source_file", "expected_answer"):
        assert field in entry, f"Missing field: {field}"
