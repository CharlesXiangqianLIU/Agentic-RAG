# knowledge-rag/tests/test_analytics_tools.py
from unittest.mock import patch, MagicMock
from agent.analytics_tools import extract_structured_data, statistical_summary, multi_hop_search


def test_extract_yield_from_chunks():
    chunks = [
        {"text": "yield: 87%", "attribution": "[Source: a.docx | Page 1 | Section: T1]"},
        {"text": "temperature: 80°C", "attribution": "[Source: a.docx | Page 1 | Section: T1]"},
    ]
    result = extract_structured_data(chunks, ["yield", "temperature"])
    assert any(e["field"] == "yield" and e["value"] == "87" for e in result)
    assert any(e["field"] == "temperature" and e["value"] == "80" for e in result)


def test_extract_returns_empty_for_no_matches():
    chunks = [{"text": "The reaction was complete.", "attribution": ""}]
    result = extract_structured_data(chunks, ["yield"])
    assert result == []


def test_extract_includes_attribution():
    chunks = [{"text": "yield = 92%", "attribution": "[Source: b.docx | Page 3 | Section: S2]"}]
    result = extract_structured_data(chunks, ["yield"])
    assert result[0]["attribution"] == "[Source: b.docx | Page 3 | Section: S2]"


def test_statistical_summary_computes_stats():
    chunks = [{"text": "yield: 70%"}, {"text": "yield: 80%"}, {"text": "yield: 90%"}]
    result = statistical_summary(chunks, "yield")
    assert "min=70" in result
    assert "max=90" in result
    assert "mean=80" in result


def test_statistical_summary_returns_empty_for_no_data():
    chunks = [{"text": "No numeric data here."}]
    result = statistical_summary(chunks, "yield")
    assert result == ""


def test_statistical_summary_detects_increasing_trend():
    chunks = [
        {"payload": {"structured_fields": {"Time": "1 h", "Yield": "60%"}}},
        {"payload": {"structured_fields": {"Time": "2 h", "Yield": "61%"}}},
        {"payload": {"structured_fields": {"Time": "3 h", "Yield": "85%"}}},
        {"payload": {"structured_fields": {"Time": "4 h", "Yield": "90%"}}},
    ]
    result = statistical_summary(chunks, "yield", "time")
    assert "trend_vs_time=increasing" in result


def test_statistical_summary_without_independent_variable_is_undetermined():
    chunks = [
        {"text": "yield: 60%"},
        {"text": "yield: 90%"},
    ]
    result = statistical_summary(chunks, "yield")
    assert "trend=undetermined" in result


def _make_mock_result(text, attribution):
    r = MagicMock()
    r.text = text
    r.attribution = attribution
    r.score = 0.9
    r.payload = {}
    return r


def test_multi_hop_search_returns_merged_results():
    mock_r = _make_mock_result("Entry 3 yield 87%", "[Source: a.docx | Page 1 | Section: T1]")
    with patch("agent.analytics_tools.hybrid_search") as mock_search, \
         patch("agent.analytics_tools.rerank") as mock_rerank:
        mock_search.return_value = [mock_r]
        mock_rerank.return_value = [mock_r]
        result = multi_hop_search("why did yield drop", ["catalyst question"])
    assert len(result) >= 1
    assert result[0]["text"] == "Entry 3 yield 87%"


def test_multi_hop_search_forwards_filters():
    mock_r = _make_mock_result("Entry 3 yield 87%", "[Source: a.docx | Page 1 | Section: T1]")
    with patch("agent.analytics_tools.hybrid_search") as mock_search, \
         patch("agent.analytics_tools.rerank") as mock_rerank:
        mock_search.return_value = [mock_r]
        mock_rerank.return_value = [mock_r]
        multi_hop_search("why did yield drop", ["catalyst question"], filters={"project_id": "PRJ-031"})
    first_call = mock_search.call_args_list[0]
    assert first_call.kwargs["filters"] == {"project_id": "PRJ-031"}


def test_multi_hop_search_deduplicates_by_attribution():
    mock_r = _make_mock_result("87%", "[Source: a.docx | Page 1 | Section: T1]")
    with patch("agent.analytics_tools.hybrid_search") as mock_search, \
         patch("agent.analytics_tools.rerank") as mock_rerank:
        mock_search.return_value = [mock_r]
        mock_rerank.return_value = [mock_r]
        result = multi_hop_search("yield query", [])
    attributions = [r["attribution"] for r in result]
    assert len(attributions) == len(set(attributions))


def test_statistical_summary_prefers_payload_structured_fields():
    """Chunk has no useful text but payload has {"Yield": "87%"}, metric is "yield"."""
    chunk = {
        "text": "no useful text here",
        "payload": {"structured_fields": {"Yield": "87%"}},
    }
    result = statistical_summary([chunk], "yield")
    assert "87" in result


def test_statistical_summary_falls_back_to_text_when_no_payload_field():
    """Chunk has payload={} and text "yield: 92%", fallback regex should find 92."""
    chunk = {
        "text": "yield: 92%",
        "payload": {},
    }
    result = statistical_summary([chunk], "yield")
    assert "92" in result


def test_extract_prefers_payload_structured_fields():
    """Test that payload structured_fields are preferred over regex on text."""
    chunk = {
        "text": "no useful text here",
        "attribution": "[Source: a.docx | Page 1 | Section: T1]",
        "payload": {"structured_fields": {"Yield": "87%"}},
    }
    result = extract_structured_data([chunk], ["yield"])
    # The regex on text would find nothing, so if result is non-empty it proves payload was used
    assert len(result) == 1
    assert result[0]["field"] == "yield"
    assert result[0]["value"] == "87"
    assert result[0]["unit"] == "%"


def test_extract_falls_back_to_regex_when_no_payload():
    """Test that regex fallback works when payload field is absent."""
    chunk = {
        "text": "yield: 92%",
        "attribution": "[Source: b.docx | Page 2 | Section: S2]",
        "payload": {},
    }
    result = extract_structured_data([chunk], ["yield"])
    assert len(result) == 1
    assert result[0]["field"] == "yield"
    assert result[0]["value"] == "92"
    assert result[0]["unit"] == "%"
