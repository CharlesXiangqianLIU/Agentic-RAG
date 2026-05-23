import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from evaluation.generate_qa_candidates import _extract_source_file, _generate_qa, generate


def test_extract_source_file_parses_attribution():
    attr = "[Source: PRJ-031.docx | Page 12 | Section: Table 1]"
    assert _extract_source_file(attr) == "PRJ-031.docx"


def test_extract_source_file_returns_raw_on_no_match():
    assert _extract_source_file("no-bracket-format") == "no-bracket-format"


def test_generate_qa_returns_dict_on_valid_json(tmp_path):
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"question": "What was the yield?", "suggested_answer": "87%", "type": "parameter_lookup"}'
    chunk = {"text": "Entry 3 yield 87%", "attribution": "[Source: a.docx | Page 1 | Section: T1]", "payload": {}}
    result = _generate_qa(chunk, mock_llm)
    assert result is not None
    assert result["question"] == "What was the yield?"
    assert result["suggested_answer"] == "87%"


def test_generate_qa_returns_none_on_invalid_json():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "not valid json at all"
    chunk = {"text": "some text", "attribution": "src", "payload": {}}
    result = _generate_qa(chunk, mock_llm)
    assert result is None


def test_generate_writes_candidates_file(tmp_path):
    """generate() writes a JSON file to the output path."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"question": "Q?", "suggested_answer": "A", "type": "parameter_lookup"}'

    mock_chunk = {"text": "87%", "attribution": "[Source: a.docx | Page 1 | Section: T1]", "payload": {}}

    with patch("evaluation.generate_qa_candidates.hybrid_search") as mock_search, \
         patch("evaluation.generate_qa_candidates.get_llm_provider", return_value=mock_llm):
        mock_search.return_value = [MagicMock(text="87%", attribution="[Source: a.docx | Page 1 | Section: T1]", payload={})]
        output = tmp_path / "candidates.json"
        results = generate(n=1, output_path=output)

    assert output.exists()
    data = json.loads(output.read_text())
    assert len(data) == len(results)


def test_generate_append_to_gold(tmp_path):
    """--append-to-gold appends candidates to qa_gold.json."""
    gold_path = tmp_path / "qa_gold.json"
    gold_path.write_text(json.dumps([{"id": "EXISTING-001"}]))

    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"question": "Q?", "suggested_answer": "A", "type": "parameter_lookup"}'

    with patch("evaluation.generate_qa_candidates.hybrid_search") as mock_search, \
         patch("evaluation.generate_qa_candidates.get_llm_provider", return_value=mock_llm):
        mock_search.return_value = [MagicMock(text="87%", attribution="[Source: a.docx | Page 1]", payload={})]
        output = tmp_path / "candidates.json"
        generate(n=1, output_path=output, append_to_gold=True, gold_path=gold_path)

    gold_data = json.loads(gold_path.read_text())
    assert len(gold_data) == 2  # original + new
    assert gold_data[0]["id"] == "EXISTING-001"
