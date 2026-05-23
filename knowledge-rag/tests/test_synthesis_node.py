# knowledge-rag/tests/test_synthesis_node.py
from agent.state import AgentState
from agent.nodes.synthesis import synthesis_node
from tests.conftest import make_agent_state


def make_state(worker_results):
    return make_agent_state(worker_results=worker_results)


def chunk(text, attribution):
    return {"text": text, "attribution": attribution, "score": 0.9, "payload": {}}


def test_synthesis_builds_evidence_map():
    wr = [{"sub_task": "q", "agent_type": "lookup",
           "chunks": [chunk("87%", "[Source: a.docx | Page 1 | Section: T1]")]}]
    result = synthesis_node(make_state(wr))
    assert "evidence_map" in result
    assert len(result["evidence_map"]) == 1


def test_synthesis_deduplicates_identical_chunks():
    c = chunk("87%", "[Source: a.docx | Page 1 | Section: T1]")
    wr = [
        {"sub_task": "q1", "agent_type": "lookup", "chunks": [c]},
        {"sub_task": "q2", "agent_type": "lookup", "chunks": [c]},
    ]
    result = synthesis_node(make_state(wr))
    assert len(result["evidence_map"]) == 1


def test_synthesis_keeps_distinct_chunks():
    wr = [
        {"sub_task": "q1", "agent_type": "lookup",
         "chunks": [chunk("87%", "[Source: a.docx | Page 1 | Section: T1]")]},
        {"sub_task": "q2", "agent_type": "comparison",
         "chunks": [chunk("75%", "[Source: b.docx | Page 2 | Section: T2]")]},
    ]
    result = synthesis_node(make_state(wr))
    assert len(result["evidence_map"]) == 2


def test_synthesis_handles_empty_worker_results():
    result = synthesis_node(make_state([]))
    assert result["evidence_map"] == {}


def test_synthesis_evidence_map_values_are_chunk_dicts():
    c = chunk("87%", "[Source: a.docx | Page 1 | Section: T1]")
    wr = [{"sub_task": "q", "agent_type": "lookup", "chunks": [c]}]
    result = synthesis_node(make_state(wr))
    for v in result["evidence_map"].values():
        assert "text" in v
        assert "attribution" in v


def test_synthesis_caps_to_top_k(monkeypatch):
    """synthesis_node caps evidence_map to SYNTHESIS_TOP_K by score."""
    import agent.nodes.synthesis as _syn
    monkeypatch.setattr(_syn, "SYNTHESIS_TOP_K", 3)

    # Create 10 workers each with 1 unique chunk (10 total unique)
    worker_results = []
    for i in range(10):
        worker_results.append({
            "sub_task": f"task{i}", "agent_type": "lookup",
            "chunks": [{"text": f"chunk{i}", "attribution": f"src{i}", "score": float(i), "payload": {}}]
        })

    state = make_agent_state(question="q", worker_results=worker_results)
    result = synthesis_node(state)
    assert len(result["evidence_map"]) == 3
    # Top 3 by score should be chunks 9, 8, 7 (scores 9.0, 8.0, 7.0)
    texts = {c["text"] for c in result["evidence_map"].values()}
    assert "chunk9" in texts
    assert "chunk8" in texts
    assert "chunk7" in texts


def test_synthesis_stable_keys_for_same_chunk_from_different_workers():
    c = chunk("87%", "[Source: a.docx | Page 1 | Section: T1]")
    wr = [
        {"sub_task": "q1", "agent_type": "lookup", "chunks": [c]},
        {"sub_task": "q2", "agent_type": "comparison", "chunks": [c]},
    ]
    result = synthesis_node(make_state(wr))
    assert len(result["evidence_map"]) == 1
    keys = list(result["evidence_map"].keys())
    assert len(keys[0]) == 16  # 16-char hex string
    assert all(c in "0123456789abcdef" for c in keys[0])  # all hex characters


# ---------------------------------------------------------------------------
# Semantic dedup (opt-in via SEMANTIC_DEDUP=1)
# ---------------------------------------------------------------------------


def test_semantic_dedup_merges_paraphrases(monkeypatch):
    """When two chunks are semantically near-identical, the lower-scored one is dropped."""
    monkeypatch.setenv("SEMANTIC_DEDUP", "1")
    monkeypatch.setenv("SEMANTIC_DEDUP_THRESHOLD", "0.92")

    # Pretend embeddings: A and A' are near-duplicates; B is distinct.
    fake_vectors = [
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],  # cosine to A ≈ 0.9999
        [0.0, 1.0, 0.0],    # cosine to A = 0.0
    ]

    def fake_embed_texts(texts):
        return fake_vectors[:len(texts)]

    import retrieval.embedder as embedder_mod
    monkeypatch.setattr(embedder_mod, "embed_texts", fake_embed_texts, raising=False)

    wr = [{
        "sub_task": "q",
        "agent_type": "lookup",
        "chunks": [
            {"text": "Revenue grew 10%.", "attribution": "a", "score": 0.9, "payload": {"source_file": "a", "page_number": 1}},
            {"text": "Revenue rose by 10 percent.", "attribution": "b", "score": 0.8, "payload": {"source_file": "b", "page_number": 1}},
            {"text": "Headcount stayed flat.", "attribution": "c", "score": 0.7, "payload": {"source_file": "c", "page_number": 1}},
        ],
    }]
    result = synthesis_node(make_state(wr))
    texts = {c["text"] for c in result["evidence_map"].values()}
    assert "Revenue grew 10%." in texts
    assert "Headcount stayed flat." in texts
    assert "Revenue rose by 10 percent." not in texts
    assert len(result["evidence_map"]) == 2


def test_semantic_dedup_falls_back_when_embedder_raises(monkeypatch):
    """If the embedder errors out, the node returns the exact-dedup result instead."""
    monkeypatch.setenv("SEMANTIC_DEDUP", "1")

    def boom(_texts):
        raise RuntimeError("model not loaded")

    import retrieval.embedder as embedder_mod
    monkeypatch.setattr(embedder_mod, "embed_texts", boom, raising=False)

    wr = [{
        "sub_task": "q",
        "agent_type": "lookup",
        "chunks": [
            {"text": "A.", "attribution": "x", "score": 0.9, "payload": {"source_file": "x", "page_number": 1}},
            {"text": "B.", "attribution": "y", "score": 0.8, "payload": {"source_file": "y", "page_number": 1}},
        ],
    }]
    result = synthesis_node(make_state(wr))
    assert len(result["evidence_map"]) == 2  # nothing dropped


def test_semantic_dedup_disabled_does_not_call_embedder(monkeypatch):
    """With SEMANTIC_DEDUP=0 the embedder is never invoked."""
    monkeypatch.setenv("SEMANTIC_DEDUP", "0")

    calls = {"n": 0}

    def counted_embed(_texts):
        calls["n"] += 1
        return [[1.0]] * len(_texts)

    import retrieval.embedder as embedder_mod
    monkeypatch.setattr(embedder_mod, "embed_texts", counted_embed, raising=False)

    wr = [{
        "sub_task": "q",
        "agent_type": "lookup",
        "chunks": [
            {"text": "A.", "attribution": "x", "score": 0.9, "payload": {"source_file": "x", "page_number": 1}},
            {"text": "B.", "attribution": "y", "score": 0.8, "payload": {"source_file": "y", "page_number": 1}},
        ],
    }]
    synthesis_node(make_state(wr))
    assert calls["n"] == 0
