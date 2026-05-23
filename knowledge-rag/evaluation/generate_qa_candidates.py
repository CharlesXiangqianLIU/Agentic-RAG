import argparse
import json
import random
import re
from pathlib import Path

from retrieval.searcher import hybrid_search
from llm import get_llm_provider

_BROAD_QUERIES = [
    "reaction yield optimization",
    "catalyst temperature conditions",
    "solvent effect selectivity",
    "conversion rate time",
    "product isolation purification",
]

_SYSTEM_PROMPT = (
    "You are an expert synthetic chemistry assistant. "
    "Generate ONE specific question and answer pair based on the provided source passage."
)


def _sample_chunks(n: int) -> list[dict]:
    seen, chunks = set(), []
    for query in _BROAD_QUERIES:
        for r in hybrid_search(query):
            key = r.attribution if hasattr(r, "attribution") else r.get("attribution", "")
            text = r.text if hasattr(r, "text") else r.get("text", "")
            attribution = key
            if attribution not in seen:
                seen.add(attribution)
                chunks.append(
                    {
                        "text": text,
                        "attribution": attribution,
                        "payload": r.payload if hasattr(r, "payload") else r.get("payload", {}),
                    }
                )
    random.shuffle(chunks)
    return chunks[:n]


def _extract_source_file(attribution: str) -> str:
    m = re.search(r"\[Source:\s*([^\|]+)", attribution)
    return m.group(1).strip() if m else attribution


def _generate_qa(chunk: dict, llm) -> dict | None:
    user_msg = (
        f"Source: {chunk['attribution']}\n\nPassage:\n{chunk['text']}\n\n"
        "Generate a question and answer in this EXACT JSON format:\n"
        '{"question": "...", "suggested_answer": "...", "type": "parameter_lookup"}'
    )
    raw = llm.complete(
        [{"role": "user", "content": user_msg}],
        system=_SYSTEM_PROMPT,
        max_tokens=512,
    )
    try:
        json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        data = json.loads(json_match.group() if json_match else raw)
        return data
    except Exception:
        return None


def generate(
    n: int = 20,
    output_path: Path = None,
    append_to_gold: bool = False,
    gold_path: Path = None,
) -> list[dict]:
    if output_path is None:
        output_path = Path(__file__).parent / "eval_set" / "qa_candidates.json"
    output_path = Path(output_path)

    chunks = _sample_chunks(n)
    llm = get_llm_provider()
    results = []
    for i, chunk in enumerate(chunks, 1):
        qa = _generate_qa(chunk, llm)
        if qa is None:
            continue
        entry = {
            "id": f"CAND-{len(results) + 1:03d}",
            "type": qa.get("type", "parameter_lookup"),
            "question": qa.get("question", ""),
            "suggested_answer": qa.get("suggested_answer", ""),
            "source_file": _extract_source_file(chunk["attribution"]),
            "source_page": None,
            "source_section": "",
            "expected_answer": "",
        }
        results.append(entry)
        print(f"  [{i}/{len(chunks)}] Generated: {entry['question'][:60]}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(results)} candidates to {output_path}")

    if append_to_gold:
        if gold_path is None:
            gold_path = Path(__file__).parent / "eval_set" / "qa_gold.json"
        gold_path = Path(gold_path)
        existing = json.loads(gold_path.read_text()) if gold_path.exists() else []
        existing.extend(results)
        gold_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        print(f"Appended {len(results)} items to {gold_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA candidate pairs from indexed chunks")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument(
        "--append-to-gold",
        action="store_true",
        help="Append generated candidates to eval_set/qa_gold.json (for review pipeline)",
    )
    args = parser.parse_args()
    generate(n=args.n_samples, output_path=args.output_path, append_to_gold=args.append_to_gold)


if __name__ == "__main__":
    main()
