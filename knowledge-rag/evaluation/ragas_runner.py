"""
Ragas evaluation runner for the Chemical RAG system.

Usage:
    python -m evaluation.ragas_runner
    python -m evaluation.ragas_runner --show-unannotated

Requires:
    - evaluation/eval_set/qa_gold.json must have items with non-empty expected_answer
    - Qdrant must be running with indexed documents
    - ANTHROPIC_API_KEY must be set in .env

Target scores (Week 4):
    - faithfulness > 0.85
    - answer_relevancy > 0.80
    - context_recall > 0.80
    - context_precision > 0.75
"""
import json
from pathlib import Path
from agent.state import AgentState

EVAL_SET_PATH = Path(__file__).parent / "eval_set" / "qa_gold.json"


def _make_eval_state(question: str) -> AgentState:
    """Build a minimal AgentState for evaluation runs."""
    return AgentState(
        question=question,
        question_type="",
        sub_tasks=[],
        current_sub_task="",
        current_agent_type="",
        worker_results=[],
        evidence_map={},
        draft_answer="",
        final_answer="",
        critic_issues=[],
        critic_round=0,
        reflection_passed=False,
        messages=[],
    )


def load_annotated_items() -> list[dict]:
    """Load only items with non-empty expected_answer (domain expert annotated)."""
    with open(EVAL_SET_PATH) as f:
        items = json.load(f)
    annotated = [i for i in items if i.get("expected_answer")]
    if not annotated:
        print(f"Warning: No annotated items found in {EVAL_SET_PATH}")
        print("Domain experts must fill in 'expected_answer' fields before evaluation.")
    return annotated


def show_unannotated() -> None:
    """Print all items still missing expected_answer to guide annotation work."""
    if not EVAL_SET_PATH.exists():
        print(f"Eval set not found: {EVAL_SET_PATH}")
        return
    with open(EVAL_SET_PATH) as f:
        items = json.load(f)
    unannotated = [i for i in items if not i.get("expected_answer")]
    annotated_count = len(items) - len(unannotated)
    print(f"Annotation progress: {annotated_count}/{len(items)} items complete\n")
    if not unannotated:
        print("All items are annotated. Ready to run evaluation.")
        return
    print(f"Unannotated items ({len(unannotated)}):")
    for item in unannotated:
        print(f"  [{item.get('id', '?')}] {item.get('question', '')[:80]}")


def run_evaluation(max_items: int = None) -> dict:
    """
    Run Ragas evaluation against the gold-standard Q&A set.

    Args:
        max_items: Limit evaluation to first N items (useful for smoke tests).

    Returns:
        dict of Ragas metric scores.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from agent.graph import build_graph

    items = load_annotated_items()
    if max_items:
        items = items[:max_items]

    if not items:
        return {}

    graph = build_graph()
    questions, answers, ground_truths, contexts = [], [], [], []

    for i, item in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {item['id']}: {item['question'][:60]}...")
        try:
            result = graph.invoke(_make_eval_state(item["question"]))
            questions.append(item["question"])
            answers.append(result["final_answer"])
            ground_truths.append(item["expected_answer"])
            contexts.append([c.get("text", "") for c in list(result.get("evidence_map", {}).values())])
        except Exception as e:
            print(f"    ERROR: {e}")

    if not questions:
        print("No results to evaluate.")
        return {}

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "ground_truth": ground_truths,
        "contexts": contexts,
    })

    print(f"\nEvaluating {len(questions)} items with Ragas...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    print("\n=== Ragas Evaluation Results ===")
    print(scores)
    return scores


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run Ragas evaluation or inspect annotation progress")
    parser.add_argument(
        "--show-unannotated",
        action="store_true",
        help="List items missing expected_answer without running evaluation",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Limit evaluation to first N items")
    args = parser.parse_args()

    if args.show_unannotated:
        show_unannotated()
        return

    print(f"Loading eval set from: {EVAL_SET_PATH}")
    items = load_annotated_items()
    print(f"Found {len(items)} annotated item(s) ready for evaluation.")
    if items:
        run_evaluation(max_items=args.max_items)
    else:
        print("\nNext steps:")
        print("  1. Open evaluation/eval_set/qa_gold.json")
        print("  2. Fill in 'expected_answer' fields (domain expert task)")
        print("  3. Re-run: python -m evaluation.ragas_runner")


if __name__ == "__main__":
    main()
