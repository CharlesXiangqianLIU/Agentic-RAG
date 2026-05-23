# Gold-Standard Q&A Evaluation Set

This directory contains the gold-standard evaluation dataset for the Chemical Process R&D Knowledge Base RAG system. The dataset will be used for RAGAS evaluation in Week 4.

## Current Status

- **Seed items:** 5
- **Target:** 50–100 items by end of Week 4
- **Next steps:** Domain expert annotation of empty `expected_answer` fields

## Schema Reference

Each Q&A item in `qa_gold.json` follows this structure:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | string | Unique identifier for the item | `"QA-001"` |
| `type` | string | Question category (see below) | `"parameter_lookup"` |
| `question` | string | The actual chemistry R&D question to ask the RAG system | `"What was the reaction yield at 80°C..."` |
| `expected_answer` | string | Ground-truth answer (provided by domain expert) | `"87%"` or `""` (empty for pending annotation) |
| `source_file` | string | Report filename, `"multiple"` for cross-report, or `""` if unknown | `"PRJ-2024-031.docx"` |
| `source_page` | integer or null | Page number in source report, or null if unknown | `42` or `null` |
| `source_section` | string | Report section name (e.g., "Scope Study", "Optimization") | `""` if unknown |

## Question Types

The evaluation set must cover four distinct types of RAG queries:

### 1. `parameter_lookup`
Direct factual retrieval of a specific value: temperature, yield, reagent, solvent, or reaction condition.

**Example:** "What was the reaction yield when using Pd(OAc)₂ at 80°C?"

**Evaluation focus:** Exact match of numerical values or reagent names; source attribution accuracy.

### 2. `cross_report_comparison`
Comparative analysis across multiple reports to identify differences, similarities, or relative performance.

**Example:** "Compare the optimal temperatures for palladium-catalyzed reactions in reports PRJ-2024-031 and PRJ-2024-032."

**Evaluation focus:** Information synthesis from multiple sources; correctness of comparative claims.

### 3. `trend_analysis`
Identification of patterns or trends across many data points (e.g., optimization series, temperature sweeps).

**Example:** "What is the general trend in yield as reaction time increases from 6 to 24 hours?"

**Evaluation focus:** Pattern recognition; reasoning over quantitative data; handling of exceptions.

### 4. `reasoning`
Causal or mechanistic analysis: why a change was made, what hypothesis it tests, what the result implies.

**Example:** "Why was the base changed from K₂CO₃ to Cs₂CO₃? What does this suggest about the mechanism?"

**Evaluation focus:** Correctness of mechanistic reasoning; justification with evidence from the reports.

## Annotation Workflow

1. **Domain Expert Review:** For each seed item with `expected_answer: ""`:
   - Read the referenced report (use `source_file` and `source_page`)
   - Provide a concise, accurate ground-truth answer
   - Verify `source_page` and `source_section` are correct
   - Update the JSON file

2. **New Item Creation:** As data ingestion proceeds, create new Q&A items following the same schema.
   - Aim for a balanced mix: roughly 15–20 of each type
   - Ensure `expected_answer` is filled in at creation time
   - Include specific source citations

3. **Version Control:** Commit updates regularly:
   ```bash
   git add evaluation/eval_set/qa_gold.json
   git commit -m "eval: add Q&A items QA-XXX to QA-YYY (20 items, domain expert reviewed)"
   ```

## RAGAS Evaluation (Week 4)

The RAGAS framework will assess:
- **Faithfulness:** Does the RAG answer use only information from retrieved chunks?
- **Answer Relevance:** How well does the answer address the question?
- **Context Relevance:** Did the retrieval system find the correct source chunks?
- **Correctness:** Does the expected_answer match domain expert evaluation?

For full details, see `../ragas_evaluation.md` (to be created in Week 4).
