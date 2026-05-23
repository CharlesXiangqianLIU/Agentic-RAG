# Writing a domain pack

A **domain pack** is a single YAML file that specialises the generic
Knowledge RAG pipeline for a specific subject area — chemistry, law,
HR, medical billing, finance, whatever — **without writing any code**.

This guide walks you through writing one from scratch, end to end, for
a fictional HR policy knowledge base. By the end you'll have a pack
that:

- expands HR acronyms (`PTO → Paid Time Off`)
- recognises HR-specific entities (employee IDs, policy refs)
- normalises common date formats
- knows to look for "duration" / "eligibility" / "approver" fields when
  comparing policies
- speaks the language of HR (the answer / classify / critic prompts
  mention employees and policies instead of generic "documents")

## Mental model

A domain pack overrides five orthogonal axes of the pipeline:

| Section            | When it fires       | Why you'd populate it                                            |
|--------------------|---------------------|------------------------------------------------------------------|
| `abbreviations`    | ingestion (index)   | Expand in-corpus shorthand so both forms are searchable.         |
| `synonym_groups`   | ingestion (index)   | Tie token clusters together — query "PTO" hits "vacation" chunks. |
| `unit_patterns`    | ingestion (index)   | Normalise date / currency / measurement formats.                  |
| `fields`           | retrieval + UI      | Drive comparison/trend hints and sidebar filters.                |
| `entity_patterns`  | retrieval (multi-hop) | Codes the system should look for when doing second-hop searches. |
| `prompt_overrides` | answer / planning   | Re-frame the LLM's role in your domain's language.               |

You don't have to populate all of them. Start with what hurts.

## Step 0 — Decide if you actually need one

Skip a domain pack if your corpus has:
- No discipline-specific jargon (general business docs, internal wiki).
- No acronyms that conflict with their expansions.
- No structured tables you need to compare across.

If you only need a different chat-UI title, just set `APP_PASSWORD`,
swap the docs, and ship. The default pack-less behaviour is genuinely
domain-agnostic.

## Step 1 — Scaffold

Copy the chemistry example as a template:

```sh
cp knowledge-rag/domain/examples/chemistry.yaml knowledge-rag/domain/hr.yaml
```

Set `DOMAIN_PACK_PATH=domain/hr.yaml` in `.env`. Restart the UI —
nothing should be different yet because the file still contains
chemistry content, but the wiring works.

Now empty out every section and fill them in one by one. You can test
each section in isolation with `pytest`.

## Step 2 — Mine acronyms from your corpus

Look at 10–20 representative source docs. Write down every acronym
that appears more than twice. For HR policies you might see:

```yaml
abbreviations:
  PTO: Paid Time Off
  FMLA: Family and Medical Leave Act
  401k: 401(k) Retirement Plan
  HRA: Health Reimbursement Arrangement
  COBRA: Consolidated Omnibus Budget Reconciliation Act
  EEO: Equal Employment Opportunity
  H1B: H-1B visa
  L&D: Learning and Development
```

**Pitfalls**:
- Don't include acronyms whose expanded form is rare. If your corpus
  always says "PTO" and never "Paid Time Off", expanding to a phrase
  nobody wrote dilutes recall.
- Acronyms ARE case-sensitive by design (`\b...\b` match). `pto` won't
  match the rule. Add lowercase variants only if your corpus uses them.
- Avoid acronyms that overlap with common words.  `HR` would tag every
  "her" in someone's name. Either omit or pre-process.

Verify:

```python
from domain.loader import load_domain_pack
pack = load_domain_pack("domain/hr.yaml")
assert pack.abbreviations["PTO"] == "Paid Time Off"
```

## Step 3 — Group synonyms

Synonyms tell the indexer "these tokens are interchangeable; whichever
the user types, hit the same chunks".

```yaml
synonym_groups:
  - [PTO, Paid Time Off, vacation, annual leave, holiday]
  - [FMLA, parental leave, maternity leave, paternity leave]
  - [base salary, base pay, fixed compensation]
  - [bonus, variable comp, incentive pay, performance pay]
  - [manager, supervisor, direct report manager, people manager]
```

A query "How many vacation days do new hires get?" will retrieve
chunks that say "PTO" or "annual leave" even if "vacation" never
appears in them. The synonyms are recorded on each chunk's metadata
at ingestion time.

**Pitfalls**:
- Don't transitively merge what shouldn't merge. "bonus" and "stock
  grant" are different even though both are non-base comp.

## Step 4 — Unit normalisation

For HR you might want to canonicalise date formats so a query for
"March 2024" matches docs that wrote "03/2024" or "Mar 2024".
Knowledge RAG applies `unit_patterns` as raw `re.sub` rules — so the
left side is a regex and the right side is a Python replacement
template (use `\1`, `\2` for groups).

```yaml
unit_patterns:
  # Normalise US date variants to ISO "YYYY-MM-DD" — first replacement
  # tries Mon DD YYYY (e.g. "March 12, 2024"). Pattern is per-pattern;
  # add more rules as you discover the formats your corpus uses.
  - ['(?i)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})', '\3-\1-\2']
  # MM/DD/YYYY → YYYY-MM-DD
  - ['(\d{1,2})/(\d{1,2})/(\d{4})', '\3-\1-\2']
  # Strip thousand separators in salary numbers (\$120,000 → \$120000)
  - ['\$(\d{1,3}(?:,\d{3})+)', '\$\1']
```

Validate at load: bad regex is caught immediately by the loader.

```sh
pytest tests/test_domain_pack.py::test_invalid_regex_in_unit_pattern_raises_at_load
```

If you want to dry-run normalisation on a real doc:

```python
from ingestion.normalizer import normalize_text
print(normalize_text("Updated March 12, 2024 — base pay $120,000."))
# 'Updated 2024-mar-12 — base pay $120000.'
```

## Step 5 — Comparison / trend fields

`fields` drives two things at runtime:

1. **Worker hints.** When a question is classified as `comparison` or
   `trend`, the worker tries to find any of these field names in the
   sub-task text and uses the first match as a hint to
   `compare_across_reports` or `statistical_summary`.
2. **UI filter sidebar.** The first two fields show up as text inputs
   in the Streamlit sidebar (more under an Advanced Filters expander).

For HR:

```yaml
fields:
  - eligibility
  - duration
  - approver
  - effective_date
  - region
  - role_level
  - benefits
```

This means a user can type "Region: AMER" in the sidebar and the
worker will pass `{"region": "AMER"}` as a metadata filter to Qdrant.
For that to actually filter anything, your ingestion sidecar JSON
needs to set `region` on each doc:

```json
{"region": "AMER", "category": "policy", "doc_type": "handbook"}
```

## Step 6 — Entity patterns for second-hop search

When the agent gets a reasoning question, the multi-hop tool extracts
"interesting tokens" from the first-pass results and uses them to seed
a second search. By default it picks up uppercase codes like
`PRJ-031`. You teach it about domain-specific identifiers via
`entity_patterns`:

```yaml
entity_patterns:
  # Employee IDs: EMP- followed by 5+ digits
  - 'EMP-\d{5,}'
  # Policy reference codes: POL- + category + 4-digit number
  - 'POL-[A-Z]{2,4}-\d{4}'
  # Statutes commonly cited in US HR docs
  - '\b29 (?:U\.S\.C\.|USC)\s+§\s*\d+'
```

Each pattern is `re.compile`d at load time — typos surface in seconds.

## Step 7 — Prompt overrides

This is where the domain pack does the most observable work — it
changes what the LLM thinks it is. The six available slots, in
descending importance for HR:

```yaml
prompt_overrides:
  answer_system: |
    You are an expert HR policy assistant working with a private corpus
    of employee handbooks, benefits guides, and policy documents.
    Answer the user's question using ONLY the provided source passages.
    After each factual claim, cite the source using:
    [Source: <filename> | Page <n> | Section: <section>]
    Use neutral language; never give the user career, financial, or
    legal advice. When sources disagree, list both and flag the
    discrepancy.

  answer_comparison_system: |
    You are an expert HR policy assistant. Answer the comparison
    question using ONLY the provided source passages. Present your
    answer as a Markdown table. Pick columns from: Document, Effective
    Date, Region, Eligibility, Duration, Approver. End with a 1-2
    sentence summary of the operational difference.

  classify_system: |
    You classify HR policy questions into exactly one of these four
    types:
    - lookup: a single specific value (PTO days, deadline, dollar amount, eligibility threshold)
    - comparison: same policy across multiple regions / years / role levels
    - trend: how a metric has changed across multiple policy revisions
    - reasoning: why a policy was changed; what the underlying rationale was
    Respond with ONLY the type word.

  plan_system: |
    You are an HR research planning assistant. Given a question and
    its type, output a JSON array of sub-tasks. Each item:
    {"task": "<search query>", "agent_type": "<lookup|comparison|trend|reasoning>"}
    Bias the sub-task phrasing toward HR jargon (eligibility, accrual,
    approver, effective date).
    Output ONLY a valid JSON array.

  critic_system: |
    You are a fact-checker for HR policy answers. For every sentence
    that asserts a number, date, or eligibility rule, verify it is
    explicitly supported by the source passages. Be strict — HR
    answers that drift from policy carry compliance risk.
    Output JSON only:
    {"overall": "PASS" | "FAIL",
     "issues": [{"claim": "...", "issue_type": "unsupported|missing_context|contradictory", "retry_query": "..."}]}
```

You don't have to fill all six. Empty slot → use the generic default.

## Step 8 — Test before you commit

Add a one-line test that pins the shape of your pack so a typo
doesn't quietly break it:

```python
# tests/test_hr_pack.py
def test_hr_pack_loads():
    from pathlib import Path
    from domain.loader import load_domain_pack
    pack = load_domain_pack(
        Path(__file__).resolve().parent.parent / "domain" / "hr.yaml"
    )
    assert pack.abbreviations["PTO"] == "Paid Time Off"
    assert "eligibility" in pack.fields
    assert "answer_system" in pack.prompt_overrides
```

Then in CI:

```sh
DOMAIN_PACK_PATH=domain/hr.yaml pytest tests/test_hr_pack.py
```

## Step 9 — Iterate from real questions

Don't hand-craft a perfect pack up front. Ship the v1, watch real
questions for a week, and grow the pack from the gaps:

| Symptom you see in logs                       | What to add to the pack          |
|-----------------------------------------------|----------------------------------|
| Answer says "I don't have information on X"  | Acronym for X to `abbreviations` |
| Two correct answers for the same intent       | Group the surface forms in `synonym_groups` |
| Critic flags 30 % of answers `[UNSUPPORTED]`  | Tighten `critic_system` prompt    |
| Sidebar filter never used                     | Remove that `fields` entry        |
| Comparison answers are wide but shallow       | Add more specific `fields` entries to drive table columns |

## Common pitfalls

1. **Overlapping abbreviations.** If you have both `IT` (Information
   Technology) and a coworker's initial `IT`, the loader can't tell.
   Pick one or namespace via context-only references.
2. **Greedy synonym groups.** Don't put everything in one giant group.
   The bigger a group, the noisier its recall.
3. **Unit patterns over-matching.** A pattern like `(\d+)\s*days?` is
   too aggressive — it'll touch any sentence with a day count. Anchor
   with surrounding context (`PTO\s+(\d+)\s+days?`).
4. **Prompts that change the answer format.** If you change the
   citation format in `answer_system`, the safety pass (`check_answer`)
   may stop recognising your sources. Keep `[Source: ... | Page ... | Section: ...]`
   stable.
5. **YAML escaping.** Regex backslashes need single-quoted YAML
   strings: `'(\d+)'` not `"(\d+)"`. Backslash + double quote in YAML
   gets eaten.

## Next steps

- Hand the pack to a domain expert; ask them to grade 20 answers.
- When you're confident, freeze the pack as `domain/hr.yaml.v1` and
  bump the version on changes — the audit log records the prompt
  version implicitly through `created_at`.
- See `docs/deployment.md` for how to ship the pack alongside the
  app image.
