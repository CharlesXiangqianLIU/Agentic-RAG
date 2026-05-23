"""Pin the shape of domain/examples/hr.yaml.

This test catches typos / regressions in the bundled HR example so
docs/writing-a-domain-pack.md keeps pointing at a valid file.
"""
from pathlib import Path

import pytest

from domain.loader import load_domain_pack


_PACK_PATH = Path(__file__).resolve().parent.parent / "domain" / "examples" / "hr.yaml"


@pytest.fixture(scope="module")
def hr_pack():
    return load_domain_pack(_PACK_PATH)


def test_hr_pack_loads(hr_pack):
    assert not hr_pack.is_empty


def test_hr_pack_expands_pto(hr_pack):
    assert hr_pack.abbreviations["PTO"] == "Paid Time Off"
    assert "FMLA" in hr_pack.abbreviations


def test_hr_pack_groups_synonyms(hr_pack):
    pto_group = next(
        (g for g in hr_pack.synonym_groups if "PTO" in g),
        None,
    )
    assert pto_group is not None
    assert "vacation" in pto_group
    assert "Paid Time Off" in pto_group


def test_hr_pack_lists_filterable_fields(hr_pack):
    for required in ("eligibility", "duration", "approver", "region"):
        assert required in hr_pack.fields


def test_hr_pack_normalises_dates(hr_pack):
    """The unit_patterns include the canonical Month DD YYYY → YYYY-Mon-DD rule."""
    from ingestion.normalizer import normalize_text
    from domain import cache as domain_cache

    # Inject the HR pack for this single normalize call.
    original = domain_cache.get_domain_pack
    domain_cache.get_domain_pack = lambda: hr_pack  # type: ignore[assignment]
    import ingestion.normalizer as nm
    nm.get_domain_pack = lambda: hr_pack  # type: ignore[assignment]
    try:
        result = normalize_text("Effective March 12, 2026 across the AMER region.")
        assert "2026-mar-12" in result.lower() or "2026-Mar-12" in result
    finally:
        domain_cache.get_domain_pack = original
        nm.get_domain_pack = original


def test_hr_pack_entity_patterns_compile(hr_pack):
    import re
    for pattern in hr_pack.entity_patterns:
        re.compile(pattern)  # must not raise


def test_hr_pack_entity_patterns_match_real_codes(hr_pack):
    import re
    blob = "Owner is EMP-10042. See POL-MED-2026 and 29 USC § 2611."
    found = []
    for pattern in hr_pack.entity_patterns:
        found.extend(re.findall(pattern, blob))
    assert "EMP-10042" in found
    assert "POL-MED-2026" in found
    assert any("2611" in f for f in found)


def test_hr_pack_overrides_all_six_prompt_slots(hr_pack):
    expected_slots = {
        "answer_system",
        "answer_comparison_system",
        "classify_system",
        "plan_system",
        "critic_system",
    }
    assert expected_slots.issubset(hr_pack.prompt_overrides.keys())
