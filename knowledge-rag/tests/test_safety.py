# tests/test_safety.py
"""Tests for agent/safety.py — check_answer and _check_entities.

With an empty domain pack the entity check is skipped entirely, leaving
only the numeric-claim check (which is domain-agnostic). The fixtures
below inject a small chemistry-flavoured pack so the existing flagging
behaviour can still be exercised.
"""
import pytest

from domain.loader import DomainPack
from agent import safety as safety_mod


@pytest.fixture
def chemistry_pack(monkeypatch):
    pack = DomainPack(
        abbreviations={
            "DCM": "Dichloromethane",
            "THF": "Tetrahydrofuran",
            "Pd": "Palladium",
        },
    )
    monkeypatch.setattr(safety_mod, "get_domain_pack", lambda: pack)
    yield pack


@pytest.fixture
def empty_pack(monkeypatch):
    monkeypatch.setattr(safety_mod, "get_domain_pack", lambda: DomainPack())
    yield


# ---------- entity check (requires populated pack) ----------


def test_check_entities_flags_absent_abbreviation(chemistry_pack):
    """DCM in sentence but not in source → should flag as unsupported."""
    sentence = "The reaction was run in DCM at room temperature."
    source_text = "the reaction was performed in methanol at low temperature."
    assert safety_mod._check_entities(sentence, source_text) is True


def test_check_entities_passes_present_entity(chemistry_pack):
    """DCM in sentence and 'dcm' in source → should not flag."""
    sentence = "The reaction was run in DCM at room temperature."
    source_text = "the reaction was performed in dcm at low temperature."
    assert safety_mod._check_entities(sentence, source_text) is False


def test_check_answer_flags_unsupported_entity(chemistry_pack):
    """Answer mentions Pd but chunks only contain THF → flagged."""
    answer = "The reaction used Pd catalyst for the coupling step."
    chunks = [{"text": "The reaction was performed in THF at room temperature."}]
    assert "[UNSUPPORTED" in safety_mod.check_answer(answer, chunks)


def test_check_answer_passes_supported_entity(chemistry_pack):
    """Answer mentions DCM and chunks contain 'dcm' → not flagged."""
    answer = "The reaction used DCM as solvent."
    chunks = [{"text": "dcm was used as the solvent in this reaction."}]
    assert "[UNSUPPORTED" not in safety_mod.check_answer(answer, chunks)


# ---------- numeric check (domain-agnostic) ----------


def test_check_answer_flags_unsupported_number(empty_pack):
    answer = "The yield was 87% under these conditions."
    chunks = [{"text": "The reaction was performed and characterized."}]
    assert "[UNSUPPORTED" in safety_mod.check_answer(answer, chunks)


def test_check_answer_passes_supported_number(empty_pack):
    answer = "The yield was 87% under these conditions."
    chunks = [{"text": "The reaction afforded the product in 87% yield."}]
    assert "[UNSUPPORTED" not in safety_mod.check_answer(answer, chunks)


# ---------- empty pack disables the entity check ----------


def test_empty_pack_does_not_flag_entities(empty_pack):
    """With no domain pack, entity tokens like 'DCM' must not be flagged."""
    answer = "The reaction was run in DCM."
    chunks = [{"text": "The reaction was performed in methanol."}]
    # No numbers, no entities the system knows about -> nothing to flag
    assert "[UNSUPPORTED" not in safety_mod.check_answer(answer, chunks)


# ---------- richer numeric formats (domain-agnostic) ----------


def test_thousands_separator_matches_either_form(empty_pack):
    # Answer uses 1,234; source spells it 1234 (no comma).
    answer = "The revenue was 1,234 dollars."
    chunks = [{"text": "The revenue was 1234 dollars."}]
    assert "[UNSUPPORTED" not in safety_mod.check_answer(answer, chunks)

    # And vice versa.
    answer2 = "The revenue was 1234 dollars."
    chunks2 = [{"text": "The revenue was 1,234 dollars."}]
    assert "[UNSUPPORTED" not in safety_mod.check_answer(answer2, chunks2)


def test_percent_with_space_is_supported_when_source_has_it(empty_pack):
    answer = "The yield was 87 % under these conditions."
    chunks = [{"text": "The reaction afforded the product in 87 % yield."}]
    assert "[UNSUPPORTED" not in safety_mod.check_answer(answer, chunks)


def test_scientific_notation_is_supported_when_source_has_it(empty_pack):
    answer = "The constant is 1.5e3."
    chunks = [{"text": "The published constant is 1.5e3 per second."}]
    assert "[UNSUPPORTED" not in safety_mod.check_answer(answer, chunks)


def test_unicode_minus_normalised(empty_pack):
    # Answer uses unicode minus, source uses ascii hyphen-minus.
    answer = "The bias was −15 mV."
    chunks = [{"text": "Measured bias: -15 mV."}]
    assert "[UNSUPPORTED" not in safety_mod.check_answer(answer, chunks)


def test_negative_number_unsupported(empty_pack):
    answer = "The bias was -15 mV."
    chunks = [{"text": "No bias reported in this run."}]
    assert "[UNSUPPORTED" in safety_mod.check_answer(answer, chunks)


def test_thousands_grouping_unsupported(empty_pack):
    answer = "Revenue was 1,234,567 dollars."
    chunks = [{"text": "Revenue figures were withheld."}]
    assert "[UNSUPPORTED" in safety_mod.check_answer(answer, chunks)
