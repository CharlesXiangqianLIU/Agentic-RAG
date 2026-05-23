# tests/test_normalizer.py
"""normalize_text + expand_synonyms behaviour, both with and without a
populated domain pack."""
import pytest

from domain.loader import DomainPack
from domain import cache as domain_cache
from ingestion.normalizer import normalize_text, expand_synonyms


@pytest.fixture
def chemistry_pack(monkeypatch):
    """Inject a chemistry-flavoured domain pack for the duration of the test."""
    pack = DomainPack(
        abbreviations={
            "DCM": "Dichloromethane",
            "THF": "Tetrahydrofuran",
            "MeOH": "Methanol",
        },
        synonym_groups=[
            frozenset({"DCM", "Dichloromethane", "methylene chloride", "CH2Cl2"}),
            frozenset({"THF", "Tetrahydrofuran"}),
        ],
        unit_patterns=[
            (r"(\d+)\s*deg\s*[Cc]", r"\1 °C"),
            (r"(\d+)\s*°\s*[Cc]", r"\1 °C"),
            (r"(\d+)\s*°\s*[Ff]", r"\1 °F"),
        ],
    )
    monkeypatch.setattr(domain_cache, "get_domain_pack", lambda: pack)
    # The normalizer module imported the symbol directly, so patch it there too.
    import ingestion.normalizer as nm
    monkeypatch.setattr(nm, "get_domain_pack", lambda: pack)
    yield pack


@pytest.fixture
def empty_pack(monkeypatch):
    pack = DomainPack()
    monkeypatch.setattr(domain_cache, "get_domain_pack", lambda: pack)
    import ingestion.normalizer as nm
    monkeypatch.setattr(nm, "get_domain_pack", lambda: pack)
    yield pack


# ---------- with a populated pack ----------


def test_unit_normalization_deg_c(chemistry_pack):
    assert "80 °C" in normalize_text("heated to 80 deg C")


def test_unit_normalization_lowercase(chemistry_pack):
    assert "°C" in normalize_text("reaction at 100°c")


def test_abbreviation_dcm(chemistry_pack):
    assert "Dichloromethane" in normalize_text("dissolved in DCM")


def test_abbreviation_thf(chemistry_pack):
    assert "Tetrahydrofuran" in normalize_text("solvent was THF")


def test_abbreviation_preserves_surrounding_text(chemistry_pack):
    result = normalize_text("Add DCM dropwise to the flask")
    assert "Dichloromethane" in result
    assert "dropwise" in result
    assert "flask" in result


def test_synonym_expansion_dcm(chemistry_pack):
    synonyms = expand_synonyms("DCM")
    assert "Dichloromethane" in synonyms
    assert "DCM" in synonyms
    assert "methylene chloride" in synonyms


def test_synonym_expansion_unknown(chemistry_pack):
    assert expand_synonyms("UnknownReagent42") == ["UnknownReagent42"]


def test_normalize_text_unknown_passes_through(chemistry_pack):
    assert "custom_reagent_xyz123" in normalize_text("custom_reagent_xyz123 was added")


# ---------- with an empty pack (default knowledge-rag behaviour) ----------


def test_empty_pack_normalize_is_passthrough(empty_pack):
    text = "heated to 80 deg C, dissolved in DCM"
    assert normalize_text(text) == text


def test_empty_pack_expand_returns_just_term(empty_pack):
    assert expand_synonyms("DCM") == ["DCM"]
    assert expand_synonyms("anything") == ["anything"]
