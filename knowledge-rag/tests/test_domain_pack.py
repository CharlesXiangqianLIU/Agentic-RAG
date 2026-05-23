import textwrap

import pytest

from domain.loader import DomainPack, load_domain_pack
from domain.cache import get_domain_pack, reset_cache


# ---------- load_domain_pack ----------


def test_empty_path_returns_default_pack():
    pack = load_domain_pack("")
    assert isinstance(pack, DomainPack)
    assert pack.is_empty
    assert pack.abbreviations == {}
    assert pack.synonym_groups == []
    assert pack.fields == []
    assert pack.unit_patterns == []
    assert pack.prompt_overrides == {}


def test_none_path_returns_default_pack():
    assert load_domain_pack(None).is_empty


def test_missing_file_returns_default_pack(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    pack = load_domain_pack(missing)
    assert pack.is_empty


def test_valid_yaml_populates_pack(tmp_path):
    yaml_text = textwrap.dedent(
        """
        abbreviations:
          DCM: Dichloromethane
          THF: Tetrahydrofuran
        synonym_groups:
          - [DCM, Dichloromethane, methylene chloride]
          - [MeOH, Methanol]
        fields:
          - yield
          - temperature
        unit_patterns:
          - ['(\\d+)\\s*deg\\s*[Cc]', '\\1 °C']
        prompt_overrides:
          answer_system: "You are an expert."
        """
    )
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(yaml_text, encoding="utf-8")

    pack = load_domain_pack(pack_file)
    assert not pack.is_empty
    assert pack.abbreviations == {"DCM": "Dichloromethane", "THF": "Tetrahydrofuran"}
    assert frozenset({"DCM", "Dichloromethane", "methylene chloride"}) in pack.synonym_groups
    assert frozenset({"MeOH", "Methanol"}) in pack.synonym_groups
    assert pack.fields == ["yield", "temperature"]
    assert pack.unit_patterns == [(r"(\d+)\s*deg\s*[Cc]", r"\1 °C")]
    assert pack.prompt_overrides["answer_system"] == "You are an expert."


def test_omitted_keys_default_to_empty(tmp_path):
    pack_file = tmp_path / "partial.yaml"
    pack_file.write_text("abbreviations:\n  X: Xenon\n", encoding="utf-8")

    pack = load_domain_pack(pack_file)
    assert pack.abbreviations == {"X": "Xenon"}
    assert pack.synonym_groups == []
    assert pack.fields == []


def test_malformed_yaml_raises(tmp_path):
    pack_file = tmp_path / "broken.yaml"
    pack_file.write_text("abbreviations: : :", encoding="utf-8")
    with pytest.raises(ValueError):
        load_domain_pack(pack_file)


def test_wrong_top_level_shape_raises(tmp_path):
    pack_file = tmp_path / "list.yaml"
    pack_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_domain_pack(pack_file)


def test_wrong_abbreviations_shape_raises(tmp_path):
    pack_file = tmp_path / "bad_abbr.yaml"
    pack_file.write_text("abbreviations:\n  - DCM\n  - THF\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_domain_pack(pack_file)


def test_wrong_unit_pattern_shape_raises(tmp_path):
    pack_file = tmp_path / "bad_units.yaml"
    pack_file.write_text("unit_patterns:\n  - [only_one_item]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_domain_pack(pack_file)


def test_invalid_regex_in_unit_pattern_raises_at_load(tmp_path):
    """A bad regex must be caught at load time, not at first use."""
    pack_file = tmp_path / "bad_regex.yaml"
    pack_file.write_text("unit_patterns:\n  - ['(unbalanced', 'x']\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid regex"):
        load_domain_pack(pack_file)


def test_entity_patterns_loaded(tmp_path):
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(
        "entity_patterns:\n  - '\\d+\\s*°C'\n  - 'ICD-[0-9]+'\n",
        encoding="utf-8",
    )
    pack = load_domain_pack(pack_file)
    assert pack.entity_patterns == [r"\d+\s*°C", r"ICD-[0-9]+"]


def test_invalid_entity_pattern_raises(tmp_path):
    pack_file = tmp_path / "bad.yaml"
    pack_file.write_text("entity_patterns:\n  - '(unbalanced'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid regex"):
        load_domain_pack(pack_file)


def test_chemistry_example_loads():
    """The shipped chemistry example pack must parse cleanly."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    pack = load_domain_pack(repo_root / "domain" / "examples" / "chemistry.yaml")
    assert pack.abbreviations["DCM"] == "Dichloromethane"
    assert any("Dichloromethane" in g for g in pack.synonym_groups)
    assert "yield" in pack.fields
    assert pack.unit_patterns  # non-empty
    assert "answer_system" in pack.prompt_overrides


# ---------- get_domain_pack (cache) ----------


def test_cache_returns_same_instance(monkeypatch):
    import config

    monkeypatch.setattr(config, "DOMAIN_PACK_PATH", "")
    reset_cache()
    a = get_domain_pack()
    b = get_domain_pack()
    assert a is b


def test_cache_picks_up_path_after_reset(monkeypatch, tmp_path):
    import config

    pack_file = tmp_path / "p.yaml"
    pack_file.write_text("abbreviations:\n  TLA: Three Letter Abbreviation\n", encoding="utf-8")

    monkeypatch.setattr(config, "DOMAIN_PACK_PATH", "")
    reset_cache()
    assert get_domain_pack().is_empty

    monkeypatch.setattr(config, "DOMAIN_PACK_PATH", str(pack_file))
    reset_cache()
    pack = get_domain_pack()
    assert pack.abbreviations == {"TLA": "Three Letter Abbreviation"}
