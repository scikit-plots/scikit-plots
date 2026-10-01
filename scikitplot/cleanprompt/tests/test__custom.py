"""
Tests for :mod:`scikitplot.cleanprompt._custom`.

Notes
-----
**Developer notes.** A custom definition gets no special trust, so these
tests mostly check that it is held to the same rules as a built-in: its
examples run, unknown keys are refused, a built-in is not silently replaced.
"""

from __future__ import annotations

import json

import pytest

from .._custom import MAX_FILE_BYTES, load_custom, with_custom
from .._exceptions import CleanPromptError
from .._packs import PackError

_HR = {
    "name": "hr",
    "version": 1,
    "summary": "HR ids.",
    "requires": ["personal"],
    "fields": [{"names": ["badge_id"], "kind": "EMPLOYEE", "role": "id"}],
}
_TICKETS = {
    "name": "tickets",
    "version": 1,
    "summary": "Tickets.",
    "extensions": [".tkt"],
    "splitter": "keyvalue",
    "round_trip": True,
    "packs": ["hr"],
}


def _write(folder, name, document):
    path = folder / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class TestLoad:
    def test_one_json_pack(self, tmp_path):
        catalog = load_custom(_write(tmp_path, "hr.json", _HR))
        assert set(catalog.packs) == {"hr"}
        assert catalog.packs["hr"].source == "hr.json"

    def test_shape_decides_pack_format_or_bundle(self, tmp_path):
        bundle = _write(tmp_path, "b.json", {"packs": [_HR], "formats": [_TICKETS]})
        catalog = load_custom([bundle])
        assert set(catalog.packs) == {"hr"} and set(catalog.formats) == {"tickets"}
        assert catalog.packs["hr"].source == "b.json#packs[0].json"

    def test_many_files(self, tmp_path):
        catalog = load_custom(
            [_write(tmp_path, "hr.json", _HR), _write(tmp_path, "t.json", _TICKETS)]
        )
        merged = with_custom([tmp_path / "hr.json", tmp_path / "t.json"])
        assert catalog.formats["tickets"].packs == ("hr",)
        assert merged.format_for("x.tkt").name == "tickets"

    def test_yaml(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        path = tmp_path / "hr.yaml"
        path.write_text(yaml.safe_dump(_HR), encoding="utf-8")
        assert [p.name for p in with_custom(path).resolve_packs("hr")] == [
            "personal",
            "hr",
        ]

    def test_yaml_tags_that_construct_objects_are_refused(self, tmp_path):
        pytest.importorskip("yaml")
        path = tmp_path / "evil.yaml"
        path.write_text(
            "name: !!python/object/apply:os.system ['true']\n", encoding="utf-8"
        )
        with pytest.raises(PackError, match="not valid YAML"):
            load_custom(path)


class TestRefusals:
    def test_missing_file(self, tmp_path):
        with pytest.raises(CleanPromptError, match="does not exist"):
            load_custom(tmp_path / "nope.json")

    def test_unknown_suffix(self, tmp_path):
        path = tmp_path / "hr.toml"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(CleanPromptError, match=".yaml, .yml or .json"):
            load_custom(path)

    def test_oversized_file_is_refused_unread(self, tmp_path):
        path = tmp_path / "big.json"
        path.write_bytes(b" " * (MAX_FILE_BYTES + 1))
        with pytest.raises(CleanPromptError, match="limit"):
            load_custom(path)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(PackError, match="not valid JSON"):
            load_custom(path)

    def test_unknown_requirement_is_caught_on_merge(self, tmp_path):
        path = _write(tmp_path, "hr.json", dict(_HR, requires=["payroll"]))
        with pytest.raises(CleanPromptError, match="payroll"):
            with_custom(path)


class TestConflicts:
    def test_redefining_a_builtin_needs_replace(self, tmp_path):
        personal = dict(_HR, name="personal", requires=[])
        path = _write(tmp_path, "p.json", personal)
        with pytest.raises(CleanPromptError, match="replace"):
            with_custom(path)
        assert (
            with_custom(path, conflict="replace").packs["personal"].source == "p.json"
        )

    def test_none_returns_the_builtins(self):
        from .._catalog import builtin_catalog

        assert with_custom(None) is builtin_catalog()
