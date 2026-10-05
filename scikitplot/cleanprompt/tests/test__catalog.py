"""
Tests for :mod:`scikitplot.cleanprompt._catalog`.

Notes
-----
**Developer notes.** ``TestCompiledMatchesYaml`` is invariant ``I10``: the
JSON the base tier reads must be exactly what the YAML compiles to. It is
the test that fails when someone edits a YAML pack and forgets to run
``packs --compile``, which is the whole reason the compiled file can be
trusted without PyYAML installed.
"""

from __future__ import annotations

import json
import shutil

import pytest

from .._catalog import (
    AT_REST_PACKS,
    COMPILED_PATH,
    CONFIG_DIR,
    at_rest_findings,
    builtin_catalog,
    canonical,
    catalog_from_documents,
    check_compiled,
    write_compiled,
)
from .._exceptions import CleanPromptError
from .._packs import PackError


class TestCompiledMatchesYaml:
    def test_no_drift(self):
        pytest.importorskip("yaml")
        assert check_compiled() == []

    def test_drift_is_reported_with_the_fix(self, tmp_path):
        pytest.importorskip("yaml")
        folder = tmp_path / "config"
        shutil.copytree(CONFIG_DIR, folder)
        target = folder / "_compiled.json"
        document = json.loads(target.read_text(encoding="utf-8"))
        document["packs"]["personal"]["summary"] = "edited by hand"
        target.write_text(json.dumps(document), encoding="utf-8")
        problems = check_compiled(folder)
        assert "pack 'personal' differs from its YAML source" in problems
        assert problems[-1].startswith("run: ")

    def test_write_then_check_is_clean(self, tmp_path):
        pytest.importorskip("yaml")
        folder = tmp_path / "config"
        shutil.copytree(CONFIG_DIR, folder)
        (folder / "_compiled.json").unlink()
        assert "missing" in check_compiled(folder)[0]
        write_compiled(folder)
        assert check_compiled(folder) == []

    def test_compiling_twice_writes_the_same_bytes(self, tmp_path):
        pytest.importorskip("yaml")
        folder = tmp_path / "config"
        shutil.copytree(CONFIG_DIR, folder)
        first = write_compiled(folder).read_bytes()
        assert write_compiled(folder).read_bytes() == first
        assert first == COMPILED_PATH.read_bytes()

    def test_compiled_file_is_json_the_stdlib_reads(self):
        assert json.loads(COMPILED_PATH.read_text(encoding="utf-8"))["schema"] == 1



def _credential_packs():
    catalog = builtin_catalog()
    return [catalog.packs[name] for name in AT_REST_PACKS]


def _whole(kind):
    """Return the joined positive example of a credential pattern, built at run time."""
    for pack in _credential_packs():
        for spec in pack.patterns:
            if spec.kind == kind:
                return spec.examples_yes[0]
    raise AssertionError(kind)


class TestNoCredentialAtRest:
    """
    Invariant ``I14``: no built-in definition file holds a whole credential.

    A positive example for a key pattern is, by construction, a string a
    secret scanner blocks. Written whole in ``secrets.yaml`` it was copied
    into ``_compiled.json`` and the push carrying both was refused. Examples
    are now written as fragments, and these tests hold the files to the
    package's own patterns so the next pattern added cannot reintroduce it.
    """

    def test_the_credential_packs_exist_and_have_patterns(self):
        """An empty guard would pass everything; make sure it is armed."""
        packs = _credential_packs()
        assert [pack.name for pack in packs] == list(AT_REST_PACKS)
        assert all(pack.patterns for pack in packs)

    def test_no_built_in_definition_file_matches(self):
        texts = {
            str(path.relative_to(CONFIG_DIR)): path.read_text(encoding="utf-8")
            for path in sorted(CONFIG_DIR.rglob("*"))
            if path.is_file() and path.suffix in {".yaml", ".yml", ".json"}
        }
        assert COMPILED_PATH.name in texts and "packs/secrets.yaml" in {
            name.replace("\\", "/") for name in texts
        }
        assert at_rest_findings(texts, _credential_packs()) == []

    def test_every_credential_example_is_stored_as_fragments(self):
        """The joined value of each positive example appears in neither file."""
        stored = COMPILED_PATH.read_text(encoding="utf-8")
        source = (CONFIG_DIR / "packs" / "secrets.yaml").read_text(encoding="utf-8")
        for pack in _credential_packs():
            for spec in pack.patterns:
                for example in spec.examples_yes:
                    assert example not in source, spec.kind
                    assert json.dumps(example)[1:-1] not in stored, spec.kind

    @pytest.mark.parametrize(
        "kind",
        [spec.kind for name in AT_REST_PACKS for spec in builtin_catalog().packs[name].patterns],
    )
    def test_a_whole_value_is_found_and_not_repeated(self, kind):
        value = _whole(kind)
        findings = at_rest_findings({"x.yaml": "first line\nv: " + value}, _credential_packs())
        assert any(f.startswith("x.yaml:2: ") and kind in f for f in findings), findings
        assert all(value not in finding for finding in findings)

    def test_findings_are_sorted_and_deterministic(self):
        text = "\n".join(_whole(kind) for kind in ("SLACK_TOKEN", "STRIPE_KEY", "SLACK_TOKEN"))
        first = at_rest_findings({"b.yaml": text, "a.yaml": text}, _credential_packs())
        assert first == sorted(first)
        assert first == at_rest_findings({"a.yaml": text, "b.yaml": text}, _credential_packs())
        assert len(first) == len(set(first))

    def test_clean_text_and_no_packs_report_nothing(self):
        assert at_rest_findings({"a.yaml": "sk_live_short and bearer of bad news"}, _credential_packs()) == []
        assert at_rest_findings({"a.yaml": _whole("STRIPE_KEY")}, []) == []
        assert at_rest_findings({}, _credential_packs()) == []

    def test_compile_refuses_a_whole_example_in_the_yaml(self, tmp_path):
        """The compiled file cannot be produced in a state a scanner refuses."""
        pytest.importorskip("yaml")
        folder = tmp_path / "config"
        shutil.copytree(CONFIG_DIR, folder)
        source = folder / "packs" / "secrets.yaml"
        text = source.read_text(encoding="utf-8")
        fragments = "[['xoxb-', '1234567890-abcdefghij']]"
        assert text.count(fragments) == 1
        source.write_text(text.replace(fragments, "['" + _whole("SLACK_TOKEN") + "']"), encoding="utf-8")
        before = (folder / "_compiled.json").read_bytes()
        with pytest.raises(PackError, match="SLACK_TOKEN-shaped") as caught:
            write_compiled(folder)
        assert any(p.startswith("packs/secrets.yaml:") for p in caught.value.problems)
        assert any(p.startswith("_compiled.json:") for p in caught.value.problems)
        assert _whole("SLACK_TOKEN") not in str(caught.value)
        assert (folder / "_compiled.json").read_bytes() == before

    def test_check_reports_a_hand_edited_compiled_file(self, tmp_path):
        pytest.importorskip("yaml")
        folder = tmp_path / "config"
        shutil.copytree(CONFIG_DIR, folder)
        target = folder / "_compiled.json"
        document = json.loads(target.read_text(encoding="utf-8"))
        for pattern in document["packs"]["secrets"]["patterns"]:
            pattern["examples_yes"] = ["".join(e) if isinstance(e, list) else e for e in pattern["examples_yes"]]
        target.write_text(json.dumps(document, sort_keys=True, indent=1) + "\n", encoding="utf-8")
        problems = check_compiled(folder)
        assert any("STRIPE_KEY-shaped" in p and p.startswith("_compiled.json:") for p in problems)
        assert problems[-1].startswith("run: ")
        write_compiled(folder)
        assert check_compiled(folder) == []


class TestBuiltinCatalog:
    def test_is_cached(self):
        assert builtin_catalog() is builtin_catalog()

    def test_all_packs_agree_on_every_field(self):
        """A field may mean one thing across the whole built-in catalog."""
        catalog = builtin_catalog()
        index = catalog.field_index(catalog.resolve_packs("all"))
        assert index["email"].kind == "EMAIL"

    def test_every_format_names_known_packs_and_unique_extensions(self):
        catalog = builtin_catalog()
        seen = {}
        for spec in catalog.formats.values():
            assert set(spec.packs) <= set(catalog.packs)
            for extension in spec.extensions:
                assert extension not in seen, (
                    extension,
                    seen.get(extension),
                    spec.name,
                )
                seen[extension] = spec.name


class TestResolvePacks:
    def test_requirements_come_first(self):
        names = [p.name for p in builtin_catalog().resolve_packs(["addressbook"])]
        assert names == ["personal", "addressbook"]

    def test_modes(self):
        catalog = builtin_catalog()
        assert catalog.resolve_packs("none") == ()
        assert len(catalog.resolve_packs("all")) == len(catalog.packs)
        csv = catalog.resolve_formats("csv")
        assert {p.name for p in catalog.resolve_packs("auto", formats=csv)} >= set(
            csv[0].packs
        )

    def test_unknown_name_suggests_the_nearest(self):
        with pytest.raises(CleanPromptError, match="patient"):
            builtin_catalog().resolve_packs(["patinet"])

    def test_mode_cannot_be_combined_with_names(self):
        with pytest.raises(CleanPromptError, match="on its own"):
            builtin_catalog().resolve_packs(["all", "patient"])


class TestResolveFormats:
    @pytest.mark.parametrize(
        ("given", "expected"), [("csv", "csv"), (".ipynb", "notebook"), ("sh", "shell")]
    )
    def test_by_name_or_extension(self, given, expected):
        (spec,) = builtin_catalog().resolve_formats(given)
        assert spec.name == expected

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("deploy/.env", "env"),
            ("NOTES.MD", "markdown"),
            ("a.tar.gz", None),
            ("Makefile", None),
        ],
    )
    def test_format_for(self, path, expected):
        found = builtin_catalog().format_for(path)
        assert (found.name if found else None) == expected


class TestFingerprint:
    def test_changes_with_the_selection(self):
        catalog = builtin_catalog()
        packs = catalog.resolve_packs(["patient", "finance"])
        formats = catalog.resolve_formats(["csv", "json"])
        assert catalog.fingerprint(packs, formats) == catalog.fingerprint(
            packs, formats
        )
        assert catalog.fingerprint(packs, formats) != catalog.fingerprint(
            packs[:1], formats
        )

    def test_canonical_is_order_independent(self):
        assert canonical({"b": 1, "a": [2]}) == canonical({"a": [2], "b": 1})


class TestConsistency:
    _FORMAT = {
        "name": "tickets",
        "version": 1,
        "summary": "Tickets.",
        "extensions": [".csv"],
        "splitter": "delimited",
        "round_trip": True,
    }

    def test_two_formats_cannot_claim_one_extension(self):
        with pytest.raises(CleanPromptError, match=".csv"):
            builtin_catalog().merge(
                catalog_from_documents({}, {"t.json": self._FORMAT}, partial=True)
            )

    def test_a_format_cannot_name_an_unknown_pack(self):
        document = dict(self._FORMAT, extensions=[".tkt"], packs=["nope"])
        with pytest.raises(CleanPromptError, match="nope"):
            catalog_from_documents({}, {"t.json": document})
