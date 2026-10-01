"""Mutation tests for the `cleanprompt` contract checker.

Notes
-----
**Developer notes.** A checker that reports PASS on a healthy tree proves very
little; it might report PASS on anything. Each test here copies the tree, breaks
exactly one property, and asserts that the checker notices *that* property. A
rule with no mutation test behind it is decoration.
"""

from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

check_contract = runpy.run_path(str(TOOLS / "check_contract.py"))
payload = check_contract["payload"]
discover_repo = check_contract["discover_repo"]
tree_fingerprint = check_contract["tree_fingerprint"]
ContractError = check_contract["ContractError"]


@pytest.fixture(scope="module")
def repo():
    """Return the wide checkout root."""
    return discover_repo(Path(__file__).resolve().parent)


@pytest.fixture()
def sandbox(tmp_path, repo):
    """Return a writable copy of the checkout."""
    target = tmp_path / "repo"
    target.mkdir()
    for name in ("scikitplot", "maintenances", "skills"):
        shutil.copytree(
            repo / name,
            target / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return target


def runtime(sandbox):
    """Return the runtime package path inside a sandbox."""
    return sandbox / "scikitplot" / "cleanprompt"


def findings_of(sandbox):
    """Return the runtime findings for a sandbox."""
    return payload(sandbox)["runtime_findings"]


def errors_of(sandbox):
    """Return the maintenance errors for a sandbox."""
    return payload(sandbox)["maintenance_errors"]


def has(items, marker):
    """Return whether any item mentions ``marker``."""
    return any(marker in item for item in items)


class TestHealthyTree:
    """The baseline: a clean tree must be clean."""

    def test_runtime_is_clean(self, repo):
        report = payload(repo)
        assert report["runtime_findings"] == []
        assert report["runtime_status"] == "PASS"

    def test_maintenance_is_clean(self, repo):
        report = payload(repo)
        assert report["maintenance_errors"] == []
        assert report["maintenance_status"] == "PASS"

    def test_release_is_unverified_not_passed(self, repo):
        """Structural green is not release green, and must not read as it."""
        assert payload(repo)["release_status"] == "UNVERIFIED"

    def test_inventory_is_reported(self, repo):
        inventory = payload(repo)["inventory"]
        assert inventory["runtime_modules"] > 10
        assert inventory["focused_test_modules"] > 10

    def test_fingerprint_is_stable(self, repo):
        assert tree_fingerprint(repo) == tree_fingerprint(repo)

    def test_sandbox_matches_the_original(self, sandbox, repo):
        assert tree_fingerprint(sandbox) == tree_fingerprint(repo)

    def test_tool_caches_do_not_change_the_fingerprint(self, sandbox):
        before = tree_fingerprint(sandbox)
        for cache in (".ruff_cache", ".pytest_cache", "__pycache__"):
            (runtime(sandbox) / cache).mkdir(exist_ok=True)
            (runtime(sandbox) / cache / "entry").write_text("x")
        assert tree_fingerprint(sandbox) == before
        (runtime(sandbox) / "new_module.py").write_text("x = 1\n")
        assert tree_fingerprint(sandbox) != before



def _whole_credential(repo, kind):
    """Join the stored fragments of one credential example, at run time."""
    compiled = json.loads(
        (repo / "scikitplot" / "cleanprompt" / "_config" / "_compiled.json").read_text(
            encoding="utf-8"
        )
    )
    for pattern in compiled["packs"]["secrets"]["patterns"]:
        if pattern["kind"] == kind:
            example = pattern["examples_yes"][0]
            assert isinstance(example, list), "credential examples are stored as fragments"
            return "".join(example)
    raise AssertionError(kind)


class TestCredentialsAtRest:
    """CP-REST-001: nothing committed for this subsystem holds a whole credential."""

    @pytest.mark.parametrize(
        "relative",
        [
            "maintenances/cleanprompt/_maintenance/DESIGN.md",
            "skills/cleanprompt/SKILL.md",
            "scikitplot/cleanprompt/README.md",
            "scikitplot/cleanprompt/tests/test__packs.py",
        ],
    )
    def test_a_whole_value_is_caught_wherever_it_is_written(self, sandbox, repo, relative):
        value = _whole_credential(repo, "STRIPE_KEY")
        path = sandbox / relative
        path.write_text(
            path.read_text(encoding="utf-8") + "\nexample: " + value + "\n",
            encoding="utf-8",
        )
        errors = [e for e in errors_of(sandbox) if "CP-REST-001" in e]
        assert len(errors) == 1 and relative in errors[0] and "STRIPE_KEY" in errors[0]
        assert value not in errors[0]

    def test_a_gallery_script_is_covered(self, sandbox, repo):
        folder = sandbox / "galleries" / "examples" / "cleanprompt"
        folder.mkdir(parents=True)
        (folder / "plot_x.py").write_text(
            "TOKEN = '" + _whole_credential(repo, "SLACK_TOKEN") + "'\n", encoding="utf-8"
        )
        assert has(errors_of(sandbox), "CP-REST-001: galleries/examples/cleanprompt/plot_x.py:1")

    def test_fragments_are_accepted(self, sandbox):
        path = sandbox / "skills" / "cleanprompt" / "SKILL.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\nexamples_yes: [['sk_live_', '0123456789abcdefABCDEFGH']]\n",
            encoding="utf-8",
        )
        assert not has(errors_of(sandbox), "CP-REST-001")

    def test_an_unarmed_check_is_an_error_not_a_pass(self, sandbox):
        catalog = runtime(sandbox) / "_catalog.py"
        source = catalog.read_text(encoding="utf-8")
        assert 'AT_REST_PACKS = ("secrets",)' in source
        catalog.write_text(source.replace('AT_REST_PACKS = ("secrets",)', "AT_REST_PACKS = ()"), encoding="utf-8")
        assert has(errors_of(sandbox), "CP-REST-002")

    def test_a_missing_credential_pack_is_an_error(self, sandbox):
        compiled = runtime(sandbox) / "_config" / "_compiled.json"
        document = json.loads(compiled.read_text(encoding="utf-8"))
        del document["packs"]["secrets"]
        compiled.write_text(json.dumps(document), encoding="utf-8")
        assert has(errors_of(sandbox), "CP-REST-002")


class TestTierMutations:
    """The import contract is the submodule's central claim."""

    def test_module_scope_optional_import_is_caught(self, sandbox):
        path = runtime(sandbox) / "_engine.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "from __future__ import annotations",
                "from __future__ import annotations\n\nimport spacy",
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-TIER-001")

    def test_deferred_optional_import_is_accepted(self, sandbox):
        path = runtime(sandbox) / "_engine.py"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n\ndef _later():\n    import spacy\n    return spacy\n",
            encoding="utf-8",
        )
        assert not has(findings_of(sandbox), "CP-TIER-001")

    def test_sibling_submodule_import_is_caught(self, sandbox):
        path = runtime(sandbox) / "_types.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "from __future__ import annotations",
                "from __future__ import annotations\n\nfrom scikitplot.corpus import x",
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-INDEP-001")

    def test_a_deferred_corpus_import_in_the_bridge_is_permitted(self, sandbox):
        """The one sanctioned sibling import: _corpus.py, inside a function."""
        (runtime(sandbox) / "_corpus.py").write_text(
            '"""Bridge."""\n\n\ndef read():\n'
            '    """Read."""\n'
            "    from scikitplot.corpus import DocumentReader\n"
            "    return DocumentReader\n",
            encoding="utf-8",
        )
        assert not has(findings_of(sandbox), "CP-INDEP-001")

    def test_a_module_scope_corpus_import_in_the_bridge_is_caught(self, sandbox):
        (runtime(sandbox) / "_corpus.py").write_text(
            '"""Bridge."""\n\nfrom scikitplot.corpus import DocumentReader\n',
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-INDEP-001")

    def test_a_deferred_corpus_import_elsewhere_is_caught(self, sandbox):
        """The exception is one module, not a licence for deferral anywhere."""
        path = runtime(sandbox) / "_types.py"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n\ndef _later():\n    from scikitplot.corpus import x\n    return x\n",
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-INDEP-001")

    def test_another_sibling_in_the_bridge_is_caught(self, sandbox):
        (runtime(sandbox) / "_corpus.py").write_text(
            '"""Bridge."""\n\n\ndef read():\n'
            '    """Read."""\n'
            "    from scikitplot.mcp import x\n"
            "    return x\n",
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-INDEP-001")

    def test_developer_plane_import_is_caught(self, sandbox):
        path = runtime(sandbox) / "_types.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "from __future__ import annotations",
                "from __future__ import annotations\n\nimport maintenances.cleanprompt",
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-PLANE-001")


class TestFacadeMutations:
    """The lazy facade must keep its exact shape."""

    def test_optional_name_in_all_is_caught(self, sandbox):
        """The exact regression CP-017: an optional name appended to __all__."""
        path = runtime(sandbox) / "__init__.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '    # metadata\n    "__version__",',
                '    # metadata\n    "__version__",\n    "spacy_detector",',
                1,
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-FACADE-002")

    def test_aggregated_all_is_resolved_not_reported_empty(self, repo):
        """A checker that cannot read the contract it guards is worthless."""
        assert not has(payload(repo)["runtime_findings"], "CP-FACADE-001")
        assert not has(payload(repo)["runtime_findings"], "CP-FACADE-004")

    def test_unresolvable_all_is_reported_as_such(self, sandbox):
        path = runtime(sandbox) / "__init__.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "__all__ = []", "__all__ = list(_dynamic())", 1
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-FACADE-004")

    def test_missing_required_symbol_is_caught(self, sandbox):
        """Dropping a private module's export drops it from the facade too."""
        path = runtime(sandbox) / "_engine.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"Redactor",\n', "", 1),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-FACADE-001")

    def test_missing_getattr_is_caught(self, sandbox):
        path = runtime(sandbox) / "__init__.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "def __getattr__(", "def _disabled_getattr("
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-FACADE-003")

    def test_missing_dir_is_caught(self, sandbox):
        path = runtime(sandbox) / "__init__.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "def __dir__(", "def _disabled_dir("
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-FACADE-003")


class TestCapabilityMutations:
    """Capability truth."""

    def test_missing_vocabulary_member_is_caught(self, sandbox):
        path = runtime(sandbox) / "_capabilities.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace('BROKEN = "BROKEN"', 'BROKEN = "X"'),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-CAPS-001")

    def test_find_spec_usage_is_caught(self, sandbox):
        path = runtime(sandbox) / "_capabilities.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "def probe(tier: str)",
                "def _bad(name):\n"
                "    from importlib.util import find_spec\n"
                "    return find_spec(name)\n\n\n"
                "def probe(tier: str)",
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-CAPS-002")

    def test_the_explanatory_docstring_is_not_a_false_positive(self, repo):
        """The module explains why find_spec is unsuitable; that is not usage."""
        source = (
            repo / "scikitplot" / "cleanprompt" / "_capabilities.py"
        ).read_text(encoding="utf-8")
        assert "find_spec" in source
        assert not has(payload(repo)["runtime_findings"], "CP-CAPS-002")


class TestSafetyMutations:
    """Dynamic evaluation and swallowed errors."""

    def test_eval_is_caught(self, sandbox):
        path = runtime(sandbox) / "_types.py"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n\ndef _bad(s):\n    return eval(s)\n",
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-SAFE-001")

    def test_bare_except_is_caught(self, sandbox):
        path = runtime(sandbox) / "_types.py"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n\ndef _bad():\n    try:\n        pass\n    except:\n        return None\n",
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-SAFE-002")

    def test_silent_pass_is_caught(self, sandbox):
        path = runtime(sandbox) / "_types.py"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n\ndef _bad():\n    try:\n        pass\n    except ValueError:\n        pass\n",
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-SAFE-002")


class TestStructureMutations:
    """Layout and ownership."""

    def test_missing_runtime_module_is_caught(self, sandbox):
        (runtime(sandbox) / "_vault.py").unlink()
        assert has(findings_of(sandbox), "CP-STRUCT")

    def test_orphaned_source_module_is_caught(self, sandbox):
        (runtime(sandbox) / "_extra.py").write_text(
            '"""Extra."""\n\nfrom __future__ import annotations\n\n__all__ = []\n',
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-TEST-001")

    def test_missing_regression_module_is_caught(self, sandbox):
        (runtime(sandbox) / "tests" / "test_regressions.py").unlink()
        assert has(findings_of(sandbox), "CP-TEST-002")

    def test_missing_module_docstring_is_caught(self, sandbox):
        path = runtime(sandbox) / "_extra.py"
        path.write_text("from __future__ import annotations\n\n__all__ = []\n", encoding="utf-8")
        (runtime(sandbox) / "tests" / "test__extra.py").write_text(
            '"""Owns _extra."""\n', encoding="utf-8"
        )
        assert has(findings_of(sandbox), "CP-DOC-001")

    def test_missing_package_is_caught(self, sandbox):
        shutil.rmtree(runtime(sandbox))
        assert has(findings_of(sandbox), "package is missing")


class TestFrontendMutations:
    """The two-frontend contract."""

    def test_missing_commands_table_is_caught(self, sandbox):
        path = runtime(sandbox) / "_cli.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "COMMANDS: tuple[Command, ...] = (", "_RENAMED: tuple = (", 1
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-FRONT-001")

    def test_cli_not_dispatching_through_load_runner_is_caught(self, sandbox):
        path = runtime(sandbox) / "_cli.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace("load_runner", "_bypass"),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-FRONT-001")

    def test_load_runner_dropping_a_frontend_is_caught(self, sandbox):
        """CP-047: the property is the dispatch path, not an import line."""
        path = runtime(sandbox) / "_frontends.py"
        source = path.read_text(encoding="utf-8")
        start = source.index("def load_runner(")
        end = source.index("\ndef ", start + 1) if "\ndef " in source[start + 1 :] else len(source)
        body = source[start:end].replace("run_click", "run_argparse")
        path.write_text(source[:start] + body + source[end:], encoding="utf-8")
        assert has(findings_of(sandbox), "CP-FRONT-001")

    def test_an_unused_runner_import_is_not_required(self, sandbox):
        """The tree a linter produces must pass: dead imports prove nothing."""
        assert not has(findings_of(sandbox), "CP-FRONT-001")

    def test_module_scope_click_import_is_caught(self, sandbox):
        path = runtime(sandbox) / "_frontends.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "from __future__ import annotations",
                "from __future__ import annotations\n\nimport click",
                1,
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-FRONT-002")

    def test_missing_diagnosis_function_is_caught(self, sandbox):
        path = runtime(sandbox) / "_diagnostics.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "def suggest_terms(", "def _renamed_suggest(", 1
            ),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-DIAG-001")

    def test_web_tier_ignoring_the_diagnosis_is_caught(self, sandbox):
        """The defect that prompted the rewrite must stay impossible."""
        path = runtime(sandbox) / "_app.py"
        source = path.read_text(encoding="utf-8")
        path.write_text(source.replace("diagnose", "_no_diagnosis"), encoding="utf-8")
        assert has(findings_of(sandbox), "CP-DIAG-002")


class TestWebMutations:
    """The web tier's construction-time contract."""

    def test_module_level_app_is_caught(self, sandbox):
        path = runtime(sandbox) / "_app.py"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n\napp = Flask(__name__)\n",
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-WEB-001")

    def test_missing_constant_time_compare_is_caught(self, sandbox):
        path = runtime(sandbox) / "_app.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace("compare_digest", "_plain_equals"),
            encoding="utf-8",
        )
        assert has(findings_of(sandbox), "CP-WEB-002")


class TestMaintenanceMutations:
    """The plane's own consistency."""

    def test_missing_document_is_caught(self, sandbox):
        (sandbox / "maintenances" / "cleanprompt" / "_maintenance" / "DESIGN.md").unlink()
        assert has(errors_of(sandbox), "DESIGN.md")

    def test_missing_skill_is_caught(self, sandbox):
        (sandbox / "skills" / "cleanprompt" / "SKILL.md").unlink()
        assert has(errors_of(sandbox), "SKILL.md")

    def test_shallow_skill_is_caught(self, sandbox):
        (sandbox / "skills" / "cleanprompt" / "SKILL.md").write_text(
            "# too short\n", encoding="utf-8"
        )
        assert has(errors_of(sandbox), "too shallow")

    def test_executable_field_in_metadata_is_caught(self, sandbox):
        path = sandbox / "maintenances" / "cleanprompt" / "MAINTENANCE.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["command"] = "rm -rf /"
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        assert has(errors_of(sandbox), "executable field")

    def test_closed_finding_without_a_regression_is_caught(self, sandbox):
        path = sandbox / "maintenances" / "cleanprompt" / "REVIEW.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["findings"][0].pop("regression", None)
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        assert has(errors_of(sandbox), "closed with no named regression")

    def test_unavailable_lane_without_a_reason_is_caught(self, sandbox):
        path = (
            sandbox / "maintenances" / "cleanprompt" / "_maintenance" / "EVIDENCE.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        for lane in document["lanes"]:
            if lane["status"] == "UNAVAILABLE":
                lane.pop("reason")
                break
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        assert has(errors_of(sandbox), "UNAVAILABLE with no reason")

    def test_stale_fingerprint_is_caught(self, sandbox):
        path = (
            sandbox / "maintenances" / "cleanprompt" / "_maintenance" / "EVIDENCE.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        document["runtime_tree_fingerprint"] = "0" * 64
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        assert has(errors_of(sandbox), "runtime_tree_fingerprint does not match")

    def test_malformed_json_is_caught(self, sandbox):
        (sandbox / "maintenances" / "cleanprompt" / "REVIEW.json").write_text(
            "{not json", encoding="utf-8"
        )
        assert has(errors_of(sandbox), "not valid JSON")


class TestDiscovery:
    """Locating the checkout."""

    def test_finds_the_root(self, repo):
        assert (repo / "scikitplot").is_dir()

    def test_falls_back_to_the_tool_s_own_checkout(self, tmp_path, repo):
        """An unrelated start still resolves, via the checker's own location.

        Notes
        -----
        **Developer notes.** Deliberate: the checker is usually invoked by path
        from an arbitrary working directory, so it searches upward from the
        caller *and* from itself. Asserting a refusal here would encode the
        opposite contract.
        """
        assert discover_repo(tmp_path) == repo

    def test_refuses_when_neither_start_is_inside_a_checkout(self, tmp_path):
        """With the tool copied outside any checkout, discovery must fail."""
        isolated = tmp_path / "isolated"
        isolated.mkdir()
        shutil.copy(TOOLS / "check_contract.py", isolated / "check_contract.py")
        module = runpy.run_path(str(isolated / "check_contract.py"))
        with pytest.raises(module["ContractError"], match="wide repository root"):
            module["discover_repo"](tmp_path)
