# scikitplot/tests/test_usage_skills.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The usage skills of the ``scikitplot`` plugin match the installed package (Rule 165).

``plugins/scikitplot/skills/`` is generated from each listed module's
``__all__`` and docstrings by ``tools/maint_tools/generate_usage_skills.py``.
A change to a public API or a docstring makes this test fail until the skills
are regenerated with ``python tools/maint_tools/generate_usage_skills.py sync``.

Notes
-----
The first two tests are the contract with the repository (skills current,
every public subpackage listed or excluded). The rest pin the generator's own
behaviour (error codes and their guideline, stale classification, skill lint,
command line) without writing to the checkout.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "tools/maint_tools/generate_usage_skills.py"

pytestmark = pytest.mark.skipif(
    not GENERATOR.is_file() or not (REPO / "plugins/scikitplot").is_dir(),
    reason="not a scikit-plots checkout (the generator and the plugin live there)",
)

#: Codes that mean "this environment cannot evaluate the package", not "the skills are wrong".
_ENVIRONMENT_CODES = ("E-ENV", "E-IMPORT")


def _load(register: bool):
    spec = importlib.util.spec_from_file_location("_generate_usage_skills", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


@pytest.fixture(scope="module")
def generator():
    module = _load(register=True)
    try:
        yield module
    finally:
        sys.modules.pop("_generate_usage_skills", None)


def _skip_if_environment(generator, messages):
    """Skip when the package cannot be read here (optional dependency missing)."""
    text = " ".join(messages)
    if any(f"[{code}]" in text for code in _ENVIRONMENT_CODES) or "do not resolve here" in text:
        pytest.skip(f"an optional dependency is missing here: {text}")


# ----------------------------------------------------------------------- repository contract


def test_the_usage_skills_are_current(generator):
    report = generator.check(REPO)
    _skip_if_environment(generator, report.errors)
    guide = generator.remediation(report.errors)
    steps = "\n".join(
        f"[{g['code']}] {g['title']}: " + " | ".join(g["steps"]) for g in guide
    )
    assert report.ok, (
        "usage skills are stale or invalid. Run "
        "`python tools/maint_tools/generate_usage_skills.py explain` to see why, then "
        "`... sync` (never edit generated files).\n"
        + "\n".join(report.errors)
        + "\n"
        + steps
    )
    assert report.stale == []


def test_build_and_current_agree_with_check(generator):
    """``build``/``current`` (the pre-existing API) still describe the same tree as ``check``."""
    try:
        wanted = generator.build(REPO)
    except generator.SkillError as exc:
        _skip_if_environment(generator, exc.messages)
        raise
    have = generator.current(REPO)
    stale = sorted(p for p in set(wanted) | set(have) if wanted.get(p) != have.get(p))
    assert stale == generator.check(REPO).stale


def test_every_public_subpackage_is_listed_or_excluded(generator):
    """A new public subpackage gets a usage skill or a stated reason (Rule 165)."""
    policy = generator.load_policy(REPO)
    known = set(policy["usage_modules"]) | set(policy["usage_excluded"])
    package = REPO / "scikitplot"
    public = {
        f"scikitplot.{p.name}"
        for p in package.iterdir()
        if p.is_dir()
        and not p.name.startswith(("_", "."))
        and (p / "__init__.py").is_file()
        and p.name not in ("tests",)
    }
    assert sorted(public - known) == []


# ----------------------------------------------------------------------- loading


def test_the_generator_loads_without_being_registered_in_sys_modules():
    """
    Regression: ``exec_module`` of an unregistered module broke ``@dataclass``.

    The dataclasses resolve ``from __future__ import annotations`` strings through
    ``sys.modules``; the generator must import either way and leave nothing behind.
    """
    sys.modules.pop("_generate_usage_skills", None)
    module = _load(register=False)
    assert "_generate_usage_skills" not in sys.modules
    assert module.Report([], [], [], {}).ok is True


# ----------------------------------------------------------------------- report and guideline


def test_report_ok_means_no_errors(generator):
    assert generator.Report([], ["w"], ["p"], {}).ok is True
    assert generator.Report(["e"], [], [], {}).ok is False


def test_every_code_the_generator_can_emit_has_a_guideline(generator):
    codes = {
        value
        for name, value in vars(generator).items()
        if re.fullmatch(r"[EW]_[A-Z]+", name) and isinstance(value, str)
    }
    assert codes, "no E_*/W_* constants found"
    assert codes == set(generator.GUIDE), "every code needs exactly one guideline entry"
    for code, (title, steps) in generator.GUIDE.items():
        assert title and steps and all(isinstance(s, str) and s for s in steps), code


def test_remediation_is_ordered_unique_and_ignores_unknown_codes(generator):
    messages = [
        "[E-STALE] a",
        "[W-DOC] b",
        "[E-STALE] again",
        "[E-NOPE] unknown",
        "untagged text",
    ]
    assert [e["code"] for e in generator.remediation(messages)] == ["E-STALE", "W-DOC"]


def test_skill_error_carries_its_code_and_tagged_message(generator):
    exc = generator.SkillError("boom", generator.E_API)
    assert exc.code == "E-API"
    assert exc.messages == ["[E-API] boom"]
    many = generator.SkillError("a; b", generator.E_IMPORT, messages=["[E-IMPORT] a", "[E-API] b"])
    assert many.messages == ["[E-IMPORT] a", "[E-API] b"]


def test_stale_files_are_classified_missing_outdated_orphaned(generator):
    wanted = {"a": "1", "b": "2", "c": "3"}
    have = {"b": "2", "c": "changed", "z": "orphan"}
    assert generator.classify_stale(wanted, have) == {
        "missing": ["a"],
        "outdated": ["c"],
        "orphaned": ["z"],
    }


# ----------------------------------------------------------------------- policy and lint


def test_load_policy_rejects_malformed_tables(generator, tmp_path):
    def policy(body: str):
        (tmp_path / "pyproject.toml").write_text(
            "[tool.scikitplot.maintenance.skills]\n" + body, encoding="utf-8"
        )
        return generator.load_policy(tmp_path)

    assert policy('usage_modules = ["scikitplot"]')["usage_modules"] == ["scikitplot"]
    with pytest.raises(generator.SkillError, match="twice"):
        policy('usage_modules = ["scikitplot", "scikitplot"]')
    with pytest.raises(generator.SkillError, match="both"):
        policy('usage_modules = ["scikitplot.a"]\nusage_excluded = {"scikitplot.a" = "why"}')
    with pytest.raises(generator.SkillError, match="starting with 'scikitplot'"):
        policy('usage_modules = ["numpy"]')
    with pytest.raises(generator.SkillError, match="reasons"):
        policy('usage_modules = []\nusage_excluded = {"scikitplot.a" = ""}')
    (tmp_path / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")
    with pytest.raises(generator.SkillError, match="has no"):
        generator.load_policy(tmp_path)


def _skill(name: str, description: str = '"Use it."', link: str = "") -> dict:
    text = f"---\nname: {name}\ndescription: {description}\n---\n\n# x\n{link}\n"
    return {"SKILL.md": text}


def test_lint_accepts_a_wellformed_skill(generator):
    errors, warnings = generator.lint_files("scikitplot.x", "scikitplot-x", _skill("scikitplot-x"))
    assert (errors, warnings) == ([], [])


@pytest.mark.parametrize(
    ("name", "files", "needle"),
    [
        ("Scikitplot_X", _skill("Scikitplot_X"), "lower-case"),
        ("scikitplot-x", _skill("other"), "differs from the folder"),
        ("scikitplot-x", _skill("scikitplot-x", description='""'), "description"),
        ("scikitplot-x", _skill("scikitplot-x", link="[a](references/API.md)"), "not rendered"),
        ("scikitplot-x", {}, "was not rendered"),
        ("scikitplot-x", {"SKILL.md": "no frontmatter"}, "no frontmatter"),
    ],
)
def test_lint_rejects_a_broken_skill(generator, name, files, needle):
    errors, _ = generator.lint_files("scikitplot.x", name, files)
    assert errors and all(e.startswith("[E-LINT]") for e in errors)
    assert any(needle in e for e in errors)


def test_lint_api_warns_about_weak_skills_without_failing(generator):
    row = {"name": "f", "kind": "function", "signature": "`f(...)`", "summary": ""}
    warnings = generator.lint_api("scikitplot.x", {"rows": [row], "children": [], "example": ""})
    codes = {w.split("]")[0][1:] for w in warnings}
    assert codes == {"W-DOC", "W-SIGNATURE", "W-EXAMPLE"}
    empty = generator.lint_api("scikitplot.y", {"rows": [], "children": [], "example": ""})
    assert [w.split("]")[0][1:] for w in empty] == ["W-EMPTY"]


# ----------------------------------------------------------------------- command line


def test_guide_command_lists_every_code(generator):
    out, err = io.StringIO(), io.StringIO()
    assert generator.main(["guide"], stdout=out, stderr=err) == 0
    assert all(f"[{code}]" in out.getvalue() for code in generator.GUIDE)
    as_json = io.StringIO()
    assert generator.main(["guide", "--json"], stdout=as_json, stderr=err) == 0
    assert {e["code"] for e in json.loads(as_json.getvalue())} == set(generator.GUIDE)


def test_check_json_has_the_report_fields(generator):
    out, err = io.StringIO(), io.StringIO()
    status = generator.main(["--repo-root", str(REPO), "check", "--json"], stdout=out, stderr=err)
    payload = json.loads(out.getvalue())
    assert set(payload) == {"ok", "errors", "warnings", "stale", "counts", "remediation"}
    _skip_if_environment(generator, payload["errors"])
    assert status == (0 if payload["ok"] else 1)


def test_repo_root_is_accepted_before_and_after_the_command(generator):
    first, second = io.StringIO(), io.StringIO()
    err = io.StringIO()
    a = generator.main(["--repo-root", str(REPO), "list", "--json"], stdout=first, stderr=err)
    b = generator.main(["list", "--json", "--repo-root", str(REPO)], stdout=second, stderr=err)
    assert a == b
    if a == 0:
        assert first.getvalue() == second.getvalue()


def test_a_directory_without_pyproject_is_a_usage_error(generator, tmp_path):
    err = io.StringIO()
    assert generator.main(["--repo-root", str(tmp_path), "check"], stdout=io.StringIO(), stderr=err) == 2
    assert "no pyproject.toml" in err.getvalue()
