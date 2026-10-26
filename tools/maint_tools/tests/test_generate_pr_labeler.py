# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for tools/maint_tools/generate_pr_labeler.py.

Covers the glob matcher (and, when Node and minimatch are installed, a
cross-check against minimatch itself), the actions/labeler v7 grouping and
matching semantics, the negated-glob lint, the title grammar, policy
validation, stale detection, and finally the repository gate: the checked-in
configuration must pass ``check`` and be in sync.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "generate_pr_labeler.py"
SPEC = importlib.util.spec_from_file_location("generate_pr_labeler", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

REPO = Path(__file__).resolve().parents[3]

try:
    import yaml  # noqa: F401

    HAVE_YAML = True
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    HAVE_YAML = False


def _groups(raw):
    return MODULE.normalise_label_config("x", raw)


def _policy_data():
    data = MODULE._load_pyproject(REPO)
    return copy.deepcopy(data)


def _labeler_table(data):
    return data["tool"]["scikitplot"]["maintenance"]["labeler"]


def _labeler_tables():
    """Return only the ``[tool.scikitplot.maintenance.labeler*]`` tables of pyproject.toml."""
    keep, out = False, []
    for line in (REPO / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            keep = stripped.lstrip("[").startswith("tool.scikitplot.maintenance.labeler")
        if keep:
            out.append(line)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- globs


class GlobTests(unittest.TestCase):
    def assertMatches(self, glob, path, expected=True):
        self.assertIs(MODULE.compile_glob(glob)(path), expected, f"{glob!r} vs {path!r}")

    def test_star_stays_inside_one_segment(self):
        self.assertMatches("scikitplot/*", "scikitplot/a.py")
        self.assertMatches("scikitplot/*", "scikitplot/sub/a.py", False)

    def test_leading_globstar_matches_zero_or_more_directories(self):
        self.assertMatches("**/conftest.py", "conftest.py")
        self.assertMatches("**/conftest.py", "a/b/conftest.py")
        self.assertMatches("**/tests/**/*", "tests/a.py")
        self.assertMatches("**/tests/**/*", "x/tests/y/a.py")
        self.assertMatches("**/tests/**/*", "x/tests", False)

    def test_middle_globstar(self):
        self.assertMatches("docs/**/*.rst", "docs/a.rst")
        self.assertMatches("docs/**/*.rst", "docs/x/y/a.rst")
        self.assertMatches("docs/**/*.rst", "docsx/a.rst", False)

    def test_dot_true_matches_dotfiles_and_dot_directories(self):
        self.assertMatches(".github/**/*", ".github/workflows/a.yml")
        self.assertMatches("**/*", ".pre-commit-config.yaml")
        self.assertMatches("*", ".flake8")

    def test_question_mark_is_one_character(self):
        self.assertMatches("a?.py", "ab.py")
        self.assertMatches("a?.py", "a.py", False)
        self.assertMatches("a?.py", "a/.py", False)

    def test_literal_characters_are_not_regex(self):
        self.assertMatches("a.b", "axb", False)
        self.assertMatches("lang-c_1.txt", "lang-c_1.txt")

    def test_negation_inverts(self):
        self.assertMatches("!docs/**/*", "src/a.py")
        self.assertMatches("!docs/**/*", "docs/a.rst", False)

    def test_refused_syntax(self):
        for glob in (
            "",
            "!!a",
            "/abs",
            "./rel",
            "dir/",
            "docs/**",
            "a/**/**/b",
            "a**b",
            "a//b",
            "a/../b",
            "{a,b}",
            "[ab]",
            "+(a)",
            "a b",
        ):
            with self.subTest(glob=glob), self.assertRaises(MODULE.LabelerError):
                MODULE.validate_glob(glob)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_matches_minimatch_when_available(self):
        script = (
            "let mm;try{mm=require('minimatch')}catch(e){process.exit(3)}"
            "const m=mm.minimatch||mm;const c=JSON.parse(require('fs').readFileSync(0,'utf8'));"
            "process.stdout.write(JSON.stringify(c.map(([g,p])=>m(p,g,{dot:true}))))"
        )
        globs = ["**/tests/**/*", "**/conftest.py", "docs/**/*.rst", "scikitplot/*",
                 ".github/**/*", "a?.py", "!docs/**/*", "**/*", "*", "x/*/y/**/*.p?"]
        paths = ["conftest.py", "a/tests/b.py", "tests/x", "docs/a.rst", "docs/x/a.rst",
                 "scikitplot/a.py", "scikitplot/b/c.py", ".github/w.yml", "ab.py", "a.py",
                 ".flake8", "x/q/y/z/a.py", "x/q/y/a.pyx", "src/a.py"]
        cases = [[g, p] for g in globs for p in paths]
        result = subprocess.run(  # noqa: S603
            ["node", "-e", script],  # noqa: S607
            input=json.dumps(cases), capture_output=True, text=True, check=False,
        )
        if result.returncode == 3:
            self.skipTest("minimatch is not installed for node")
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = json.loads(result.stdout)
        actual = [MODULE.compile_glob(g)(p) for g, p in cases]
        mismatches = [c for c, e, a in zip(cases, expected, actual) if e != a]
        self.assertEqual(mismatches, [])


# --------------------------------------------------------------------------- labeler v7


class LabelerSemanticsTests(unittest.TestCase):
    def test_bare_keys_join_one_any_group(self):
        groups = _groups([
            {"changed-files": [{"any-glob-to-any-file": "a/*"}]},
            {"head-branch": "^docs/"},
        ])
        self.assertEqual([g.kind for g in groups], ["any"])
        self.assertEqual(len(groups[0].bases), 2)
        self.assertTrue(MODULE.label_matches(groups, ["b/x"], head_branch="docs/fix"))
        self.assertTrue(MODULE.label_matches(groups, ["a/x"], head_branch="main"))
        self.assertFalse(MODULE.label_matches(groups, ["b/x"], head_branch="main"))

    def test_explicit_groups_are_and_ed(self):
        groups = _groups([
            {"any": [{"changed-files": [{"any-glob-to-any-file": "src/**/*"}]}]},
            {"all": [{"base-branch": "^main$"}]},
        ])
        self.assertEqual([g.kind for g in groups], ["any", "all"])
        self.assertTrue(MODULE.label_matches(groups, ["src/a"], base_branch="main"))
        self.assertFalse(MODULE.label_matches(groups, ["src/a"], base_branch="dev"))

    def test_one_any_object_with_several_keys_checks_each_key(self):
        groups = _groups([
            {"any": [{"base-branch": "^release$", "changed-files": [{"any-glob-to-any-file": "a/*"}]}]},
        ])
        self.assertTrue(MODULE.label_matches(groups, ["a/x"], base_branch="main"))
        self.assertTrue(MODULE.label_matches(groups, ["b/x"], base_branch="release"))
        self.assertFalse(MODULE.label_matches(groups, ["b/x"], base_branch="main"))

    def test_options_inside_one_changed_files_list_are_or_ed_in_any(self):
        groups = _groups([{"changed-files": [
            {"any-glob-to-any-file": ["LICENSE"]},
            {"all-globs-to-any-file": ["docs/**/*", "!docs/frag/**/*"]},
        ]}])
        self.assertTrue(MODULE.label_matches(groups, ["LICENSE"]))
        self.assertTrue(MODULE.label_matches(groups, ["docs/a.rst"]))
        self.assertFalse(MODULE.label_matches(groups, ["docs/frag/1.rst"]))

    def test_four_options(self):
        files = ["a/x.py", "b/y.py"]
        cases = {
            "any-glob-to-any-file": (["a/*"], True),
            "all-globs-to-any-file": (["**/*.py", "a/*"], True),
            "any-glob-to-all-files": (["a/*", "**/*.py"], True),
            "all-globs-to-all-files": (["**/*.py", "a/*"], False),
        }
        for option, (globs, expected) in cases.items():
            with self.subTest(option=option):
                groups = _groups([{"changed-files": [{option: globs}]}])
                self.assertIs(MODULE.label_matches(groups, files), expected)

    def test_old_documentation_rule_matched_almost_everything(self):
        old = _groups([{"changed-files": [{"any-glob-to-any-file": [
            "docs/**/*", "!docs/source/whats_new/upcoming_changes/**/*"]}]}])
        self.assertTrue(MODULE.label_matches(old, ["scikitplot/corpus/_base.py"]))
        self.assertTrue(MODULE.lint_label("documentation", old))

    def test_new_documentation_rule(self):
        new = _groups([{"changed-files": [
            {"any-glob-to-any-file": ["LICENSE"]},
            {"all-globs-to-any-file": ["docs/**/*", "!docs/source/whats_new/upcoming_changes/**/*"]},
        ]}])
        self.assertFalse(MODULE.label_matches(new, ["scikitplot/corpus/_base.py"]))
        self.assertFalse(MODULE.label_matches(
            new, ["docs/source/whats_new/upcoming_changes/scikitplot.corpus/864.bugfix.rst"]))
        self.assertTrue(MODULE.label_matches(new, ["docs/source/index.rst"]))
        self.assertEqual(MODULE.lint_label("documentation", new), [])

    def test_lint_allows_except_idiom_only_inside_all(self):
        ok = _groups([
            {"any": [{"changed-files": [{"any-glob-to-any-file": "src/**/*"}]}]},
            {"all": [{"changed-files": [{"all-globs-to-all-files": ["!src/vendored/**/*"]}]}]},
        ])
        self.assertEqual(MODULE.lint_label("x", ok), [])
        bad = _groups([{"changed-files": [{"all-globs-to-all-files": ["!src/**/*"]}]}])
        self.assertTrue(MODULE.lint_label("x", bad))

    def test_unknown_keys_and_options_are_errors(self):
        for raw in (
            [{"changed-file": [{"any-glob-to-any-file": "a"}]}],
            [{"changed-files": [{"any-glob-to-any-files": "a"}]}],
            [{"any": [{"head_branch": "x"}]}],
            {"changed-files": []},
            [],
            [{"changed-files": [{"any-glob-to-any-file": "{a,b}"}]}],
        ):
            with self.subTest(raw=raw), self.assertRaises(MODULE.LabelerError):
                _groups(raw)

    def test_no_files_never_matches_changed_files(self):
        groups = _groups([{"all": [{"changed-files": [{"all-globs-to-all-files": ["!x/**/*"]}]}]}])
        self.assertFalse(MODULE.label_matches(groups, []))


# --------------------------------------------------------------------------- titles


TITLE_CASES = {
    "BUG: x": ["00 - Bug"],
    "[Bug]: crash": ["00 - Bug"],
    "bug: lower": ["00 - Bug"],
    "BUG, TST: flaky": ["00 - Bug", "05 - Testing"],
    "BUG/ENH: two": ["00 - Bug", "01 - Enhancement"],
    "ENH+DOC: x": ["01 - Enhancement", "04 - Documentation"],
    "MAINT(ci): pin": ["03 - Maintenance"],
    "  DOC: lead space": ["04 - Documentation"],
    "DOCS: guide": ["04 - Documentation"],
    "[Doc]: page": ["04 - Documentation"],
    "[BUG] crash": ["00 - Bug"],
    "ENH:x": ["01 - Enhancement"],
    "fix: typo": ["00 - Bug"],
    "WIP:": ["25 - WIP"],
    "TYP: stubs": ["static typing"],
    "CI: cache": ["CI"],
    "Critical: data loss": ["critical"],
    "an URGENT fix": ["critical"],
    "query planner": ["query"],
    "array  types": ["array types"],
    # No explicit terminator, or a longer word: no label.
    "DOCKER image": [],
    "Fix typo": [],
    "Bug in reader": [],
    "FIXME: later": [],
    "APIs overview": [],
    "RELEASE notes": [],
    "SPECIAL CI chars": [],
    "ENH: add CIFAR loader": ["01 - Enhancement"],
    "WIP": [],
    "ENH-x": [],
    "(BUG): x": [],
    "[Tag]: new tag": [],
    "urgently needed": [],
    "queries are slow": [],
    "Merge branch main": [],
    "": [],
}


class TitleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = MODULE.load_policy(MODULE._load_pyproject(REPO))
        cls.rules = [(label, MODULE.js_regex(regex)) for label, regex in MODULE.title_rules(cls.policy)]

    def labels(self, title):
        text = f"{title}\n\n"
        return sorted(label for label, pattern in self.rules if pattern.search(text))

    def test_cases(self):
        for title, expected in TITLE_CASES.items():
            with self.subTest(title=title):
                self.assertEqual(self.labels(title), sorted(expected))

    def test_every_token_gives_its_label(self):
        for prefix in self.policy.title_prefixes:
            with self.subTest(token=prefix.token):
                self.assertIn(prefix.label, self.labels(f"{prefix.token}: x"))
                self.assertIn(prefix.label, self.labels(f"[{prefix.token.title()}]: x"))
                self.assertEqual(self.labels(f"{prefix.token}X: x"), [])

    def test_js_regex_flags(self):
        self.assertTrue(MODULE.js_regex("/abc/i").search("ABC"))
        self.assertFalse(MODULE.js_regex("abc").search("ABC"))
        with self.assertRaises(MODULE.LabelerError):
            MODULE.js_regex("/abc/g")

    def test_rendered_yaml_round_trips(self):
        if not HAVE_YAML:
            self.skipTest("PyYAML is not installed")
        rules = MODULE.title_rules(self.policy)
        loaded = yaml.safe_load(MODULE.render_title_rules(rules))
        self.assertEqual(loaded, {label: [regex] for label, regex in rules})


# --------------------------------------------------------------------------- policy


class PolicyTests(unittest.TestCase):
    def test_bad_token(self):
        data = _policy_data()
        _labeler_table(data)["title_prefixes"]["bug"] = "00 - Bug"
        with self.assertRaises(MODULE.LabelerError):
            MODULE.load_policy(data)

    def test_unknown_satellite_placeholder(self):
        data = _policy_data()
        _labeler_table(data)["satellites"]["scikitplot"] = ["x/{module}/**/*"]
        with self.assertRaises(MODULE.LabelerError):
            MODULE.load_policy(data)

    def test_bad_keyword(self):
        data = _policy_data()
        _labeler_table(data)["title_keywords"] = [{"label": "x", "words": ["a.*"]}]
        with self.assertRaises(MODULE.LabelerError):
            MODULE.load_policy(data)

    def test_keyword_label_clashing_with_prefix_label(self):
        data = _policy_data()
        _labeler_table(data)["title_keywords"] = [{"label": "00 - Bug", "words": ["crash"]}]
        policy = MODULE.load_policy(data)
        with self.assertRaises(MODULE.LabelerError):
            MODULE.title_rules(policy)

    def test_max_label_length_must_be_positive_int(self):
        for value in (0, True, "50"):
            data = _policy_data()
            _labeler_table(data)["max_label_length"] = value
            with self.subTest(value=value), self.assertRaises(MODULE.LabelerError):
                MODULE.load_policy(data)

    def test_missing_table(self):
        with self.assertRaises(MODULE.LabelerError):
            MODULE.load_policy({"tool": {}})


class ModuleRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = MODULE.repository_files(REPO)
        cls.data = _policy_data()

    def rules(self, data):
        return MODULE.module_rules(REPO, data, MODULE.load_policy(data), self.files)

    def test_stale_alias_is_an_error(self):
        data = copy.deepcopy(self.data)
        _labeler_table(data)["label_aliases"]["scikitplot.does_not_exist"] = "x"
        with self.assertRaises(MODULE.LabelerError):
            self.rules(data)

    def test_long_label_without_alias_is_an_error(self):
        data = copy.deepcopy(self.data)
        _labeler_table(data)["label_aliases"] = {}
        with self.assertRaises(MODULE.LabelerError):
            self.rules(data)

    def test_lower_limit_is_enforced(self):
        data = copy.deepcopy(self.data)
        _labeler_table(data)["max_label_length"] = 10
        with self.assertRaises(MODULE.LabelerError):
            self.rules(data)

    def test_unknown_owner_root_is_an_error(self):
        data = copy.deepcopy(self.data)
        _labeler_table(data)["rollup_owner_roots"] = []
        with self.assertRaises(MODULE.LabelerError):
            self.rules(data)

    def test_a_skill_glob_is_the_mirrored_skill_folder(self):
        """Skill folders are flat (Rules 164, 165); folders come from the metadata."""
        rules = {rule.owner: rule for rule in self.rules(self.data)}
        annoy = rules["scikitplot.annoy"].globs
        for glob in (
            "skills/scikitplot-annoy-dev/**/*",
            "agents/scikitplot-annoy-dev.md",
            "plugins/scikitplot/skills/scikitplot-annoy/**/*",
            "plugins/scikitplot/agents/scikitplot-annoy.md",
        ):
            self.assertIn(glob, annoy)
        source = rules["scikitplot.cexternals._annoy"].globs
        self.assertIn("skills/scikitplot-cexternals-annoy-dev/**/*", source)
        self.assertNotIn("skills/scikitplot-annoy-dev/**/*", source)
        parent = rules["scikitplot._externals"].globs
        for folder in ("sphinx-ai-assistant", "sphinx-feedback", "sphinx-youtube-gallery"):
            self.assertIn(f"skills/scikitplot-{folder}-dev/**/*", parent)
        self.assertIn("plugins/scikitplot/skills/scikitplot-sphinx-ext/**/*", parent)
        for rule in rules.values():
            self.assertFalse(any(g.startswith("skills/_") for g in rule.globs), rule.owner)

    def test_skill_placeholders_expand_per_mirrored_skill(self):
        mirrors = {
            "skill": {"a": "scikitplot-a-dev", "a/_b": "scikitplot-b-dev", "ab": "scikitplot-ab-dev"},
            "usage": {"": "scikitplot", "a": "scikitplot-a"},
        }
        expand = MODULE._expand_satellite
        self.assertEqual(
            expand("skills/{skill}/**/*", "a", "scikitplot.a", mirrors),
            ["skills/scikitplot-a-dev/**/*", "skills/scikitplot-b-dev/**/*"],
        )
        self.assertEqual(expand("p/{usage}.md", "a", "scikitplot.a", mirrors), ["p/scikitplot-a.md"])
        self.assertEqual(expand("skills/{skill}/**/*", "zz", "scikitplot.zz", mirrors), [])
        self.assertEqual(expand("maintenances/{rel}/**/*", "a", "scikitplot.a", mirrors), ["maintenances/a/**/*"])
        with self.assertRaises(MODULE.LabelerError):
            expand("{skill}/{usage}", "a", "scikitplot.a", mirrors)

    def test_shape_of_generated_rules(self):
        rules = {rule.label: rule for rule in self.rules(self.data)}
        self.assertEqual(rules["tools"].globs, ("tools/**/*",))
        self.assertEqual(rules["scikitplot (top-level)"].globs, ("scikitplot/*",))
        labels = list(rules)
        self.assertEqual(len(labels), len(set(labels)))
        for rule in rules.values():
            self.assertLessEqual(len(rule.label), 50)
            self.assertEqual(len(rule.globs), len(rule.origins))
            if rule.owner.startswith("libs.") and rule.owner != "libs._tools":
                self.assertIn("ships", rule.origins, rule.owner)


# --------------------------------------------------------------------------- check / sync


@unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
class SyncCheckTests(unittest.TestCase):
    """A tiny repository; module discovery is replaced by a fixed rule list."""

    RULES = [MODULE.ModuleRule("pkg/a", "pkg.a", ("pkg/a/**/*",), ("tree",))]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        policy = MODULE.load_policy(MODULE._load_pyproject(REPO))
        (self.root / "pyproject.toml").write_text(_labeler_tables(), encoding="utf-8")
        for relative in (policy.module_template, policy.title_template):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / relative, target)
        curated = self.root / policy.curated_config
        curated.parent.mkdir(parents=True, exist_ok=True)
        curated.write_text("x:\n- changed-files:\n  - any-glob-to-any-file: pkg/**/*\n", encoding="utf-8")
        self.policy = policy
        self.patch = mock.patch.object(MODULE, "module_rules", return_value=self.RULES)
        self.patch.start()
        self.files = ["pkg/a/m.py"]

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_sync_then_check(self):
        self.assertEqual(
            sorted(MODULE.sync(self.root, self.files, apply=False)),
            sorted([self.policy.module_config, self.policy.title_config]),
        )
        report = MODULE.check(self.root, self.files)
        self.assertFalse(report.ok)
        self.assertEqual(len(report.stale), 2)
        MODULE.sync(self.root, self.files, apply=True)
        self.assertEqual(MODULE.sync(self.root, self.files, apply=False), [])
        report = MODULE.check(self.root, self.files)
        self.assertTrue(report.ok, report.errors)
        text = (self.root / self.policy.module_config).read_text(encoding="utf-8")
        self.assertIn("GENERATED FILE", text)
        self.assertIn('"pkg/a":', text)

    def test_hand_edit_is_stale(self):
        MODULE.sync(self.root, self.files, apply=True)
        path = self.root / self.policy.title_config
        path.write_text(path.read_text(encoding="utf-8") + "\n# edit\n", encoding="utf-8")
        report = MODULE.check(self.root, self.files)
        self.assertEqual(report.stale, [self.policy.title_config])

    def test_curated_problems(self):
        MODULE.sync(self.root, self.files, apply=True)
        curated = self.root / self.policy.curated_config
        curated.write_text(
            '"pkg/a":\n- changed-files:\n  - any-glob-to-any-file: pkg/**/*\n'
            "neg:\n- changed-files:\n  - any-glob-to-any-file: ['!pkg/**/*']\n"
            "dead:\n- changed-files:\n  - any-glob-to-any-file: nope/*\n",
            encoding="utf-8",
        )
        report = MODULE.check(self.root, self.files)
        joined = "\n".join(report.errors)
        self.assertIn("also a generated module label", joined)
        self.assertIn("negated glob", joined)
        self.assertTrue(any("nope/*" in w for w in report.warnings))
        strict = MODULE.check(self.root, self.files, strict=True)
        self.assertTrue(any("nope/*" in e for e in strict.errors))

    def test_referenced_label_must_exist(self):
        MODULE.sync(self.root, self.files, apply=True)
        templates = self.root / ".github" / "ISSUE_TEMPLATE"
        templates.mkdir(parents=True)
        (templates / "t.yml").write_text('labels: ["Not A Label"]\n', encoding="utf-8")
        report = MODULE.check(self.root, self.files)
        self.assertTrue(any("Not A Label" in e for e in report.errors))

    def test_template_needs_one_placeholder(self):
        target = self.root / self.policy.title_template
        target.write_text("no placeholder\n", encoding="utf-8")
        with self.assertRaises(MODULE.LabelerError):
            MODULE.sync(self.root, self.files, apply=False)


# --------------------------------------------------------------------------- repository gate


@unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
class RepositoryGateTests(unittest.TestCase):
    def test_check_passes_and_generated_files_are_in_sync(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        status = MODULE.main(["--repo-root", str(REPO), "check", "--json"], stdout=stdout, stderr=stderr)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(status, 0, payload["errors"] or stderr.getvalue())
        self.assertEqual(payload["stale"], [])

    def test_explain_samples(self):
        policy = MODULE.load_policy(MODULE._load_pyproject(REPO))
        result = MODULE.explain(REPO, ["scikitplot/corpus/_base.py"], "BUG: x")
        self.assertNotIn("documentation", result[policy.curated_config])
        self.assertIn("scikitplot/corpus", result[policy.module_config])
        self.assertEqual(result[policy.title_config], ["00 - Bug"])
        result = MODULE.explain(REPO, ["scikitplot/_typing.pyi"], None)
        self.assertIn("static typing", result[policy.curated_config])
        self.assertNotIn("lang:cython", result[policy.curated_config])


if __name__ == "__main__":
    unittest.main()
