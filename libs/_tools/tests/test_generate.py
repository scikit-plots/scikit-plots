# libs/_tools/tests/test_generate.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of ``libs/_tools/generate.py``: generated files and their inputs."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .. import generate, registry, staging

ROOT = staging.repo_root()
MAP = staging.load_distributions(ROOT)
META = generate.read_root_metadata(ROOT)
PACKAGES = registry.by_distribution()


def _parse_toml(text: str) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(text)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class TestReadVersion:
    def _root(self, tmp_path, source):
        (tmp_path / "scikitplot").mkdir()
        (tmp_path / "scikitplot" / "__init__.py").write_text(source, encoding="utf-8")
        return tmp_path

    def test_reads_the_literal(self, tmp_path):
        assert generate.read_version(self._root(tmp_path, '__version__ = "1.2.dev3"\n')) == "1.2.dev3"

    def test_takes_the_first_module_level_assignment(self, tmp_path):
        source = (
            '# __version__ = "0.0.0"\n'
            '__version__ = "1.0"\n'
            "try:\n    from .version import __version__\nexcept ImportError:\n    pass\n"
            '__version__ = "2.0"\n'
        )
        assert generate.read_version(self._root(tmp_path, source)) == "1.0"

    def test_does_not_execute_the_package(self, tmp_path):
        source = '__version__ = "1.0"\nraise RuntimeError("the package was imported")\n'
        assert generate.read_version(self._root(tmp_path, source)) == "1.0"

    @pytest.mark.parametrize("source", ["x = 1\n", "def f():\n    __version__ = '1'\n"])
    def test_no_assignment_is_refused(self, tmp_path, source):
        with pytest.raises(ValueError, match="no module-level __version__"):
            generate.read_version(self._root(tmp_path, source))

    @pytest.mark.parametrize("source", ["__version__ = 1\n", "__version__ = ''\n", "__version__ = f()\n"])
    def test_a_non_literal_or_empty_value_is_refused(self, tmp_path, source):
        with pytest.raises(ValueError, match="not a non-empty string literal"):
            generate.read_version(self._root(tmp_path, source))

    def test_the_real_version_is_pep_440_like(self):
        import re

        assert re.match(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$", META.version)


class TestRequirementName:
    @pytest.mark.parametrize(
        ("requirement", "name"),
        [
            ("numpy", "numpy"),
            ("numpy>=2.0.0", "numpy"),
            ('numpy>=1.20,!=1.24.0; python_version < "3.9"', "numpy"),
            ("numpy <= 2.4.6; python_version == '3.13'", "numpy"),
            ("scikit-learn>=1.3.0rc1", "scikit-learn"),
            ("typing_extensions>=4.0.0", "typing-extensions"),
            ("pdfminer.six", "pdfminer-six"),
            ("scikit-plots[core]", "scikit-plots"),
            ("  Pillow>=6", "pillow"),
        ],
    )
    def test_names(self, requirement, name):
        assert generate.requirement_name(requirement) == name

    @pytest.mark.parametrize("bad", ["", ">=1.0", "[extra]", "; python_version < '3'"])
    def test_a_string_without_a_name_is_refused(self, bad):
        with pytest.raises(ValueError, match="cannot read a project name"):
            generate.requirement_name(bad)


class TestInheritRequirements:
    ROOT_LINES = (
        'numpy>=1.20,!=1.24.0; python_version < "3.9"',
        "numpy <= 2.4.6; python_version == '3.13'",
        "scipy>=1.7.0",
        "scikit-learn>=1.3.0rc1",
    )

    def test_every_marker_line_is_copied_verbatim(self):
        assert generate.inherit_requirements(["numpy"], self.ROOT_LINES) == list(self.ROOT_LINES[:2])

    def test_any_spelling_of_the_name_matches(self):
        assert generate.inherit_requirements(["scikit_learn"], self.ROOT_LINES) == [
            "scikit-learn>=1.3.0rc1"
        ]

    def test_result_follows_the_requested_order(self):
        lines = generate.inherit_requirements(["scipy", "numpy"], self.ROOT_LINES)
        assert lines[0] == "scipy>=1.7.0" and len(lines) == 3

    def test_nothing_requested_is_nothing_inherited(self):
        assert generate.inherit_requirements([], self.ROOT_LINES) == []

    def test_an_undeclared_name_is_refused_and_the_table_is_named(self):
        with pytest.raises(ValueError) as excinfo:
            generate.inherit_requirements(["pandas"], self.ROOT_LINES, "[build-system].requires")
        assert "'pandas'" in str(excinfo.value)
        assert "[build-system].requires" in str(excinfo.value)


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", registry.PACKAGES, ids=lambda p: p.distribution)
class TestRenderPyproject:
    def test_is_valid_toml_with_the_expected_metadata(self, package):
        document = _parse_toml(generate.render_pyproject(package, MAP, META))
        project = document["project"]
        assert project["name"] == package.distribution
        assert project["version"] == META.version
        assert project["description"] == MAP.get(package.distribution).summary
        assert project["requires-python"] == (package.requires_python or META.requires_python)
        assert project["authors"] == list(META.authors)
        assert project["urls"] == META.urls
        assert project["dynamic"] == ["license"]
        assert "license" not in project
        assert document["build-system"]["build-backend"] == "setuptools.build_meta"
        assert document["tool"]["setuptools"]["include-package-data"] is False

    def test_readme_names_the_generated_description(self, package):
        project = _parse_toml(generate.render_pyproject(package, MAP, META))["project"]
        directory = registry.directory_of(package.distribution)
        assert project["readme"]["file"] == generate.readme_name(directory)
        assert (ROOT / "libs" / directory / project["readme"]["file"]).is_file()

    def test_depends_on_the_core_and_never_on_the_full_distribution(self, package):
        project = _parse_toml(generate.render_pyproject(package, MAP, META))["project"]
        names = [generate.requirement_name(line) for line in project["dependencies"]]
        if package.distribution == MAP.CORE:
            assert names == []
        else:
            assert names[0] == MAP.CORE
            assert project["dependencies"][0] == f"{MAP.CORE}>={META.version}"
        every = project["dependencies"] + [
            line for lines in project.get("optional-dependencies", {}).values() for line in lines
        ]
        assert MAP.FULL not in {generate.requirement_name(line) for line in every}

    def test_no_requirement_is_pinned(self, package):
        project = _parse_toml(generate.render_pyproject(package, MAP, META))["project"]
        every = project["dependencies"] + [
            line for lines in project.get("optional-dependencies", {}).values() for line in lines
        ]
        assert not [line for line in every if "==" in line.split(";")[0]]

    def test_extras_are_the_root_extras_verbatim(self, package):
        optional = _parse_toml(generate.render_pyproject(package, MAP, META))["project"].get(
            "optional-dependencies", {}
        )
        for extra in package.extras:
            assert optional[extra] == list(META.extras[extra])
        for extra, siblings in package.siblings:
            assert optional[extra] == [f"{sibling}>={META.version}" for sibling in siblings]
        assert set(optional) == set(package.extras) | {extra for extra, _ in package.siblings}

    def test_only_the_core_declares_the_console_script(self, package):
        project = _parse_toml(generate.render_pyproject(package, MAP, META))["project"]
        if package.scripts:
            assert project["scripts"] == META.scripts
        else:
            assert "scripts" not in project

    def test_classifiers_are_inherited_and_none_describes_the_full_build_only(self, package):
        project = _parse_toml(generate.render_pyproject(package, MAP, META))["project"]
        own = set(package.classifiers)
        for classifier in project["classifiers"]:
            assert classifier in META.classifiers or classifier in own
        assert "Framework :: Matplotlib" not in project["classifiers"]
        assert "Programming Language :: Fortran" not in project["classifiers"]
        assert len(project["classifiers"]) == len(set(project["classifiers"]))

    def test_build_requirements(self, package):
        requires = _parse_toml(generate.render_pyproject(package, MAP, META))["build-system"]["requires"]
        assert requires[: len(registry.BUILD_REQUIRES)] == list(registry.BUILD_REQUIRES)
        names = {generate.requirement_name(line) for line in requires}
        assert ("cython" in names) == ("cython" in package.build_inherit)


class TestRenderPyprojectRefusals:
    def test_an_extra_the_root_does_not_declare_is_refused(self):
        package = PACKAGES["scikit-plots-mcp"]._replace(extras=("no-such-extra",))
        with pytest.raises(ValueError, match="not declared in the root"):
            generate.render_pyproject(package, MAP, META)

    def test_an_extra_that_installs_the_full_distribution_is_refused(self):
        # The root's "test" extra refers back to scikit-plots itself.
        selfish = next(
            name for name, lines in META.extras.items()
            if any(generate.requirement_name(line) == MAP.FULL for line in lines)
        )
        package = PACKAGES["scikit-plots-mcp"]._replace(extras=(selfish,))
        with pytest.raises(ValueError, match="would install the full distribution"):
            generate.render_pyproject(package, MAP, META)

    def test_an_extra_name_used_twice_is_refused(self):
        package = PACKAGES["scikit-plots-mcp"]._replace(
            extras=("mcp",), siblings=(("mcp", ("scikit-plots-corpus",)),)
        )
        with pytest.raises(ValueError, match="both a root extra and a sibling extra"):
            generate.render_pyproject(package, MAP, META)

    def test_an_unknown_sibling_is_refused(self):
        package = PACKAGES["scikit-plots-mcp"]._replace(siblings=(("x", ("scikit-plots-nope",)),))
        with pytest.raises(KeyError):
            generate.render_pyproject(package, MAP, META)

    def test_an_undeclared_inherited_dependency_is_refused(self):
        package = PACKAGES["scikit-plots-mcp"]._replace(inherit=("not-a-root-dependency",))
        with pytest.raises(ValueError, match=r"not declared in the root \[project.dependencies\]"):
            generate.render_pyproject(package, MAP, META)


# ---------------------------------------------------------------------------
# setup.py, MANIFEST.in, READMEs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", registry.PACKAGES, ids=lambda p: p.distribution)
class TestRenderSetup:
    def test_is_valid_python_3_8(self, package):
        ast.parse(generate.render_setup(package), feature_version=(3, 8))

    def test_names_its_distribution_and_licence(self, package):
        tree = ast.parse(generate.render_setup(package))
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"DISTRIBUTION", "LICENSE", "EXTENSIONS", "PACKAGE"}
        }
        assert constants["DISTRIBUTION"] == package.distribution
        assert constants["LICENSE"] == package.license
        assert constants["PACKAGE"] == "scikitplot"
        assert [entry["name"] for entry in constants["EXTENSIONS"]] == [
            ext.name for ext in package.extensions
        ]
        for entry, ext in zip(constants["EXTENSIONS"], package.extensions):
            assert tuple(entry["sources"]) == ext.sources
            assert tuple(entry["templates"]) == ext.templates
            assert entry["cxx_standard"] == ext.cxx_standard

    def test_has_no_placeholder_left(self, package):
        assert "@@" not in generate.render_setup(package)

    def test_never_passes_a_cpu_specific_flag(self, package):
        # A wheel must run on machines other than the one that built it.
        source = generate.render_setup(package)
        assert "-march" not in source and "-mtune" not in source and "/arch:" not in source


class TestRenderSetupIsOneScript:
    def test_scripts_differ_only_in_their_constants(self):
        def body(package):
            lines = generate.render_setup(package).splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith("CYTHON_DIRECTIVES"))
            return [l for l in lines[start:] if package.distribution not in l]

        bodies = {tuple(body(package)) for package in registry.PACKAGES}
        assert len(bodies) == 1


class TestOtherFiles:
    @pytest.mark.parametrize("package", registry.PACKAGES, ids=lambda p: p.distribution)
    def test_manifest_includes_what_the_sdist_needs(self, package):
        directory = registry.directory_of(package.distribution)
        lines = generate.render_manifest(package).splitlines()
        assert "include LICENSE.txt" in lines
        assert f"include {generate.readme_name(directory)}" in lines
        assert "graft scikitplot" in lines

    @pytest.mark.parametrize("package", registry.PACKAGES, ids=lambda p: p.distribution)
    def test_package_description_states_how_to_install_it(self, package):
        text = generate.render_readme_package(package, MAP, META)
        assert f"pip install {package.distribution}" in text
        assert package.distribution.replace("-", "_") in text
        assert package.example.strip() in text
        for extra in package.extras:
            assert f'"{package.distribution}[{extra}]"' in text

    @pytest.mark.parametrize("package", registry.PACKAGES, ids=lambda p: p.distribution)
    def test_directory_readme_lists_what_is_owned(self, package):
        text = generate.render_readme_directory(package, MAP)
        for tree in MAP.get(package.distribution).trees:
            assert f"`scikitplot/{tree}/`" in text
        assert f"pip install ./libs/{registry.directory_of(package.distribution)}" in text

    def test_readme_name(self):
        assert generate.readme_name("rank-bm25") == "README_RANK-BM25.md"
        assert generate.readme_name("skinny") == "README_SKINNY.md"

    def test_gitignore_covers_everything_a_build_can_leave(self):
        lines = generate.render_gitignore().splitlines()
        for pattern in ("/*/scikitplot", "/*/LICENSE.txt", "/*/build/", "/*/*.egg-info/"):
            assert pattern in lines
        # With a trailing slash the rule would match a directory only and let
        # a ``libs/<name>/scikitplot`` *link* be committed.
        assert "/*/scikitplot/" not in lines
        assert "node_modules/" in lines  # the Pyodide probe's ``npm ci``
        assert "!mlflow" in lines  # libs/mlflow must not be caught by a repository-wide rule

    def test_libs_readme_lists_every_distribution(self):
        text = generate.render_libs_readme(list(PACKAGES.values()), MAP)
        for name in PACKAGES:
            assert f"`{name}`" in text


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class TestDriver:
    def test_every_output_is_text_ending_in_a_newline(self):
        for relative, content in generate.expected_files(ROOT).items():
            assert content.endswith("\n"), relative
            assert "\r" not in content, relative

    def test_outputs_are_the_expected_set(self):
        expected = {"libs/README.md", "libs/.gitignore"}
        for name in PACKAGES:
            directory = registry.directory_of(name)
            expected |= {
                f"libs/{directory}/{leaf}"
                for leaf in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md",
                             generate.readme_name(directory))
            }
        assert set(generate.expected_files(ROOT)) == expected

    def test_generation_is_deterministic(self):
        assert generate.expected_files(ROOT) == generate.expected_files(ROOT)

    def test_the_committed_files_are_up_to_date(self):
        """The drift guard: what is on disk is what the inputs produce."""
        assert generate.check(ROOT) == [], "run: python -m libs._tools generate"

    def test_a_changed_input_is_reported_as_stale(self, monkeypatch):
        changed = tuple(
            p._replace(keywords=(*p.keywords, "changed")) if p.distribution == MAP.CORE else p
            for p in registry.PACKAGES
        )
        monkeypatch.setattr(registry, "PACKAGES", changed)
        assert generate.check(ROOT) == ["libs/skinny/pyproject.toml"]

    def test_registries_that_disagree_are_refused(self, monkeypatch):
        monkeypatch.setattr(registry, "PACKAGES", registry.PACKAGES[:-1])
        with pytest.raises(ValueError, match="disagree"):
            generate.expected_files(ROOT)

    def test_generate_writes_nothing_when_up_to_date(self):
        before = {
            path: (ROOT / path).stat().st_mtime_ns for path in generate.expected_files(ROOT)
        }
        assert generate.generate(ROOT) == []
        after = {path: (ROOT / path).stat().st_mtime_ns for path in before}
        assert after == before


class TestClassifiersFollowThePythonFloor:
    @pytest.mark.parametrize("package", registry.PACKAGES, ids=lambda p: p.distribution)
    def test_no_classifier_names_a_python_the_distribution_excludes(self, package):
        project = _parse_toml(generate.render_pyproject(package, MAP, META))["project"]
        prefix = "Programming Language :: Python :: "
        versions = [
            c[len(prefix):] for c in project["classifiers"]
            if c.startswith(prefix) and c[len(prefix):][0].isdigit() and "." in c[len(prefix):]
        ]
        assert versions, "no Python version classifier at all"
        for version in versions:
            assert registry.python_satisfies(project["requires-python"], version), version

    def test_a_narrower_floor_drops_the_older_versions(self):
        package = PACKAGES["scikit-plots-mlflow"]
        classifiers = _parse_toml(generate.render_pyproject(package, MAP, META))["project"][
            "classifiers"
        ]
        assert "Programming Language :: Python :: 3.11" in classifiers
        assert "Programming Language :: Python :: 3.10" not in classifiers
        # Unversioned classifiers are untouched.
        assert "Programming Language :: Python :: 3 :: Only" in classifiers


class TestPythonLiteral:
    """Plain data is rendered as source a formatter leaves alone."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "None"),
            (True, "True"),
            (False, "False"),
            (0, "0"),
            (17, "17"),
            ("c++17", '"c++17"'),
            ('a "quoted" \\ name', '"a \\"quoted\\" \\\\ name"'),
            ("café", '"caf\\u00e9"'),
            ([], "[]"),
            ((), "[]"),
            ({}, "{}"),
            (["a"], '[\n    "a",\n]'),
            (("a", None), '[\n    "a",\n    None,\n]'),
            ({"k": []}, '{\n    "k": [],\n}'),
            ({"k": [1]}, '{\n    "k": [\n        1,\n    ],\n}'),
        ],
    )
    def test_rendering(self, value, expected):
        assert generate._python_literal(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            None,
            [],
            [{"name": "a.b", "sources": ("x.cc",), "macros": [["M", None], ["N", "1"]]}],
            {"deep": {"deeper": [[], {}, [True, False, 3, "s"]]}},
        ],
    )
    def test_the_literal_evaluates_back_to_the_data(self, value):
        def as_lists(item):
            if isinstance(item, (list, tuple)):
                return [as_lists(part) for part in item]
            if isinstance(item, dict):
                return {key: as_lists(part) for key, part in item.items()}
            return item

        assert ast.literal_eval(generate._python_literal(value)) == as_lists(value)

    def test_the_starting_indent_is_applied_to_continuation_lines(self):
        assert generate._python_literal(["a"], indent=4) == '[\n        "a",\n    ]'

    @pytest.mark.parametrize("value", [1.5, b"bytes", {1, 2}, object(), [1.5]])
    def test_other_types_are_refused(self, value):
        with pytest.raises(TypeError, match="cannot render"):
            generate._python_literal(value)

    def test_a_non_string_key_is_refused(self):
        with pytest.raises(TypeError, match="dictionary keys must be str"):
            generate._python_literal({1: "a"})


class TestCount:
    @pytest.mark.parametrize(
        ("number", "expected"), [(1, "a thing"), (2, "2 things"), (0, "0 things")]
    )
    def test_count(self, number, expected):
        assert generate._count(number, "thing") == expected


class TestGeneratedFilesPassTheRepositoryLinter:
    """
    Generated files must survive the repository's own pre-commit hooks.

    A formatter that rewrote a generated file would make ``check`` report
    drift on every commit, so the generator has to emit what the formatter
    would have written.
    """

    @staticmethod
    def _ruff(*args):
        """Run the repository's ``ruff``; skip the test where there is none."""
        executable = shutil.which("ruff") or shutil.which(
            "ruff", path=str(Path(sys.executable).parent)
        )
        if executable is None:
            pytest.skip("ruff is not installed")
        return subprocess.run(
            [executable, *args],
            cwd=staging.repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )

    def _scripts(self):
        root = staging.repo_root()
        return [
            str(root / "libs" / registry.directory_of(package.distribution) / "setup.py")
            for package in registry.PACKAGES
        ]

    def test_every_setup_script_is_lint_clean(self):
        done = self._ruff("check", "--no-cache", *self._scripts())
        assert done.returncode == 0, done.stdout + done.stderr

    def test_every_setup_script_is_already_formatted(self):
        done = self._ruff("format", "--check", "--no-cache", *self._scripts())
        assert done.returncode == 0, done.stdout + done.stderr

    def test_every_readme_example_is_already_formatted(self):
        root = staging.repo_root()
        readmes = [str(path) for path in sorted((root / "libs").glob("*/README*.md"))]
        assert readmes
        done = self._ruff("format", "--check", "--no-cache", *readmes)
        assert done.returncode == 0, done.stdout + done.stderr
