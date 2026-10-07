# libs/_tools/tests/test_verify.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of ``libs/_tools/verify.py``: how observations become results.

The environment checks themselves are exercised by running ``verify``; these
tests cover the logic that turns what was observed into ``PASS`` or ``FAIL``,
which must be right for a green run to mean anything.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile

from pathlib import PurePosixPath

import pytest

from .. import generate, registry, staging, verify

MAP = staging.load_distributions(staging.repo_root())
PACKAGES = registry.by_distribution()
VERSION = "0.5.dev0"


def _statuses(results):
    return {result.check: result.status for result in results}


def _probe(**overrides):
    """Return a probe document for a healthy core + mcp installation."""
    data = {
        "import_error": None,
        "log": "",
        "version": VERSION,
        "dir": 48,
        "report": {
            "flavor": "partial",
            "installed": {"scikit-plots-skinny": VERSION, "scikit-plots-mcp": VERSION},
            "available": {},
            "core_api": MAP.CORE_API,
            "problems": [],
            "notes": [],
        },
        "core_api": {
            "core": MAP.CORE_API,
            "declared": {"scikit-plots-mcp": MAP.CORE_API, "scikit-plots-skinny": None},
        },
        "modules": {
            "scikitplot._cli": None,
            "scikitplot.logging": None,
            "scikitplot.mcp": None,
            "scikitplot.mcp._core": None,
            "scikitplot.mcp._server": {"kind": "missing", "name": "pydantic"},
            "scikitplot._cli._frontends._click": {"kind": "missing", "name": "click"},
        },
    }
    data.update(overrides)
    return data


NAMES = ["scikit-plots-mcp", "scikit-plots-skinny"]


def _check(data):
    return verify._check_probe(
        data, "raw output", NAMES, label="label", python="3.13", version=VERSION
    )


class TestCheckProbe:
    def test_a_healthy_installation_passes_every_check(self):
        results = _check(_probe())
        assert {r.status for r in results} == {verify.PASS}
        assert len(results) == 7

    def test_an_optional_third_party_module_is_reported_not_failed(self):
        (row,) = [r for r in _check(_probe()) if r.check == "every shipped module imports"]
        assert row.status == verify.PASS
        assert "click" in row.detail and "pydantic" in row.detail
        assert "4 imported, 2 need an optional package" in row.detail

    def test_no_probe_output_fails(self):
        (row,) = verify._check_probe(
            None, "Traceback ... boom", NAMES, label="label", python="3.13", version=VERSION
        )
        assert (row.check, row.status) == ("probe ran", verify.FAIL)
        assert "boom" in row.detail

    def test_a_failed_import_of_the_package_fails(self):
        (row,) = _check(_probe(import_error="ModuleNotFoundError: No module named 'numpy'"))
        assert (row.check, row.status) == ("import scikitplot", verify.FAIL)

    def test_any_log_output_on_import_fails(self):
        results = _statuses(_check(_probe(log="W scikitplot BOOM! :: Error importing\n")))
        assert results["import scikitplot is silent"] == verify.FAIL

    def test_a_wrong_version_fails(self):
        assert _statuses(_check(_probe(version="0.4.0")))["version matches"] == verify.FAIL

    def test_a_dir_that_raised_fails(self):
        results = _statuses(_check(_probe(dir="ModuleNotFoundError: scikitplot.api")))
        assert results["dir(scikitplot) works"] == verify.FAIL

    @pytest.mark.parametrize(
        "report",
        [
            {"flavor": "source", "installed": {}, "available": {}, "problems": []},
            {"flavor": "full", "installed": {"scikit-plots": VERSION}, "available": {}, "problems": []},
            {"flavor": "partial", "installed": {"scikit-plots-skinny": VERSION},
             "available": {}, "problems": []},  # mcp was asked for and is not there
            {"flavor": "partial",
             "installed": {"scikit-plots-skinny": VERSION, "scikit-plots-mcp": VERSION},
             "available": {}, "problems": ["different versions"]},
            {"flavor": "partial",
             "installed": {"scikit-plots": VERSION, "scikit-plots-skinny": VERSION,
                           "scikit-plots-mcp": VERSION},
             "available": {}, "problems": []},
        ],
    )
    def test_an_incoherent_installation_fails(self, report):
        results = _statuses(_check(_probe(report=report)))
        assert results["doctor: partial flavor, no problem"] == verify.FAIL

    @pytest.mark.parametrize(
        "declared",
        [
            {"scikit-plots-mcp": None, "scikit-plots-skinny": None},
            {"scikit-plots-mcp": MAP.CORE_API + 1, "scikit-plots-skinny": None},
            {"scikit-plots-mcp": MAP.CORE_API, "scikit-plots-skinny": MAP.CORE_API},
            {"scikit-plots-mcp": "expected one entry-point group", "scikit-plots-skinny": None},
            {"scikit-plots-skinny": None},
        ],
        ids=["silent part", "other number", "core states one", "malformed", "not probed"],
    )
    def test_a_wrong_core_api_statement_fails(self, declared):
        data = _probe(core_api={"core": MAP.CORE_API, "declared": declared})
        results = _statuses(_check(data))
        assert results["each part states the core API of the installed core"] == verify.FAIL

    def test_the_core_api_row_names_the_number(self):
        (row,) = [r for r in _check(_probe()) if r.check.startswith("each part states")]
        assert row.status == verify.PASS
        assert f"core API {MAP.CORE_API}, stated by 1 part(s)" == row.detail

    def test_a_module_that_raises_on_import_fails(self):
        modules = dict(_probe()["modules"])
        modules["scikitplot.mcp._core"] = {"kind": "error", "name": "TypeError: slots"}
        results = _check(_probe(modules=modules))
        (row,) = [r for r in results if r.check == "every shipped module imports"]
        assert row.status == verify.FAIL
        assert "scikitplot.mcp._core -> TypeError: slots" in row.detail

    def test_a_sibling_part_imported_at_module_level_fails(self):
        """A missing ``scikitplot`` module is never an "optional package"."""
        modules = dict(_probe()["modules"])
        modules["scikitplot.mcp._core"] = {"kind": "missing", "name": "scikitplot.corpus"}
        results = _check(_probe(modules=modules))
        (row,) = [r for r in results if r.check == "every shipped module imports"]
        assert row.status == verify.FAIL
        assert "scikitplot.corpus" in row.detail

    def test_a_part_that_needs_an_optional_package_to_import_fails(self):
        modules = dict(_probe()["modules"])
        modules["scikitplot.mcp"] = {"kind": "missing", "name": "pydantic"}
        results = _statuses(_check(_probe(modules=modules)))
        assert results["each part imports with base dependencies only"] == verify.FAIL
        # ... although, as a module, needing an optional package is allowed.
        assert results["every shipped module imports"] == verify.PASS

    def test_a_part_the_probe_never_saw_fails(self):
        """Its package is not among the installed files: a wheel without its part."""
        modules = {k: v for k, v in _probe()["modules"].items() if k != "scikitplot.mcp"}
        results = _check(_probe(modules=modules))
        (row,) = [r for r in results if r.check == "each part imports with base dependencies only"]
        assert row.status == verify.FAIL
        assert "scikitplot.mcp: not among the installed modules" in row.detail


class TestWheelHelpers:
    def _wheel(self, tmp_path, members):
        path = tmp_path / "scikit_plots_x-1.0-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        return path

    def test_wheel_files_digest_the_package_and_skip_the_metadata(self, tmp_path):
        wheel = self._wheel(tmp_path, {
            "scikitplot/x/__init__.py": "A = 1\n",
            "scikit_plots_x-1.0.dist-info/METADATA": "Name: scikit-plots-x\n",
            "scikit_plots_x-1.0.dist-info/RECORD": "",
        })
        assert verify._wheel_files(wheel) == {
            "scikitplot/x/__init__.py": hashlib.sha256(b"A = 1\n").hexdigest()
        }

    def test_wheel_metadata_reads_a_dist_info_member(self, tmp_path):
        wheel = self._wheel(tmp_path, {"scikit_plots_x-1.0.dist-info/METADATA": "Name: x\n"})
        assert verify._wheel_metadata(wheel, "METADATA") == "Name: x\n"
        assert verify._wheel_metadata(wheel, "entry_points.txt") == ""

    @pytest.mark.parametrize(
        ("path", "native"),
        [
            ("scikitplot/a/b.cpython-313-x86_64-linux-gnu.so", True),
            ("scikitplot/a/b.cp313-win_amd64.pyd", True),
            ("scikitplot/a/libb.dylib", True),
            ("scikitplot/a/b.py", False),
            ("scikitplot/a/b.so.txt", False),
        ],
    )
    def test_is_native(self, path, native):
        assert verify._is_native(path) is native

    def test_wheel_of_requires_exactly_one(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="expected exactly one wheel"):
            verify._wheel_of("scikit-plots-x", tmp_path)
        self._wheel(tmp_path, {})
        assert verify._wheel_of("scikit-plots-x", tmp_path).name.startswith("scikit_plots_x-")


class TestSmallHelpers:
    def test_tail_keeps_the_last_non_empty_lines(self):
        assert verify._tail("a\n\nb\n  \nc\n", lines=2) == "b | c"
        assert verify._tail("") == ""

    def test_clean_env_drops_what_redirects_imports(self, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "/somewhere")
        monkeypatch.setenv("VIRTUAL_ENV", "/venv")
        env = verify._clean_env()
        assert "PYTHONPATH" not in env and "VIRTUAL_ENV" not in env
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"

    def test_selected_keeps_declaration_order_and_accepts_any_spelling(self):
        chosen = verify._selected(["scikit_plots_mcp", "scikit-plots-rank_bm25"])
        assert [d.name for d in chosen] == ["scikit-plots-rank-bm25", "scikit-plots-mcp"]
        assert verify._selected(None) == list(MAP.DISTRIBUTIONS)

    def test_selected_refuses_an_unknown_name(self):
        with pytest.raises(KeyError):
            verify._selected(["scikit-plots-nope"])

    def test_test_requirements_name_the_distribution_with_its_test_extras(self):
        requirements = verify._test_requirements(PACKAGES["scikit-plots-mcp"])
        assert requirements[0] == "scikit-plots-mcp[mcp]"
        assert "pytest-asyncio" in requirements
        assert any(r.startswith("pytest>=") for r in requirements)

    def test_test_requirements_without_extras(self):
        assert verify._test_requirements(PACKAGES["scikit-plots-rank-bm25"])[0] == (
            "scikit-plots-rank-bm25"
        )

    def test_result_is_a_plain_record(self):
        row = verify.Result("check", "target", "3.13", verify.PASS, "detail")
        assert row._asdict() == {
            "check": "check", "target": "target", "python": "3.13",
            "status": "PASS", "detail": "detail",
        }

    def test_the_plugin_file_exists_beside_the_module(self):
        from pathlib import Path

        assert Path(verify.__file__).with_name(verify.PLUGIN + ".py").is_file()

    def test_a_missing_uv_is_explained(self, monkeypatch):
        monkeypatch.setattr(verify.shutil, "which", lambda name: None)
        with pytest.raises(SystemExit, match="needs `uv` on PATH"):
            verify._require_uv()


class TestPythonGatedModules:
    """A module of a tier that needs a newer Python is gated, not failed."""

    def _modules(self):
        modules = dict(_probe()["modules"])
        modules["scikitplot.mcp._server"] = {
            "kind": "error", "name": "ImportError: cannot import name 'Annotated'"
        }
        return modules

    def _row(self, python):
        results = verify._check_probe(
            _probe(modules=self._modules()),
            "raw",
            NAMES,
            label="label",
            python=python,
            version=VERSION,
        )
        (row,) = [r for r in results if r.check == "every shipped module imports"]
        return row

    def test_below_the_tier_floor_the_failure_is_expected(self):
        row = self._row("3.8")
        assert row.status == verify.PASS
        assert "1 gated by Python version" in row.detail
        assert "scikitplot.mcp._server" in row.detail

    def test_at_the_tier_floor_the_same_failure_is_a_failure(self):
        row = self._row("3.10")
        assert row.status == verify.FAIL
        assert "scikitplot.mcp._server -> ImportError" in row.detail

    def test_only_declared_modules_are_gated(self):
        modules = self._modules()
        modules["scikitplot.mcp._core"] = {"kind": "error", "name": "TypeError: x"}
        results = verify._check_probe(
            _probe(modules=modules), "raw", NAMES, label="label", python="3.8", version=VERSION
        )
        (row,) = [r for r in results if r.check == "every shipped module imports"]
        assert row.status == verify.FAIL
        assert "scikitplot.mcp._core" in row.detail and "_server" not in row.detail.split("fail:")[1]


def _wheel(tmp_path, name, metadata, entry_points=None):
    """Write a wheel that holds only the metadata the checks read."""
    stem = name.split("-")[0] + "-" + VERSION
    wheel = tmp_path / name
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{stem}.dist-info/METADATA", metadata)
        if entry_points is not None:
            archive.writestr(f"{stem}.dist-info/entry_points.txt", entry_points)
    return wheel


class TestWheelMetadataProblems:
    """A wheel's metadata is compared with what the registry declares."""

    META = generate.read_root_metadata(staging.repo_root())
    PACKAGE = PACKAGES["scikit-plots-rank-bm25"]

    def _metadata(self, **overrides):
        floor = self.PACKAGE.requires_python or self.META.requires_python
        fields = {
            "Name": self.PACKAGE.distribution,
            "Version": self.META.version,
            "Requires-Python": floor,
        }
        fields.update(overrides)
        lines = [f"{key}: {value}" for key, value in fields.items()]
        lines.append(f"Requires-Dist: {MAP.CORE}>={self.META.version}")
        return "\n".join(lines) + "\n"

    #: What the generator writes for this distribution (see ``CORE_API``).
    STATEMENT = f"[{MAP.parts_group()}]\nrank_bm25 = scikitplot.rank_bm25\n"

    def _problems(self, tmp_path, metadata, name=None, entry_points=STATEMENT):
        name = name or f"scikit_plots_rank_bm25-{self.META.version}-py3-none-any.whl"
        wheel = _wheel(tmp_path, name, metadata, entry_points)
        problems, _summary = verify._wheel_metadata_problems(
            wheel, self.PACKAGE, self.META
        )
        return problems

    def test_right_metadata_has_no_problem(self, tmp_path):
        assert self._problems(tmp_path, self._metadata()) == []

    def test_a_non_canonical_name_is_a_problem(self, tmp_path):
        problems = self._problems(tmp_path, self._metadata(Name="scikit_plots_rank_bm25"))
        assert problems == ["Name is not the canonical project name"]

    def test_a_wrong_version_is_a_problem(self, tmp_path):
        problems = self._problems(tmp_path, self._metadata(Version="0.0.1"))
        assert problems == [f"Version is not {self.META.version}"]

    def test_a_wrong_python_floor_is_a_problem(self, tmp_path):
        problems = self._problems(
            tmp_path, self._metadata(**{"Requires-Python": ">=3.99"})
        )
        assert len(problems) == 1 and problems[0].startswith("Requires-Python is not")

    def test_a_missing_core_requirement_is_a_problem(self, tmp_path):
        metadata = self._metadata().replace(f"Requires-Dist: {MAP.CORE}", "X-Dropped: x")
        assert self._problems(tmp_path, metadata) == [
            "core requirement missing or unexpected"
        ]

    def test_requiring_the_full_distribution_is_a_problem(self, tmp_path):
        metadata = self._metadata() + f"Requires-Dist: {MAP.FULL}\n"
        assert self._problems(tmp_path, metadata) == ["requires the full distribution"]

    def test_an_unexpected_console_script_is_a_problem(self, tmp_path):
        problems = self._problems(
            tmp_path,
            self._metadata(),
            entry_points=self.STATEMENT + "[console_scripts]\nx = y:z\n",
        )
        assert problems == ["console script missing or unexpected"]

    @pytest.mark.parametrize(
        "entry_points",
        [
            None,
            "",
            f"[{MAP.PARTS_GROUP}{MAP.CORE_API + 1}]\nrank_bm25 = scikitplot.rank_bm25\n",
            f"[{MAP.parts_group()}]\nrank_bm25 = scikitplot.other\n",
            f"[{MAP.parts_group()}]\nRank_BM25 = scikitplot.rank_bm25\n",
            STATEMENT + f"[{MAP.PARTS_GROUP}{MAP.CORE_API + 1}]\nx = scikitplot.x\n",
        ],
        ids=["no file", "empty", "other number", "other module", "other case", "two numbers"],
    )
    def test_a_wrong_core_api_statement_is_a_problem(self, tmp_path, entry_points):
        (problem,) = self._problems(tmp_path, self._metadata(), entry_points=entry_points)
        assert problem.startswith("core API statement is ")

    def test_the_core_states_no_core_api(self, tmp_path):
        package = PACKAGES[MAP.CORE]
        floor = package.requires_python or self.META.requires_python
        metadata = (
            f"Name: {MAP.CORE}\nVersion: {self.META.version}\n"
            f"Requires-Python: {floor}\n"
        )
        scripts = "[console_scripts]\nscikitplot = scikitplot._cli:main\n"
        name = f"scikit_plots_skinny-{self.META.version}-py3-none-any.whl"

        def problems(entry_points):
            wheel = _wheel(tmp_path, name, metadata, entry_points)
            return verify._wheel_metadata_problems(wheel, package, self.META)[0]

        assert problems(scripts) == []
        (problem,) = problems(scripts + self.STATEMENT)
        assert problem.startswith("core API statement is ")

    def test_a_platform_tag_on_a_pure_distribution_is_a_problem(self, tmp_path):
        name = f"scikit_plots_rank_bm25-{self.META.version}-cp313-cp313-linux_x86_64.whl"
        assert self._problems(tmp_path, self._metadata(), name=name) == [
            "wheel tag does not match pure/compiled"
        ]


class TestEntryPointGroups:
    def test_empty_text_has_no_group(self):
        assert verify._entry_point_groups("") == {}

    def test_names_keep_case_and_dots_and_values_keep_colons(self):
        text = "[g.one]\nA.b = pkg.mod:attr\n\n[g2]\nc = d\n"
        assert verify._entry_point_groups(text) == {
            "g.one": {"A.b": "pkg.mod:attr"},
            "g2": {"c": "d"},
        }


class TestImplicitEncodings:
    """Text I/O that would use the locale's encoding is found statically."""

    @pytest.mark.parametrize(
        "line",
        [
            "path.read_text()",
            "path.write_text(data)",
            "open(name)",
            "open(name, 'w')",
            "open(name, mode='a+')",
            "open(name, 'r', 1)",
            "with open(name) as handle: pass",
        ],
    )
    def test_reported(self, line):
        assert verify._implicit_encodings(f"x = 1\n{line}\n") == [2]

    @pytest.mark.parametrize(
        "line",
        [
            "path.read_text(encoding='utf-8')",
            "path.read_text('utf-8')",
            "path.write_text(data, encoding='utf-8')",
            "path.write_text(data, 'utf-8')",
            "open(name, encoding='utf-8')",
            "open(name, 'rb')",
            "open(name, mode='wb')",
            "open(name, 'r', -1, 'utf-8')",
            # Not decidable from the text alone.
            "open(name, mode)",
            "open(*arguments)",
            "path.read_text(**options)",
            # Another object's method: zipfile members have no encoding.
            "archive.open(member)",
            "os.open(name, flags)",
        ],
    )
    def test_not_reported(self, line):
        assert verify._implicit_encodings(line + "\n") == []

    def test_lines_are_ascending_and_source_that_does_not_parse_is_skipped(self):
        assert verify._implicit_encodings("b.read_text()\n\na.read_text()\n") == [1, 3]
        assert verify._implicit_encodings("def broken(:\n") == []

    def test_every_distribution_is_clean(self):
        results = verify.check_text_encoding()
        assert [r.target for r in results] == [d.name for d in MAP.DISTRIBUTIONS]
        assert [(r.target, r.detail) for r in results if r.status != verify.PASS] == []

    def test_a_finding_names_file_and_line_and_the_fix(self, tmp_path, monkeypatch):
        package = tmp_path / "scikitplot"
        (package / "alpha").mkdir(parents=True)
        (package / "alpha" / "__init__.py").write_text("", encoding="utf-8")
        (package / "alpha" / "io.py").write_text(
            "from pathlib import Path\nPath('x').read_text()\n", encoding="utf-8"
        )

        class _Dist:
            name = "scikit-plots-alpha"

        class _Map:
            DISTRIBUTIONS = (_Dist,)

        monkeypatch.setattr(staging, "repo_root", lambda: tmp_path)
        monkeypatch.setattr(staging, "load_distributions", lambda root=None: _Map)
        monkeypatch.setattr(
            staging,
            "iter_owned_files",
            lambda dist, package_dir: [
                PurePosixPath("alpha/__init__.py"),
                PurePosixPath("alpha/io.py"),
                PurePosixPath("alpha/data.txt"),
            ],
        )
        (result,) = verify.check_text_encoding()
        assert result.status == verify.FAIL
        assert "alpha/io.py:2" in result.detail and 'encoding="utf-8"' in result.detail


class TestBuildableHere:
    """A compiled distribution is only built by an interpreter it supports."""

    META = generate.read_root_metadata(staging.repo_root())

    def test_running_python_is_major_dot_minor(self):
        assert verify._running_python() == f"{sys.version_info[0]}.{sys.version_info[1]}"

    @pytest.mark.parametrize("name", [p.distribution for p in registry.PACKAGES if not p.extensions])
    def test_a_pure_distribution_is_always_buildable(self, name, monkeypatch):
        monkeypatch.setattr(verify, "_running_python", lambda: "3.8")
        assert verify._buildable_here(PACKAGES[name], self.META) is True

    @pytest.mark.parametrize(("python", "buildable"), [("3.8", False), ("3.9", False), ("3.10", True), ("3.14", True)])
    def test_a_compiled_distribution_follows_its_python_floor(
        self, python, buildable, monkeypatch
    ):
        package = PACKAGES["scikit-plots-annoy"]
        assert package.requires_python == ">=3.10"
        monkeypatch.setattr(verify, "_running_python", lambda: python)
        assert verify._buildable_here(package, self.META) is buildable


class TestDeclaresFloor:
    """Only a requirement that states a lowest version is tested at it."""

    @pytest.mark.parametrize(
        ("requirement", "expected"),
        [
            ("numpy>=2.0.0", True),
            ("scikit-learn>=1.3.0rc1", True),
            ("pkg~=1.4", True),
            ('numpy>=1.20,!=1.24.0; python_version < "3.9"', True),
            ("typing_extensions", False),
            ("numpy <= 2.4.6; python_version == '3.13'", False),
            ("pkg!=1.0", False),
            ("pkg<3", False),
            # A ``>=`` in the marker is about the interpreter, not the package.
            ('pkg; python_version >= "3.9"', False),
            ('pkg<3; python_version >= "3.9"', False),
        ],
    )
    def test_declares_floor(self, requirement, expected):
        assert verify._declares_floor(requirement) is expected



class TestIgnoreOptions:
    """Declared repository-only and Python-gated tests are left out, nothing else."""

    def _package(self, *entries, gated=()):
        return PACKAGES["scikit-plots-rank-bm25"]._replace(
            test_ignore=entries, test_gated=gated
        )

    def test_nothing_is_ignored_by_default(self):
        assert verify._ignore_options(self._package(), "rank_bm25", "3.13") == []

    def test_an_entry_becomes_a_glob_with_the_platform_separator(self, monkeypatch):
        monkeypatch.setattr(verify.os, "sep", "\\")
        options = verify._ignore_options(
            self._package("rank_bm25/tests/test_a.py"), "rank_bm25", "3.13"
        )
        assert options == ["--ignore-glob=*\\scikitplot\\rank_bm25\\tests\\test_a.py"]

    def test_only_entries_of_the_tree_under_test_apply(self):
        package = self._package("rank_bm25/tests/test_a.py", "rank_bm25_other/test_b.py")
        options = verify._ignore_options(package, "rank_bm25", "3.13")
        assert options == [
            "--ignore-glob=*"
            + verify.os.sep
            + verify.os.sep.join(["scikitplot", "rank_bm25", "tests", "test_a.py"])
        ]

    @pytest.mark.parametrize(
        ("python", "left_out"),
        [
            ("3.8", ["rank_bm25/tests/services", "rank_bm25/tests/test_new.py"]),
            ("3.9", ["rank_bm25/tests/services"]),
            ("3.10", []),
            ("3.15", []),
        ],
    )
    def test_a_gated_path_is_left_out_below_its_floor_only(self, python, left_out):
        package = self._package(
            gated=(
                ("rank_bm25/tests/services", ">=3.10"),
                ("rank_bm25/tests/test_new.py", ">=3.9"),
                ("another_tree/tests", ">=3.99"),
            )
        )
        assert verify._ignored_entries(package, "rank_bm25", python) == left_out

    def test_ignored_and_gated_entries_are_both_applied(self):
        package = self._package(
            "rank_bm25/tests/test_a.py", gated=(("rank_bm25/tests/b", ">=3.10"),)
        )
        assert verify._ignored_entries(package, "rank_bm25", "3.9") == [
            "rank_bm25/tests/test_a.py",
            "rank_bm25/tests/b",
        ]
        assert len(verify._ignore_options(package, "rank_bm25", "3.9")) == 2

    def test_every_registry_entry_is_reachable(self):
        for package in registry.PACKAGES:
            trees = MAP.get(package.distribution).trees
            # Python 0.0 is below every floor: every gated path applies.
            found = [e for tree in trees for e in verify._ignored_entries(package, tree, "0.0")]
            declared = list(package.test_ignore) + [p for p, _ in package.test_gated]
            assert sorted(found) == sorted(declared), package.distribution


class TestResidueChecks:
    """A path that was there before the build is not the build's residue."""

    def test_a_clean_tree_passes_both_checks(self, monkeypatch):
        monkeypatch.setattr(verify, "_residue", lambda: [])
        (start,), present = verify.check_clean_start()
        (after,) = verify.check_no_residue(present)
        assert (start.status, after.status, present) == (verify.PASS, verify.PASS, [])

    def test_a_committed_path_fails_the_start_check_with_the_remedy(self, monkeypatch):
        monkeypatch.setattr(verify, "_residue", lambda: ["libs/annoy/scikitplot"])
        (start,), present = verify.check_clean_start()
        assert start.status == verify.FAIL
        assert present == ["libs/annoy/scikitplot"]
        assert "git rm -r --cached libs/annoy/scikitplot" in start.detail

    def test_a_path_present_before_is_not_reported_as_left_behind(self, monkeypatch):
        monkeypatch.setattr(verify, "_residue", lambda: ["libs/annoy/scikitplot"])
        (after,) = verify.check_no_residue(["libs/annoy/scikitplot"])
        assert after.status == verify.PASS

    def test_new_residue_is_reported_beside_an_old_path(self, monkeypatch):
        monkeypatch.setattr(
            verify, "_residue", lambda: ["libs/annoy/scikitplot", "libs/mcp/build"]
        )
        (after,) = verify.check_no_residue(["libs/annoy/scikitplot"])
        assert after.status == verify.FAIL
        assert after.detail == "left: ['libs/mcp/build']"

    def test_residue_of_the_real_tree_is_listed_with_forward_slashes(self, tmp_path):
        # Read-only on the real tree: whatever is listed must be a path under
        # ``libs/<name>/`` written with ``/`` on every platform.
        for path in verify._residue():
            assert path.startswith("libs/") and "\\" not in path


class TestFailureReporting:
    """A failed test run names every failing test and keeps the whole output."""

    OUTPUT = (
        "....F..E\n"
        "=========================== short test summary info ===========================\n"
        "FAILED pkg/tests/test_a.py::test_one - AssertionError: assert 1 == 2\n"
        "ERROR pkg/tests/test_b.py::test_two - PermissionError: [Errno 13] denied\n"
        "FAILED pkg/tests/test_a.py::test_one - AssertionError: assert 1 == 2\n"
        "2 failed, 5 passed, 1 error in 0.50s\n"
    )

    def test_every_failure_and_error_is_listed_once_in_order(self):
        assert verify._failed_test_ids(self.OUTPUT) == [
            "FAILED pkg/tests/test_a.py::test_one - AssertionError: assert 1 == 2",
            "ERROR pkg/tests/test_b.py::test_two - PermissionError: [Errno 13] denied",
        ]

    def test_a_passing_run_has_none(self):
        assert verify._failed_test_ids("....\n4 passed in 0.1s\n") == []

    def test_only_the_short_summary_names_a_failure(self):
        # A log record of level ERROR before the summary, and standard error
        # appended after the totals, start with the same words.
        output = (
            "ERROR    pkg.app:app.py:2908 Record storage target unavailable\n"
            "FAILED to connect (a line of captured output)\n"
            + self.OUTPUT
            + "\n--- stderr ---\nERROR something written to standard error\n"
        )
        assert verify._failed_test_ids(output) == verify._failed_test_ids(self.OUTPUT)

    def test_output_without_a_summary_names_nothing(self):
        assert verify._failed_test_ids("ERROR    a log record\n5 passed in 1s\n") == []

    def test_the_detail_has_totals_failures_and_the_log(self, tmp_path):
        detail = verify._failure_detail(
            self.OUTPUT, "2 failed, 5 passed, 1 error in 0.50s", tmp_path / "run.txt"
        )
        parts = detail.split(" | ")
        assert parts[0] == "2 failed, 5 passed, 1 error in 0.50s"
        assert parts[1].startswith("FAILED pkg/tests/test_a.py::test_one")
        assert parts[2].startswith("ERROR pkg/tests/test_b.py::test_two")
        assert parts[-1] == "full output: test-logs/run.txt"

    def test_more_failures_than_the_limit_are_counted_not_dropped(self, tmp_path):
        many = "= short test summary info =\n" + "".join(
            f"FAILED t.py::test_{i} - boom\n" for i in range(55)
        )
        detail = verify._failure_detail(many, "55 failed in 1s", tmp_path / "run.txt")
        parts = detail.split(" | ")
        named = [part for part in parts if part.startswith("FAILED ")]
        assert len(named) == verify.MAX_NAMED_FAILURES
        assert f"... and {55 - verify.MAX_NAMED_FAILURES} more" in parts

    def test_a_run_that_never_started_shows_the_end_of_its_output(self, tmp_path):
        detail = verify._failure_detail(
            "ERROR: usage: pytest [options]\nunknown option", "", tmp_path / "run.txt"
        )
        assert "pytest did not report totals" in detail
        assert "unknown option" in detail

    @pytest.mark.parametrize(
        ("label", "target", "expected"),
        [
            (
                "scikit-plots-annoy [lowest]",
                "scikitplot.annoy",
                "scikit-plots-annoy--lowest---scikitplot.annoy--py3.12.txt",
            ),
            ("together", "scikitplot.mcp", "together--scikitplot.mcp--py3.12.txt"),
            ('a:b/c\\d*e?f"g<h>i|j', "t", "a-b-c-d-e-f-g-h-i-j--t--py3.12.txt"),
        ],
    )
    def test_log_file_names_are_valid_everywhere(self, label, target, expected):
        assert verify._log_file_name(label, target, "3.12") == expected


class TestImportFaults:
    """Which import outcomes are faults, with and without optional packages."""

    NAMES = ["scikit-plots-skinny", "scikit-plots-mcp"]

    def _modules(self, **outcomes):
        modules = {"scikitplot.mcp": None, "scikitplot.mcp._a": None}
        modules.update(outcomes)
        return modules

    def test_everything_imported_is_no_fault(self):
        bad, missing, gated, optional = verify._import_faults(
            self._modules(), self.NAMES, "3.13"
        )
        assert (bad, missing, gated, optional) == ({}, {}, {}, [])

    def test_a_missing_third_party_package_is_not_a_fault(self):
        modules = self._modules(**{"scikitplot.mcp._b": {"kind": "missing", "name": "yaml"}})
        bad, missing, _, optional = verify._import_faults(modules, self.NAMES, "3.13")
        assert bad == {}
        assert missing == {"scikitplot.mcp._b": "yaml"}
        assert optional == ["yaml"]

    def test_any_other_exception_is_a_fault(self):
        modules = self._modules(
            **{"scikitplot.mcp._b": {"kind": "error", "name": "TypeError: unsupported |"}}
        )
        bad, _, _, _ = verify._import_faults(modules, self.NAMES, "3.9")
        assert bad == {"scikitplot.mcp._b": "TypeError: unsupported |"}

    def test_a_missing_module_of_the_package_itself_is_a_fault(self):
        modules = self._modules(
            **{"scikitplot.mcp._b": {"kind": "missing", "name": "scikitplot.corpus"}}
        )
        bad, _, _, optional = verify._import_faults(modules, self.NAMES, "3.13")
        assert bad == {"scikitplot.mcp._b": "scikitplot.corpus"}
        assert optional == []

    def test_a_gated_module_is_neither(self):
        modules = self._modules(
            **{"scikitplot.mcp._server": {"kind": "error", "name": "ImportError: x"}}
        )
        bad, missing, gated, _ = verify._import_faults(modules, self.NAMES, "3.8")
        assert (bad, missing) == ({}, {})
        assert gated == {"scikitplot.mcp._server": ">=3.10"}


class TestProbeWithOptionalPackages:
    """The second probe reaches what the first could not."""

    NAMES = ["scikit-plots-skinny", "scikit-plots-mcp"]

    def _result(self, monkeypatch, data, raw=""):
        monkeypatch.setattr(verify, "_probe", lambda env, names: (data, raw))
        return verify._check_probe_with_optional_packages(
            object(), self.NAMES, label="label", python="3.9"
        )

    def test_a_clean_probe_passes(self, monkeypatch):
        result = self._result(
            monkeypatch, {"import_error": None, "modules": {"scikitplot.mcp": None}}
        )
        assert result.status == verify.PASS
        assert result.check == "every shipped module imports with its optional packages"

    def test_an_error_behind_an_optional_import_fails(self, monkeypatch):
        modules = {"scikitplot.mcp._b": {"kind": "error", "name": "TypeError: | on types"}}
        result = self._result(monkeypatch, {"import_error": None, "modules": modules})
        assert result.status == verify.FAIL
        assert "scikitplot.mcp._b -> TypeError" in result.detail

    def test_a_package_that_is_still_absent_is_reported_not_failed(self, monkeypatch):
        modules = {"scikitplot.mcp._b": {"kind": "missing", "name": "torch"}}
        result = self._result(monkeypatch, {"import_error": None, "modules": modules})
        assert result.status == verify.PASS
        assert "['torch']" in result.detail

    def test_no_probe_output_fails(self, monkeypatch):
        assert self._result(monkeypatch, None, "Traceback boom").status == verify.FAIL
