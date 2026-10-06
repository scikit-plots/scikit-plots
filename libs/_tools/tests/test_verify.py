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
            "problems": [],
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
        assert len(results) == 6

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

    def _problems(self, tmp_path, metadata, name=None, entry_points=None):
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
            tmp_path, self._metadata(), entry_points="[console_scripts]\nx = y:z\n"
        )
        assert problems == ["console script missing or unexpected"]

    def test_a_platform_tag_on_a_pure_distribution_is_a_problem(self, tmp_path):
        name = f"scikit_plots_rank_bm25-{self.META.version}-cp313-cp313-linux_x86_64.whl"
        assert self._problems(tmp_path, self._metadata(), name=name) == [
            "wheel tag does not match pure/compiled"
        ]


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
    """Declared repository-only tests are left out, and nothing else."""

    def _package(self, *entries):
        return PACKAGES["scikit-plots-rank-bm25"]._replace(test_ignore=entries)

    def test_nothing_is_ignored_by_default(self):
        assert verify._ignore_options(self._package(), "rank_bm25") == []

    def test_an_entry_becomes_a_glob_with_the_platform_separator(self, monkeypatch):
        monkeypatch.setattr(verify.os, "sep", "\\")
        options = verify._ignore_options(
            self._package("rank_bm25/tests/test_a.py"), "rank_bm25"
        )
        assert options == ["--ignore-glob=*\\scikitplot\\rank_bm25\\tests\\test_a.py"]

    def test_only_entries_of_the_tree_under_test_apply(self):
        package = self._package("rank_bm25/tests/test_a.py", "rank_bm25_other/test_b.py")
        options = verify._ignore_options(package, "rank_bm25")
        assert options == [
            "--ignore-glob=*"
            + verify.os.sep
            + verify.os.sep.join(["scikitplot", "rank_bm25", "tests", "test_a.py"])
        ]

    def test_every_registry_entry_is_reachable(self):
        for package in registry.PACKAGES:
            trees = MAP.get(package.distribution).trees
            found = [o for tree in trees for o in verify._ignore_options(package, tree)]
            assert len(found) == len(package.test_ignore), package.distribution
