# .github/scripts/tests/test_uv_project_from_reqs.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for uv_project_from_reqs.py and the committed docker/env_uv store."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uv_project_from_reqs import main, read_requirements, render_pyproject  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[3]
STORE = REPOSITORY / "docker" / "env_uv"
SCRIPTS = REPOSITORY / ".github" / "scripts"

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None


def _write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


class TestReadRequirements(unittest.TestCase):
    def test_comments_blanks_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "r.txt", "# head\n\nnumpy>=2  # why\nscipy\nnumpy>=2\n")
            self.assertEqual(read_requirements(path), ["numpy>=2", "scipy"])

    def test_pip_options_are_refused(self):
        for line in ("-r other.txt", "-e .", "--index-url https://example.invalid"):
            with self.subTest(line=line), tempfile.TemporaryDirectory() as tmp:
                path = _write(Path(tmp), "r.txt", f"numpy\n{line}\n")
                with self.assertRaisesRegex(ValueError, "r.txt:2: pip option"):
                    read_requirements(path)


class TestRender(unittest.TestCase):
    def test_same_inputs_same_text(self):
        args = ("3.11", ["numpy>=2"], ["pytest"])
        self.assertEqual(render_pyproject(*args), render_pyproject(*args))

    def test_pins_one_minor_version_and_is_not_a_package(self):
        text = render_pyproject("3.12", ["numpy"], [])
        self.assertIn('requires-python = "==3.12.*"', text)
        self.assertIn('name = "scikit-plots-env-py312"', text)
        self.assertIn("package = false", text)

    def test_dev_leaves_out_runtime_entries(self):
        text = render_pyproject("3.11", ["numpy"], ["numpy", "pytest"])
        dev = text.split("[dependency-groups]")[1]
        self.assertNotIn('"numpy"', dev)
        self.assertIn('"pytest"', dev)

    @unittest.skipIf(tomllib is None, "tomllib needs Python 3.11")
    def test_markers_and_quotes_survive_as_toml(self):
        runtime = ['numpy>=2; python_version >= "3.9"', "voyager @ git+https://x.invalid/v.git"]
        data = tomllib.loads(render_pyproject("3.11", runtime, ["a\\b"]))
        self.assertEqual(data["project"]["dependencies"], runtime)
        self.assertEqual(data["dependency-groups"]["dev"], ["a\\b"])

    def test_bad_version(self):
        for bad in ("3", "311", "2.7", "3.11.2", ""):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "py_version"):
                render_pyproject(bad, [], [])

    def test_main_writes_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = _write(root, "r.txt", "numpy\n")
            dev = _write(root, "d.txt", "pytest\n")
            self.assertEqual(main([str(runtime), str(dev), str(root / "py311"), "3.11"]), 0)
            self.assertIn('"numpy"', (root / "py311" / "pyproject.toml").read_text())
            self.assertEqual(main(["too", "few"]), 2)


class TestCommittedStore(unittest.TestCase):
    """The committed projects match what the workflow writes today."""

    def test_root_copy_is_one_of_the_versions(self):
        root = (STORE / "pyproject.toml").read_bytes()
        lock = (STORE / "uv.lock").read_bytes()
        matches = [
            d.name
            for d in STORE.glob("py3*")
            if (d / "pyproject.toml").read_bytes() == root and (d / "uv.lock").read_bytes() == lock
        ]
        self.assertEqual(len(matches), 1, matches)

    def test_every_version_has_a_project_and_a_lock(self):
        versions = sorted(d.name for d in STORE.glob("py3*") if d.is_dir())
        self.assertTrue(versions)
        for name in versions:
            with self.subTest(version=name):
                self.assertTrue((STORE / name / "pyproject.toml").is_file())
                self.assertGreater((STORE / name / "uv.lock").stat().st_size, 0)

    def test_projects_match_the_requirements(self):
        try:
            import packaging  # noqa: F401  # flatten_reqs.py evaluates markers with it
        except ModuleNotFoundError:
            self.skipTest("packaging is not installed")
        for folder in sorted(d for d in STORE.glob("py3*") if d.is_dir()):
            version = f"3.{folder.name[3:]}"
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                flat = Path(tmp)
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS / "flatten_reqs.py"),
                        str(REPOSITORY / "requirements" / "default.txt"),
                        str(REPOSITORY / "requirements" / "all.txt"),
                        str(flat / "r.txt"),
                        str(flat / "d.txt"),
                        version,
                    ],
                    check=True,
                    capture_output=True,
                )
                expected = render_pyproject(
                    version, read_requirements(flat / "r.txt"), read_requirements(flat / "d.txt")
                )
                self.assertEqual(
                    (folder / "pyproject.toml").read_text(encoding="utf-8"),
                    expected,
                    "requirements changed: run .github/workflows/ci_uv_artifact.yml",
                )


if __name__ == "__main__":
    unittest.main()
