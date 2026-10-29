# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for tools/citation/citation_ref.py.

The last test is the gate: the repository's own CITATION.bib and CITATION.cff
must be what the templates render for the declared version.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "citation_ref.py"
SPEC = importlib.util.spec_from_file_location("citation_ref", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

REPO = Path(__file__).resolve().parents[3]
SHA = "0123456789abcdef0123456789abcdef01234567"
HAS_JINJA = importlib.util.find_spec("jinja2") is not None


def _checkout(root: Path, version: str) -> Path:
    """A minimal checkout: the version file and the version-string helper."""
    (root / "scikitplot" / "_build_utils").mkdir(parents=True)
    (root / "scikitplot" / "__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    shutil.copy(
        REPO / "scikitplot" / "_build_utils" / "gitversion.py",
        root / "scikitplot" / "_build_utils",
    )
    return root


def _main(*argv):
    out, err = io.StringIO(), io.StringIO()
    status = MODULE.main(list(argv), stdout=out, stderr=err)
    return status, out.getvalue(), err.getvalue()


class ProvenanceTests(unittest.TestCase):
    def test_kind_follows_pep_440(self):
        kinds = {
            "0.5.dev0": "development",
            "0.5.1rc0": "pre-release",
            "0.4.0.post11": "post-release",
            "0.4.0": "release",
            "1!2.0.dev3": "development",
        }
        for version, kind in kinds.items():
            self.assertEqual(MODULE.Provenance(version).kind, kind, version)

    def test_build_uses_the_gitversion_format(self):
        build = MODULE.Provenance("0.5.dev0", SHA, "2026-10-28").build
        self.assertEqual(build, "0.5.dev0+git.20261028.0123456")
        self.assertEqual(MODULE.Provenance("0.4.0").build, "0.4.0")

    def test_invalid_input_is_refused(self):
        for args in (
            ("banana",),
            ("0.5.dev0", "zzzzzzz", "2026-10-28"),
            ("0.5.dev0", SHA, None),
            ("0.5.dev0", SHA, "28.10.2026"),
            ("0.5+local",),  # a local segment is the build, not the version
        ):
            with self.assertRaises(MODULE.CitationError, msg=args):
                MODULE.Provenance(*args)


@unittest.skipUnless(HAS_JINJA, "rendering needs jinja2")
class RenderTests(unittest.TestCase):
    def test_two_entries_keep_the_historic_key_first(self):
        bib = MODULE.render(MODULE.Provenance("0.5.dev0", SHA, "2026-10-28"))["bib"]
        self.assertLess(
            bib.index("@software{The-scikit-plots-developers_2024,"),
            bib.index("@software{scikit-plots_v0.5.dev0,"),
        )
        self.assertEqual(
            bib.count("@software"), 2
        )  # config/_citation.py reads the first
        self.assertIn(f"/tree/{SHA}", bib)
        self.assertIn("build 0.5.dev0+git.20261028.0123456", bib)
        self.assertIn(
            f"% provenance: version=0.5.dev0 commit={SHA} date=2026-10-28", bib
        )

    def test_a_known_release_gets_its_doi(self):
        files = MODULE.render(MODULE.Provenance("0.4.0", SHA, "2025-05-01"))
        self.assertIn("doi      = { 10.5281/zenodo.13367095 }", files["bib"])
        self.assertIn('value: "10.5281/zenodo.13367095"', files["cff"])
        self.assertIn("scikit-plots.github.io/stable", files["cff"])

    def test_without_git_no_date_is_claimed(self):
        files = MODULE.render(MODULE.Provenance("0.4.0.post11"))
        version_entry = files["bib"].split("@software{scikit-plots_v0.4.0.post11,")[1]
        self.assertNotIn("year", version_entry)
        self.assertNotIn("\ndate-released:", files["cff"])
        self.assertNotIn("\ncommit:", files["cff"])


@unittest.skipUnless(HAS_JINJA, "rendering needs jinja2")
class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.root = _checkout(self.tmp, "0.5.dev0")

    def _sync(self, *extra):
        return _main(
            "--repo-root",
            str(self.root),
            "sync",
            "--commit",
            SHA,
            "--date",
            "2026-10-28",
            *extra,
        )

    def test_sync_is_a_preview_until_apply(self):
        status, out, _ = self._sync()
        self.assertEqual(status, 0)
        self.assertIn("would change: CITATION.bib, CITATION.cff", out)
        self.assertFalse((self.root / "CITATION.bib").exists())
        self.assertEqual(self._sync("--apply")[0], 0)
        self.assertTrue((self.root / "CITATION.cff").is_file())
        self.assertIn("would change: nothing", self._sync()[1])  # idempotent

    def test_check_tracks_version_and_edits_not_head(self):
        status, out, _ = _main("--repo-root", str(self.root), "check")
        self.assertEqual(status, 1)  # no files yet: stale, not an error
        self._sync("--apply")
        self.assertEqual(_main("--repo-root", str(self.root), "check")[0], 0)
        init = self.root / "scikitplot" / "__init__.py"
        init.write_text('__version__ = "0.5.dev1"\n', encoding="utf-8")
        status, out, _ = _main("--repo-root", str(self.root), "check")
        self.assertEqual(status, 1)
        self.assertIn("declares 0.5.dev1", out)
        init.write_text('__version__ = "0.5.dev0"\n', encoding="utf-8")
        cff = self.root / "CITATION.cff"
        cff.write_text(
            cff.read_text(encoding="utf-8").replace("BSD-3-Clause", "MIT"),
            encoding="utf-8",
        )
        status, out, _ = _main("--repo-root", str(self.root), "check", "--json")
        self.assertEqual(status, 1)
        self.assertEqual(
            json.loads(out)["problems"],
            ["CITATION.cff differs from its template rendering: run sync --apply"],
        )

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_head_is_read_from_git(self):
        def git(*args):
            subprocess.run(
                ["git", "-C", str(self.root), *args], check=True, capture_output=True
            )

        git("init", "-q")
        git("add", "-A")
        git(
            "-c",
            "user.email=a@b.example",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "x",
        )
        commit, date = MODULE.read_git_head(self.root)
        self.assertRegex(commit, r"^[0-9a-f]{40}$")
        self.assertRegex(date, r"^\d{4}-\d{2}-\d{2}$")
        status, out, _ = _main("--repo-root", str(self.root), "render")
        self.assertEqual(status, 0)
        self.assertIn(f'commit: "{commit}"', out)

    def test_errors_exit_2_with_a_message(self):
        status, _, err = _main(
            "--repo-root",
            str(self.root),
            "render",
            "--commit",
            "xyz",
            "--date",
            "2026-10-28",
        )
        self.assertEqual(status, 2)
        self.assertIn("is not a hexadecimal hash", err)
        status, _, err = _main(
            "--repo-root", str(self.root), "render", "--no-git", "--commit", SHA
        )
        self.assertEqual(status, 2)


@unittest.skipUnless(HAS_JINJA, "rendering needs jinja2")
class RepositoryGate(unittest.TestCase):
    def test_the_repository_citation_files_are_current(self):
        status, out, err = _main("--repo-root", str(REPO), "check")
        self.assertEqual(status, 0, out + err)


if __name__ == "__main__":
    unittest.main()
