"""
Structural checks on the published `cleanprompt` gallery examples.

Notes
-----
**User notes.** These do not execute the examples — that is lane 23, and it
takes minutes because it runs six subprocesses in sandboxed homes. These are
the cheap checks that catch the mistakes which are easy to make and silent to
miss, so they can run on every commit.

**Developer notes — what is being defended, and why each rule is here.**

A gallery script is code that runs during somebody else's documentation build,
in their process, on their machine. Two of the three rules below exist because
the consequence of breaking them is invisible in review.

*Every script confines its vault.* A vault holds the removed values in clear
text, and the CLI's default location is the platform state directory. A script
that does not set ``CLEANPROMPT_VAULT`` leaves those values in the
documentation builder's home. The execution lane proves the absence of leftover
files; this proves the *intent* is present in the source, which is what a
reviewer can act on.

*Every script cleans up.* A ``TemporaryDirectory`` that is never cleaned is a
directory of clear-text values surviving the build.

*Every folder Sphinx-Gallery renders has a ``README.txt``.* Without one, the
folder is silently skipped and the examples never appear — a documentation
build that succeeds having published nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

GALLERY = (
    Path(__file__).resolve().parents[3].parent / "galleries" / "examples" / "cleanprompt"
)


def _scripts():
    """Return the gallery scripts, or skip when the gallery is not checked out."""
    if not GALLERY.is_dir():
        pytest.skip("gallery not present in this checkout: {0}".format(GALLERY))
    found = sorted(GALLERY.glob("plot_*.py"))
    if not found:
        pytest.skip("no gallery scripts in {0}".format(GALLERY))
    return found


class TestGalleryStructure:
    """The shape Sphinx-Gallery requires."""

    def test_the_folder_has_a_readme(self):
        """Without one, the folder is skipped and nothing is published."""
        _scripts()
        readme = GALLERY / "README.txt"
        assert readme.is_file()
        assert readme.read_text(encoding="utf-8").strip(), "README.txt is empty"

    def test_the_readme_declares_a_label_and_a_title(self):
        _scripts()
        text = (GALLERY / "README.txt").read_text(encoding="utf-8")
        assert ".. _cleanprompt_examples:" in text
        assert "=" * 8 in text

    def test_every_script_has_a_title_and_a_currentmodule(self):
        for script in _scripts():
            text = script.read_text(encoding="utf-8")
            assert text.startswith('"""'), script.name
            assert ".. currentmodule:: scikitplot.cleanprompt" in text, script.name

    def test_every_script_carries_the_licence_header(self):
        for script in _scripts():
            text = script.read_text(encoding="utf-8")
            assert "SPDX-License-Identifier: BSD-3-Clause" in text, script.name

    def test_every_script_declares_tags(self):
        for script in _scripts():
            assert ".. tags::" in script.read_text(encoding="utf-8"), script.name

    def test_every_script_declares_a_level(self):
        """The learning path in the README is only real if the tags agree."""
        levels = set()
        for script in _scripts():
            text = script.read_text(encoding="utf-8")
            found = [
                one
                for one in ("beginner", "intermediate", "advanced")
                if "level: {0}".format(one) in text
            ]
            assert len(found) == 1, "{0}: levels {1}".format(script.name, found)
            levels.add(found[0])
        assert levels == {"beginner", "intermediate", "advanced"}


class TestGallerySafety:
    """The rules that keep a documentation build from holding somebody's data."""

    def test_every_script_that_shells_out_confines_its_vault(self):
        """
        Only the command line resolves a *default* vault path.

        Notes
        -----
        **Developer notes.** The first version of this rule fired on any script
        containing the word ``encode``, which flagged the pure-API example — a
        script that writes no vault at all, because :func:`encode` and
        :class:`Session` keep it in memory and only the CLI resolves a path on
        the filesystem. A rule that reports a file which cannot have the defect
        teaches people to ignore it, so the condition is the one that is
        actually true: a script that starts the CLI in a subprocess must say
        where the vault goes.

        The complementary direction — that nothing else writes one either — is
        not assertable from the source, and is measured by lane 23 instead,
        which runs each script with ``HOME`` redirected and fails on any file
        left behind.
        """
        for script in _scripts():
            text = script.read_text(encoding="utf-8")
            if "subprocess.run" not in text and "subprocess.Popen" not in text:
                continue
            assert 'os.environ["CLEANPROMPT_VAULT"]' in text, (
                "{0} starts the CLI, which resolves a vault in the platform "
                "state directory by default; point CLEANPROMPT_VAULT at a "
                "TemporaryDirectory in the first cell".format(script.name)
            )

    def test_every_script_cleans_up_its_workspace(self):
        for script in _scripts():
            text = script.read_text(encoding="utf-8")
            if "TemporaryDirectory" not in text:
                continue
            assert ".cleanup()" in text, (
                "{0} creates a temporary workspace and never removes it".format(
                    script.name
                )
            )

    def test_no_script_hard_codes_a_reachable_domain(self):
        """A published page must not carry an address that could reach anyone."""
        allowed = ("example.com", "example.org", "example.net", "example.invalid")
        import re

        pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        offenders = []
        for script in _scripts():
            for address in pattern.findall(script.read_text(encoding="utf-8")):
                if not any(address.lower().endswith(host) for host in allowed):
                    offenders.append("{0}: {1}".format(script.name, address))
        assert offenders == []

    def test_no_script_hard_codes_a_routable_address(self):
        """RFC 5737 and RFC 3849 reserve the ranges a document may print."""
        import re

        reserved = ("192.0.2.", "198.51.100.", "203.0.113.", "127.0.0.1", "0.0.0.0")
        pattern = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
        offenders = []
        for script in _scripts():
            for address in pattern.findall(script.read_text(encoding="utf-8")):
                if not any(address.startswith(prefix) for prefix in reserved):
                    offenders.append("{0}: {1}".format(script.name, address))
        assert offenders == []


class TestGalleryLane:
    """The execution lane exists and is recorded."""

    def test_the_probe_is_checked_in(self):
        probe = Path(__file__).resolve().parents[1] / "evidence" / "probe_gallery.py"
        assert probe.is_file()

    def test_the_lane_is_recorded(self):
        import json

        evidence = Path(__file__).resolve().parents[1] / "EVIDENCE.json"
        lanes = json.loads(evidence.read_text(encoding="utf-8"))["lanes"]
        recorded = {lane["id"]: lane for lane in lanes}
        assert "gallery_examples" in recorded
        assert recorded["gallery_examples"]["status"] == "PASS"


class TestGalleryReStructuredText:
    """Heading hierarchy, which is easy to get wrong and silent when it is."""

    @staticmethod
    def _headings(script):
        """Return (underline_character, title) for every heading in a script."""
        lines = script.read_text(encoding="utf-8").splitlines()
        found = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("#") or index == 0:
                continue
            body = stripped[1:].strip()
            if len(body) < 3 or len(set(body)) != 1 or body[0] not in "=-^\"~":
                continue
            previous = lines[index - 1].strip()
            title = previous[1:].strip() if previous.startswith("#") else ""
            if title:
                found.append((body, title))
        return found

    def test_every_underline_is_long_enough(self):
        """docutils warns and drops the section when it is short."""
        short = []
        for script in _scripts():
            for underline, title in self._headings(script):
                if len(underline) < len(title):
                    short.append("{0}: {1}".format(script.name, title))
        assert short == []

    def test_no_body_heading_uses_the_title_level(self):
        """
        The docstring title owns ``=``.

        Notes
        -----
        **Developer notes.** A body section underlined with ``=`` becomes a
        sibling of the page title rather than a child of it, so the page
        renders with two top-level sections and the sidebar shows a flat list.
        Nothing fails; the document is simply wrong, which is why this is
        asserted rather than reviewed.
        """
        offenders = []
        for script in _scripts():
            for underline, title in self._headings(script):
                if underline[0] == "=":
                    offenders.append("{0}: {1}".format(script.name, title))
        assert offenders == []

    def test_the_heading_levels_are_consistent_within_a_script(self):
        """A level's character must not change once it has been established."""
        for script in _scripts():
            seen = []
            for underline, _title in self._headings(script):
                character = underline[0]
                if character not in seen:
                    seen.append(character)
            assert seen in ([], ["-"], ["-", "^"]), "{0}: {1}".format(script.name, seen)

    def test_every_script_compiles(self):
        """A syntax error in an example is a failed documentation build."""
        for script in _scripts():
            compile(script.read_text(encoding="utf-8"), str(script), "exec")
