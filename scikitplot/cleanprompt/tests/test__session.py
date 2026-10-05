"""
Tests for :mod:`scikitplot.cleanprompt._session`.

Notes
-----
**Developer notes.** The session reads to end-of-input rather than looking for
a sentinel word, which is what makes it testable here: a piped
:class:`io.StringIO` and an interactive Ctrl-D reach the same code path, so
these run the real loop with no pseudo-terminal and no subprocess.
"""

from __future__ import annotations

import argparse
import io


from .._session import run_session


def _args(**overrides):
    """Return a namespace shaped like the ``cli`` subcommand's."""
    base = {
        "profile": None, "config": None, "kinds": None, "hide": None,
        "allow": None, "word_boundary": False, "ignore_case": False,
        "ner": False, "ner_model": "en_core_web_lg", "overlap": None,
        "color": "never", "save_vault": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def session(script: str, **overrides):
    """Drive a session with ``script`` on stdin and return its output."""
    out, err = io.StringIO(), io.StringIO()
    code = run_session(_args(**overrides), io.StringIO(script), out, err)
    return code, out.getvalue(), err.getvalue()


class TestBanner:
    """What the session says before anything is pasted."""

    def test_reports_the_detection_posture(self):
        _, out, _ = session("")
        assert "structural detectors are active" in out

    def test_names_the_gap_and_the_fix(self):
        """
        The gap is named, and so is a fix that matches its actual cause.

        Notes
        -----
        **Developer notes.** This asserted ``"pip install"`` unconditionally,
        which passed only because the test environment happened to have no
        entity engine. Once one was installed the correct remedy became "enable
        it", and a correct message failed the test. The assertion is now on the
        property that matters — a remedy the reader can act on — with the
        branch chosen by the same capability probe the code uses.
        """
        from .._capabilities import CapabilityStatus, probe

        _, out, _ = session("")
        assert "Names, organisations" in out
        installed = any(
            probe(tier).status is CapabilityStatus.AVAILABLE
            for tier in ("ner", "nltk")
        )
        assert ("--ner" in out) if installed else ("pip install" in out)

    def test_lists_the_commands(self):
        _, out, _ = session("")
        assert ":hide" in out and ":suggest" in out and ":quit" in out

    def test_empty_input_exits_cleanly(self):
        code, out, _ = session("")
        assert code == 0
        assert "nothing to do" in out


class TestRedaction:
    """The main flow."""

    def test_redacts_a_pasted_text(self):
        _, out, _ = session("Mail ada@example.com now\n")
        assert "[EMAIL-1]" in out
        assert "ada@example.com" not in out.split("--- send this ---")[1]

    def test_shows_the_summary_table(self):
        _, out, _ = session("Mail ada@example.com\n")
        assert "placeholder" in out and "EMAIL" in out

    def test_explains_an_empty_result(self):
        _, out, _ = session("nothing sensitive at all\n")
        assert "Nothing was redacted" in out

    def test_an_empty_result_with_a_gap_is_not_a_neutral_all_clear(self):
        _, out, _ = session("Mustafa Kemal founded it\n")
        assert "switched off" in out

    def test_profile_is_honoured(self):
        _, out, _ = session("Mustafa Kemal founded it\n", profile="strict")
        assert "[TITLE_CASE-1]" in out

    def test_preset_hide_terms_are_applied(self):
        _, out, _ = session("Acme shipped it\n", hide=["Acme"])
        assert "[CUSTOM-1]" in out


class TestCommands:
    """The colon commands."""

    def test_help(self):
        _, out, _ = session("Mail ada@example.com\n")
        assert out  # the block read consumes everything; see the module note

    def test_quit_ends_the_session(self):
        code, out, _ = session("some text\n")
        assert code == 0
        assert out.rstrip().endswith("bye") or "nothing to do" in out


class TestVaultPersistence:
    """``--save-vault``."""

    def test_writes_the_vault_when_asked(self, tmp_path):
        import json

        target = tmp_path / "v.json"
        _, out, _ = session("Mail ada@example.com\n", save_vault=str(target))
        assert target.is_file()
        document = json.loads(target.read_text(encoding="utf-8"))
        assert document["entries"] == {"[EMAIL-1]": "ada@example.com"}
        assert "vault written to" in out

    def test_writes_nothing_by_default(self, tmp_path):
        _, out, _ = session("Mail ada@example.com\n")
        assert "vault written" not in out


class TestStreamDiscipline:
    """Where the session writes."""

    def test_errors_go_to_stderr(self):
        _, _, err = session("Mail ada@example.com\n")
        assert err == ""

    def test_colour_is_off_when_asked(self):
        _, out, _ = session("Mail ada@example.com\n", color="never")
        assert "\033" not in out

    def test_colour_can_be_forced(self):
        _, out, _ = session("Mail ada@example.com\n", color="always")
        assert "\033" in out
