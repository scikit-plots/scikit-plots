"""
Tests for :mod:`scikitplot.cleanprompt._cli`.

Notes
-----
**Developer notes.** The entry point is exercised by calling :func:`main` with
an argument list and injected streams, so every branch is reachable without a
subprocess. ``__main__.py`` contains no logic of its own and is covered here;
that it stays trivial is asserted by :class:`TestModuleEntryPoint`.
"""

from __future__ import annotations

import io
import json
import os
import stat
import sys

import pytest

from .._cli import VAULT_FORMAT, build_parser, main


def _run(args, stdin=""):
    """Run the CLI and return ``(status, stdout, stderr)``."""
    out, err = io.StringIO(), io.StringIO()
    status = main(args, stdin=io.StringIO(stdin), stdout=out, stderr=err)
    return status, out.getvalue(), err.getvalue()


class TestParser:
    """Argument surface."""

    def test_builds(self):
        assert build_parser() is not None

    def test_is_rebuilt_each_time(self):
        assert build_parser() is not build_parser()

    def test_subcommands(self):
        parser = build_parser()
        for command in ("doctor", "kinds", "inspect", "scan", "docker"):
            assert parser.parse_args([command]) is not None
        for command in ("redact", "restore"):
            assert parser.parse_args([command, "--vault", "v"]) is not None

    def test_redact_no_longer_requires_a_vault_path(self):
        """It defaults to the state-directory vault; see TestDefaultVaultPath."""
        assert build_parser().parse_args(["redact"]) is not None

    def test_a_vault_path_is_still_accepted(self):
        assert build_parser().parse_args(["redact", "--vault", "v"]) is not None


class TestNoCommand:
    """Invoked with nothing to do."""

    def test_prints_help_and_returns_two(self):
        status, out, _ = _run([])
        assert status == 2
        assert "redact" in out


class TestKinds:
    """The ``kinds`` subcommand."""

    def test_text_output(self):
        status, out, _ = _run(["kinds"])
        assert status == 0
        assert "EMAIL" in out and "PHONE" in out

    def test_json_output(self):
        status, out, _ = _run(["kinds", "--format", "json"])
        payload = json.loads(out)["kinds"]
        assert status == 0
        assert payload["EMAIL"]["intent"]
        assert payload["CREDIT_CARD"]["validated"] is True
        assert payload["MAC"]["validated"] is False

    def test_lists_profiles(self):
        _, out, _ = _run(["kinds", "--format", "json"])
        assert set(json.loads(out)["profiles"]) == {"minimal", "balanced", "strict"}

    def test_title_case_is_listed_but_off_by_default(self):
        _, out, _ = _run(["kinds", "--format", "json"])
        entry = json.loads(out)["kinds"]["TITLE_CASE"]
        assert entry["enabled_by_default"] is False


class TestDoctor:
    """The ``doctor`` subcommand, which replaced ``capabilities``."""

    def test_text_output(self):
        status, out, _ = _run(["doctor"])
        assert status == 0
        for tier in ("ner", "nltk", "web", "crypto"):
            assert tier in out

    def test_json_output(self):
        status, out, _ = _run(["doctor", "--format", "json"])
        payload = json.loads(out)
        assert status == 0
        assert set(payload["tiers"]) == {"ner", "nltk", "web", "crypto"}
        assert payload["tiers"]["ner"]["install_hint"].startswith("pip install")
        assert payload["status"] in ("ok", "degraded")

    def test_reports_blind_spots(self):
        """The whole point: what is NOT being detected is first-class."""
        _, out, _ = _run(["doctor", "--format", "json"])
        payload = json.loads(out)
        assert "blind_spots" in payload
        assert payload["healthy"] == (payload["status"] == "ok")

    def test_reports_the_active_configuration(self):
        _, out, _ = _run(["doctor", "--format", "json"])
        config = json.loads(out)["configuration"]
        assert len(config["policy_fingerprint"]) == 16
        assert len(config["grammar_fingerprint"]) == 16

    def test_capabilities_remains_as_an_alias(self):
        """The old name keeps working; it just is not the documented one."""
        status, out, _ = _run(["capabilities", "--format", "json"])
        assert status == 0
        assert "tiers" in json.loads(out)

    @pytest.mark.parametrize("fmt", ["text", "json"])
    def test_every_stdlib_format_renders(self, fmt):
        status, out, _ = _run(["doctor", "--format", fmt])
        assert status == 0
        assert out.strip()

    def test_can_diagnose_a_text(self, tmp_path):
        source = tmp_path / "t.txt"
        source.write_text("Mustafa Kemal wrote to ada@example.com", encoding="utf-8")
        _, out, _ = _run(["doctor", "--in", str(source), "--format", "json"])
        report = json.loads(out)["text_report"]
        assert report["would_redact"] >= 1
        assert report["outcome"] in ("ok", "warning", "alert")


class TestRedactAndRestore:
    """The round trip through the file system."""

    def test_stdin_to_stdout(self, tmp_path):
        vault = tmp_path / "v.json"
        status, out, err = _run(
            ["redact", "--vault", str(vault), "--color", "never"],
            stdin="mail ada@example.com",
        )
        assert status == 0
        assert out.strip() == "mail [EMAIL-1]"
        assert "redacted 1 value(s)" in err
        assert vault.is_file()

    def test_file_to_file(self, tmp_path):
        source = tmp_path / "in.txt"
        source.write_text("mail ada@example.com", encoding="utf-8")
        target = tmp_path / "out.txt"
        vault = tmp_path / "v.json"
        status, _, _ = _run(
            ["redact", "--in", str(source), "--out", str(target), "--vault", str(vault)]
        )
        assert status == 0
        assert target.read_text(encoding="utf-8") == "mail [EMAIL-1]"

    def test_full_round_trip(self, tmp_path):
        original = "Ann met Anna; mail ada@example.com or call +1 555 010 4477"
        vault = tmp_path / "v.json"
        _, redacted, _ = _run(
            [
                "redact",
                "--vault",
                str(vault),
                "--hide",
                "Ann",
                "--hide",
                "Anna",
                "--color",
                "never",
            ],
            stdin=original,
        )
        status, restored, _ = _run(
            ["restore", "--vault", str(vault)], stdin=redacted.rstrip("\n")
        )
        assert status == 0
        assert restored.rstrip("\n") == original

    def test_kinds_narrowing(self, tmp_path):
        vault = tmp_path / "v.json"
        _, out, _ = _run(
            ["redact", "--vault", str(vault), "--kinds", "EMAIL", "--color", "never"],
            stdin="mail a@x.com call +1 555 010 4477",
        )
        assert "[EMAIL-1]" in out
        assert "+1 555 010 4477" in out

    def test_ignore_case(self, tmp_path):
        vault = tmp_path / "v.json"
        _, out, _ = _run(
            [
                "redact",
                "--vault",
                str(vault),
                "--ignore-case",
                "--hide",
                "acme",
                "--color",
                "never",
            ],
            stdin="Acme and ACME",
        )
        assert out.strip() == "[CUSTOM-1] and [CUSTOM-1]"

    def test_word_boundary(self, tmp_path):
        vault = tmp_path / "v.json"
        _, out, _ = _run(
            [
                "redact",
                "--vault",
                str(vault),
                "--hide",
                "class",
                "--word-boundary",
                "--color",
                "never",
            ],
            stdin="classic",
        )
        assert out.strip() == "classic"

    def test_quiet_suppresses_the_summary(self, tmp_path):
        vault = tmp_path / "v.json"
        _, _, err = _run(
            ["redact", "--vault", str(vault), "--quiet"], stdin="mail a@x.com"
        )
        assert err == ""

    def test_reveal_is_opt_in(self, tmp_path):
        vault = tmp_path / "v.json"
        _, _, without = _run(["redact", "--vault", str(vault)], stdin="mail a@x.com")
        _, _, with_reveal = _run(
            ["redact", "--vault", str(vault), "--reveal"], stdin="mail a@x.com"
        )
        assert "a@x.com" not in without
        assert "a@x.com" in with_reveal

    def test_strict_restore_fails_on_an_unknown_label(self, tmp_path):
        vault = tmp_path / "v.json"
        _run(["redact", "--vault", str(vault)], stdin="mail a@x.com")
        status, _, err = _run(
            ["restore", "--vault", str(vault), "--strict"], stdin="see [EMAIL-9]"
        )
        assert status == 1
        assert "not present in the vault" in err

    def test_lenient_restore_reports_the_unknown_label(self, tmp_path):
        vault = tmp_path / "v.json"
        _run(["redact", "--vault", str(vault)], stdin="mail a@x.com")
        status, out, err = _run(
            ["restore", "--vault", str(vault)], stdin="see [EMAIL-9]"
        )
        assert status == 0
        assert "[EMAIL-9]" in out
        assert "unknown" in err


class TestStreamDiscipline:
    """Results on stdout, diagnostics on stderr."""

    def test_no_diagnostics_on_stdout(self, tmp_path):
        vault = tmp_path / "v.json"
        _, out, err = _run(
            ["redact", "--vault", str(vault), "--color", "never"], stdin="mail a@x.com"
        )
        assert out.strip() == "mail [EMAIL-1]"
        assert "vault:" in err

    def test_output_is_pipe_safe(self, tmp_path):
        """A non-terminal stdout gets no escape sequences."""
        vault = tmp_path / "v.json"
        _, out, _ = _run(["redact", "--vault", str(vault)], stdin="mail a@x.com")
        assert "\033" not in out


class TestVaultFile:
    """The on-disk vault document."""

    def test_structure(self, tmp_path):
        vault = tmp_path / "v.json"
        _run(["redact", "--vault", str(vault)], stdin="mail a@x.com")
        document = json.loads(vault.read_text(encoding="utf-8"))
        assert document["format"] == VAULT_FORMAT
        assert document["entries"] == {"[EMAIL-1]": "a@x.com"}
        assert len(document["grammar_fingerprint"]) == 16
        assert document["tag_style"]["prefix"] == "["

    @pytest.mark.skipif(os.name == "nt", reason="POSIX modes only")
    def test_is_created_owner_readable_only(self, tmp_path):
        vault = tmp_path / "v.json"
        _run(["redact", "--vault", str(vault)], stdin="mail a@x.com")
        mode = stat.S_IMODE(os.stat(str(vault)).st_mode)
        assert mode == 0o600

    def test_malformed_json_is_refused(self, tmp_path):
        vault = tmp_path / "v.json"
        vault.write_text("{not json", encoding="utf-8")
        status, _, err = _run(["restore", "--vault", str(vault)], stdin="x")
        assert status == 1
        assert "not valid JSON" in err

    def test_wrong_format_version_is_refused(self, tmp_path):
        vault = tmp_path / "v.json"
        vault.write_text(json.dumps({"format": 99, "entries": {}}), encoding="utf-8")
        status, _, err = _run(["restore", "--vault", str(vault)], stdin="x")
        assert status == 1
        assert "format" in err

    def test_non_object_document_is_refused(self, tmp_path):
        vault = tmp_path / "v.json"
        vault.write_text("[]", encoding="utf-8")
        status, _, err = _run(["restore", "--vault", str(vault)], stdin="x")
        assert status == 1
        assert "not a JSON object" in err

    def test_malformed_entries_are_refused(self, tmp_path):
        vault = tmp_path / "v.json"
        vault.write_text(
            json.dumps({"format": VAULT_FORMAT, "entries": {"[A-1]": 5}}),
            encoding="utf-8",
        )
        status, _, err = _run(["restore", "--vault", str(vault)], stdin="x")
        assert status == 1
        assert "malformed" in err

    def test_missing_vault_file_is_reported(self, tmp_path):
        status, _, err = _run(
            ["restore", "--vault", str(tmp_path / "absent.json")], stdin="x"
        )
        assert status == 1
        assert "error:" in err


class TestErrorHandling:
    """Exit statuses and messages."""

    def test_limit_error_returns_one(self, tmp_path):
        vault = tmp_path / "v.json"
        status, _, err = _run(
            ["redact", "--vault", str(vault), "--kinds", "NOPE"], stdin="x"
        )
        assert status == 1
        assert "unknown pattern kind" in err

    def test_capability_error_prints_the_hint(self, tmp_path, monkeypatch):
        from .. import _capabilities

        monkeypatch.setattr(_capabilities, "_installed_version", lambda _n: None)
        vault = tmp_path / "v.json"
        status, _, err = _run(["redact", "--vault", str(vault), "--ner"], stdin="x")
        # 69 is EX_UNAVAILABLE: a missing OPTIONAL capability is a different
        # outcome from a general failure, and a caller scripting this needs to
        # tell "install something" apart from "it broke".
        assert status == 69
        assert "hint: pip install" in err

    def test_unreadable_input_returns_one(self, tmp_path):
        status, _, err = _run(
            [
                "redact",
                "--in",
                str(tmp_path / "missing.txt"),
                "--vault",
                str(tmp_path / "v"),
            ]
        )
        assert status == 1
        assert "error:" in err

    def test_strict_overlap_choice_is_accepted(self, tmp_path):
        vault = tmp_path / "v.json"
        status, _, _ = _run(
            [
                "redact",
                "--vault",
                str(vault),
                "--overlap",
                "STRICT",
                "--color",
                "never",
            ],
            stdin="plain text",
        )
        assert status == 0


class TestModuleEntryPoint:
    """``python -m scikitplot.cleanprompt``."""

    def test_delegates_without_logic_of_its_own(self):
        import ast
        import pathlib

        path = pathlib.Path(__file__).resolve().parent.parent / "__main__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]

    def test_runs_as_a_module(self):
        import subprocess

        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        completed = subprocess.run(
            [sys.executable, "-m", "scikitplot.cleanprompt", "kinds"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        assert completed.returncode == 0
        assert "EMAIL" in completed.stdout


def _run_both(args, stdin=""):
    """
    Run the CLI through both frontends and return the argparse result.

    Notes
    -----
    **Developer notes.** Every assertion about the new input handling is made
    against both frontends, because the two get there by different routes:
    argparse binds a variadic positional with ``nargs="*"`` and click with
    ``nargs=-1``, and they disagree about what "absent" looks like. ``CP-021``
    was exactly this class of divergence, so a new positional is checked in
    both from the day it lands rather than in whichever one happened to be
    installed.
    """
    results = {}
    for frontend in ("argparse", "click"):
        out, err = io.StringIO(), io.StringIO()
        status = main(
            ["--frontend", frontend, *args],
            stdin=io.StringIO(stdin),
            stdout=out,
            stderr=err,
        )
        results[frontend] = (status, out.getvalue(), err.getvalue())
    assert results["argparse"] == results["click"], "frontends disagree: {0}".format(
        results
    )
    return results["argparse"]


class _Tty(io.StringIO):
    """A stream that claims to be a terminal."""

    def isatty(self):
        return True


class TestTextOnTheCommandLine:
    """
    Text passed directly, which is what a first-time user types.

    Notes
    -----
    **Developer notes.** These commands took ``--in`` or standard input and
    nothing else, so ``cleanprompt inspect "some text"`` was rejected with
    "Got unexpected extra argument" — a message that names the text as the
    problem and does not say what to do instead. The obvious invocation now
    works.
    """

    def test_inspect_accepts_text_directly(self):
        status, out, _ = _run_both(["inspect", "mail ada@example.com"])
        assert status == 0
        assert "[EMAIL-1]" in out

    def test_an_unquoted_phrase_arrives_intact(self):
        """A shell splits on spaces before this program sees anything."""
        status, out, _ = _run_both(
            ["inspect", "--reveal", "mail", "ada@example.com", "now"]
        )
        assert status == 0
        assert "ada@example.com" in out

    def test_scan_accepts_text_directly(self):
        status, _, _ = _run_both(["scan", "mail ada@example.com"])
        assert status == 3  # EXIT_FOUND

    def test_scan_is_clean_on_text_with_nothing_in_it(self):
        status, _, _ = _run_both(["scan", "the cat sat on the mat"])
        assert status == 0

    def test_redact_accepts_text_directly(self, tmp_path):
        vault = tmp_path / "v.json"
        status, out, _ = _run_both(
            ["redact", "--vault", str(vault), "mail ada@example.com"]
        )
        assert status == 0
        assert out.strip() == "mail [EMAIL-1]"

    def test_restore_accepts_text_directly(self, tmp_path):
        vault = tmp_path / "v.json"
        _run(["redact", "--vault", str(vault), "mail ada@example.com"])
        status, out, _ = _run(["restore", "--vault", str(vault), "I mailed [EMAIL-1]."])
        assert status == 0
        assert out.strip() == "I mailed ada@example.com."

    def test_text_and_in_together_is_an_error(self, tmp_path):
        """Guessing which one was meant would silently ignore the other."""
        path = tmp_path / "p.txt"
        path.write_text("from the file", encoding="utf-8")
        status, _, err = _run_both(["inspect", "--in", str(path), "direct text"])
        assert status == 1
        assert "--in" in err and "one or the other" in err

    def test_in_still_works_on_its_own(self, tmp_path):
        path = tmp_path / "p.txt"
        path.write_text("mail ada@example.com", encoding="utf-8")
        status, out, _ = _run_both(["inspect", "--in", str(path)])
        assert status == 0
        assert "[EMAIL-1]" in out

    def test_stdin_still_works_on_its_own(self):
        status, out, _ = _run_both(["inspect"], stdin="mail ada@example.com")
        assert status == 0
        assert "[EMAIL-1]" in out

    def test_explicit_dash_still_reads_stdin(self):
        status, out, _ = _run_both(
            ["inspect", "--in", "-"], stdin="mail ada@example.com"
        )
        assert status == 0
        assert "[EMAIL-1]" in out


class TestInteractiveStdinHint:
    """Falling through to standard input must not look like a hang."""

    def _run_tty(self, args, stdin=""):
        out, err = io.StringIO(), io.StringIO()
        status = main(args, stdin=_Tty(stdin), stdout=out, stderr=err)
        return status, out.getvalue(), err.getvalue()

    def test_a_terminal_is_told_what_is_happening(self):
        _, _, err = self._run_tty(["inspect"], stdin="mail ada@example.com")
        assert "reading from standard input" in err
        assert "Ctrl-D" in err

    def test_the_hint_names_the_other_ways_in(self):
        _, _, err = self._run_tty(["inspect"], stdin="x")
        assert "--in notes.txt" in err
        assert "<<'END'" in err

    def test_the_hint_names_the_command_that_was_run(self):
        """It is meant to be pasted, so a representative name will not do."""
        _, _, err = self._run_tty(["scan"], stdin="x")
        assert "cleanprompt scan" in err

    def test_a_pipe_is_not_told_anything(self):
        """A hint on standard error would be noise in every script."""
        _, _, err = _run_both(["inspect"], stdin="mail ada@example.com")
        assert "reading from standard input" not in err

    def test_the_hint_goes_to_stderr_not_stdout(self):
        """Standard output carries the result and must stay clean in a pipe."""
        _, out, err = self._run_tty(["inspect"], stdin="mail ada@example.com")
        assert "reading from standard input" in err
        assert "reading from standard input" not in out

    def test_text_on_the_command_line_prints_no_hint(self):
        _, _, err = self._run_tty(["inspect", "mail ada@example.com"])
        assert "reading from standard input" not in err


class TestRoundtrip:
    """The one-shot demonstration of both halves."""

    SAMPLE = "Mail ada@example.com from 192.168.1.10."

    def test_shows_every_stage(self):
        status, out, _ = _run_both(["roundtrip", self.SAMPLE])
        assert status == 0
        for stage in ("1 ·", "2 ·", "3 ·", "4 ·", "5 ·"):
            assert stage in out

    def test_the_safe_text_holds_no_value(self):
        _, out, _ = _run_both(["roundtrip", self.SAMPLE])
        sent = out.split("2 ·")[1].split("3 ·")[0]
        assert "ada@example.com" not in sent
        assert "[EMAIL-1]" in sent

    def test_the_restored_text_holds_the_values_again(self):
        _, out, _ = _run_both(["roundtrip", self.SAMPLE])
        restored = out.split("5 ·")[1]
        assert "ada@example.com" in restored
        assert "192.168.1.10" in restored

    def test_the_stand_in_reply_is_labelled_as_one(self):
        """Presenting a fixed string as a model's answer would be a lie."""
        _, out, _ = _run_both(["roundtrip", self.SAMPLE])
        assert "NOT from a model" in out

    def test_a_real_reply_is_not_labelled_as_simulated(self):
        _, out, _ = _run_both(
            ["roundtrip", self.SAMPLE, "--reply", "I mailed [EMAIL-1]."]
        )
        assert "NOT from a model" not in out
        assert "the model's reply" in out

    def test_a_real_reply_is_restored(self):
        _, out, _ = _run_both(
            ["roundtrip", self.SAMPLE, "--reply", "I mailed [EMAIL-1]."]
        )
        assert "I mailed ada@example.com." in out

    def test_a_reply_can_come_from_a_file(self, tmp_path):
        path = tmp_path / "reply.txt"
        path.write_text("Done: [EMAIL-1]", encoding="utf-8")
        _, out, _ = _run_both(["roundtrip", self.SAMPLE, "--reply-in", str(path)])
        assert "Done: ada@example.com" in out

    def test_values_are_hidden_unless_reveal_is_given(self):
        _, out, _ = _run_both(["roundtrip", "Mail ada@example.com"])
        table = out.split("3 ·")[1].split("4 ·")[0]
        assert "ada@example.com" not in table

    def test_reveal_shows_them(self):
        _, out, _ = _run_both(["roundtrip", "Mail ada@example.com", "--reveal"])
        table = out.split("3 ·")[1].split("4 ·")[0]
        assert "ada@example.com" in table

    def test_writes_no_vault_file(self, tmp_path):
        """A demonstration that leaves a file behind leaves a liability."""
        before = set(tmp_path.iterdir())
        _run_both(["roundtrip", self.SAMPLE])
        assert set(tmp_path.iterdir()) == before

    def test_reports_an_exact_round_trip(self):
        _, out, _ = _run_both(["roundtrip", self.SAMPLE])
        assert "round trip exact: yes" in out

    def test_text_with_nothing_in_it_says_so(self):
        status, out, _ = _run_both(["roundtrip", "the cat sat on the mat"])
        assert status == 0
        assert "nothing" in out.lower()

    def test_json_output_carries_every_stage(self):
        status, out, _ = _run_both(["roundtrip", self.SAMPLE, "--format", "json"])
        payload = json.loads(out)
        assert status == 0
        assert set(payload) >= {
            "original",
            "safe",
            "entries",
            "reply",
            "reply_is_simulated",
            "restored",
            "round_trip_exact",
        }
        assert payload["round_trip_exact"] is True
        assert payload["reply_is_simulated"] is True

    def test_json_marks_a_real_reply_as_real(self):
        _, out, _ = _run_both(
            ["roundtrip", self.SAMPLE, "--reply", "ok [EMAIL-1]", "--format", "json"]
        )
        assert json.loads(out)["reply_is_simulated"] is False

    def test_json_safe_field_holds_no_value(self):
        _, out, _ = _run_both(["roundtrip", self.SAMPLE, "--format", "json"])
        assert "ada@example.com" not in json.loads(out)["safe"]

    def test_demo_is_an_alias(self):
        status, out, _ = _run_both(["demo", self.SAMPLE])
        assert status == 0
        assert "1 ·" in out

    def test_it_reads_stdin_like_every_other_command(self):
        status, out, _ = _run_both(["roundtrip"], stdin=self.SAMPLE)
        assert status == 0
        assert "[EMAIL-1]" in out

    def test_an_unresolvable_placeholder_is_named(self):
        _, out, _ = _run_both(["roundtrip", self.SAMPLE, "--reply", "see [EMAIL-9]"])
        assert "unresolved placeholders: [EMAIL-9]" in out


class TestOptionGrammar:
    """
    ``--`` and the long/short option forms, asserted through both frontends.

    Notes
    -----
    **Developer notes.** Two POSIX conventions, and this submodule owes both.

    A standalone ``--`` ends the options: everything after it is a positional
    argument even when it begins with a dash. That matters here more than in
    most tools, because the positional is *arbitrary user text* — a prompt
    about command-line flags, a password beginning with a hyphen, a filename
    like ``-rf``. Without ``--`` those are unpassable.

    A double dash before a word is a long option, as against a single-dash
    short one. Both frontends render both spellings from one declaration, so
    the assertions below run through each and compare.
    """

    def _both(self, argv, stdin=""):
        """Return the argparse result, having checked click agrees."""
        results = {}
        for frontend in ("argparse", "click"):
            out, err = io.StringIO(), io.StringIO()
            status = main(
                ["--frontend", frontend, *argv],
                stdin=io.StringIO(stdin),
                stdout=out,
                stderr=err,
            )
            results[frontend] = (status, out.getvalue())
        assert results["argparse"][0] == results["click"][0], (
            "frontends disagree on exit status for {0}: {1}".format(argv, results)
        )
        return results["argparse"]

    # -- standalone --, the options delimiter ------------------------------

    def test_double_dash_passes_ordinary_text(self):
        status, out = self._both(["roundtrip", "-f", "json", "--", "mail a@b.co"])
        assert status == 0
        assert json.loads(out)["original"] == "mail a@b.co"

    def test_double_dash_passes_text_that_looks_like_an_option(self):
        """Without it, '--secret' is parsed as a flag and the command fails."""
        status, out = self._both(
            ["roundtrip", "-f", "json", "--", "--secret is a@b.co"]
        )
        assert status == 0
        assert json.loads(out)["original"] == "--secret is a@b.co"

    def test_double_dash_passes_the_classic_dash_rf(self):
        """The canonical example: a value that would otherwise be read as -r -f."""
        status, out = self._both(["roundtrip", "-f", "json", "--", "-rf a@b.co"])
        assert status == 0
        assert json.loads(out)["original"] == "-rf a@b.co"

    def test_only_the_first_double_dash_is_consumed(self):
        """A '--' inside the text is text, which prose about CLIs contains."""
        status, out = self._both(
            ["roundtrip", "-f", "json", "--", "pass -- to rm, then a@b.co"]
        )
        assert status == 0
        assert json.loads(out)["original"] == "pass -- to rm, then a@b.co"

    def test_text_that_is_exactly_a_double_dash(self):
        status, out = self._both(["roundtrip", "-f", "json", "--", "--"])
        assert status == 0
        assert json.loads(out)["original"] == "--"

    def test_options_before_the_delimiter_still_apply(self):
        status, out = self._both(
            ["roundtrip", "-f", "json", "--kinds", "EMAIL", "--", "-x a@b.co"]
        )
        payload = json.loads(out)
        assert status == 0
        assert payload["original"] == "-x a@b.co"
        assert [e["kind"] for e in payload["entries"]] == ["EMAIL"]

    def test_what_follows_the_delimiter_is_never_an_option(self):
        """'--format json' after '--' is text, not a format request."""
        status, out = self._both(["roundtrip", "--", "--format", "json"])
        assert status == 0
        assert "1 ·" in out  # rendered as text, not as JSON

    def test_without_the_delimiter_a_dashy_argument_is_refused(self):
        """It must fail rather than be silently swallowed or misread."""
        status, _ = self._both(["inspect", "--secret is a@b.co"])
        assert status == 2

    def test_the_delimiter_works_for_every_text_command(self, tmp_path):
        for argv in (
            ["inspect", "--", "-x a@b.co"],
            ["scan", "--", "-x nothing here"],
            ["roundtrip", "--", "-x a@b.co"],
            ["redact", "--vault", str(tmp_path / "v.json"), "--", "-x a@b.co"],
        ):
            status, _ = self._both(argv)
            assert status in (0, 3), "{0} -> {1}".format(argv, status)

    # -- long and short option spellings -----------------------------------

    @pytest.mark.parametrize(
        ("short", "long_", "value"),
        [
            ("-f", "--format", "json"),
            ("-k", "--kinds", "EMAIL"),
        ],
    )
    def test_short_and_long_spellings_agree(self, short, long_, value):
        with_short = self._both(["inspect", short, value, "mail a@b.co"])
        with_long = self._both(["inspect", long_, value, "mail a@b.co"])
        assert with_short == with_long

    def test_short_input_and_output_options(self, tmp_path):
        source = tmp_path / "in.txt"
        source.write_text("mail ada@example.com", encoding="utf-8")
        target = tmp_path / "out.txt"
        status, _ = self._both(
            [
                "redact",
                "-i",
                str(source),
                "-o",
                str(target),
                "--vault",
                str(tmp_path / "v.json"),
            ]
        )
        assert status == 0
        assert target.read_text(encoding="utf-8") == "mail [EMAIL-1]"

    def test_short_quiet_option(self, tmp_path):
        out, err = io.StringIO(), io.StringIO()
        status = main(
            ["redact", "--vault", str(tmp_path / "v.json"), "-q", "mail a@b.co"],
            stdin=io.StringIO(""),
            stdout=out,
            stderr=err,
        )
        assert status == 0
        assert err.getvalue() == ""

    def test_version_has_a_short_spelling(self):
        for argv in (["-V"], ["--version"]):
            status, out = self._both(argv)
            assert status == 0
            assert "1.0.0" in out

    def test_help_has_both_spellings(self):
        for argv in (["inspect", "-h"], ["inspect", "--help"]):
            status, out = self._both(argv)
            assert status == 0
            assert "--format" in out

    def test_attached_and_separated_values_agree(self):
        """'--format=json' and '--format json' are the same request."""
        assert self._both(["inspect", "--format=json", "mail a@b.co"]) == self._both(
            ["inspect", "--format", "json", "mail a@b.co"]
        )

    def test_an_option_value_beginning_with_a_dash_uses_the_attached_form(self):
        """The portable spelling: both frontends accept '--hide=-value'."""
        status, out = self._both(
            ["inspect", "--reveal", "--hide=-secret", "mind the -secret"]
        )
        assert status == 0
        assert "-secret" in out

    def test_a_dash_value_without_the_attached_form_is_explained(self):
        """argparse refuses it; the message must name the form that works."""
        out, err = io.StringIO(), io.StringIO()
        status = main(
            ["--frontend", "argparse", "inspect", "--hide", "-secret", "x"],
            stdin=io.StringIO(""),
            stdout=out,
            stderr=err,
        )
        assert status == 2
        assert "--hide=-your-value" in err.getvalue()
        assert "--" in err.getvalue()


class TestNoOptionAbbreviation:
    """
    ``CP-032`` — an abbreviated long option must fail in both frontends.

    Notes
    -----
    **Developer notes.** argparse accepts any unambiguous prefix by default and
    click accepts none, so ``--form json`` succeeded or failed depending on
    which library was installed. Beyond the divergence, an abbreviation is a
    latent break in its own right: it stops being unambiguous the moment a new
    option shares its prefix, so a script that worked for a year fails on an
    upgrade that added a feature it never used.
    """

    def _status(self, frontend, argv):
        out, err = io.StringIO(), io.StringIO()
        return main(
            ["--frontend", frontend, *argv],
            stdin=io.StringIO(""),
            stdout=out,
            stderr=err,
        )

    @pytest.mark.parametrize(
        "argv",
        [
            ["inspect", "--form", "json", "x"],
            ["inspect", "--forma", "json", "x"],
            ["inspect", "--rev", "x"],
            ["inspect", "--no-sug", "x"],
            ["inspect", "--ner-eng", "nltk", "x"],
        ],
    )
    def test_abbreviations_are_refused_by_both(self, argv):
        assert self._status("argparse", argv) == 2
        assert self._status("click", argv) == 2

    @pytest.mark.parametrize(
        "argv",
        [
            ["inspect", "--format", "json", "x"],
            ["inspect", "--reveal", "x"],
            ["inspect", "--no-suggest", "x"],
            ["inspect", "--ner-engine", "none", "x"],
        ],
    )
    def test_full_spellings_are_accepted_by_both(self, argv):
        assert self._status("argparse", argv) == 0
        assert self._status("click", argv) == 0


@pytest.fixture()
def state_home(tmp_path, monkeypatch):
    """Point the default vault at a temporary state directory."""
    monkeypatch.delenv("CLEANPROMPT_VAULT", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    # Windows keeps state under %LOCALAPPDATA% and never reads XDG_STATE_HOME;
    # without this the test would use, and change, the real user profile.
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "state"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path / "state" / "cleanprompt" / "vault.json"


class TestDefaultVaultPath:
    """
    ``--vault`` has a default, and it is not the working directory.

    Notes
    -----
    **Developer notes.** The location is a security decision, not a taste one.
    A vault holds the removed values in clear text, and this tool is used from
    inside checkouts — the session that prompted the default ran from a
    checked-out branch. Defaulting to the working directory would put a
    plain-text secrets file one ``git add .`` away from a commit.
    """

    def test_the_default_is_not_the_working_directory(
        self, state_home, tmp_path, monkeypatch
    ):
        from .._cli import default_vault_path

        monkeypatch.chdir(tmp_path)
        assert not default_vault_path().startswith(str(tmp_path) + os.sep + "vault")
        assert "cleanprompt" in default_vault_path()

    def test_the_environment_variable_wins(self, tmp_path, monkeypatch):
        from .._cli import default_vault_path

        target = tmp_path / "mine.json"
        monkeypatch.setenv("CLEANPROMPT_VAULT", str(target))
        assert default_vault_path() == str(target)

    def test_an_explicit_option_wins_over_the_environment(self, tmp_path, monkeypatch):
        from .._cli import resolve_vault_path

        monkeypatch.setenv("CLEANPROMPT_VAULT", str(tmp_path / "env.json"))
        assert resolve_vault_path(str(tmp_path / "flag.json")) == str(
            tmp_path / "flag.json"
        )

    def test_a_user_path_is_expanded(self, monkeypatch, tmp_path):
        from .._cli import resolve_vault_path

        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_vault_path("~/v.json") == str(tmp_path / "v.json")

    def test_redact_needs_no_vault_option(self, state_home):
        status, out, err = _run(["redact", "mail ada@example.com"])
        assert status == 0
        assert out.strip() == "mail [EMAIL-1]"
        assert state_home.is_file()
        assert "vault:" in err

    def test_restore_needs_no_vault_option(self, state_home):
        _run(["redact", "-q", "mail ada@example.com"])
        status, out, _ = _run(["restore", "-q", "sent to [EMAIL-1]"])
        assert status == 0
        assert out.strip() == "sent to ada@example.com"

    def test_the_directory_is_created_owner_only(self, state_home):
        _run(["redact", "-q", "mail ada@example.com"])
        mode = stat.S_IMODE(os.stat(state_home.parent).st_mode)
        assert mode & 0o077 == 0, oct(mode)

    def test_the_file_is_created_owner_only(self, state_home):
        _run(["redact", "-q", "mail ada@example.com"])
        mode = stat.S_IMODE(os.stat(state_home).st_mode)
        assert mode & 0o077 == 0, oct(mode)

    def test_restoring_with_no_vault_yet_is_explained(self, state_home):
        status, _, err = _run(["restore", "see [EMAIL-1]"])
        assert status == 1
        assert "no vault at" in err
        assert "clean" in err

    def test_the_note_names_the_path_and_the_mode(self, state_home):
        _, _, err = _run(["redact", "mail ada@example.com"])
        assert "vault:" in err
        assert "overwrite" in err

    def test_quiet_silences_the_note(self, state_home):
        _, _, err = _run(["redact", "-q", "mail ada@example.com"])
        assert err == ""


class TestVaultMode:
    """``--vault-mode append`` keeps a placeholder meaning one thing."""

    def test_overwrite_is_the_default_for_redact(self, state_home):
        _run(["redact", "-q", "mail ada@example.com"])
        _run(["redact", "-q", "mail bob@example.com"])
        document = json.loads(state_home.read_text(encoding="utf-8"))
        assert document["entries"] == {"[EMAIL-1]": "bob@example.com"}

    def test_append_keeps_the_label_a_value_already_had(self, state_home):
        _run(["redact", "-q", "--vault-mode", "append", "mail ada@example.com"])
        _, out, _ = _run(
            ["redact", "-q", "--vault-mode", "append", "again ada@example.com"]
        )
        assert out.strip() == "again [EMAIL-1]"

    def test_append_numbers_a_new_value_after_the_old_ones(self, state_home):
        _run(["redact", "-q", "--vault-mode", "append", "mail ada@example.com"])
        _, out, _ = _run(
            ["redact", "-q", "--vault-mode", "append", "cc bob@example.com"]
        )
        assert out.strip() == "cc [EMAIL-2]"

    def test_append_accumulates_the_vault(self, state_home):
        _run(["redact", "-q", "--vault-mode", "append", "mail ada@example.com"])
        _run(["redact", "-q", "--vault-mode", "append", "cc bob@example.com"])
        document = json.loads(state_home.read_text(encoding="utf-8"))
        assert document["entries"] == {
            "[EMAIL-1]": "ada@example.com",
            "[EMAIL-2]": "bob@example.com",
        }

    def test_a_reply_spanning_several_turns_restores(self, state_home):
        """The reason append exists: one conversation, many prompts."""
        _run(["redact", "-q", "--vault-mode", "append", "mail ada@example.com"])
        _run(["redact", "-q", "--vault-mode", "append", "cc bob@example.com"])
        _, out, _ = _run(["restore", "-q", "sent [EMAIL-1], cc [EMAIL-2]"])
        assert out.strip() == "sent ada@example.com, cc bob@example.com"

    def test_no_placeholder_ever_maps_to_two_values(self, state_home):
        for text in (
            "mail ada@example.com",
            "cc bob@example.com",
            "and eve@example.com",
        ):
            _run(["redact", "-q", "--vault-mode", "append", text])
        document = json.loads(state_home.read_text(encoding="utf-8"))
        values = list(document["entries"].values())
        assert len(values) == len(set(values)) == 3

    def test_append_on_a_missing_vault_is_an_ordinary_first_run(self, state_home):
        status, out, _ = _run(
            ["redact", "-q", "--vault-mode", "append", "mail ada@example.com"]
        )
        assert status == 0
        assert out.strip() == "mail [EMAIL-1]"

    def test_the_index_is_written(self, state_home):
        _run(["redact", "-q", "mail ada@example.com"])
        document = json.loads(state_home.read_text(encoding="utf-8"))
        assert document["format"] == 2
        assert document["index"] == [
            {"label": "[EMAIL-1]", "kind": "EMAIL", "ordinal": 1}
        ]

    def test_the_index_carries_no_value(self, state_home):
        """Labels and categories are already in the text that was sent."""
        _run(["redact", "-q", "mail ada@example.com"])
        document = json.loads(state_home.read_text(encoding="utf-8"))
        assert "ada@example.com" not in json.dumps(document["index"])

    def test_appending_to_an_indexless_vault_is_refused(self, state_home, tmp_path):
        """Guessing would map one placeholder onto two different values."""
        old = tmp_path / "old.json"
        old.write_text(
            json.dumps(
                {
                    "format": 1,
                    "encrypted": False,
                    "entries": {"[EMAIL-1]": "old@example.com"},
                    "grammar_fingerprint": None,
                }
            ),
            encoding="utf-8",
        )
        status, _, err = _run(
            ["redact", "--vault", str(old), "--vault-mode", "append", "a@b.co"]
        )
        assert status == 1
        assert "cannot be appended to" in err
        assert "collide" in err

    def test_an_old_vault_still_restores(self, tmp_path):
        """Refusing to append must not mean refusing to read."""
        old = tmp_path / "old.json"
        old.write_text(
            json.dumps(
                {
                    "format": 1,
                    "encrypted": False,
                    "entries": {"[EMAIL-1]": "old@example.com"},
                    "grammar_fingerprint": None,
                }
            ),
            encoding="utf-8",
        )
        status, out, _ = _run(["restore", "-q", "--vault", str(old), "see [EMAIL-1]"])
        assert status == 0
        assert out.strip() == "see old@example.com"


class TestCleanCommand:
    """Text in, a prompt you can paste, nothing else on standard output."""

    def test_stdout_carries_only_the_redacted_text(self, state_home):
        status, out, _ = _run(["clean", "mail ada@example.com from 192.168.1.10."])
        assert status == 0
        assert out == "mail [EMAIL-1] from [IPV4-1].\n"

    def test_no_value_reaches_standard_output(self, state_home):
        _, out, _ = _run(["clean", "mail ada@example.com"])
        assert "ada@example.com" not in out

    def test_it_reads_a_heredoc_shaped_paste(self, state_home):
        paste = (
            "Mustafa Kemal wrote to ataturk@example.com\nfrom the Republic of Turkey.\n"
        )
        _, out, _ = _run(["clean"], stdin=paste)
        assert "ataturk@example.com" not in out
        assert out.count("\n") == paste.count("\n")

    def test_the_vault_note_is_a_single_line(self, state_home):
        """One line, not the table `redact` prints."""
        _, _, err = _run(["clean", "mail ada@example.com"])
        notes = [line for line in err.splitlines() if line.startswith("vault:")]
        assert len(notes) == 1
        assert "placeholder" not in err  # no summary table

    def test_quiet_leaves_stderr_empty(self, state_home):
        _, _, err = _run(["clean", "-q", "mail ada@example.com"])
        assert err == ""

    def test_it_defaults_to_append(self, state_home):
        _run(["clean", "-q", "mail ada@example.com"])
        _, out, _ = _run(["clean", "-q", "again ada@example.com"])
        assert out.strip() == "again [EMAIL-1]"

    def test_overwrite_can_be_asked_for(self, state_home):
        _run(["clean", "-q", "mail ada@example.com"])
        _run(["clean", "-q", "--vault-mode", "overwrite", "mail bob@example.com"])
        document = json.loads(state_home.read_text(encoding="utf-8"))
        assert document["entries"] == {"[EMAIL-1]": "bob@example.com"}

    def test_the_reply_restores_with_no_arguments(self, state_home):
        """The whole loop, with no path typed anywhere."""
        _run(["clean", "-q", "mail ada@example.com and bob@example.com"])
        _, out, _ = _run(["restore", "-q", "I mailed [EMAIL-1] and [EMAIL-2]."])
        assert out.strip() == "I mailed ada@example.com and bob@example.com."

    def test_a_high_severity_gap_is_still_reported(self, state_home):
        """Quiet is not silent about the text not having been checked."""
        _, _, err = _run(["clean", "the cat sat on the mat"])
        assert "not detected" in err

    def test_prompt_is_an_alias(self, state_home):
        status, out, _ = _run(["prompt", "-q", "mail ada@example.com"])
        assert status == 0
        assert out.strip() == "mail [EMAIL-1]"

    def test_it_can_write_to_a_file(self, state_home, tmp_path):
        target = tmp_path / "safe.txt"
        status, out, _ = _run(
            ["clean", "-q", "-o", str(target), "mail ada@example.com"]
        )
        assert status == 0
        assert out == ""
        assert target.read_text(encoding="utf-8") == "mail [EMAIL-1]"

    def test_both_frontends_agree(self, state_home):
        results = []
        for frontend in ("argparse", "click"):
            out, err = io.StringIO(), io.StringIO()
            status = main(
                ["--frontend", frontend, "clean", "-q", "mail ada@example.com"],
                stdin=io.StringIO(""),
                stdout=out,
                stderr=err,
            )
            results.append((status, out.getvalue(), err.getvalue()))
        assert results[0] == results[1]


class TestEncodeDecodePair:
    """
    ``encode`` and ``decode``: the two halves, named for each other.

    Notes
    -----
    **Developer notes.** ``encode`` used to be an alias of ``redact``, which
    writes a vault to a named path and prints a table of what it removed.
    ``decode`` was an alias of ``restore``. So the two names that read as a
    pair behaved nothing like one: one was verbose and file-oriented, the other
    minimal. A user asking for "encode to clean, decode that AI chat answer"
    was asking for the symmetry the names already implied.

    ``encode`` is now the minimal half — text in, a pasteable prompt out — and
    ``redact`` keeps the explicit, scripted behaviour under its own name.
    """

    def test_encode_puts_only_the_prompt_on_stdout(self, state_home):
        status, out, _ = _run(["encode", "mail ada@example.com from 192.168.1.10."])
        assert status == 0
        assert out == "mail [EMAIL-1] from [IPV4-1].\n"

    def test_decode_puts_the_values_back(self, state_home):
        _run(["encode", "-q", "mail ada@example.com"])
        status, out, _ = _run(["decode", "-q", "I mailed [EMAIL-1]."])
        assert status == 0
        assert out.strip() == "I mailed ada@example.com."

    def test_the_pair_needs_no_paths(self, state_home):
        """The whole workflow, with nothing typed but the text."""
        _, safe, _ = _run(["encode", "-q", "Ada mailed ada@example.com"])
        assert "ada@example.com" not in safe
        _, back, _ = _run(["decode", "-q", "Sent to [EMAIL-1], thanks."])
        assert back.strip() == "Sent to ada@example.com, thanks."

    def test_encode_is_no_longer_the_verbose_command(self, state_home):
        """It must not print redact's summary table."""
        _, _, err = _run(["encode", "mail ada@example.com"])
        assert "placeholder" not in err

    def test_redact_keeps_its_own_behaviour(self, state_home):
        """Moving the alias must not have changed the command it left."""
        _, _, err = _run(["redact", "mail ada@example.com"])
        assert "placeholder" in err  # the summary table is still there

    def test_clean_and_prompt_remain_as_aliases(self, state_home):
        for name in ("clean", "prompt"):
            status, out, _ = _run([name, "-q", "mail ada@example.com"])
            assert status == 0
            assert out.strip() == "mail [EMAIL-1]"

    def test_restore_remains_as_an_alias(self, state_home):
        _run(["encode", "-q", "mail ada@example.com"])
        status, out, _ = _run(["restore", "-q", "sent [EMAIL-1]"])
        assert status == 0
        assert out.strip() == "sent ada@example.com"

    def test_the_two_names_appear_together_in_help(self):
        _, out, _ = _run(["--help"])
        assert "encode" in out and "decode" in out

    def test_each_names_the_other(self):
        """A reader who finds one must be told the other exists."""
        _, encode_help, _ = _run(["encode", "-h"])
        _, decode_help, _ = _run(["decode", "-h"])
        assert "decode" in encode_help
        assert "encode" in decode_help

    def test_both_frontends_agree(self, state_home):
        results = []
        for frontend in ("argparse", "click"):
            out, err = io.StringIO(), io.StringIO()
            status = main(
                ["--frontend", frontend, "encode", "-q", "mail ada@example.com"],
                stdin=io.StringIO(""),
                stdout=out,
                stderr=err,
            )
            results.append((status, out.getvalue(), err.getvalue()))
        assert results[0] == results[1]

    def test_a_multi_turn_conversation_decodes_throughout(self, state_home):
        _run(["encode", "-q", "mail ada@example.com"])
        _run(["encode", "-q", "cc bob@example.com"])
        _, out, _ = _run(["decode", "-q", "Sent [EMAIL-1], copied [EMAIL-2]."])
        assert out.strip() == "Sent ada@example.com, copied bob@example.com."


class TestForget:
    """Deleting the vault, which the default path made necessary."""

    def test_it_does_nothing_without_force(self, state_home):
        _run(["encode", "-q", "mail ada@example.com"])
        status, _, err = _run(["forget"])
        assert status == 0
        assert state_home.is_file()
        assert "would delete" in err
        assert "--force" in err

    def test_force_deletes_the_vault(self, state_home):
        _run(["encode", "-q", "mail ada@example.com"])
        status, _, err = _run(["forget", "--force"])
        assert status == 0
        assert not state_home.exists()
        assert "deleted" in err

    def test_it_reports_categories_but_never_a_value(self, state_home):
        _run(["encode", "-q", "mail ada@example.com from 192.168.1.10."])
        _, _, err = _run(["forget"])
        assert "EMAIL=1" in err and "IPV4=1" in err
        assert "ada@example.com" not in err
        assert "192.168.1.10" not in err

    def test_the_note_does_not_overclaim(self, state_home):
        """Unlinking is not shredding, and the message must not imply it is."""
        _run(["encode", "-q", "mail ada@example.com"])
        _, _, err = _run(["forget", "--force"])
        assert "may survive" in err
        assert "--encrypt" in err

    def test_forgetting_nothing_is_not_an_error(self, state_home):
        status, _, err = _run(["forget", "--force"])
        assert status == 0
        assert "nothing to forget" in err

    def test_decode_after_forget_says_so(self, state_home):
        _run(["encode", "-q", "mail ada@example.com"])
        _run(["forget", "--force"])
        status, _, err = _run(["decode", "sent [EMAIL-1]"])
        assert status == 1
        assert "no vault at" in err

    def test_an_unreadable_vault_can_still_be_deleted(self, state_home):
        """The user who most wants it gone must not be blocked by a parse error."""
        state_home.parent.mkdir(parents=True, exist_ok=True)
        state_home.write_text("{ not json", encoding="utf-8")
        status, _, err = _run(["forget", "--force"])
        assert status == 0
        assert not state_home.exists()
        assert "unreadable" in err

    def test_it_can_target_an_explicit_vault(self, state_home, tmp_path):
        target = tmp_path / "other.json"
        _run(["encode", "-q", "--vault", str(target), "mail ada@example.com"])
        assert target.is_file()
        _run(["forget", "--force", "--vault", str(target)])
        assert not target.exists()

    def test_the_alias_works(self, state_home):
        _run(["encode", "-q", "mail ada@example.com"])
        status, _, _ = _run(["clear-vault", "--force"])
        assert status == 0
        assert not state_home.exists()

    def test_both_frontends_agree(self, state_home):
        results = []
        for frontend in ("argparse", "click"):
            out, err = io.StringIO(), io.StringIO()
            status = main(
                ["--frontend", frontend, "forget"],
                stdin=io.StringIO(""),
                stdout=out,
                stderr=err,
            )
            results.append((status, out.getvalue(), err.getvalue()))
        assert results[0] == results[1]


class TestEncryptedVault:
    """
    ``--encrypt`` on the commands that write a vault.

    Notes
    -----
    **Developer notes.** ``--encrypt`` was on ``redact`` only, and needed the
    ``crypto`` tier. So ``forget``'s advice — "use ``--encrypt`` for values
    that must not be recoverable" — pointed at a flag the reader's command did
    not have, and which would have failed anyway without a compiled dependency
    installed. Both halves of that are now fixed: the flag is on ``encode``,
    and the default cipher is the standard-library one.
    """

    @pytest.fixture()
    def key(self, monkeypatch):
        monkeypatch.setenv("CLEANPROMPT_VAULT_KEY", "a test passphrase")

    def test_encode_accepts_encrypt(self, state_home, key):
        status, out, _ = _run(["encode", "--encrypt", "-q", "mail ada@example.com"])
        assert status == 0
        assert out.strip() == "mail [EMAIL-1]"

    def test_the_value_is_not_in_the_file(self, state_home, key):
        _run(["encode", "--encrypt", "-q", "mail ada@example.com"])
        assert "ada@example.com" not in state_home.read_text(encoding="utf-8")

    def test_the_round_trip_still_works(self, state_home, key):
        _run(["encode", "--encrypt", "-q", "mail ada@example.com"])
        _, out, _ = _run(["decode", "-q", "sent to [EMAIL-1]"])
        assert out.strip() == "sent to ada@example.com"

    def test_the_document_names_its_cipher_and_parameters(self, state_home, key):
        _run(["encode", "--encrypt", "-q", "mail ada@example.com"])
        document = json.loads(state_home.read_text(encoding="utf-8"))
        assert document["cipher"] == "cleanprompt-hmac-v1"
        assert document["kdf"]["name"] in ("scrypt", "pbkdf2_hmac_sha256")
        assert document["kdf"]["salt"]

    def test_it_records_the_construction_not_the_option_word(self, state_home, key):
        """'portable' is a choice; the document must say which one ran."""
        _run(["encode", "--encrypt", "--cipher", "portable", "-q", "a@b.co"])
        document = json.loads(state_home.read_text(encoding="utf-8"))
        assert document["cipher"] != "portable"

    def test_the_wrong_passphrase_is_refused(self, state_home, key, monkeypatch):
        _run(["encode", "--encrypt", "-q", "mail ada@example.com"])
        monkeypatch.setenv("CLEANPROMPT_VAULT_KEY", "the wrong one")
        status, _, err = _run(["decode", "sent [EMAIL-1]"])
        assert status == 1
        assert "passphrase is wrong" in err

    def test_a_missing_passphrase_is_explained(self, state_home, monkeypatch):
        monkeypatch.delenv("CLEANPROMPT_VAULT_KEY", raising=False)
        status, _, err = _run(["encode", "--encrypt", "mail ada@example.com"])
        assert status == 1
        assert "CLEANPROMPT_VAULT_KEY" in err
        assert "--new-key" in err

    def test_an_unencrypted_vault_still_works(self, state_home):
        """The flag is opt-in and must not have changed the default path."""
        _run(["encode", "-q", "mail ada@example.com"])
        document = json.loads(state_home.read_text(encoding="utf-8"))
        assert document["encrypted"] is False
        _, out, _ = _run(["decode", "-q", "sent [EMAIL-1]"])
        assert out.strip() == "sent ada@example.com"

    def test_append_works_on_an_encrypted_vault(self, state_home, key):
        _run(["encode", "--encrypt", "-q", "mail ada@example.com"])
        _, out, _ = _run(["encode", "--encrypt", "-q", "again ada@example.com"])
        assert out.strip() == "again [EMAIL-1]"

    def test_forget_reports_an_encrypted_vault_without_the_key(
        self, state_home, key, monkeypatch
    ):
        """Deleting must not require being able to read it."""
        _run(["encode", "--encrypt", "-q", "mail ada@example.com"])
        monkeypatch.delenv("CLEANPROMPT_VAULT_KEY", raising=False)
        status, _, err = _run(["forget", "--force"])
        assert status == 0
        assert not state_home.exists()

    def test_new_key_needs_nothing_installed(self):
        status, out, _ = _run(["doctor", "--new-key"])
        assert status == 0
        assert len(out.strip().split("-")) == 4

    def test_redact_still_takes_encrypt(self, state_home, key, tmp_path):
        vault = tmp_path / "v.json"
        status, _, _ = _run(
            ["redact", "-q", "--vault", str(vault), "--encrypt", "mail ada@example.com"]
        )
        assert status == 0
        assert "ada@example.com" not in vault.read_text(encoding="utf-8")

    def test_both_frontends_agree(self, state_home, key):
        results = []
        for frontend in ("argparse", "click"):
            out, err = io.StringIO(), io.StringIO()
            status = main(
                ["--frontend", frontend, "encode", "--encrypt", "-q", "a@b.co"],
                stdin=io.StringIO(""),
                stdout=out,
                stderr=err,
            )
            results.append((status, out.getvalue(), err.getvalue()))
        assert results[0] == results[1]

    def test_a_fernet_vault_on_a_machine_without_the_tier_is_explained(
        self, state_home, tmp_path, monkeypatch
    ):
        """The message must name the cipher and the portable alternative."""
        vault = tmp_path / "f.json"
        vault.write_text(
            json.dumps(
                {
                    "format": 2,
                    "encrypted": True,
                    "cipher": "fernet",
                    "kdf": None,
                    "entries": {"[EMAIL-1]": "gAAAAA-not-a-real-token"},
                    "index": [],
                    "grammar_fingerprint": None,
                }
            ),
            encoding="utf-8",
        )
        from .. import _capabilities as caps

        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        status, _, err = _run(["decode", "--vault", str(vault), "see [EMAIL-1]"])
        assert status == 1
        assert "Fernet" in err
        assert "--cipher portable" in err

    def test_an_unknown_cipher_is_refused(self, tmp_path):
        vault = tmp_path / "x.json"
        vault.write_text(
            json.dumps(
                {
                    "format": 2,
                    "encrypted": True,
                    "cipher": "rot13",
                    "entries": {"[EMAIL-1]": "zzz"},
                    "index": [],
                    "grammar_fingerprint": None,
                }
            ),
            encoding="utf-8",
        )
        status, _, err = _run(["decode", "--vault", str(vault), "see [EMAIL-1]"])
        assert status == 1
        assert "unknown cipher" in err


class TestStyleTravelsWithTheVault:
    """
    ``decode`` must stay a no-argument command in either style.

    Notes
    -----
    **Developer notes.** Found by running the round's own evidence capture
    rather than by a test, which is the honest order of events. ``encode
    --style surrogate`` writes a vault whose grammar fingerprint differs, and a
    bare ``decode`` assembled the default grammar from nothing, so it was
    refused: "this vault was issued under placeholder grammar 1a85… but
    restoration was asked for grammar 0e7c…".

    The refusal was correct — a vault written in one style must not be read as
    the other — and the situation was still wrong. The vault records its own
    ``tag_style``; requiring the user to restate on the way back something the
    file already holds is making them carry the program's state. The whole
    point of the pair is that neither half needs an argument.
    """

    def test_a_surrogate_vault_decodes_with_no_arguments(self, state_home):
        status, out, _ = _run(
            ["encode", "--style", "surrogate", "-q", "mail ada@example.com"]
        )
        assert status == 0
        stand_in = out.strip().split()[-1]
        assert "@example.invalid" in stand_in

        status, back, _ = _run(["decode", "-q", "I wrote to {0}.".format(stand_in)])
        assert status == 0
        assert back.strip() == "I wrote to ada@example.com."

    def test_a_placeholder_vault_still_decodes_with_no_arguments(self, state_home):
        _run(["encode", "-q", "mail ada@example.com"])
        status, back, _ = _run(["decode", "-q", "I mailed [EMAIL-1]."])
        assert status == 0
        assert back.strip() == "I mailed ada@example.com."

    def test_leniency_still_applies_to_a_surrogate_vault(self, state_home):
        """Its credential placeholders can be rewritten by the model too."""
        _run(["encode", "--style", "surrogate", "-q", "card 4242 4242 4242 4242"])
        status, back, _ = _run(["decode", "-q", "card [CREDIT_CARD_1] on file"])
        assert status == 0
        assert "4242 4242 4242 4242" in back

    def test_a_vault_with_no_recorded_grammar_falls_back(self, state_home, tmp_path):
        """An older document must not crash the reader."""
        vault = tmp_path / "old.json"
        vault.write_text(
            json.dumps(
                {
                    "format": 2,
                    "encrypted": False,
                    "entries": {"[EMAIL-1]": "ada@example.com"},
                    "index": [{"label": "[EMAIL-1]", "kind": "EMAIL", "ordinal": 1}],
                    "grammar_fingerprint": None,
                }
            ),
            encoding="utf-8",
        )
        status, out, _ = _run(["decode", "-q", "--vault", str(vault), "see [EMAIL-1]"])
        assert status == 0
        assert out.strip() == "see ada@example.com"

    def test_a_malformed_grammar_does_not_raise(self, state_home, tmp_path):
        vault = tmp_path / "bad.json"
        vault.write_text(
            json.dumps(
                {
                    "format": 2,
                    "encrypted": False,
                    "entries": {"[EMAIL-1]": "ada@example.com"},
                    "index": [],
                    "tag_style": {"nonsense": True},
                    "grammar_fingerprint": None,
                }
            ),
            encoding="utf-8",
        )
        status, out, _ = _run(["decode", "-q", "--vault", str(vault), "see [EMAIL-1]"])
        assert status == 0
        assert out.strip() == "see ada@example.com"

    def test_both_frontends_agree(self, state_home):
        results = []
        for frontend in ("argparse", "click"):
            out, err = io.StringIO(), io.StringIO()
            status = main(
                [
                    "--frontend",
                    frontend,
                    "encode",
                    "--style",
                    "surrogate",
                    "-q",
                    "a@b.co",
                ],
                stdin=io.StringIO(""),
                stdout=out,
                stderr=err,
            )
            results.append((status, out.getvalue(), err.getvalue()))
        assert results[0] == results[1]


class TestPacksCommand:
    """``packs``: list, show, check."""

    def test_list_names_packs_and_formats(self):
        status, out, _ = _run_both(["packs", "-f", "json"])
        data = json.loads(out)
        assert status == 0
        assert "patient" in data["packs"] and "notebook" in data["formats"]
        assert data["formats"]["csv"]["round_trip"] is True

    def test_show_one(self):
        status, out, _ = _run(["packs", "--show", "patient", "-f", "json"])
        assert status == 0
        assert json.loads(out)["name"] == "patient"

    def test_show_unknown_is_an_error(self):
        status, _, err = _run(["packs", "--show", "patinet"])
        assert status != 0
        assert "patinet" in err

    def test_check_reports_a_bad_custom_file(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"name": "X"}', encoding="utf-8")
        status, out, _ = _run(
            ["packs", "--check", "--pack-file", str(bad), "-f", "json"]
        )
        data = json.loads(out)
        assert status == 1
        assert data["status"] == "problems"

    def test_check_is_clean_for_the_builtins(self):
        status, out, _ = _run(["packs", "--check", "-f", "json"])
        assert status == 0, out


class TestBatchCommand:
    """``batch``: a folder or zip in, a safe copy out, and back."""

    def _tree(self, root):
        root.mkdir()
        (root / "a.csv").write_text(
            "name,email\nAnn Lee,ann@example.com\n", encoding="utf-8"
        )
        (root / "b.json").write_text('{"mrn": 12345678}', encoding="utf-8")
        return root

    def test_encode_then_decode_a_folder(self, tmp_path):
        source = self._tree(tmp_path / "src")
        vault = tmp_path / "vault.json"
        status, out, _ = _run_both(
            [
                "batch",
                str(source),
                "--out",
                str(tmp_path / "out"),
                "--pack",
                "all",
                "--vault",
                str(vault),
                "-f",
                "json",
            ]
        )
        assert status == 0
        assert json.loads(out)["encoded"] == 2
        safe = (tmp_path / "out" / "a.csv").read_text(encoding="utf-8")
        assert "ann@example.com" not in safe and "Ann Lee" not in safe
        status, _, _ = _run(
            [
                "batch",
                str(tmp_path / "out"),
                "--decode",
                "--out",
                str(tmp_path / "back"),
                "--vault",
                str(vault),
            ]
        )
        assert status == 0
        for name in ("a.csv", "b.json"):
            assert (tmp_path / "back" / name).read_bytes() == (
                source / name
            ).read_bytes()

    def test_refused_files_make_the_run_fail(self, tmp_path):
        source = self._tree(tmp_path / "src")
        (source / "broken.json").write_text("{", encoding="utf-8")
        status, out, _ = _run(
            [
                "batch",
                str(source),
                "--out",
                str(tmp_path / "out"),
                "--vault",
                str(tmp_path / "v.json"),
                "-f",
                "json",
            ]
        )
        assert status == 1
        assert json.loads(out)["refused"] == 1
        assert not (tmp_path / "out" / "broken.json").exists()

    def test_a_zip(self, tmp_path):
        import zipfile

        archive = tmp_path / "in.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("a.csv", "email\nann@example.com\n")
        status, _, _ = _run(
            [
                "batch",
                str(archive),
                "--out",
                str(tmp_path / "out.zip"),
                "--vault",
                str(tmp_path / "v.json"),
            ]
        )
        assert status == 0
        with zipfile.ZipFile(tmp_path / "out.zip") as handle:
            assert b"ann@example.com" not in handle.read("a.csv")

    def test_the_vault_restores_through_decode(self, tmp_path):
        source = self._tree(tmp_path / "src")
        vault = tmp_path / "vault.json"
        _run(
            [
                "batch",
                str(source),
                "--out",
                str(tmp_path / "out"),
                "--vault",
                str(vault),
            ]
        )
        safe = (tmp_path / "out" / "a.csv").read_text(encoding="utf-8")
        status, out, _ = _run(["decode", "--vault", str(vault), "-q", safe])
        assert status == 0
        assert "ann@example.com" in out

    def test_a_bad_selection_is_one_error_listing_every_problem(self, tmp_path):
        source = self._tree(tmp_path / "src")
        status, _, err = _run(
            [
                "batch",
                str(source),
                "--out",
                str(tmp_path / "out"),
                "--pack",
                "patinet",
                "--file-format",
                "docz",
            ]
        )
        assert status != 0
        assert "patinet" in err and "docz" in err

    def test_neither_folder_nor_zip(self, tmp_path):
        status, _, err = _run(
            ["batch", str(tmp_path / "missing"), "--out", str(tmp_path / "o")]
        )
        assert status != 0
        assert "neither a folder nor a .zip" in err


class TestBatchDryRun:
    """``batch --dry-run``: what a folder holds, without writing or remembering."""

    def _tree(self, root):
        root.mkdir()
        (root / "a.csv").write_text(
            "name,email\nAnn Lee,ann@example.com\n", encoding="utf-8"
        )
        (root / "b.txt").write_text("Write to Ann Lee.\n", encoding="utf-8")
        (root / "c.bin").write_bytes(b"\xff\xfe")
        return root

    def test_reports_and_writes_nothing(self, tmp_path):
        source = self._tree(tmp_path / "src")
        vault = tmp_path / "vault.json"
        before = sorted(p.name for p in tmp_path.rglob("*"))
        status, out, _ = _run_both(
            ["batch", str(source), "--dry-run", "--vault", str(vault), "-f", "json"]
        )
        assert sorted(p.name for p in tmp_path.rglob("*")) == before
        data = json.loads(out)
        assert data["kinds"] == {"EMAIL": 1, "PERSON": 2}
        assert data["would_encoded"] == 2
        assert "Ann Lee" not in out and "ann@example.com" not in out
        assert status == (1 if data.get("refused") else 0)

    @pytest.mark.parametrize("extra", [["--out", "o"], ["--decode"]])
    def test_refuses_writing_options(self, tmp_path, extra):
        source = self._tree(tmp_path / "src")
        status, _, err = _run_both(["batch", str(source), "--dry-run", *extra])
        assert status != 0 and "--dry-run writes nothing" in err

    def test_refuses_a_zip(self, tmp_path):
        archive = tmp_path / "x.zip"
        import zipfile

        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("a.txt", "x")
        status, _, err = _run_both(["batch", str(archive), "--dry-run"])
        assert status != 0 and "unpack it first" in err

    def test_out_is_required_without_it(self, tmp_path):
        source = self._tree(tmp_path / "src")
        status, _, err = _run_both(["batch", str(source)])
        assert status != 0 and "--out is required" in err


class TestAskCommand:
    """``ask``: one guarded prompt through any command-line model."""

    def _via(self, script):
        import shlex

        return " ".join(shlex.quote(part) for part in (sys.executable, "-c", script))

    def test_the_model_sees_placeholders_and_the_user_values(self, tmp_path):
        seen = tmp_path / "seen.txt"
        script = f"import sys; d = sys.stdin.read(); open({str(seen)!r}, 'w').write(d); sys.stdout.write('Re: ' + d)"
        status, out, _ = _run_both(
            [
                "ask",
                "--via",
                self._via(script),
                "--pack",
                "patient",
                "MRN: 00412345 ann@example.com",
            ]
        )
        assert status == 0
        assert out == "Re: MRN: 00412345 ann@example.com"
        assert (
            "00412345" not in seen.read_text(encoding="utf-8")
            and "ann@example.com" not in seen.read_text(encoding="utf-8")
        )

    def test_show_sent_prints_what_left(self):
        status, _, err = _run(
            [
                "ask",
                "--via",
                self._via("import sys; sys.stdin.read()"),
                "--show-sent",
                "mail ann@example.com",
            ]
        )
        assert status == 0
        assert "mail [EMAIL-1]" in err and "ann@example.com" not in err

    def test_the_exit_status_is_the_models(self):
        status, _, _ = _run(
            ["ask", "--via", self._via("import sys; sys.exit(5)"), "hi"]
        )
        assert status == 5

    def test_a_missing_command_is_an_error(self):
        status, _, err = _run(["ask", "--via", "cleanprompt-no-such-command", "hi"])
        assert status == 1 and "could not start" in err


class TestSkillCommand:
    def test_prints_the_agent_skill(self):
        status, out, _ = _run_both(["skill"])
        assert status == 0
        assert out.startswith("---\nname: cleanprompt-guard\n")
        assert "LeakError" in out

    def test_installs_and_refuses_to_overwrite(self, tmp_path):
        status, _, _ = _run(["skill", "--write", str(tmp_path)])
        target = tmp_path / "cleanprompt-guard" / "SKILL.md"
        assert status == 0 and target.is_file()
        status, _, err = _run(["skill", "--write", str(tmp_path)])
        assert status == 1 and "--force" in err
        assert _run(["skill", "--write", str(tmp_path), "--force"])[0] == 0

    def test_every_command_the_skill_names_exists(self):
        import re

        from .._cli import COMMANDS

        out = _run(["skill"])[1]
        named = set(re.findall(r"python -m scikitplot\.cleanprompt (\w+)", out))
        known = {c.name for c in COMMANDS} | {a for c in COMMANDS for a in c.aliases}
        assert named and named <= known


class TestPlanCommand:
    def test_write_check_and_use(self, tmp_path):
        plan = tmp_path / "team.json"
        assert _run_both(["plan", "--pack", "patient", "--write", str(plan)])[0] == 0
        status, out, _ = _run(["plan", "--check", str(plan), "-f", "json"])
        assert status == 0 and json.loads(out)["status"] == "ok"
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.json").write_text('{"mrn": "00412345"}', encoding="utf-8")
        status, _, _ = _run(
            [
                "batch",
                str(source),
                "--out",
                str(tmp_path / "out"),
                "--plan",
                str(plan),
                "--vault",
                str(tmp_path / "v.json"),
            ]
        )
        assert status == 0
        assert (tmp_path / "out" / "a.json").read_text(encoding="utf-8") == '{"mrn": "[MRN-1]"}'

    def test_a_stale_plan_fails_the_check(self, tmp_path):
        plan = tmp_path / "team.json"
        _run(["plan", "--pack", "patient", "--write", str(plan)])
        document = json.loads(plan.read_text(encoding="utf-8"))
        document["fingerprint"] = "0" * 64
        plan.write_text(json.dumps(document), encoding="utf-8")
        status, out, _ = _run(["plan", "--check", str(plan), "-f", "json"])
        assert status == 1 and json.loads(out)["status"] == "stale"

    def test_a_plan_cannot_be_mixed_with_choices(self, tmp_path):
        plan = tmp_path / "team.json"
        _run(["plan", "--write", str(plan)])
        status, _, err = _run(
            ["ask", "--via", "cat", "--plan", str(plan), "--pack", "all", "hi"]
        )
        assert status == 1 and "--pack" in err


class TestVaultAcrossProcesses:
    """CP-068 and CP-069: runs sharing one vault take turns and never lose it."""

    def test_parallel_encodes_give_every_value_its_own_label(self, tmp_path):
        import concurrent.futures
        import subprocess

        vault = str(tmp_path / "vault.json")
        repo = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        def encode(index):
            done = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scikitplot.cleanprompt",
                    "encode",
                    "-q",
                    f"mail user{index}@example.com",
                    "--vault",
                    vault,
                ],
                capture_output=True,
                text=True,
                cwd=repo,
                timeout=120,
            )
            return index, done.returncode, done.stdout.strip(), done.stderr

        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            runs = list(pool.map(encode, range(6)))
        assert [run[1] for run in runs] == [0] * 6, [run[3] for run in runs]
        labels = [run[2].split()[-1] for run in runs]
        assert len(set(labels)) == 6
        for index, _, safe, _ in runs:
            assert _run(["decode", safe, "--vault", vault])[1].strip() == (
                f"mail user{index}@example.com"
            )

    def _held(self, tmp_path, monkeypatch):
        import functools

        from .. import _cli
        from .._files import locked
        from .test__files import _holder

        vault = tmp_path / "vault.json"
        _run(["encode", "-q", "mail ada@example.com", "--vault", str(vault)])
        monkeypatch.setattr(_cli, "locked", functools.partial(locked, timeout=0.3))
        return vault, _holder(vault, 30)

    def test_encode_waits_for_the_lock_and_says_why(self, tmp_path, monkeypatch):
        from .test__files import _stop as _stop_holder

        vault, child = self._held(tmp_path, monkeypatch)
        try:
            status, _, err = _run(
                ["encode", "mail bob@example.com", "--vault", str(vault)]
            )
        finally:
            _stop_holder(child)
        assert status != 0
        assert "waiting for another cleanprompt run" in err
        assert "held the vault lock" in err

    def test_forget_takes_the_same_lock(self, tmp_path, monkeypatch):
        from .test__files import _stop as _stop_holder

        vault, child = self._held(tmp_path, monkeypatch)
        try:
            status, _, _ = _run(["forget", "--force", "--vault", str(vault)])
        finally:
            _stop_holder(child)
        assert status != 0 and vault.exists()
        assert _run(["forget", "--force", "--vault", str(vault)])[0] == 0
        assert not vault.exists()


class TestFernetVaultWithoutTheTier:
    """The refusal says which of "missing" and "refused" it is (CP-092)."""

    @staticmethod
    def _refusal(monkeypatch, version):
        from .. import CleanPromptError, _capabilities
        from .._cli import _decrypt_entries

        monkeypatch.setattr(_capabilities, "_installed_version", lambda _n: version)
        with pytest.raises(CleanPromptError) as caught:
            _decrypt_entries("vault.json", {"cipher": "fernet"}, {})
        return str(caught.value)

    def test_not_installed(self, monkeypatch):
        message = self._refusal(monkeypatch, None)
        assert "cryptography is not installed" in message
        assert 'pip install "cryptography>=41"' in message
        assert "--cipher portable" in message

    def test_installed_but_too_old(self, monkeypatch):
        message = self._refusal(monkeypatch, "40.0.2")
        assert "installed 40.0.2 is outside cryptography>=41" in message
        assert 'pip install "cryptography>=41"' in message

    def test_a_vault_without_the_field_is_fernet(self, monkeypatch):
        from .. import CleanPromptError, _capabilities
        from .._cli import _decrypt_entries

        monkeypatch.setattr(_capabilities, "_installed_version", lambda _n: None)
        with pytest.raises(CleanPromptError, match="encrypted with Fernet"):
            _decrypt_entries("vault.json", {}, {})
