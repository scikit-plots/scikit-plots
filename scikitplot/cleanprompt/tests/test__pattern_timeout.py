r"""
Tests for :mod:`scikitplot.cleanprompt._pattern_timeout`.

Notes
-----
**Developer notes.** Two groups. The first needs nothing optional: the value
checks, the order a limit is chosen in (argument or plan, then
``CLEANPROMPT_PATTERN_TIMEOUT``, then none), the per-document deadline, the
plan file, the command line, and that built-in packs are never limited. The
second needs the ``timeout`` tier (the ``regex`` package) and skips with the
probe's own account when it is absent (``_tiers``): flag mapping, the same
spans as :mod:`re`, the limit itself, one budget per document, and a child
process that a runaway pattern would otherwise hang.
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from .. import _pattern_timeout as pt
from .._catalog import builtin_catalog
from .._exceptions import CleanPromptError, PatternError, PatternTimeoutError
from .._packs import PackPatternDetector, pack_detectors, pack_from_document
from .._plan import CleanPlan, FluentCleanPrompt, plan_from_dict, save_plan
from .._policy import DEFAULT_POLICY
from ._regex_fixtures import regex_fixture
from ._tiers import available, skip_reason

#: Exponential on a near-match, in ``re`` and in ``regex`` alike.
_SLOW = regex_fixture(r"^(?:a|a)*$")
_NEAR_MATCH = "a" * 40 + "!"

needs_regex = pytest.mark.skipif(
    not available("timeout"), reason=skip_reason("timeout")
)
without_regex = pytest.mark.skipif(
    available("timeout"), reason="the 'timeout' tier is installed"
)


def _pack(pattern=_SLOW, kind="SLOWX", yes=("aaa",), no=("b",)):
    item = {
        "kind": kind,
        "pattern": pattern,
        "intent": "A test pattern.",
        "examples_yes": list(yes),
        "examples_no": list(no),
        "risk": "accepted",
        "risk_reason": "a test of the time limit",
    }
    return {"name": "slow", "version": 1, "summary": "Test.", "patterns": [item]}


def _write(folder, document, name="pack.json"):
    path = folder / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def _cleaner(path, seconds, **kwargs):
    from .._runtime import Cleaner  # ruff: ignore[import-outside-top-level]

    builder = FluentCleanPrompt().custom(path).packs("slow")
    if seconds is not None:
        builder = builder.pattern_timeout(seconds)
    return Cleaner(builder.build(), **kwargs)


def _run(argv):
    from .._cli import main  # ruff: ignore[import-outside-top-level]

    out, err = io.StringIO(), io.StringIO()
    code = main(
        argv, stdin=io.StringIO(""), stdout=out, stderr=err, frontend="argparse"
    )
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    """Every test starts without the environment variable."""
    monkeypatch.delenv(pt.PATTERN_TIMEOUT_ENV, raising=False)


# ---------------------------------------------------------------------------
# values and resolution
# ---------------------------------------------------------------------------


class TestSeconds:
    @pytest.mark.parametrize("good", [1, 0.25, 30.0])
    def test_positive_numbers(self, good):
        assert pt.check_seconds(good) == float(good)

    @pytest.mark.parametrize(
        "bad", [0, -1, float("nan"), float("inf"), True, "1", None]
    )
    def test_everything_else_is_refused(self, bad):
        with pytest.raises(ValueError, match="positive number of seconds"):
            pt.check_seconds(bad)

    def test_argument_wins_over_environment(self, monkeypatch):
        monkeypatch.setenv(pt.PATTERN_TIMEOUT_ENV, "9")
        assert pt.resolve_pattern_timeout(2) == 2.0

    def test_environment_then_none(self, monkeypatch):
        assert pt.resolve_pattern_timeout() is None
        monkeypatch.setenv(pt.PATTERN_TIMEOUT_ENV, " 1.5 ")
        assert pt.resolve_pattern_timeout() == 1.5
        monkeypatch.setenv(pt.PATTERN_TIMEOUT_ENV, "")
        assert pt.resolve_pattern_timeout() is None

    @pytest.mark.parametrize("bad", ["soon", "0", "-2", "nan"])
    def test_a_bad_environment_value_names_the_variable(self, monkeypatch, bad):
        monkeypatch.setenv(pt.PATTERN_TIMEOUT_ENV, bad)
        with pytest.raises(ValueError, match=pt.PATTERN_TIMEOUT_ENV):
            pt.resolve_pattern_timeout()


class TestDeadline:
    def test_none_sets_no_deadline(self):
        with pt.pattern_deadline(None):
            assert pt._DEADLINE.get() is None

    def test_outermost_block_wins_and_is_reset(self):
        with pt.pattern_deadline(10):
            outer = pt._DEADLINE.get()
            with pt.pattern_deadline(0.001):
                assert pt._DEADLINE.get() == outer
            assert pt.active_deadline(99) == outer
        assert pt._DEADLINE.get() is None

    def test_each_thread_has_its_own(self):
        seen = []
        with pt.pattern_deadline(10):
            worker = threading.Thread(target=lambda: seen.append(pt._DEADLINE.get()))
            worker.start()
            worker.join()
        assert seen == [None]

    def test_outside_a_document_a_scan_gets_a_fresh_budget(self):
        before = time.monotonic()
        assert before + 5 <= pt.active_deadline(5) <= time.monotonic() + 5


# ---------------------------------------------------------------------------
# plan, plan file, builder, command line (no optional package needed)
# ---------------------------------------------------------------------------


class TestPlan:
    def test_unset_keeps_the_fingerprint_of_older_plans(self):
        assert "pattern_timeout" not in CleanPlan().as_dict()

    def test_set_joins_the_plan(self):
        plan = FluentCleanPrompt().pattern_timeout(2).build()
        assert plan.as_dict()["pattern_timeout"] == 2.0
        assert plan.fingerprint() != CleanPlan().fingerprint()

    def test_builder_refuses_bad_values_and_a_second_choice(self):
        with pytest.raises(CleanPromptError, match="positive number of seconds"):
            FluentCleanPrompt().pattern_timeout(0)
        with pytest.raises(CleanPromptError):
            FluentCleanPrompt().pattern_timeout(1).pattern_timeout(2)
        replaced = (
            FluentCleanPrompt()
            .pattern_timeout(1)
            .pattern_timeout(2, conflict="replace")
        )
        assert replaced.build().pattern_timeout == 2.0

    def test_plan_file(self, tmp_path):
        assert plan_from_dict({"pattern_timeout": 1.5}).pattern_timeout == 1.5
        assert plan_from_dict({"pattern_timeout": None}).pattern_timeout is None
        for bad in ("1", 0, True):
            with pytest.raises(CleanPromptError, match="pattern_timeout"):
                plan_from_dict({"pattern_timeout": bad})
        path = tmp_path / "plan.json"
        save_plan(FluentCleanPrompt().pattern_timeout(3).build(), path)
        assert json.loads(path.read_text(encoding="utf-8"))["pattern_timeout"] == 3.0

    def test_a_bad_environment_value_is_a_plan_problem(self, monkeypatch):
        monkeypatch.setenv(pt.PATTERN_TIMEOUT_ENV, "soon")
        assert any(p.startswith("pattern_timeout:") for p in CleanPlan().validate())

    def test_without_custom_packs_the_limit_needs_nothing(self):
        from .._runtime import Cleaner  # ruff: ignore[import-outside-top-level]

        plan = FluentCleanPrompt().pattern_timeout(1).build()
        assert plan.validate() == []
        assert (
            Cleaner(plan).encode_text("mail ann@example.com").text == "mail [EMAIL-1]"
        )

    def test_built_in_packs_are_never_limited(self):
        packs = builtin_catalog().packs.values()
        detectors = pack_detectors(packs, timeout=1.0)
        assert detectors
        assert all(d.timeout is None and d._bounded is None for d in detectors)


class TestCommandLine:
    def test_plan_fixes_the_limit_even_zero(self, tmp_path):
        plan = tmp_path / "plan.json"
        save_plan(FluentCleanPrompt().build(), plan)
        (tmp_path / "src").mkdir()
        argv = ["batch", str(tmp_path / "src"), "--out", str(tmp_path / "out")]
        code, _, err = _run([*argv, "--plan", str(plan), "--pattern-timeout", "0"])
        assert code == 1
        assert "remove --pattern-timeout" in err

    def test_a_non_number_is_a_usage_error(self, tmp_path):
        code, _, _ = _run(["packs", "--pattern-timeout", "soon"])
        assert code == 2

    @without_regex
    def test_custom_packs_without_the_tier_name_the_install(self, tmp_path):
        path = _write(tmp_path, _pack())
        code, _, err = _run(["packs", "--pack-file", path, "--pattern-timeout", "1"])
        assert code == 1
        assert 'pip install "regex>=2020.1"' in err

    @without_regex
    def test_custom_packs_without_the_tier_are_a_plan_problem(self, tmp_path):
        path = _write(tmp_path, _pack())
        problems = FluentCleanPrompt().custom(path).pattern_timeout(1).validate()
        assert len(problems) == 1
        assert problems[0].startswith("pattern_timeout: A time limit on custom pack")
        assert "ABSENT" in problems[0]


# ---------------------------------------------------------------------------
# with the timeout tier
# ---------------------------------------------------------------------------


class _ExpiredError(Exception):
    """Stands in for the PatternTimeoutError a detector builds."""


def _bounded(pattern, seconds):
    import regex  # ruff: ignore[import-outside-top-level]

    return pt.BoundedPattern(
        regex.compile(pattern), time.monotonic() + seconds, _ExpiredError
    )


@needs_regex
class TestBoundedEngine:
    @pytest.mark.parametrize("call", ["search", "match", "finditer"])
    def test_no_call_is_made_once_the_budget_is_spent(self, call):
        # regex reads a negative timeout as no limit at all, so the time left
        # must be checked before every call, not passed through.
        scan = getattr(_bounded("a", -0.001), call)
        with pytest.raises(_ExpiredError):
            list(scan("a")) if call == "finditer" else scan("a")

    @pytest.mark.parametrize("call", ["search", "match"])
    def test_a_timeout_drops_the_exception_chain(self, call):
        bounded = _bounded(_SLOW, 0.1)
        with pytest.raises(_ExpiredError) as caught:
            getattr(bounded, call)(_NEAR_MATCH)
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__

    def test_time_between_matches_counts_against_the_budget(self):
        # regex times only its own matching, so a slow consumer (validators,
        # policy checks) would never run out without the check per match.
        found = []
        with pytest.raises(_ExpiredError):
            for match in _bounded("a", 0.3).finditer("a" * 5):
                found.append(match)
                time.sleep(0.2)
        assert 1 <= len(found) < 5

    def test_flags_are_mapped_by_name(self):
        import regex  # ruff: ignore[import-outside-top-level]

        spec = pack_from_document(
            _pack(r"[a-z]\w*", kind="WORD", yes=["abc"], no=["1"]), source="t.json"
        )
        ascii_spec = dataclasses.replace(spec.patterns[0], flags=re.ASCII)
        compiled = pt.bounded_compile(ascii_spec, "t")
        assert compiled.flags & regex.ASCII
        assert not compiled.flags & regex.V1  # re.ASCII's bit is regex.V1
        assert compiled.findall("café") == ["caf"]

    def test_an_unmapped_flag_is_refused(self):
        class Spec:
            kind, pattern, flags = "DEMO", "a", re.DEBUG

        with pytest.raises(
            PatternError, match=r"pattern DEMO of pack 'p'.*no counterpart"
        ):
            pt.bounded_compile(Spec(), "p")

    def test_a_pattern_the_engine_rejects_is_refused(self, monkeypatch):
        import regex  # ruff: ignore[import-outside-top-level]

        def reject(*args, **kwargs):
            raise regex.error("unsupported here")

        monkeypatch.setattr(regex, "compile", reject)

        class Spec:
            kind, pattern, flags = "DEMO", "a", 0

        with pytest.raises(PatternError, match="time-limited engine rejects it"):
            pt.bounded_compile(Spec(), "p")

    def test_the_rejection_stops_the_plan(self, tmp_path, monkeypatch):
        import regex  # ruff: ignore[import-outside-top-level]

        def reject(*args, **kwargs):
            raise regex.error("unsupported here")

        monkeypatch.setattr(regex, "compile", reject)
        path = _write(tmp_path, _pack())
        problems = FluentCleanPrompt().custom(path).pattern_timeout(1).validate()
        assert problems
        assert "pattern SLOWX of pack 'slow'" in problems[0]

    @pytest.mark.parametrize(
        ("pattern", "yes", "text"),
        [
            (
                r"\bINV-\d{4,8}\b",
                ["INV-12345"],
                "INV-12345 and INV-1 and xINV-1234 INV-99999999",
            ),
            (
                r"(?i)\bMRN\s*:?\s*(?P<value>\d{4,12})\b",
                ["MRN: 00412345"],
                "MRN: 00412345, mrn 1234, MRN:12 and MRN 123456789012345",
            ),
        ],
    )
    def test_same_spans_as_re(self, tmp_path, pattern, yes, text):
        pack = pack_from_document(_pack(pattern, kind="DEMO", yes=yes), source="t.json")
        spec = pack.patterns[0]
        plain = PackPatternDetector(spec, "slow")
        bounded = PackPatternDetector(spec, "slow", timeout=5)
        assert bounded._bounded is not None
        found = [(s.start, s.end, s.text) for s in bounded.detect(text, DEFAULT_POLICY)]
        assert found == [
            (s.start, s.end, s.text) for s in plain.detect(text, DEFAULT_POLICY)
        ]
        assert found

    def test_runaway_pattern_fails_the_document_without_its_text(self, tmp_path):
        cleaner = _cleaner(_write(tmp_path, _pack()), 0.2)
        assert cleaner.encode_text("aaa").text == "[SLOWX-1]"
        started = time.monotonic()
        with pytest.raises(PatternTimeoutError) as caught:
            cleaner.encode_text(_NEAR_MATCH)
        assert time.monotonic() - started < 5
        error = caught.value
        assert (error.pack, error.kind, error.seconds, error.detector) == (
            "slow",
            "SLOWX",
            0.2,
            "pack:slow:SLOWX",
        )
        assert "aaaa" not in str(error)
        assert error.__cause__ is None
        assert error.__context__ is None or error.__suppress_context__

    def test_one_budget_per_document(self, tmp_path):
        pack = pack_from_document(_pack(), source="t.json")
        detector = PackPatternDetector(pack.patterns[0], "slow", timeout=0.25)
        moderate = "a" * 19 + "!"  # a few hundredths of a second each
        cost = time.monotonic()
        list(detector.detect(moderate, DEFAULT_POLICY))
        cost = time.monotonic() - cost
        repeats = int(0.25 / max(cost, 1e-4)) + 3
        for _ in range(repeats):  # each scan alone has its own budget
            list(detector.detect(moderate, DEFAULT_POLICY))
        with pytest.raises(PatternTimeoutError), pt.pattern_deadline(0.25):
            for _ in range(repeats):  # one document: the scans share it
                list(detector.detect(moderate, DEFAULT_POLICY))

    def test_the_cleaner_sets_the_document_deadline(self, tmp_path, monkeypatch):
        from .. import _packs  # ruff: ignore[import-outside-top-level]

        seen = []
        real = _packs.active_deadline
        monkeypatch.setattr(
            _packs,
            "active_deadline",
            lambda s: seen.append(pt._DEADLINE.get()) or real(s),
        )
        _cleaner(_write(tmp_path, _pack()), 3).encode_text("aaa")
        assert seen
        assert all(d is not None for d in seen)
        assert pt._DEADLINE.get() is None

    def test_batch_refuses_that_file_and_cleans_the_others(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        (source / "ok.txt").write_text("mail ann@example.com\n", encoding="utf-8")
        (source / "slow.txt").write_text(_NEAR_MATCH + "\n", encoding="utf-8")
        code, out, _ = _run(
            [
                "batch",
                str(source),
                "--out",
                str(tmp_path / "out"),
                "--vault",
                str(tmp_path / "vault.json"),
                "--pack-file",
                _write(tmp_path, _pack()),
                "--pack",
                "slow",
                "--pattern-timeout",
                "0.2",
                "--format",
                "json",
            ]
        )
        report = json.loads(out)
        status = {item["relative"]: item["status"] for item in report["items"]}
        assert code == 1
        assert status == {"ok.txt": "encoded", "slow.txt": "refused"}
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["ok.txt"]

    def test_a_runaway_child_stops_within_the_bound(self, tmp_path):
        path = _write(tmp_path, _pack())
        script = textwrap.dedent(f"""
            import sys, time
            from scikitplot.cleanprompt._exceptions import PatternTimeoutError
            from scikitplot.cleanprompt._plan import FluentCleanPrompt
            from scikitplot.cleanprompt._runtime import Cleaner
            plan = FluentCleanPrompt().custom({path!r}).packs("slow").pattern_timeout(0.5).build()
            started = time.monotonic()
            try:
                Cleaner(plan).encode_text("a" * 60 + "!")
            except PatternTimeoutError as exc:
                print("stopped", round(time.monotonic() - started, 2), exc.kind)
                sys.exit(0)
            sys.exit(3)
            """)
        result = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,  # without the bound this child would run for years
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        _, seconds, kind = result.stdout.split()
        assert kind == "SLOWX"
        assert float(seconds) < 5
