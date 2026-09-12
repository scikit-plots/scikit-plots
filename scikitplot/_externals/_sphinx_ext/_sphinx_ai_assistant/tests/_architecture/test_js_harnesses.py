"""
Run every Node harness as part of the Python test suite.

Why this file exists
--------------------
The ``tests/*.mjs`` harnesses hold the assertions that guard this extension's
security properties: the nonce fence, egress redaction, the injection
false-positive corpus, credential non-echo, the export registry, the keyboard
accelerators. Until this wrapper existed they ran **only when a human typed the
command** -- nothing in ``meson.build`` or ``conftest.py`` referenced them, and
there is no workflow file in this subpackage.

A test nobody runs is documentation with a misleading file extension. This
makes them a gate.

Failure modes handled deliberately
----------------------------------
**Missing ``node`` skips, it does not pass.** A silent pass would make an
environment without Node indistinguishable from one where every harness
succeeded, which is the exact failure this file exists to prevent. It skips
with a reason, so the absence is visible in the summary.

**Discovery is dynamic.** The harnesses are found by globbing, so a new one is
picked up by existing. A hardcoded list would let a harness be added and never
run -- the same defect one level up.

**An empty glob fails.** If the directory ever contains no harnesses at all,
that is a packaging bug, not a clean run.

**Harness output is non-vacuous, not format-fragile.** Historical harnesses
emit ``N passed, M failed`` while newer focused harnesses also use TAP plans,
``passed=N failed=M``, or ``N/N passed``.  The wrapper accepts those existing
reporting dialects, but still requires a positive assertion count and zero
reported failures.  A zero-exit harness that asserts nothing therefore cannot
silently turn green.

**Both static targets are supplied.** Some harnesses inspect JavaScript only;
newer UI regressions inspect the paired CSS as ``process.argv[3]``. Supplying
both paths is backwards-compatible because JavaScript-only harnesses simply
ignore the extra argument.

SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT, TESTS_ROOT

import pathlib
import re
import shutil
import subprocess

import pytest

_TESTS_DIR = TESTS_ROOT
_JS_TARGET = RUNTIME_ROOT / "_static" / "ai-assistant.js"
_CSS_TARGET = RUNTIME_ROOT / "_static" / "ai-assistant.css"
_HARNESSES = sorted(TESTS_ROOT.rglob("test_*.mjs"))

#: Wall-clock budget per harness.  Generous: these are pure string/regex work
#: with no I/O, so anything approaching this is a runaway loop, not slowness.
_TIMEOUT_S = 120


_CANONICAL_SUMMARY_RE = re.compile(r"^(\d+)\s+passed,\s*(\d+)\s+failed$")
_KEY_VALUE_SUMMARY_RE = re.compile(r"^passed=(\d+)\s+failed=(\d+)$")
_RATIO_SUMMARY_RE = re.compile(r"(?:^|:\s*)(\d+)/(\d+)\s+passed$")
_TAP_PLAN_RE = re.compile(r"^1\.\.(\d+)$")
_TAP_OK_RE = re.compile(r"^ok\s+(\d+)(?:\s+-\s+.*)?$")
_TAP_NOT_OK_RE = re.compile(r"^not ok\s+(\d+)(?:\s+-\s+.*)?$")


def _assert_non_vacuous_success(harness: pathlib.Path, stdout: str) -> None:
    """Require a recognised positive assertion report from a zero-exit harness."""
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
    assert lines, f"{harness.name} produced no output"
    summary = lines[-1]

    match = _CANONICAL_SUMMARY_RE.fullmatch(summary)
    if match:
        passed, failed = map(int, match.groups())
        assert failed == 0, f"{harness.name} reported {failed} failed assertions"
        assert passed > 0, f"{harness.name} exited 0 but asserted nothing"
        return

    match = _KEY_VALUE_SUMMARY_RE.fullmatch(summary)
    if match:
        passed, failed = map(int, match.groups())
        assert failed == 0, f"{harness.name} reported {failed} failed assertions"
        assert passed > 0, f"{harness.name} exited 0 but asserted nothing"
        return

    match = _RATIO_SUMMARY_RE.search(summary)
    if match:
        passed, total = map(int, match.groups())
        assert total > 0, f"{harness.name} exited 0 but asserted nothing"
        assert passed == total, (
            f"{harness.name} reported only {passed}/{total} passing assertions"
        )
        return

    match = _TAP_PLAN_RE.fullmatch(summary)
    if match:
        planned = int(match.group(1))
        assert planned > 0, f"{harness.name} exited 0 but asserted nothing"
        assert not any(_TAP_NOT_OK_RE.fullmatch(line) for line in lines[:-1]), (
            f"{harness.name} emitted TAP 'not ok' despite exiting 0"
        )
        ok_numbers = [
            int(ok.group(1))
            for line in lines[:-1]
            if (ok := _TAP_OK_RE.fullmatch(line)) is not None
        ]
        assert ok_numbers == list(range(1, planned + 1)), (
            f"{harness.name}: TAP plan 1..{planned} does not match successful "
            f"assertions {ok_numbers!r}"
        )
        return

    raise AssertionError(f"{harness.name}: unrecognised summary {summary!r}")


def test_harnesses_were_discovered() -> None:
    """
    At least one Node harness must exist.

    An empty glob would make this whole file pass vacuously -- the same
    "green means nothing" failure the wrapper was written to close.
    """
    assert _HARNESSES, f"no test_*.mjs harnesses found in {_TESTS_DIR}"


def test_harness_target_exists() -> None:
    """The paired static files must be where the harnesses expect them."""
    assert _JS_TARGET.is_file(), f"harness target missing: {_JS_TARGET}"
    assert _CSS_TARGET.is_file(), f"harness target missing: {_CSS_TARGET}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("harness", _HARNESSES, ids=lambda p: p.relative_to(TESTS_ROOT).as_posix())
def test_node_harness(harness: pathlib.Path) -> None:
    """
    Run one Node harness and require a clean, non-vacuous exit.

    Parameters
    ----------
    harness : pathlib.Path
        Path to a ``test_*.mjs`` file. The wrapper supplies JS, CSS, and runtime-root arguments.

    Raises
    ------
    AssertionError
        With the harness's own output attached on process failure, or when a
        zero-exit harness reports no positive assertion evidence.
    """
    result = subprocess.run(
        ["node", str(harness), str(_JS_TARGET), str(_CSS_TARGET), str(RUNTIME_ROOT)],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            f"{harness.name} failed (exit {result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    _assert_non_vacuous_success(harness, result.stdout)
