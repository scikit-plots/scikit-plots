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

SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_TARGET = _TESTS_DIR.parent / "_static" / "ai-assistant.js"
_HARNESSES = sorted(_TESTS_DIR.glob("test_*.mjs"))

#: Wall-clock budget per harness.  Generous: these are pure string/regex work
#: with no I/O, so anything approaching this is a runaway loop, not slowness.
_TIMEOUT_S = 120


def test_harnesses_were_discovered() -> None:
    """
    At least one Node harness must exist.

    An empty glob would make this whole file pass vacuously -- the same
    "green means nothing" failure the wrapper was written to close.
    """
    assert _HARNESSES, f"no test_*.mjs harnesses found in {_TESTS_DIR}"


def test_harness_target_exists() -> None:
    """The file under test must be where the harnesses expect it."""
    assert _TARGET.is_file(), f"harness target missing: {_TARGET}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("harness", _HARNESSES, ids=lambda p: p.name)
def test_node_harness(harness: pathlib.Path) -> None:
    """
    Run one Node harness and require a clean exit.

    Parameters
    ----------
    harness : pathlib.Path
        Path to a ``test_*.mjs`` file.

    Raises
    ------
    AssertionError
        With the harness's own output attached.  The harnesses print one
        ``FAIL <name>`` line per failure with got/want, so forwarding stdout
        verbatim is more useful than any summary this wrapper could invent.
    """
    result = subprocess.run(
        ["node", str(harness), str(_TARGET)],
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

    # A harness that exits 0 having asserted nothing is the vacuous case this
    # suite has already been bitten by twice.  Every harness ends with a
    # "<n> passed, <m> failed" line; require a positive count.
    tail = result.stdout.strip().splitlines()
    assert tail, f"{harness.name} produced no output"
    summary = tail[-1]
    assert "passed" in summary, f"{harness.name}: unrecognised summary {summary!r}"
    passed = int(summary.split()[0])
    assert passed > 0, f"{harness.name} exited 0 but asserted nothing"
