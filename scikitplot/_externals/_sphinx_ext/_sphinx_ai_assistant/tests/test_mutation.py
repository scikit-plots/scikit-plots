"""
Mutation testing gate.

Runs every entry in :mod:`tests._mutants` against the harness that is supposed
to catch it. A mutant that survives means that harness no longer guards the
defect it was written for.

This is the check that green-on-first-run cannot give you. Over this project's
history it has caught, among others: a harness that rebuilt the array it was
testing, an ordering assertion made vacuous by ``indexOf`` returning -1, a
truthiness check that conflated *absent* with *present but falsy*, and an
injection threshold that hid a pattern loose enough to fire on ordinary API
documentation.

Mechanism
---------
Each mutant is written to a temporary copy of ``ai-assistant.js`` and the named
harness is pointed at it via ``argv[2]``. Nothing in the working tree is
modified, so an interrupted run leaves nothing to clean up and no restore step
can be skipped.

SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

import sys

import pytest

_TESTS_DIR = pathlib.Path(__file__).resolve().parent

# The catalogue is a sibling module, not a package member: this directory is a
# test directory, not an importable package, so pytest's rootdir does not put
# it on the path. Inserting it here keeps `pytest tests/` working with no
# PYTHONPATH ceremony -- a gate that needs a special invocation is a gate CI
# will eventually be configured without.
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _mutants import MUTANTS  # noqa: E402
_TARGET = _TESTS_DIR.parent / "_static" / "ai-assistant.js"
_TIMEOUT_S = 120


def _ids(m: dict) -> str:
    return m["id"]


def test_catalogue_is_not_empty() -> None:
    """An empty catalogue would make this file pass vacuously."""
    assert MUTANTS, "the mutant catalogue is empty"


def test_mutant_ids_are_unique() -> None:
    """Duplicate ids make a failure report ambiguous."""
    ids = [m["id"] for m in MUTANTS]
    assert len(set(ids)) == len(ids)


def test_every_mutant_is_documented() -> None:
    """
    ``why`` is not decoration.

    A mutant without a stated consequence becomes unmaintainable the moment
    the code around it moves: the next reader cannot tell whether it still
    guards anything real, and deletes it.
    """
    for mutant in MUTANTS:
        assert mutant.get("why", "").strip(), f"{mutant['id']}: missing 'why'"
        assert mutant.get("harness", "").endswith(".mjs"), (
            f"{mutant['id']}: 'harness' must name a .mjs file"
        )


@pytest.mark.parametrize("mutant", MUTANTS, ids=_ids)
def test_mutant_anchor_is_unique(mutant: dict) -> None:
    """
    The ``find`` text must appear exactly once in the source.

    Zero matches means the mutant has rotted: the code it targeted has moved
    or been rewritten, and it is silently testing nothing. Several matches
    means it would mutate an arbitrary one of them, so a pass would not mean
    what it appears to.

    Checked separately from the run below so a rotted catalogue reports as a
    catalogue problem rather than as a surviving mutant, which would send
    someone looking in the wrong file.
    """
    source = _TARGET.read_text(encoding="utf-8")
    count = source.count(mutant["find"])
    assert count == 1, (
        f"{mutant['id']}: anchor found {count} times, expected exactly 1. "
        f"The mutant has rotted and is no longer testing anything."
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("mutant", MUTANTS, ids=_ids)
def test_mutant_is_caught(mutant: dict) -> None:
    """
    Applying the mutant must make its harness fail.

    Parameters
    ----------
    mutant : dict
        Entry from :data:`tests._mutants.MUTANTS`.

    Raises
    ------
    AssertionError
        When the harness still passes, with the mutant's ``why`` attached --
        the surviving mutant is only half the message; what it lets through is
        the other half.
    """
    source = _TARGET.read_text(encoding="utf-8")
    assert source.count(mutant["find"]) == 1, f"{mutant['id']}: anchor not unique"
    mutated = source.replace(mutant["find"], mutant["replace"], 1)
    assert mutated != source, f"{mutant['id']}: replacement is a no-op"

    harness = _TESTS_DIR / mutant["harness"]
    assert harness.is_file(), f"{mutant['id']}: no such harness {harness.name}"

    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / "ai-assistant.js"
        target.write_text(mutated, encoding="utf-8")

        result = subprocess.run(
            ["node", str(harness), str(target)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )

    if result.returncode == 0:
        raise AssertionError(
            f"MUTANT SURVIVED: {mutant['id']}\n"
            f"  harness : {mutant['harness']} still passes with this applied\n"
            f"  breaks  : {mutant['why']}\n"
            f"  find    : {mutant['find'][:120]!r}\n"
            f"  replace : {mutant['replace'][:120]!r}\n"
            f"The harness no longer guards this defect."
        )

    # A crash is not a catch. The harness must REPORT the failure, or a
    # regression that throws would abort every later assertion in that file
    # while still looking like a successful detection.
    assert "failed" in result.stdout, (
        f"{mutant['id']}: {mutant['harness']} exited non-zero without a failure "
        f"summary -- it crashed rather than failing.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
