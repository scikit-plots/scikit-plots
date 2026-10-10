"""
Round 26 probe: does the pattern-risk check agree with measured run time?

Run from the wide checkout root::

    python -B maintenances/cleanprompt/_maintenance/evidence/probe_round26.py

For each pattern, a near-miss input is built (a run of the repeated unit,
then a character that makes the match fail) and matched in a child process
at growing sizes, with a time limit. A pattern is *measured slow* when the
time grows by more than ``GROWTH`` from the smallest to the largest size and
the largest exceeds ``FLOOR`` (so timer noise is never growth), or the child
is killed. The probe then compares that measurement with what
``analyse_pattern`` reports, and with the core patterns and built-in packs,
which must report nothing.

A disagreement is printed as ``FAIL``; the last line is
``TOTAL FAILURES: N``. Timings are evidence for this machine, not
constants: the verdicts are what must hold.
"""

import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from scikitplot.cleanprompt._catalog import builtin_catalog  # noqa: E402
from scikitplot.cleanprompt._pattern_risk import analyse_pattern, pack_findings  # noqa: E402
from scikitplot.cleanprompt._patterns import PATTERNS  # noqa: E402
from scikitplot.cleanprompt.tests._regex_fixtures import regex_fixture  # noqa: E402

SIZES = (12, 18, 26)
LIMIT = 4.0  # seconds per child before it counts as slow
GROWTH = 20.0
FLOOR = 0.005  # seconds: below this, a ratio is timer noise, not growth

#: (pattern, prefix, repeated unit, failing tail, expected: True = reported).
#: Units are doubled where growth is Fibonacci-like rather than doubling, so
#: the largest size is far from timer noise on any machine.
#: The second block is the round-26 independent review's: shapes the first
#: version of the check passed (verbose mode, inline flags, escaped
#: characters, a repeated unit, a bounded outer repetition) and linear shapes
#: it flagged (separators one group up, fixed counts).
CASES = [
    (regex_fixture(r"^(a+)+$"), "", "a", "b", True),
    (regex_fixture(r"^(a*)*$"), "", "a", "b", True),
    (regex_fixture(r"^(a|aa)+$"), "", "aa", "b", True),
    (regex_fixture(r"^(\w+\s?)*$"), "", "a", "!", True),
    (regex_fixture(r"^(?:[A-Z]+\d*)+-\d+$"), "", "A", "!", True),
    (r"^(\w+,)+$", "", "a,", "!", False),
    (r"^(?:\.\w+)+$", "", ".a", "!", False),
    (r"^(?:[A-Z]+-)+\d+$", "", "A-", "!", False),
    (r"^\bEMP-\d{6}\b$", "", "E", "!", False),
    # round 26 review
    (regex_fixture(r"(?x) ^ (?: \w+ \s? )+ $  # verbose"), "", "a", "!", True),
    (regex_fixture(r"^(?:\w+,\w+)+$"), "a,", "aaaa,", "a!", True),
    (regex_fixture(r"(?i)^(?:a+A)+$"), "", "aa", "!", True),
    (regex_fixture(r"^(?:[\x41-\x5a]+\x4b)+$"), "", "KK", "!", True),
    (regex_fixture(r"^(?:\w+\s?){1,40}$"), "", "a", "!", True),
    (r"^(?:(?:\w+)-)+$", "", "aa-", "!", False),
    (r"^(?:([a-z0-9]+)\.)+[a-z]{2,}$", "", "ab.", "!", False),
    (r"^(?:[ ]?[A-Z0-9]{4}){2,7}$", "", "ABCD", "!", False),
]
if sys.version_info >= (3, 11):  # possessive and atomic forms compile from 3.11
    CASES += [
        (r"^(?:\w++\s?)+$", "", "a", "!", False),
        (r"^(?:(?>\w+)\s?)+$", "", "a", "!", False),
    ]

_CHILD = """
import re, sys, time
pattern, prefix, unit, tail, n = sys.argv[1:5] + [int(sys.argv[5])]
compiled = re.compile(pattern)
text = prefix + unit * n + tail
start = time.perf_counter()
compiled.match(text)
print(time.perf_counter() - start)
"""


def measure(pattern, prefix, unit, tail, n):
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c", _CHILD, pattern, prefix, unit, tail, str(n)],
            capture_output=True,
            text=True,
            timeout=LIMIT,
            check=True,
        )
    except subprocess.TimeoutExpired:
        return None
    return float(done.stdout.strip())


failures = 0


def check(cid, desc, ok, detail=""):
    global failures
    failures += 0 if ok else 1
    print(f"{'ok  ' if ok else 'FAIL'} {cid} {desc} {detail}".rstrip())


for pattern, prefix, unit, tail, expected in CASES:
    times = [measure(pattern, prefix, unit, tail, n) for n in SIZES]
    killed = any(t is None for t in times)
    first = max(times[0] or 0.0, 1e-6)
    growth = float("inf") if killed else (times[-1] / first)
    slow = killed or (growth > GROWTH and times[-1] > FLOOR)
    reported = bool(analyse_pattern(pattern))
    shown = ["killed" if t is None else f"{t:.4f}s" for t in times]
    check(
        "R26-RISK",
        f"{pattern!s:28} measured {'slow' if slow else 'linear':6} reported={reported}",
        reported == expected and slow == expected,
        f"sizes={SIZES} times={shown}",
    )

start = time.perf_counter()
core = {kind: analyse_pattern(spec.pattern, spec.flags) for kind, spec in PATTERNS.items()}
check("R26-CORE", "core patterns report nothing", not any(core.values()),
      str({k: [r.rule for r in v] for k, v in core.items() if v}))
packs = pack_findings(builtin_catalog().packs.values(), include_builtin=True)
check("R26-PACKS", "built-in packs report nothing", packs == [],
      str([(f.pack, f.kind, f.risk.rule) for f in packs]))
check("R26-COST", "analysing every built-in pattern takes under a second",
      time.perf_counter() - start < 1.0, f"{time.perf_counter() - start:.3f}s")

print(f"TOTAL FAILURES: {failures}")
