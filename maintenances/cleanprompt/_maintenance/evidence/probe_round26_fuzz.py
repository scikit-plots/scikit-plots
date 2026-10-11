"""
Round 26 fuzz: is the pattern-risk check sound, and how precise is it?

Run from the wide checkout root::

    python -B maintenances/cleanprompt/_maintenance/evidence/probe_round26_fuzz.py [SECONDS]

Random patterns are generated from a fixed seed (atoms, classes, every
quantifier form, nested and alternated groups). Each compilable one is
analysed, then matched in a child process with a two-second limit on eight
near-miss inputs (a run of ``a``, ``a ``, ``a-``, ``a.``, ``1``, `` ``,
``x`` or ``ab``, then ``!``) at 28 repetitions.

* **Soundness** — a pattern the check passes that is killed is a miss and
  is printed as ``FAIL``. The first version of the round-26 fix had three
  (nullable groups treated as separators, optional elements trading
  characters); the line ``TOTAL FAILURES: 0`` is the gate.
* **Precision** — of the patterns it reports, how many are killed on these
  generic inputs. The rest are either harmless or need inputs tailored to
  them; this ratio is why the default is *warn*, never *refuse*. It is
  reported, not asserted.
"""

import pathlib
import random
import re
import subprocess
import sys
import time
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from scikitplot.cleanprompt._pattern_risk import analyse_pattern  # noqa: E402

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
ATOMS = ["a", "\\w", "\\d", "\\s", ".", "[a-z]", "[^x]", "\\.", "-", " ", "\\b", "x"]
QUANTS = ["", "", "+", "*", "?", "{2}", "{1,5}", "{2,}"]
CHILD = (
    "import re,sys\np=re.compile(sys.argv[1])\n"
    "for u in ['a','a ','a-','a.','1',' ','x','ab']:\n p.match(u*28+'!')"
)
rng = random.Random(26)


def gen(depth=0):
    parts = []
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.35 and depth < 3:
            inner = "|".join(gen(depth + 1) for _ in range(rng.randint(1, 2)))
            parts.append("(?:%s)" % inner + rng.choice(["+", "*", "{1,9}", ""]))
        else:
            parts.append(rng.choice(ATOMS) + rng.choice(QUANTS))
    return "".join(parts)


def killed(pattern):
    try:
        subprocess.run(
            [sys.executable, "-I", "-c", CHILD, pattern], timeout=2, capture_output=True
        )
    except subprocess.TimeoutExpired:
        return True
    return False


clean = flagged = misses = confirmed = 0
deadline = time.time() + BUDGET
while time.time() < deadline:
    pattern = "^" + gen() + "$"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            re.compile(pattern)
        except re.error:
            continue
    if analyse_pattern(pattern):
        if flagged < 60:  # precision sample, bounded so soundness gets the time
            flagged += 1
            confirmed += killed(pattern)
        continue
    clean += 1
    if killed(pattern):
        misses += 1
        print(f"FAIL passed but slow: {pattern!r}")

print(f"soundness: {clean} passed patterns timed, {misses} slow")
print(f"precision: {flagged} reported patterns timed, {confirmed} slow on generic inputs")
print(f"TOTAL FAILURES: {misses}")
