# `scikitplot.random` maintenance: active tasks

Round 33 work that changes `scikitplot.random`. The campaign's status and
open decisions are in [`../../corpus/todo/todo.md`](../../corpus/todo/todo.md).
Completed tasks are archived under [`../_maintenance/history/`](../_maintenance/history/).

Rules that apply to every increment: [`plugins/scikitplot-dev/skills/scikitplot-engineering-rules/`](../../../plugins/scikitplot-dev/skills/scikitplot-engineering-rules/SKILL.md).

## Task: round 33 / Run 6 / 6a - unbiased bounded integers, a locked spawn (R6-1, R6-2)

### Context
- Review (`modules/15_random.md`):
  - F-33: `index(n)` is `kiss() % n` (`kissrandom.h`), biased for large `n`
    (60.3% of draws below 1,294,967,296 at `n = 3e9`, uniform 43.2%), and
    the 32-bit generator never reaches 2**32 or more;
  - F-34: the docstring promises `ValueError` for `n < 0`; the `size_t`
    parameter raises `OverflowError`;
  - F-35: `KissSeedSequence.spawn` reads, builds, then adds to
    `n_children_spawned` with no lock; the modules are compiled
    `freethreading_compatible`, so two threads can mint one spawn key.
  - `KissGenerator.integers` uses `raw % range_size`: biased in principle,
    and wrong for a range above 2**64.
- Decision D5 (open): the shared `kissrandom.h` also drives Annoy's tree
  sampling. Option (a), recommended, keeps the header and Annoy unchanged;
  (b) changes both. This task implements (a) in `scikitplot.random` only,
  which (b) would extend, so it does not presume the decision.
- Base: 5c (`b34c5c5`).

### Design
- One rule for every bounded draw in `random`: rejection, then modulo.
  With `w` the generator's width (32 or 64 bits) and `limit = 2**w - (2**w
  mod n)`, draw `v` until `v < limit`, return `v mod n`. Unbiased, exact,
  one division; a draw that is not rejected returns what the modulo did, so
  seeded streams are unchanged except where a draw is rejected (probability
  below `n / 2**w`, so never for small `n` in practice).
- `Kiss32Random.index(n)` for `n > 2**32`: two 32-bit draws make one 64-bit
  value (high word first), then the 64-bit rule.
- `index(n)`: `n` is any integer (`operator.index`); `n < 0` raises
  `ValueError` as documented; `n` above `SIZE_MAX` raises `OverflowError`;
  `n == 0` returns 0 (unchanged).
- `KissGenerator.integers`: the 64-bit rule for ranges up to 2**64; a
  wider range raises `ValueError`.
- `spawn`: a `threading.Lock` (not re-entered) around reading and advancing
  `n_children_spawned`; children are built outside it from the reserved
  range; the lock is not part of the state, and is created again when a
  state is restored without `__init__`.
- Gate: chi-square uniformity at several `n` (2**31 + 1, 3e9, 2**33, 7)
  with seeded draws and a fixed significance; the maximum reaches 2**32 for
  `n = 2**33`; small-`n` seeded streams equal the previous modulo output;
  8 threads x 3,000 spawns give 24,000 distinct keys; Annoy's outputs for a
  fixed seed are byte-identical (D5 (a)).

### Results review - 6a (2026-10-17)
- Before/after seeded outputs (`r34/run6a/stream.py`): `index` at small `n`
  for both generators, raw `kiss()`, `integers` scalar and array, and an
  Annoy index's neighbours for a fixed seed: all byte-identical (D5 (a)
  holds: Annoy unchanged). At `n = 3e9` the share of 32-bit `index` draws
  below 1,294,967,296 is now 0.4317 (uniform 0.432; was 0.603); the maximum
  of `index(2**33)` reaches 8,589,908,199.
- Review (`r34/review6a/`): the thresholds matched independent references
  at every edge (`2**32 - 1`, `2**32`, `2**32 + 1`, `2**64 - 1`), and
  termination holds (rejection below one half). 5 findings, fixed with
  tests: [High] `integers` with numpy bounds raised `OverflowError` on every
  draw (bounds now `operator.index`; float bounds refused); [Med] the spawn
  test could not tell the unlocked code from the locked (it now runs at a
  1 us switch interval: the old code gives about 90 duplicate keys); [Low]
  two uniformity cases tested termination only (labelled; a 64-bit
  old-modulo check added); the stream-compatibility wording; `bool`
  accepted, stubs typed `SupportsIndex`. Re-review: all hold; 2 Low
  (`bool` in `integers`, `randint` Raises), fixed.
- Negative proof: the final tests on the base build: 19 failed, 28 passed;
  final: 47 passed. `scikitplot/random` and `scikitplot/annoy`: 954 passed
  after the change.
- Suites (with `scikitplot/random` added): 10015 passed + the baseline 15.
- Gates against `r34/base5c` (from `8c3bf4a`): clean; labeler PASS.
- Not run: a free-threaded build (no `py313t` scikitplot build here); the
  lock is verified by the switch-interval race on the GIL build.
- `random` is in no partial distribution, so the CI harness does not build
  it; the editable build was rebuilt from the changed `.pyx` for every test.
- CI harness on an untouched snapshot (`snap6a`, with `scikit-plots-annoy`):
  203 passed, 4 failed, 3 skipped. The 4 failures are the corpus legs, each
  5170 passed and the same 14 DNS tests as 3b; the skips are the annoy wheel
  on 3.9/3.13 (built for 3.12 only) and the mcp extra on 3.9. The annoy
  wheel ships 124 files, each identical to the source tree.
