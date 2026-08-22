# History — `scikitplot.annoy`

Read only when the *why* behind a current rule is unclear. Nothing here is
current source truth.

---

## This submodule

A prior campaign completed a deep review and fix pass: CY-002, CY-020, a native-error-ownership sweep (RAII migration), a `__contains__`/`__getitem__` negative-index fix, CY-003, CY-001, CY-009 concurrency corrections. It fixed a macOS-15 Intel link failure (illegal text relocation from `__builtin_cpu_supports("f16c")`; resolved by removing `-Wl,-ld_classic` and adding a CPUID probe) and rewrote the scalar f16 converters, which had been truncating subnormals, to IEEE-754 round-to-nearest-even. It built the float-type system: `supported_dtypes()`, float80 wired, multiprecision 256/512 gated behind `ANNOY_ENABLE_MULTIPRECISION`. **CY-015 (`annoymodule.cc` modernization) was deferred and remains open.** The ~35 `RUN*.md` files are that campaign's per-run evidence.

---

## What the Corpus campaign changed for this family

Corpus completed review (R00–R16) and implementation (IMPL-01–IMPL-18) before
this family's campaign began. Three outcomes matter here.

**`ANNBackend` → `VectorIndexBackend`.** Renamed because "ANN" names an
algorithm class, and `BruteForceBackend` is exact — so the old name was already
wrong for one of the four implementations.

**Eleven declarative capability members.** The backend contract answered three of
eleven required properties; it now declares `metric`, `score_semantics`,
`dimension`, `dtype`, `supports_persistence`, `thread_safety` and
`memory_profile`. Annoy's backend must answer all of them **truthfully** — a
declaration that outruns the implementation is the defect class two campaigns
have been removing.

**`ANNIndexArtifact` exists.** Review run R06 found Annoy could fill only one of
its four stated roles, because three needed an on-disk index Corpus had no way to
describe. Corpus now writes an artifact directory — native payload, versioned
ordinal→doc_id sidecar, embedding manifest — published atomically. Wiring Annoy's
native `save`/`load` into `write_native`/`open_native` is now a contained change.

One consequence to watch: Corpus currently declares `supports_persistence=True`
for every backend, flipped when the artifact format landed. If Annoy's native
persistence does not round-trip, **that declaration is wrong** and must change.

---

## The maintenance set

`LIVE_V1` mirrors `scikitplot/corpus/_maintenance/` and
`scikitplot/mcp/_maintenance/`. The principle carried over:

> A document that describes what a script can check should be replaced by the
> script; one that records why cannot be.

Hence `check_trackers.py`, which turned the family's defining coupling — nine
relative-path references into another submodule's headers — from an undocumented
convention into a checked rule.
