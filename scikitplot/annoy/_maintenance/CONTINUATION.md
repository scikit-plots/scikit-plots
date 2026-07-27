<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Continuation & export — resume the annoy review in a fresh chat/sandbox

This doc makes the campaign **restartable**. If the working chat is cleared or the
sandbox resets, follow this to continue with the exact same method (strict
grounding, minimal root-cause fixes, tiered always-green gate, per-run evidence,
cumulative drop-in export). Nothing here relies on chat memory.

## 1. What to upload to a new chat (checklist)

The submodule export alone **cannot build** (see rule L-BUILD-ROOT). Re-upload:

- [ ] `annoy_submodules_through_run<N>.zip` — the latest cumulative drop-in
      (this hub travels inside it, under `scikitplot/annoy/_maintenance/`).
- [ ] `ANNOY_DEEP_SEMANTIC_REVIEW_GUIDE.md` — the finding source of truth.
- [ ] Repo-root build files (not in the submodule export):
      `pyproject.toml`, root `meson.build`, `meson.options`, `meson_cpu.zip`,
      `_build_utils.zip` (extracts to `scikitplot/_build_utils/`).

## 2. Standing prompt to paste (condensed)

> Lead Data Scientist / MLOps reviewer. Continue the `annoy` subsystem deep review
> finding-by-finding, same discipline as `scikitplot.cython`: strict grounding /
> zero hallucination, minimal-impact root-cause fixes, Conventional Commits,
> NumPyDoc, tiered always-green gate, per-turn evidence, no `==` pinning. Carry the
> prevention rules in `annoy/MAINTAINING.md` (L-SEC…L-DOC, L-HARNESS, L-GATE-SCOPE,
> L-BUILD-ROOT, L-NATIVE-SENTINEL). Scope: `annoy` + `cexternals/_annoy` + `memmap`
> + `random` (neighbors + nc last). Restructure in place only (no file moves).
> One contract per run; ring-failure blocks the run. Read
> `_maintenance/ANNOY_REVIEW_PLAYBOOK.md` for order and `_maintenance/todo.md` for
> the next target. First: rebuild + confirm the ring baseline (recipe in
> `RUN0b_MESON_GATE.md`), then stop and wait for "continue".

## 3. Build + gate bring-up (condensed; full recipe in RUN0b_MESON_GATE.md)

```
pip install --break-system-packages cython meson ninja meson-python tempita pybind11 pytest hypothesis
# assemble repo root: pyproject.toml, meson.build, meson.options, meson_cpu/, scikitplot/_build_utils/
# scaffolds: CITATION.bib (placeholder) + tools/f2py/generate_f2pymod.py (stub)
meson setup builddir -Dallow-noblas=true
ninja -C builddir \
  scikitplot/random/_kiss/kiss_random.*.so \
  scikitplot/memmap/_memmap/mem_map.*.so \
  scikitplot/cexternals/_annoy/annoylib.*.so \
  scikitplot/annoy/_annoy/annoylib.*.so
# stage built .so + generated config/__config__.py, config/_citation.py, version.py into the source tree
# gate (per submodule): PYTHONPATH=. python _maintenance/gate.py scikitplot/<sub>/tests
```
Expected ring baseline: **561 passed** (kiss 105, memmap 49, annoy/_annoy 333,
annoy metric 42, annoy core 32). Heavy/concurrency suites are CI-tier (OOM in a
sandbox) — see RUN0b deferrals.

## 4. Per-run workflow

Follow `MAINTAINING.md` → "Change workflow": reproduce/ground → minimal
root-cause fix (keep externed C signatures) → permanent regression test (mock
native/Windows logic on host) → tiered always-green gate → `.pyi` parity →
document (review log + prevention rule). One contract per run.

## 5. Export / drop-in procedure (run at each end-of-run)

Produce the cumulative, source-only drop-in the maintainer applies to their repo:

```
# from the working repo root, staging the four scoped submodule trees (with all
# edits) under scikitplot/, then stripping build artifacts:
mkdir -p export/scikitplot
for d in annoy cexternals/_annoy memmap random; do
  mkdir -p export/scikitplot/$(dirname $d); cp -a scikitplot/$d export/scikitplot/$d
done
cp -a scikitplot/annoy.pxd export/scikitplot/ 2>/dev/null || true
find export -type d -name __pycache__ -prune -exec rm -rf {} +
find export -type d -name '*.so.p'   -prune -exec rm -rf {} +
find export -type f \( -name '*.so' -o -name '*.o' -o -name '*.pyc' \) -delete
# include an EXPORT_README.md at export/ root (overlay + changes + wheel-exclude note)
cd export && zip -qr ../annoy_submodules_through_run<N>.zip . && cd ..
```

Rules for the drop-in:
- **Cumulative** — every change from the start, not just this run's diff.
- **Source-only** — no build artifacts; maintainer/CI rebuilds via §3.
- **Affected surface expands** — if a run touches a module outside the four
  scoped dirs, add that module to the export so it stays a complete drop-in.
- **Verified** — round-trip: fresh extract must still compile/gate green.

## 6. Current status

- Runs closed: R1-R11 (Part I 6.x + build-warn), R12 CY-008, R13 CY-006, R14 CY-005, R15 CY-007, R16 CY-010, R17 CY-011, R18 CY-017, R19 CY-018 (+CY-014 no-op), R20 CY-013, R21 CY-019, R22 CY-012, R23 CY-004. Review log in `MAINTAINING.md`.
- Latest drop-in: `annoy_submodules_through_run23.zip`.
- Next target (`todo.md`): BUILD-WARN-001 (`annoymodule.cc:4051`), §6.13 unsafe
  integer comparisons, §6.10/CY-017 exception propagation, or stage the
  §6.2/§6.3 mmap-lifetime state machine with a documented invariant.
