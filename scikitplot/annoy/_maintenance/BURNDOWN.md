<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# annoy review campaign — burndown (remaining / expected runs)

As of Run 23. Source of truth: `ANNOY_DEEP_SEMANTIC_REVIEW_GUIDE.md` Part I (20
findings §6.1–6.20) + Part II CY register (CY-001–020). Total inventory = 40
findings; **25 closed** across 23 runs (CY-014 checked, no-op) (10 Part I + 2 Part II + R5 build-warning +
the two runs bundle nothing extra), **28 open**.

## Closed (15)
Part I: 6.1(R1), 6.5(R10), 6.7(R11), 6.8(R6), 6.9(R8), 6.10(R7), 6.11(R2),
6.13(R3), 6.15(R4), 6.16(R9). Extra: BUILD-WARN-001(R5).
Part II: CY-008(R12), CY-006(R13), CY-005(R14), CY-007(R15), CY-010(R16), CY-011(R17), CY-017(R18), CY-018(R19), CY-013(R20), CY-019(R21), CY-012(R22), CY-004(R23).

## Open work, grouped by how it must be handled

### Group A — standard code-fix runs (executable now, ~1 finding/run)
These reproduce → root-cause fix → regression test → gate → package, like R1–R13.
- CY-015 P2 — HTML/CSS/link logic inflates compiled boundary (extract)
- 6.20 P2 — exports and generated metadata can drift
- CY-006 follow-up — reject zero/negative for positive params (numeric validation)

**≈ 10–11 runs.** Rough near-term order (impact-first): CY-010
→ CY-011 → CY-017 → CY-014/018 → CY-013/019 → CY-012/015 → 6.20.

### Group B — design decisions (need a maintainer choice before code)
- 6.6 P1 — one format contract for on-disk build vs normal save
- 6.14 P1 — KISS modulo bias: needs public-guarantee decision + legacy mode
- CY-001 P1/P2 — multiple impl templates can hide fixes (generation integrity)
- CY-002 P2 — `.pxi` reachability / source-of-truth
- CY-003 P1/P2 — installed `.pxd` implies downstream ABI (public Cython API)
- CY-016 P2 — 160 specializations lack support-tier/test budget

**≈ 6 decision items** (each may spawn 0–1 implementation run once decided).

### Group C — CI / environment-gated (cannot close in this root sandbox)
Sandbox runs as root and OOMs on large/concurrency suites, so these need
thread-stress + ASan/TSan, Windows CI, or fault injection to verify.
- 6.2 P0 — mmap ops race with explicit close (concurrency)
- 6.3 P0 — exported NumPy view may outlive the mapping (lifetime)
- 6.4 P0 — worker exceptions / raw locks can terminate or deadlock (concurrency)
- 6.12 P1 — Windows mmap adapter completeness (Windows CI)
- CY-009 P0/P1 — same-object GIL lock policy (concurrency)
- CY-020 P2 — generated compile-warning volume (build-quality, CI budget)
- Fault-injection follow-ups for already-closed R1/R10/R11 (ENOSPC, truncate,
  crash, Windows).

**≈ 7–8 items**, delivered as design/test scaffolding + CI jobs, not sandbox runs.

### Group D — end-of-campaign (readability + pruning, done last)
- 6.17 P2 — `annoymodule.cc` excessive responsibility (in-place split)
- 6.18 P2 — `annoylib.h` combines unrelated core domains (in-place split)
- 6.19 P2 — Meson compiles the binding into both a static lib and the extension
- Clutter pruning: Go/Lua/rockspec, `_v0`/`_review`, `backup_template/`.

**≈ 3–4 items.**

## Expected-runs summary
- **Executable now (Group A):** ~12–13 more standard runs.
- **After decisions (Group B):** ~6 decisions → up to ~6 implementation runs.
- **CI-gated (Group C):** ~7–8 items, land as CI/test scaffolding.
- **Cleanup (Group D):** ~3–4 items at the very end.

So at the current one-finding-per-run cadence, roughly **12–13 runs** close the
remaining directly-fixable defects; full 40/40 closure additionally needs ~6
maintainer decisions and ~8 CI-gated verifications that fall outside the sandbox.
