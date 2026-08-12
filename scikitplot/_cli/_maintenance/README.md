# `scikitplot._cli/_maintenance`

Permanent maintenance scaffolding for the scikit-plots CLI runtime. These files
carry the *why*, the *exact contract*, and the *tracked work* behind the rebuild
described in [`../MAINTAINING.md`](../MAINTAINING.md).

They are documentation, not importable code. Nothing in the CLI imports this
directory at runtime. A companion test (see `CONTRACT.md` §7) may read these files
to assert that the documented findings and invariants stay in sync with the code.

## Contents

| File | Purpose |
| --- | --- |
| [`DECISIONS.md`](./DECISIONS.md) | Architecture Decision Records. The argparse-first inversion, the frontend-selection policy, and the list of `CLI_SUBMODULE_DESIGN_GUIDE.md` sections these decisions supersede. |
| [`CONTRACT.md`](./CONTRACT.md) | The precise technical contract: the neutral `Param`/`CommandSpec`/`Context` IR, both frontend builders, the parity rules, the argparse-3.8 edge cases, and a runnable reference kernel. |
| [`FINDINGS.md`](./FINDINGS.md) | Tracked defects (`CLI-FE-00x`) with root-cause analysis and status. The refactor closes these. |

## Reading order

1. `../MAINTAINING.md` — big picture and invariants.
2. `DECISIONS.md` — why argparse is the base and click is an adapter.
3. `CONTRACT.md` — what exactly to build, with a reference kernel.
4. `FINDINGS.md` — what is wrong today and what "done" means per defect.

## Status legend (used in `FINDINGS.md`)

```text
OPEN        identified, not started
PLANNED     design agreed, captured in CONTRACT.md
IN-PROGRESS implementation underway
PARTIAL     fixed on one path; remaining work noted
CLOSED      fixed and verified with evidence
DEFERRED    intentionally postponed with rationale
```

## Workflow

- Every change to `_cli` starts by reading `../MAINTAINING.md` and `DECISIONS.md`.
- New commands add a `CommandSpec` to `registry.py` plus a neutral handler; they
  do not add click-only logic (invariant §3.2 in `MAINTAINING.md`).
- Any invariant break requires a new ADR here before the code change.
- Findings move to `CLOSED` only with pasted test evidence.


> Start from [`../RULESET.md`](../RULESET.md) — the CLI continuation contract.
