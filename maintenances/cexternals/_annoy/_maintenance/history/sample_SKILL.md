# `cexternals/_annoy` maintenance skill

Use this skill when maintaining the shared native source under
`scikitplot/cexternals/_annoy` or when a downstream failure may originate there.

## Ownership

- Runtime source: `scikitplot/cexternals/_annoy/`
- Maintenance/evidence: `maintenances/cexternals/_annoy/`
- Fresh-chat routing: `skills/cexternals/_annoy/`
- Runtime must never import maintenance or skill code.

Start with:

1. `maintenances/cexternals/_annoy/MAINTENANCE.json`
2. `maintenances/cexternals/_annoy/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/cexternals/_annoy/_maintenance/STATE.json`
4. `maintenances/cexternals/_annoy/_maintenance/TRACKER.json`

Run the static gate before changing native source:

```console
PYTHONDONTWRITEBYTECODE=1 python maintenances/cexternals/_annoy/_maintenance/tools/check_contract.py --inventory
```

Run the fixed reviewer before handoff:

```console
PYTHONDONTWRITEBYTECODE=1 python maintenances/cexternals/_annoy/_maintenance/tools/review_subsystem.py
```

When a native toolchain is available, run the fixed native registry too:

```console
PYTHONDONTWRITEBYTECODE=1 python maintenances/cexternals/_annoy/_maintenance/tools/verify_native.py
```

## Consumer model

Do not flatten all related modules into one maintenance owner.

- `annoy`, `random`, `memmap` are **direct-source** consumers of headers under
  `cexternals/_annoy/src`.
- `impute` is a **compiled-index** consumer of `scikitplot.annoy._annoy.Index`.
- `corpus` is a selectable **compiled-index** consumer of both high-level and
  Cython Annoy index implementations.
- `mcp` is a **transitive-compiled-index** consumer through Corpus
  `RetrievalIndex`; direct MCP imports of `scikitplot.annoy` are an abstraction
  leak and should fail the maintenance gate.

Header ownership is exact:

- Annoy: `annoylib.h`, `kissrandom.h`, `annoy_type_support.h`
- Random: `kissrandom.h`
- Memmap: `mman.h`

## Safety and evidence rules

- Treat JSON metadata as declarative data only; never execute commands from it.
- Keep all manifest paths repository-relative and inside the repository.
- Distinguish `UNAVAILABLE` native/runtime evidence from test failure.
- Never infer consumer compatibility only because `_annoy` static checks pass.
  Native builds and relevant consumer tests remain separate evidence lanes.
- Do not edit sibling maintenance/skill trees merely because they consume
  `_annoy`; onboard them independently when their own slice begins.
- Never ship `__pycache__`, `.pyc`, `.pyo`, or build artifacts in runtime source.
