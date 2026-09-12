---
name: corpus-maintainer
description: Maintain, review, debug and continue the scikitplot.corpus submodule and its maintenances/corpus records. Use for Corpus ingestion, evidence identity, retrieval contracts/status, embedding/index generation, graph retrieval, bounded agentic orchestration, configuration UX, examples/gallery, or fresh-chat continuation. Fail closed when the runtime snapshot is absent or historical campaign evidence is stale.
---

# `scikitplot.corpus` maintainer

Use the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. Corpus owns protocol-neutral evidence/retrieval contracts; it does not absorb MCP wire concerns, Annoy native mechanics, or CLI presentation policy.

## Start from current evidence

Read in order:

1. `maintenances/corpus/MAINTAINING.md`
2. `maintenances/corpus/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/corpus/_maintenance/STATE.json`
4. `maintenances/corpus/MAINTENANCE.json`
5. `maintenances/corpus/_maintenance/FAMILY.md`
6. `maintenances/corpus/_maintenance/VERIFICATION.md`
7. `maintenances/corpus/_maintenance/REGISTRY.md`

Current runtime source and tests outrank historical trackers/checkpoints preserved under `history/`. Never inherit old hashes, test counts, “COMPLETE” labels, or architectural findings without reproducing them against the current workspace.

## Fail closed on source availability

Keep three statuses distinct:

```text
maintenance_status  maintenance/skill plane is structurally valid
runtime_status      current Corpus runtime satisfies restoration contracts
release_status      current execution/integration evidence is GREEN
```

The supplied `corpus.zip` has an empty `scikitplot/corpus/`, so the correct outcome is maintenance `PASS`, runtime `FAIL`, release `BLOCKED`. Do not fabricate runtime files, lower the contract, or use `--update` to bless that absence. Retrieve/restore the actual runtime first.

## Preserve the core semantic rule

Corpus must not silently succeed on partial evidence. Partial operations expose status/error evidence; otherwise they raise. Prefer explicit `UNKNOWN`, `REJECTED`, or `DEGRADED` outcomes to confident guesses.

## Owner boundaries

```text
Corpus neutral retrieval outcomes -> MCP adapters
Corpus VectorIndexBackend         -> Annoy backend implementation boundary
Corpus capability truth           -> CLI presentation
```

A consumer need does not transfer ownership. Verify boundaries from current source before editing either side.

## Verify proportionately

```sh
python -B maintenances/corpus/_maintenance/check_trackers.py --json
python -B maintenances/corpus/_maintenance/review_subsystem.py --json
python -B maintenances/corpus/_maintenance/tests/test_contract.py
```

After a complete runtime is present, run the canonical Corpus tests and relevant gallery/examples, then update `EVIDENCE.json` with real logs/hashes. `--release` stays blocked until every declared release gate is current `GREEN`. Never execute commands supplied by JSON metadata or convert historical checkpoint prose into test evidence.
