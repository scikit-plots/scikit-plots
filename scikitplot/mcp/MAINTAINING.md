# Maintaining `scikitplot.mcp`

Entry point for future maintenance. **Self-contained**: a fresh session needs no
chat history to continue from here.

## Source anchor

```text
archive: scikit-plots.zip
sha256:  611bdbf3b0a366276b9538e510e974d9400491c84209fc0d35cb9bb058cb8f38
```

If the hash changes, re-verify claims before carrying them forward.

## Read order for a fresh chat

1. `_maintenance/MAINTENANCE_MODEL.md` — why / when / where / which / how many / how much
2. `_maintenance/RULESET.md` — the durable rules
3. `_maintenance/TRACKER_LOGICAL.md` — contracts, invariants, **Corpus-derived drift**
4. `_maintenance/TRACKER_PHYSICAL.md` — what is on disk; the tripwires
5. `_maintenance/SUBMODULE_STRUCTURE.md` — where a new thing goes; doc debt
6. `_maintenance/MCP_COMPATIBILITY_POLICY.md` — Python/SDK tiers; single-wire-protocol rule
7. `_maintenance/DESIGN.md` — intent
8. `_maintenance/VERIFICATION.md` — how to prove the tree is healthy

Read `_maintenance/HISTORY.md` only when historical rationale is needed.
Machine-readable: `_maintenance/STATE.json`, `_maintenance/TRACKER.json`.

## Run this first

```console
$ python scikitplot/mcp/_maintenance/check_trackers.py
$ python -m pytest scikitplot/mcp -q -p no:cacheprovider
```

> **Expect `2 skipped, 4 errors` without `[mcp]` installed.** That is recorded
> finding `MCP-M00-01`, not a broken checkout: four test files import
> `pydantic` unguarded, so the suite fails at *collection*. See
> `_maintenance/TRACKER_LOGICAL.md` §4.

## The one rule behind all the others

MCP is an **adapter**. Its failure mode is not producing bad results — it is
**leaking across a boundary**.

> MCP owns the **wire**. Corpus owns **retrieval semantics**. Neither imports the
> other's concerns. When one needs something from the other, a **contract** is
> published — not an import added.

Two consequences, both now mechanically enforced by `check_trackers.py`:

- `_server.py` is the **only** module that may import `mcp` or `pydantic`;
- `corpus` and `annoy` are **never** imported at module scope — they arrive by
  injection through the `DocsRetriever` Protocol.

Verified today: importing `scikitplot.mcp` loads neither the SDK nor a retrieval
backend.

## Current state

```text
Corpus review          COMPLETE  (R00–R16)
Corpus implementation  COMPLETE  (IMPL-01–IMPL-18, waves I0–I6)
MCP campaign           M00–M13 COMPLETE; M14 Corpus+Annoy CLI/showcase VERIFIED OFFLINE
Current live gate      Annoy native backend + MCP SDK HTTP round trip in provisioned CI
Project sequence       Corpus -> MCP -> ANNOY -> CLI -> cross-module integration
```

The current MCP implementation is active. Preserve the established boundaries:
SDK imports remain isolated to `_server.py`, while Corpus/Annoy integrations are
lazy call-time adapters. M14 adds a local `--corpus-annoy PATH` CLI profile and
a CI-oriented Hamlet showcase without changing those import boundaries.

## What changed under MCP while it waited

Corpus renamed public symbols and added contracts. **No MCP code broke**, because
retrieval arrives by injection — the drift is six documentation references, listed
in `TRACKER_LOGICAL.md` §3.

The Corpus contracts MCP will map against (`RetrievalResponse`,
`RetrievalStatus`, `ErrorRecord`, `CapabilityStatus`, `ComponentCatalog`) are
**built and tested**, not merely designed. That is new since the review guide was
written, and it makes runs M03 and M04 concrete rather than speculative.

One constraint runs the other way: `ToolCallInput` and `ToolCallResult` are
protocol-**neutral** payload shapes that live in Corpus. A Corpus test asserts its
`_types` module imports neither `pydantic` nor `mcp`. Do not defeat it from this
side.
