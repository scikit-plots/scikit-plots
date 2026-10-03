# Fresh-chat handoff — MCP

MCP owns the wire, adapter validation, transport wiring, capability reporting,
and MCP-specific client integration. Corpus owns retrieval semantics; Annoy owns
vector/native mechanics. Do not collapse those owners into MCP.

Start with:

```sh
python -B maintenances/mcp/_maintenance/check_trackers.py --json
python -B maintenances/mcp/_maintenance/review_subsystem.py --json
```

The supplied 2026-09-12 snapshot has a healthy static runtime architecture but
an obsolete inherited maintenance plane. The new gate resolves the wide
repository explicitly instead of treating `maintenances/mcp` as runtime.

Release is separate from structural health. A release claim needs current
official-SDK in-memory, stdio, Streamable HTTP, Corpus+Annoy, and packaging-extra
evidence. Missing optional dependencies or a sliced archive are `UNAVAILABLE`,
not PASS.

Historical M00–M14 checkpoints explain earlier decisions only. Never copy their
status line forward without re-running the present gate/evidence lanes.

## Lexical leg and passage merging (2026-09-28)

The BM25 leg must call `SQLiteStorage.search_text`, never
`StorageQuery.full_text` (a phrase filter). A new field on the `search_docs`
output needs a matching field on the closed models in `_server.py`, or the
typed path raises; `tests/test_passage_dedup.py` shows the pattern. The gate
was already FAIL on arrival (stale fingerprint/inventory) — regenerate the
lanes before refreshing it, as `VERIFICATION.md` says.

## Retrieval modes (2026-09-28, CX-04)

`--corpus-mode semantic|hybrid|keyword`. The hybrid and keyword modes delegate
to corpus `RetrievalIndex.search(query, config=..., query_embedding=...)`.
`config` is keyword-only, so passing it positionally fails every query. Do not
reimplement fusion in MCP. The CLI stays `strict=True`. To test with Annoy,
the archive needs `scikitplot/externals` plus root `__version__`/`get_config`;
a missing root raises `ImportError` (not `ModuleNotFoundError`), so skip on
`ImportError`.

Packaging lesson (2026-09-28): running the Annoy suite writes `.ann`/`.tree`
files next to its tests (git-ignored upstream). When shipping an archive built
from a working tree, compare its file list with the input archive and require
the difference to equal the intended change set. A plain directory walk
shipped 14 generated index files before this check caught them.
