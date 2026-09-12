# Fresh-chat handoff

This subsystem owns `scikitplot/cexternals/_annoy`; maintenance lives under
`maintenances/cexternals/_annoy`; its skill lives under `skills/cexternals/_annoy`.
The authoritative input archive identity is in [STATE.json](STATE.json).
Do not infer current truth from an older chat or from `history/` and `_backup/`.

1. Read [MAINTAINING.md](../MAINTAINING.md), [STATE.json](STATE.json),
   [FAMILY.md](FAMILY.md), and [VERIFICATION.md](VERIFICATION.md).
2. Run `python -B maintenances/cexternals/_annoy/_maintenance/check_trackers.py`.
3. Run `python -B maintenances/cexternals/_annoy/_maintenance/review_subsystem.py --json`.
4. Select the capability owner before editing; observe related modules without
   taking over their maintenance. Add other module skills one at a time.

The prior tools used the wrong root, tracked maintenance backups as source, and
had no _annoy skill route. Those defects are repaired in this slice. Exact
runtime hashes and observed dependency evidence replace approximate LOC gates.

The source contract has three direct header consumers. Annoy also imports the
native `Annoy` type and plotting helper. Impute imports the Cython index, Corpus
selects high-level or Cython indexes, and MCP delegates retrieval to Corpus.
Comments and docstring examples are not executable dependency evidence.

This revision changes maintenance and skill files only. All runtime bytes,
including existing cache files, remain unchanged. Onboard each user-selected
submodule independently. Native verification is outside this slice; optional
native tools and release gates are available for a later runtime review.
Historical sample test claims are not current PASS evidence.

STATE.json records the missing native link_args forwarding and the stale memmap
doc example for later scopes. No production behavior or runtime data changed.
