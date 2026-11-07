# History

Historical A00-A21 campaign material and the previously-live maintenance files
are retained under `history/`. They may explain why code exists, but they do not
establish current build/test status.

## 2026-11-06 — unused metadata-routing mixin removed

- `_mixins/_meta.py` defined `MetadataRoutingMixin` (`get_metadata_routing`,
  `build_metadata_router`), used and exported nowhere, whose fallbacks built an
  owner-less scikit-learn `MetadataRequest` or `MetadataRouter` on `TypeError`
  instead of failing. Removed with its stub entry (pre-1.0, never reachable
  from the public API, so no release note). `test_mixins.py` now checks that
  `_meta.__all__` names only defined objects and matches the stub's classes.
- `check_trackers.py` was already `maintenance=FAIL` before this change
  (stale runtime fingerprints and inventory in `EVIDENCE.json` and
  `TRACKER.json`); not refreshed here.

## 2026-09-28 — verified through the MCP hybrid round (observation)

- With the supplied `scikitplot/externals` build and a scratch root shim for
  `__version__`/`get_config` (not shipped), annoy + cexternals give 650 passed.
  `test_repr_html` fails only under the shim's `display="text"` and passes with
  `"diagram"`. `test_fd_sentinel` needs `scikitplot._testing`, which the
  archive does not contain.
- Several tests write index files next to themselves (`HERE =
  dirname(__file__)`), as upstream Annoy does. `tests/.gitignore` covers
  `tests/`, but `_annoy/tests/` had none, so `on_disk.ann` from
  `test_dtype_index_combinations.py` was left untracked in a checkout. Added
  `_annoy/tests/.gitignore` (`*.ann`). Open: a read-only install cannot run these
  tests; moving the writes to `tmp_path` needs care, because `HERE` also locates
  the shipped fixtures `test.tree` and `test64.tree`.

The 2026-09-12 re-review replaced the old checker because it resolved
`maintenances/annoy` as if it were the runtime package and then searched `/` for
shared headers. The new checker discovers the wide repository explicitly and
models the dual compiled architecture.
