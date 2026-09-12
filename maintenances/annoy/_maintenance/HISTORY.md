# History

Historical A00-A21 campaign material and the previously-live maintenance files
are retained under `history/`. They may explain why code exists, but they do not
establish current build/test status.

The 2026-09-12 re-review replaced the old checker because it resolved
`maintenances/annoy` as if it were the runtime package and then searched `/` for
shared headers. The new checker discovers the wide repository explicitly and
models the dual compiled architecture.
