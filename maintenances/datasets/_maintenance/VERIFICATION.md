# Verification ladder

1. **Static contract:** inventory, API surface, ownership planes, known-defect guards, evidence integrity.
2. **Curated loader tests:** `_load_dataset` cache/name/post-processing tests under a real package root or minimal isolation shell.
3. **Exporter tests:** deterministic hash/random/stratified/streaming behavior, manifests, CLI and stub parity.
4. **General loader tests:** local files, archives, uploads, URL routing, SQLite/DuckDB/SQLAlchemy defaults, cleanup and failure behavior. This lane is currently missing from the shipped test suite.
5. **Optional backend integration:** Parquet engines, Excel/Feather, DuckDB, SQLAlchemy drivers, async drivers. Missing dependencies are `UNAVAILABLE`, never green.
6. **Network/cache integration:** dataset-name fetch, download, interrupted download/cache recovery and offline-cache behavior.
7. **Complete package + supported platforms:** root exports/logger/optional-deps integration and platform filesystem semantics.

Release stays blocked while owned runtime findings are red or levels 4-7 are unavailable.
