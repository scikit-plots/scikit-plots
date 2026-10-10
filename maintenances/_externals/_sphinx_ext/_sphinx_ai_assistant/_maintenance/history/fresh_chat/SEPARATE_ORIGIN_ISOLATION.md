# Fresh-chat handoff — Run 22 / B41 Separate-Origin Isolation

Input release: Run 21 / B40 exact final SHA-256 `c2ed066b6b45da3985ff1373a888255c3d03e93352df867c81471614fd393976`.

B41 adds optional fail-closed separate-origin execution. Parent docs run only `_static/ai-assistant-isolation-host.js`; frame bootstrap is `_static/ai-assistant-isolated.html` + `ai-assistant-isolated-frame.js`; the mature full runtime loads in the frame only after exact source/origin/version/channel validation and MessagePort transfer. Storage is namespaced by parent origin. Host config/endpoints are snapshotted at bridge startup; late global mutation cannot alter INIT, and secret/prototype-pollution keys are rejected.

Current frozen working-tree gates: B41 11/11 Python tests, 34/34 browser assertions, Node 45/45, mutation/privacy 212/212, runnable non-Sphinx 754 passed / 3 skipped, 68/68 two-root Python compile, JS/TOML/release-subject/maintenance GREEN, and Sphinx-inclusive 1220 passed / 3 skipped / 5 failed / 62 errors with missing `sphinx` as the only failure/error family. Candidate → prefinal → exact-final-byte packaging remains pending.

Do not claim full hostile-parent integrity. `SEC-P1-42` and deployment evidence `SEC-P1-43` remain open. Same-origin compatibility mode does not inherit isolated-mode `SEC-P1-41` closure.

Controlled Run 21 → Run 22 diff is 9 added / 20 modified / 0 removed = 29 paths; prepackage membership is 288 files under exactly the two canonical roots.

Candidate exact-byte acceptance is GREEN: 11 B41 Python, 34 browser assertions, 45 Node, 212 mutation/privacy, 754 passed / 3 skipped runnable, 68-file compile, config/release/maintenance GREEN, Sphinx 1220 / 3 / 5 / 62 missing-`sphinx`-only, 288-file clean two-root archive. Metadata-bearing prefinal remains next.

Metadata-bearing prefinal exact-byte acceptance is GREEN with the same 11 / 34 / 45 / 212 / 754+3 / 68 acceptance plane and 1220 / 3 / 5 / 62 missing-`sphinx` boundary. Final metadata freeze is ready; immutable final build and exact-final-byte acceptance are recorded externally without rewriting these bytes.
