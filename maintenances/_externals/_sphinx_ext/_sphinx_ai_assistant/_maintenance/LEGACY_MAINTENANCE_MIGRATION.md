# Migration map for existing `_maintenance/` artifacts

The anchored source already contains useful maintenance/research files. Do not
delete them blindly when overlaying this package.

| Existing artifact | Recommended disposition |
|---|---|
| `CONFIG_ARCHITECTURE.md` | keep as domain detail, but reconcile every target claim with current tracker status |
| `SECURITY_FINDINGS_INDEX.md` | keep until B02 revalidation absorbs each finding into live registry/checkpoints |
| `discovery_contract.json` | keep; pair with a formal schema and make it a tested client/server contract |
| `EXT_SETTINGS.md` | keep as active prototype/design evidence until B10 adjudicates promotion |
| `_ext_settings.js` / `test_ext_settings.js` | keep as maintenance prototype/test asset until explicitly promoted |
| `FREE_PROXY_SOLUTIONS.md` | move to `history/` once live routing decisions no longer depend on it; research is not current architecture truth |
| screenshots/mockups | history/design evidence only |

## Migration rule

Do not create duplicate renamed copies such as `CONFIG_ARCHITECTURE_V2.md`.
Either update the durable live document or archive the superseded evidence with a
short pointer from `HISTORY.md`.

## Missing-evidence rule

If a live index references a “full archive” or plan that is not present in the
actual repository snapshot, record that as a maintenance finding. A fresh chat
must not depend on an inaccessible previous conversation to recover security
evidence.
