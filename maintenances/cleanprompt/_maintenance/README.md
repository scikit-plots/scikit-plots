# `maintenances/cleanprompt`

Developer-only plane. Nothing here is imported by runtime code, and an
architecture test enforces that.

| File | What it holds |
|---|---|
| `DESIGN.md` | Big picture: goals, data flow, invariants, failure modes, defect register, tier contract |
| `FRESH_CHAT_HANDOFF.md` | Where to start on a cold read |
| `SUBMODULE_STRUCTURE.md` | The file layout contract |
| `FAMILY.md` | Which module owns which question |
| `VERIFICATION.md` | The verification ladder and what each lane proves |
| `MAINTENANCE_MODEL.md` | How this plane is kept honest |
| `HISTORY.md` | What happened and when |
| `STATE.json` | Current status per plane |
| `EVIDENCE.json` | Per-lane evidence with artifacts |
| `evidence/` | The logs and the probe scripts that produced them |
| `tools/`, `tests/` | Contract checker, reviewer and their mutation tests |

The defect register and the plane-level status live one directory up, in
`REVIEW.json` and `MAINTENANCE.json`.

Three probe scripts in `evidence/` produce the logs that the lanes cite, and
they are not interchangeable: `probe_isolation.py` proves that no third-party
package loads, `probe_negative.py` runs the randomized scale probe that found
`CP-015` and `CP-016`, and `probe_engines.py` exercises real spaCy and real
NLTK. Only the last needs an engine installed, and it exits cleanly when none
is. A lane nobody ran stays `UNAVAILABLE`; `python_platform_matrix` is the one
currently in that state, and `VERIFICATION.md` says why.
