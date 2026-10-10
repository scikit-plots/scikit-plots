# Resume here

This file is the working log of the round in progress. It exists so that a
fresh chat — one with no history, no transcript and no memory of the last
session — can continue the work exactly where it stopped, without redoing a
step or trusting a claim nobody re-ran.

`FRESH_CHAT_HANDOFF.md` explains the subsystem. This file says **where the
work is**. Read this one first, then the handoff.

## How to use this file

1. Read **Current round** and **Step log**. A step is done only when its
   evidence column names a command that was run and what it printed.
2. Re-run the **Baseline commands** before touching anything. If a number
   differs from **Last verified numbers**, the tree changed since this file was
   written: find out why before continuing.
3. Do the **Next action**. When a step finishes, update its row *in the same
   change* as the code — a log written afterwards from memory is a claim with
   no evidence behind it.
4. Keep **Open ledger** equal to the notes under
   `upcoming_changes/scikitplot/cleanprompt/`. `tests/test_resume.py` fails
   when they disagree, so a note cannot be forgotten and a closed one cannot
   linger here.
5. At the end of a round, move the round's summary to `HISTORY.md`, reset the
   step log for the next round, and leave **Next action** pointing at it.

## Baseline commands

Run from the wide checkout root (the directory holding `scikitplot/`,
`maintenances/` and `skills/`).

```sh
python -B maintenances/cleanprompt/_maintenance/check_trackers.py --json
python -B -m pytest maintenances/cleanprompt/_maintenance/tests -q -p no:cacheprovider
python -B -m pytest scikitplot/cleanprompt/tests -q -p no:cacheprovider
python -B -m pytest docs/source/user_guide/cleanprompt -q -p no:cacheprovider
```

The package suite needs the parent `scikitplot` importable. In a source
checkout without a compiled build, run it under a stand-in parent: a directory
holding `scikitplot/__init__.py` (empty) and `scikitplot/cleanprompt` (a copy
or link of the real package), with a `pytest.ini` that sets
`filterwarnings = error`. That is how rounds 23–25 were verified, and
`tests/_isolated.py` measures import isolation the same way.

**Never edit the tree while a run reads it** (`tasks/lessons.md`, rules 1 and
22): verify against a frozen copy, or wait for the run to finish.

## Current round

Round twenty-five is **complete** (2026-10-10): *truthful readiness, safe
deployment files, a Unicode detection view, and a guide of several pages*.
Summary in `HISTORY.md`; plan and results in `tasks/todo.md` under
"cleanprompt round 25"; reasoning in `DESIGN.md` section 24. Findings
`CP-093` … `CP-103` are closed in `REVIEW.json`.

Inputs that are **not** in the tree and need not be re-fetched: an internal
review (`scikit_plots_cleanprompt_review_2026-10-10`, CP-NEW-01..08), an
external comparison (`cleanprompt_external_research_2026-10-09`) and a
ChatGPT planning log. Every finding taken from them was reproduced on the tree
first (`evidence/probe_round25.py`).

## Step log

| # | Step | Status | Evidence |
|---|---|---|---|
| 1 | Plan, this file, skill read-first, continuity test | done | `tests/test_resume.py` 6 passed; `RESUME.md` required by `check_contract.py` |
| 2 | CP-093 readiness, CP-094 web builder | done | `test__engines.TestReadiness`, `test__cli.TestDoctorAgreesWithTheRun`, `test__app.TestEntityDetectionUsesTheSharedBuilder`; live: spaCy 3.8.16 without model and NLTK 3.10.3 without data → `doctor --ner` healthy=False with the download remedy, `inspect --ner` exit 69 with the same; with model and data → healthy=True for spacy, nltk, both |
| 3 | CP-095/096/097 container files, debug guard | done | `test__serve.TestContainerFilesDeriveFromTheRuntime`, `TestDebugIsLoopbackOnly`; `probe_round25.py` lines CP-095..097 |
| 4 | CP-098 detection view | done | `_canonical.detection_view`, `Redactor._view_spans`; `test__canonical.TestDetectionView`, `test__engine.TestDetectionView` (every pattern example salted); probe `V25` 1000 salted values, 0 failures |
| 5 | Regression tests and negative probes | done | `test_regressions.TestCP093..TestCP099`; `probe_negative.py` TOTAL FAILURES: 0; CP-099 (reinstall hint drift) found and fixed on the way |
| 6 | Gallery | done | moderate (readiness), advanced (detection view, "nine core runtime invariants"), recipes (loopback files, debug refusal), packs (pattern trust), README.txt; `probe_gallery.py` 9/9 ok bare and with every tier (the sandbox hides NLTK data: moderate reports a remedy-bearing SKIP for nltk/both) |
| 7 | Multi-page user guide, README inventory | done | index + 10 pages under `docs/source/user_guide/cleanprompt/`, every example executed against the CLI/API; `test_user_guide_sync.py` 9 passed (all pages, toctree, contract phrases, no hand count); README: grouped commands, real `encode`/`decode` transcript, `en_core_web_sm`; `_maintenance/tests/test_documented_cli.py` (CP-101) 3 passed |
| 8 | upcoming_changes ledger | done | README note removed (implemented); `_cli` hint implemented (`scikitplot/_cli/tests/test_registry.py` 3 passed), note `blocked` on a PR number; release fragments drafted in `round-25-release-fragments.md`; six new `open` notes for the next slices |
| 9 | Verification ladder, evidence refresh | done | see **Last verified numbers**; `check_trackers.py`: maintenance PASS, runtime PASS, release UNVERIFIED (lane 27, Windows/macOS, not measured); reviewer exit 0 |
| 10 | Independent review, drop-in package | done | two fresh agents (runtime, docs) started; both were cut off by a rate limit, and the runtime agent's last lead was followed up by hand: it was real (`CP-102`), and testing the fix found `CP-103`. The docs agent produced no findings before stopping; every guide example had been executed by hand. Drop-in: `scikit_plots_cleanprompt_round25_dropin.zip` |

## Last verified numbers

Final round-25 tree, 2026-10-10 (frozen copy identical to the work tree):

- package suite, no optional tier: 2654 passed, 88 skipped (3 runs and 2
  shuffled orders agree; also with live logging at INFO)
- spaCy 3.8.16 and NLTK 3.10.3 installed, no model and no data: 2665 passed,
  77 skipped
- every tier (model and data present): 2732 passed, 10 skipped
- CPython 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14 (pytest only): all green
- probes: negative (no tier, every tier), isolation, fuzz (4000 documents),
  scale, live engines, round 25 — 0 failures; gallery 9/9 in both installations
- maintenance plane: 88 passed; guide sync: 9 passed
- `check_trackers.py`: maintenance PASS, runtime PASS, release UNVERIFIED
- project CLI `_cli` suite in a stand-in parent: 155 passed (+3 new); its 23
  failures are environmental and identical on the original tree

## Next action

Round twenty-six. Start read-only, as the ChatGPT planning log proposed, with
an evidence pack before code — but scoped to the open notes rather than to the
whole subsystem again (round 25 already re-proved the rest):

1. **Detectors + names together** (the log's "detectors + NER + policy"
   target, narrowed by what round 25 measured):
   `invisible-characters-in-keyvalue-and-code-names.md`,
   `email-addresses-with-non-ascii-letters.md`,
   `phone-pattern-groupings-and-ranges.md`. For each detector touched, record
   kind → implementation → validator → known limits → attack cases → change →
   evidence, as the log suggested.
2. `restore-alters-near-placeholders-in-the-source.md` (restoration
   adversarial review).
3. Then the capability notes, in order: custom-pattern run time, surrogate
   generator slice A, per-kind actions, vehicle/locale packs, look-alike
   characters, coverage matrix.

Reset the step log above for round 26 before starting.

## Open ledger

Every note under `upcoming_changes/scikitplot/cleanprompt/`, with its status.
`tests/test_resume.py` compares this list with the directory. Suggested order
for the next rounds: invisible characters in key-value/code names, email
addresses, PHONE groupings, near-placeholders on restore, custom-pattern run
time, surrogate generator (slice A then B), per-kind actions, vehicle/locale
packs, look-alike characters, coverage matrix.

- `upcoming_changes/scikitplot/cleanprompt/round-25-release-fragments.md` — blocked
- `upcoming_changes/scikitplot/cleanprompt/email-addresses-with-non-ascii-letters.md` — open
- `upcoming_changes/scikitplot/cleanprompt/guard-custom-pattern-runtime.md` — open
- `upcoming_changes/scikitplot/cleanprompt/customizable-surrogate-generator.md` — open
- `upcoming_changes/scikitplot/cleanprompt/per-kind-action-policy.md` — open
- `upcoming_changes/scikitplot/cleanprompt/vehicle-and-locale-identifier-packs.md` — open
- `upcoming_changes/scikitplot/cleanprompt/look-alike-characters-in-detection-view.md` — open
- `upcoming_changes/scikitplot/cleanprompt/generated-coverage-matrix.md` — open
- `upcoming_changes/scikitplot/cleanprompt/invisible-characters-in-keyvalue-and-code-names.md` — open
- `upcoming_changes/scikitplot/cleanprompt/phone-pattern-groupings-and-ranges.md` — open
- `upcoming_changes/scikitplot/cleanprompt/restore-alters-near-placeholders-in-the-source.md` — open

Outside this directory but owned by this round:
`upcoming_changes/scikitplot/_cli/align-cleanprompt-optional-tier-install-hint.md`
(blocked on a PR number for its fragment).

## Decisions waiting for the maintainer

1. A pull-request number, to turn the drafted fragments into files.
2. For `guard-custom-pattern-runtime.md`: refuse nested unbounded patterns in
   custom packs by default (with an acknowledgement flag), or warn only.
3. For `customizable-surrogate-generator.md`: whether slice B (a Python
   provider protocol) is wanted, or data sets only.
