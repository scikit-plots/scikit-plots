<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: cleanprompt round 25 - truthful readiness, safe deployment files, Unicode detection view, multi-page guide

> Fresh chat? Read `maintenances/cleanprompt/_maintenance/RESUME.md` first: it holds the
> step-by-step state of this task, the last verified numbers and the next action, so the work
> continues without this chat's history.

### Context
- Goal: make every surface of `scikitplot.cleanprompt` say what it actually does before adding
  capability (the review's governing principle), then close one leak class and expand the docs.
- Inputs (read-only, not in the tree): `scikit_plots_cleanprompt_review_2026-10-10` (internal
  review, findings CP-NEW-01..08) and `cleanprompt_external_research_2026-10-09` (external
  comparison; a backlog, not a target to clone).
- Affected: `scikitplot/cleanprompt/{_engines,_diagnostics,_app,_serve,_cli,_canonical,_engine}.py`
  and tests; `galleries/examples/cleanprompt/`; `docs/source/user_guide/cleanprompt/`;
  `maintenances/cleanprompt/`; `skills/cleanprompt/SKILL.md`; `upcoming_changes/`.

### Baseline (measured 2026-10-09 on the uploaded tree, frozen copy, stand-in parent)
- Suite: 2468 passed, 79 skipped (spaCy and NLTK absent), 38.6 s.
- `check_trackers.py --json`: runtime PASS, maintenance FAIL, release BLOCKED -
  `EVIDENCE runtime_tree_fingerprint does not match` (review CP-NEW-01 confirmed).
- Reproduced on the untouched tree (`repro_p0.py`, see RESUME.md):
  - `container_files(with_ner=True)` installs `en_core_web_lg`; runtime default is `en_core_web_sm`.
  - Dockerfile run comment publishes `-p 5000:5000` (all interfaces); compose uses `127.0.0.1:`.
  - compose sets `CLEANPROMPT_NER: "1"`, which no code reads.
  - `resolve_bind` has no notion of `debug`; `--docker --debug` reaches `app.run(debug=True)` on 0.0.0.0.
  - `create_app(enable_ner=True, ner_engine="nltk", language="tr", model_size="lg")` calls
    `spacy_detector(model=None)`.
  - `encode()` leaves `ada\u200b@example.com`, full-width addresses, `+1 555\u00a00100`,
    `+1‑555‑0100` and `192.0.2.\u200b10` unredacted; `4111\u200b1111 1111 1111` becomes
    `4111\u200b[PHONE-1]` (card prefix leaked, rest mislabelled).

### Design decisions (what / why)
- **Readiness is not installation (CP-093).** An engine is *ready* when its tier is installed,
  it supports the language, and its assets exist. spaCy assets are checked from distribution
  metadata (no import); NLTK corpora can only be checked by importing NLTK, so that check runs
  only where the caller is about to load NLTK anyway (`check_assets=True`: explicit requests,
  `doctor`, the web app with NER on). `auto` never selects an engine known to be unready;
  `build_detectors(required=True)` refuses any requested engine that is not ready, naming each
  engine's reason and remedy. One function decides, so `doctor`, `inspect`, `encode` and the web
  app cannot disagree.
- **One builder (CP-094).** `create_app` builds entity detectors through `build_detectors`, the
  same call the CLI and API use; no web-specific construction remains.
- **Generated files derive from the runtime (CP-095, CP-096).** The model the image installs is
  `resolve_model(language, size)`, and the image passes it explicitly (`--ner-engine spacy
  --ner-model <model>`), so a later default change cannot split them. Every generated launch
  line publishes on `127.0.0.1`. Dead configuration is removed, not documented.
- **Debug is refused off loopback (CP-097).** Werkzeug's debugger executes code; the app has no
  authentication. `resolve_bind(..., debug=True)` raises for any non-loopback host, including
  container mode.
- **Detection view (CP-098).** Structural and literal detectors also run over a *view* of the
  text: format characters (Unicode category Cf: zero-width, bidi controls, soft hyphen) removed
  and each remaining character folded by the same same-length table `_canonical` already uses
  for "the same value" (ASCII control whitespace kept, so lines stay lines). View spans are
  mapped back through an offset map onto the original text and joined to the original-text
  spans before overlap resolution. The source is never rewritten; the vault holds the original
  surface, so restoration is exact. Adding spans can only widen redaction (overlaps merge).
  Entity engines keep reading the original text. No view pass when the view equals the text.
- **Docs.** The single 649-line page becomes a guide of several pages; the sync test reads all
  of them. The README stops carrying a hand-counted command total.
- **Continuity.** `RESUME.md` in the maintenance plane is the step log a fresh chat starts from;
  a maintenance test fails when it and the `upcoming_changes` ledger disagree.
- **Not this round (recorded as upcoming_changes notes with full designs):** customizable
  surrogate generator, per-kind action policy, VIN and locale packs, custom-regex time guard,
  generated coverage matrix.

### Implementation Steps
- [x] 1. Plan, RESUME.md, skill read-first, continuity test (Est 30 min)
- [x] 2. CP-093 readiness + CP-094 web builder, tests first (Est 90 min)
- [x] 3. CP-095/096/097 container files and debug guard, tests first (Est 45 min)
- [x] 4. CP-098 detection view, tests per class first (Est 120 min)
- [x] 5. Regression tests + negative probes for CP-093..CP-098 (Est 45 min)
- [x] 6. Gallery updates (Est 60 min)
- [x] 7. Multi-page user guide + README inventory + sync test (Est 120 min)
- [x] 8. upcoming_changes: close the two implemented notes, add next-slice notes (Est 45 min)
- [x] 9. Verification ladder; refresh REVIEW/STATE/EVIDENCE/HISTORY/DESIGN/VERIFICATION (Est 90 min)
- [x] 10. Independent review by a fresh agent; drop-in package (Est 45 min)

### Acceptance Criteria
- [x] With spaCy installed and no model, `doctor --ner` is not healthy and names the model
      download; `inspect --ner` fails with the same remedy; `auto` does not pick spaCy.
- [x] With NLTK installed and no corpora, `doctor --ner --ner-engine nltk` is not healthy and
      names the corpora; `inspect` agrees.
- [x] `create_app` constructs exactly the detectors `build_detectors` would for its arguments.
- [x] Generated image: installed model == requested model; every launch line is loopback.
- [x] `--debug` with any non-loopback bind exits with a clear refusal.
- [x] Every obfuscation in the baseline list is redacted and restores byte-exact; ordinary text
      produces byte-identical output to the baseline.
- [x] Suite green in repeated runs and both frontends; contract checker maintenance + runtime PASS.
- [x] Gallery executes hermetically; guide sync test covers every page.

### Risks & Mitigation
- Risk: readiness check makes `build_detectors` import NLTK where it did not → Mitigation:
  only under `check_assets=True`; `test_construction_imports_nothing` stays.
- Risk: view pass changes output for ordinary text → Mitigation: no pass when view == text;
  byte-identity test against fixed expectations; additive spans only.
- Risk: test stubs assumed "installed == ready" → Mitigation: rewrite those tests to the
  corrected contract (a stubbed model), never delete.

### Verification Steps
- Focused files, then whole suite x3 + shuffled order; engines venv with and without assets;
  probes (`probe_negative`, `probe_isolation`, `probe_gallery`); `check_trackers --json`.

## Results Review - cleanprompt round 25 - 2026-10-10

### Implementation Summary
- Findings closed: `CP-093` … `CP-103` (six from the internal review, each
  reproduced first; five found by this round's own verification, one of which —
  `CP-102` — the round introduced and its independent review caught).
- Runtime files changed: `_engines.py`, `_nltk.py`, `_ner.py`, `_canonical.py`,
  `_engine.py`, `_detectors.py`, `_packs.py`, `_diagnostics.py`, `_cli.py`,
  `_app.py`, `_serve.py`, `_custom.py` (docstring), `README.md`; project CLI:
  `scikitplot/_cli/registry.py`.
- Tests: new or extended in `test__engines`, `test__cli`, `test__app`,
  `test__serve`, `test__canonical`, `test__engine`, `test__nltk`, `test__ner`,
  `test__diagnostics`, `test__runtime`, `test_regressions`, `tests/_tiers.py`;
  `scikitplot/_cli/tests/test_registry.py` (new); maintenance
  `test_resume.py`, `test_documented_cli.py`, `test_review.py` (new).
- Docs: guide split into index + ten pages; gallery: moderate, advanced,
  recipes, packs, README.txt.
- Maintenance: `RESUME.md` (new), DESIGN §24, REVIEW, STATE, EVIDENCE,
  VERIFICATION lanes 42–45 and lane 13, HISTORY, handoff, skill, reviewer
  staleness check; ledger: two notes closed or blocked, eleven added.

### Verification Evidence
- Suite: 2654 passed / 88 skipped (no tier; 3 runs + 2 shuffled + live
  logging agree), 2665 / 77 (engines without data), 2732 / 10 (every tier).
- CPython 3.8–3.14: green.
- Probes: negative (both installations), isolation, fuzz 4000, scale, engines,
  round 25: 0 failures. Gallery 9/9 twice. Maintenance 88 passed. Guide 9 passed.
- `check_trackers.py`: maintenance PASS, runtime PASS, release UNVERIFIED.
- Logs: `maintenances/cleanprompt/_maintenance/evidence/*.log` (round 25 headers).

### Deviations from Plan
- `CP-099`…`CP-103` were not in the plan; each was found by executing a step
  (live engines, executing every doc example, the independent review).
- The independent review agents were cut off by a rate limit; the runtime
  agent's last lead was followed by hand and was a real regression (`CP-102`).
- Six live-model tests that failed with spaCy installed but no model (present
  on the uploaded tree) now skip on readiness with the remedy in the reason.

### Technical Debt / decisions for the maintainer
- Towncrier fragments drafted, blocked on a PR number
  (`upcoming_changes/scikitplot/cleanprompt/round-25-release-fragments.md`).
- Open, with designs: key-value/code names and JSON strings through the view;
  non-ASCII email addresses; PHONE groupings; near-placeholders on restore;
  custom-pattern run time; surrogate generator; per-kind actions;
  vehicle/locale packs; look-alike characters; coverage matrix.
- Decisions: refuse vs warn for nested unbounded custom patterns; whether the
  surrogate generator gets a Python provider protocol (slice B).

### Approval Checklist
- [x] Meets all acceptance criteria
- [x] No regression: every pre-existing test passes; changed expectations
      rewritten to the corrected contract, none deleted
- [x] Documentation updated (guide, gallery, README, skill, maintenance plane)
- [x] Tests added for every finding, each able to fail on the old tree
- [x] Evidence refreshed only from lanes that ran on this tree
