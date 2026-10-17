<!-- Archived verbatim from tasks/todo.md at 471e46c (increment 4R).
     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py
     checks every archived part against records_manifest.json. -->

## Task: cleanprompt round 26 - pattern-risk check (warn by default, user-tuned), custom surrogate sets, generator design

### Context
- Goal: close `guard-custom-pattern-runtime.md` step 1 and
  `customizable-surrogate-generator.md` slice A, with the maintainer's
  decisions: PR number **864**; risky custom patterns are **always warned
  about**, with quick options and policy tuning (ignore / warn / refuse,
  per pattern, per run, per team, per machine); the generator scope is
  Claude's call — a sectioned, future-oriented design, then slice A.
- Rationale: one custom pattern could stall a run silently; surrogates were
  English-only and closed.
- Affected systems: `_pattern_risk.py` (new), `_surrogate_sets.py` (new),
  `_packs.py`, `_patterns.py`, `_custom.py`, `_plan.py`, `_runtime.py`,
  `_policy.py`, `_engine.py`, `_surrogates.py`, `_cli.py`, `__init__.py`,
  README, guide (5 pages), gallery (2 examples), maintenance plane, ledger.

### Implementation Steps
- [x] 1. Release fragments for round 25 under PR 864; promote the two notes
- [x] 2. `_pattern_risk.py`: parser, overlap by public `re`, three rules,
      `not-analysed`; calibration (EMAIL false positive found and fixed by
      reading round to the group's leading separator)
- [x] 3. Pack schema `risk: accepted` + `risk_reason`; `PatternSpec.risk_reason`
- [x] 4. Policy surfaces: `load_custom/with_custom(pattern_risk=)`,
      `CleanPlan.pattern_risk` (+ validate refuse, cleaner warn),
      `FluentCleanPrompt.pattern_risk`, `--pattern-risk` (5 commands, both
      frontends), stderr `warning:` block, `packs --check` listing
- [x] 5. `GENERATOR_DESIGN.md` (floor, invariants G1-G6, slices, locale,
      growth paths, rejected alternatives)
- [x] 6. Slice A: `_surrogate_sets.py`, `TagStyle.surrogates/surrogate_set`,
      engine refusal without the set, plan/CLI `--surrogates` (7 commands)
- [x] 7. `CP-105` (held two-word name shown with a dot) found via the gallery
      and fixed in `_engine._shows_held`
- [x] 8. Tests: `test__pattern_risk.py`, `test__surrogate_sets.py`,
      regressions CP-104/CP-105; flaky xdist test root-caused (Rule 55)
- [x] 9. Docs, gallery, README, fragments, ledger (2 promoted, 2 new open)
- [x] 10. Verification ladder, evidence refresh, drop-in

### Acceptance Criteria
- [x] Built-in patterns and packs report no finding; measured run time agrees
      with every verdict (`probe_round26.py`)
- [x] Default grammar, policy and plan digests unchanged (pinned)
- [x] Every pre-existing test passes; no expectation deleted
- [x] Both frontends agree; abbreviations refused
- [x] Every new doc example executed

### Risks & Mitigation
- Risk: false positives annoy users -> warn (never refuse) by default,
  acceptance in the pack, `ignore` mode; quadratic optional-separator case
  deliberately not reported.
- Risk: a set weakens safety -> fixed floor; set only proposes; run-time
  re-check of every candidate.
- Risk: new fields invalidate saved vaults/plans -> omitted while default;
  digests pinned.

## Results Review - cleanprompt round 26 - 2026-10-10

### Implementation Summary
- Findings closed: `CP-104` (pattern risk), `CP-105` (held name with a
  dot), `CP-106` (append mixed grammars), `CP-107` (OverflowError in pack
  validation); slice A of the generator.
- New modules: `_pattern_risk.py`, `_surrogate_sets.py`; changed: `_packs`,
  `_patterns`, `_custom`, `_plan`, `_runtime`, `_policy`, `_engine`,
  `_surrogates`, `_cli`, `__init__`, README; tests: `test__pattern_risk`,
  `test__surrogate_sets` (new), `test_regressions`, `test__runtime`.
- Docs: files_and_packs, how_it_works, python_api, troubleshooting,
  security_and_limits; gallery: packs, moderate; fragments: security
  feature/fix, cleanprompt feature/fix.
- Maintenance: `GENERATOR_DESIGN.md` (new), DESIGN §25, handoff rules 22–25,
  REVIEW, STATE, EVIDENCE lanes 46–47, VERIFICATION, HISTORY, RESUME, skill,
  lessons 55–57; probes `probe_round26.py`, `probe_round26_fuzz.py`; ledger:
  2 promoted, 3 new open notes.

### Verification Evidence
- Suite: 2888/88 (no tier; 2 runs, 2 shuffled orders, xdist), 2899/77
  (engines without data), 2966/10 (every tier); CPython 3.8–3.14 green.
- Probes: negative (both installations), isolation, fuzz 4000, scale,
  engines, round 25, round 26 (22 measured verdicts), round-26 fuzz: 0
  failures. Gallery 9/9 twice. Guide 9 passed.
- `check_trackers.py`: maintenance PASS, runtime PASS, release UNVERIFIED.

### Deviations from Plan
- The independent review (12 findings) and a hand re-run of its brief
  (`CP-107`, NFKC over-refusal) and a soundness fuzz (3 misses) each forced
  a re-plan of the analyser or the floor; recorded in DESIGN §25.
- A second review pass was cut off by a rate limit.

### Technical Debt / decisions for the maintainer
- Ordinary-word set entries (new note), runtime bound (new note), provider
  protocol (new note); GitHub handle for fragments.

### Approval Checklist
- [x] Meets all acceptance criteria
- [x] No regression; no expectation deleted (one test's sample value changed, root cause recorded)
- [x] Documentation updated
- [x] Tests added for every finding, each able to fail on the old tree
- [x] Evidence refreshed only from lanes that ran on this tree
