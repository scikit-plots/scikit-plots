# Patterns and metrics

Recurring patterns behind the rules in [`rules/`](rules/) and the
review counts per round, moved verbatim from `tasks/lessons.md` at
471e46c. Append new observations under the matching heading.

## Pattern Analysis
- Pattern: process-wide state changed by a test and observed by a later, unrelated one (root logger, docutils registries).
- Occurrences: 2 (round 30 root logging, round 32 Sphinx registries). A third form exists but is latent: under pytest < 9, a conftest outside `rootdir` gets base id `""`, so the AI-assistant package's autouse fixtures (and its service root handler) apply to every package; `verify.py` runs with `cwd` = the environment, which keeps `rootdir` above `site-packages`.
- Root Cause: the victim fails, not the culprit; a guard must fail the test that changed the state.
- Pattern: a declaration (dependency floor, Python floor, licence, "pure Python") that nothing executes.
- Occurrences: 8 (scikit-learn floor, Python floors, missing extras named by a CLI hint, two root floors that cannot be combined, a test suite's Python floor; round 4: Python 3.8 for four Sphinx extensions, `n_jobs` behind a macro no build defined, a script that exits on import).
- Root Cause: metadata is written once by hand and only the newest environment is ever installed.
- Pattern: two paths that must compute the same thing, written separately and never compared.
- Occurrences in round 33, all silent wrong answers that every existing test passed:
  - sparse vs dense BM25;
  - native vs Python edit distance (hash vs equality);
  - the builder's embedding vs the engine's;
  - three RRF implementations with different tie orders (reported, Run 2).
- Root Cause: the second path was added for speed or convenience, and each path got its own example test, not a test against the other.
- Prevention: one implementation, or a randomised differential test that holds the paths equal (Rules 79, 83).
- Pattern: a declaration that nothing executes. Round 33 adds 2 occurrences (10 in all): a provenance guard whose tag no producer set (Rule 82), and CLI capability declarations that nothing enforces (reported, Run 2).

- Pattern: an input or output path tested only with the shapes the author had in mind. Round 33 increment 2a: a pandas `Series` row (Rule 93), a broken rather than absent optional library (Rule 94), line-break characters and aliases in reports (Rule 95), lazy results (Rule 96). All four were found by the independent review, none by the author's tests.
- Root Cause: tests were written from the implementation's branches, not from the set of values the public signature admits.
- Prevention: for each public input, list the admitted types and the format's full value space before writing tests; fuzz where the space is large.

## Effectiveness Metrics
- Total lessons: 120
- Round 33, Run 3 / increment 3a (names and configs): independent review found 14 issues (4 medium, 10 low); 13 fixed, 1 documented; 22 tests fail on the reviewed version. F1 and F2 are Rule 116's subject.
- Round 33, Run 2 / increment 2g (fallback vocabulary): independent review found 13 issues (1 high, 6 medium, 6 low); 10 fixed, 3 kept and documented (F3, F8, F11); 16 tests fail on the reviewed version. The high one is Rule 115's subject.
- Round 33, Run 2 / increment 2f (one filter): independent review found 10 issues (1 high, 4 medium, 5 low) plus one design point; all fixed, 18 tests fail on the reviewed version. The high one and two medium ones are Rule 114's subject.
- Round 33, Run 2 / increment 2e (one rank fusion): independent review found 10 issues (2 medium, 8 low), all fixed; 16 tests fail on the reviewed version. The two medium ones are Rule 112's subject.
- Round 33, Run 2 / increment 2d (durable identities, UUIDv7):
  - Independent review found 12 issues (1 high, 4 medium, 7 low); all fixed, 23 tests fail on the reviewed version.
  - The high one is Rule 108's subject: a format change refused by the parser and swallowed by every reader's "skip malformed" loop.
  - Process repeat: Rule 8 (a `pkill -f` pattern on the same command line killed the shell). Successor mechanism: suites run through a script that never kills; a stop uses the recorded pid only.
- Round 33:
  - Product near-repeats: Rule 68 (validation tested at one entry point; Rule 84 covers all of them) and Rule 41 (a guard's fixture tagged by hand; Rule 82 builds it through the producer).
  - Seven new product subjects: native equality, threshold boundaries, native allocation, cache identity, scoring paths, fallback provenance and native-memory lifetime. Each one has a test that fails on the uploaded code.
  - Process repeats: Rules 1 and 22 (the tree was edited during a run) and Rules 3 and 8 (a process pattern matched its own command). Their successor is Rule 87, a mechanism instead of a reminder.
  - Two first fixes were incomplete and were completed in the same round after a probe:
    - the semantic chunker covered the result but not each chunk, nor a failing model;
    - the memmap fix covered views but not GIL-free methods or `mprotect`.
  - Rules 85 and 86 now name the whole surface.
  - An independent review of Run 1 (Rule 92) found 12 issues.
    - 8 are fixed with tests that fail on the reviewed version.
    - 3 were documentation or changelog errors and are corrected: the NumPy `frombuffer` claim (Rule 91), the declared-model precedence, and the free-threading caveat in the memmap fragment.
    - 1 is user misuse, now documented: calling `arr.base.release()` on an array in use.
    - Fixing them found one more: `model_revision` was recorded but never loaded.
- Round 33, increment 2c (document export and load-back):
  - Independent review found 9 issues (2 medium, 7 low); all fixed, 41 tests fail on the reviewed version.
  - Process repeat: Rule 87. A test file was copied into the CI-harness snapshot while the harness ran, so two wheel-equality checks failed; the run was repeated on a fresh, untouched snapshot. A snapshot is never edited; a change means a new snapshot.
  - Process repeat: Rule 52. The file tool decoded `\u2028`/`\u2029` escapes in a new test into the characters, and the character scan (Cf/Zs only) missed them; the scan now covers Zl, Zp and Cc, and found nothing else in any file of the round.
- Round 33, increment 2b (LLM judges, answer harness, agreement):
  - Independent review (Rule 92) found 10 issues (3 medium, 7 low); all fixed, 26 tests fail on the reviewed version.
  - The old-SDK run found what tests against one SDK version could not (Rule 102).
  - Process repeat: Rules 87/99 in a new form: a maintenance fingerprint refreshed before the last review fix shipped stale in 2a (Rule 101).
  - Rule 54 held: two guide values written before execution were wrong (0.75, 0.818) and were corrected by running the page.
- Round 33, increment 2a (evaluation harness, scorers, reports, rankers):
  - Independent review (Rule 92) of new, non-native code found 21 issues (1 high, 6 medium); all fixed, each with a test that fails on the reviewed version except one wording correction.
  - Process repeats: Rule 52 (a literal NBSP and a combining accent reached a test module; rewritten as escapes), Rule 54 (three comment values in the new guide were wrong until the blocks were executed), Rule 87 (files edited during background suites; now Rule 99), Rule 91 (a copyright year cited from memory, checked against the commit).
  - User correction: fragments reused round 32's PR number (864); renamed to 865 (Rule 100).
  - The CI harness found one more: a marker registered only by the root conftest warned in the partial distribution (fixed in `corpus/conftest.py`).
- Round 32: Rule 69 repeated (process-wide state, now docutils instead of logging): written as words, applied to one package; Rule 73 carries an autouse guard over every extension. Rule 75 retires a gate that hid a shipped defect.
- Round 31: no repeat; a semantic bug in the new matcher (elif chain over one object's keys) was caught by writing a test for each documented labeler rule.
- Round 30: Rule 66's subject recurred through delivery (the renamed file could not be deleted by the archive), now Rule 67 with a CI check.
- Round 29: Rule 62 repeated in product code (SQLite connections), now Rule 65 with the exact idiom.
- Round 26: Rule 52 repeated (escapes written through the file tool arrived as the characters in a new module, its tests and a docstring); caught by the Rule 52 scan before any test ran. The scan now runs on every file written in a round before the ladder.
- Repeat occurrences: 8 (Rule 3 twice before it was written; once more in the form Rule 8 now covers; Rule 1 once, now Rule 14; Rule 1 twice more in round 4, now Rule 22 and a guard script; in round 5 a fix for one reporting mistake made the opposite one, now Rule 33; in round 6 Rule 24's subject, text files across platforms, returned as line endings instead of encodings, now Rule 38; in round 7 Rule 24 again, for pipes, now Rule 45, and Rule 28, for a switch that was never shown a positive, now Rule 47)
- Trend: Windows failures per run: 11 rows, then 5 rows (62 tests), then 1 row (2 tests). The two repeats of round 7 are both rules that were written and then applied too narrowly; their successors carry a mechanical check.
- Increment 4R (2026-10-16/17): 125 rules after Rules 124 and 125; the total above was counted at 120.
