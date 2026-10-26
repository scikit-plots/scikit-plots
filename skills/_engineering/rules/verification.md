# Verification and evidence

Categories: Verification, Evidence, Diagnostics, Gates, Analysis, Process, Proof

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 1: Verification - never change the tree while a verification run is in progress
- **When:** a build or `python -m libs._tools verify` run is executing against the working tree.
- **Then:** make no edit, `git stash`, checkout or regeneration until it exits. For a baseline comparison use `git worktree add <dir> HEAD` (a separate directory), never `git stash`.
- **Verified by:** the run's start and end times bracket no file modification (`git status` identical before and after).
- **Added:** 2026-10-06. Root cause: a `git stash`/`pop` during a matrix run changed the files being staged, so that run's result described no real state and had to be discarded.

### Rule 2: Verification - a wheel that was rebuilt must be reinstalled, not re-read from a cache
- **When:** installing a locally built wheel whose version did not change (`0.5.dev0` rebuilt).
- **Then:** pass `--refresh-package <name>` (uv) or `--no-cache-dir --force-reinstall` (pip) for every local distribution.
- **Verified by:** `grep` the changed line in the installed file under `site-packages` before trusting any result of that environment.
- **Added:** 2026-10-06. Root cause: uv caches by wheel file name; an unchanged version means an unchanged name, so the old build was installed and a fix looked ineffective.

### Rule 9: Verification - a long run goes to the background, never under the tool's time limit
- **When:** a build or verification can take longer than the command time limit (10 minutes).
- **Then:** start it with `nohup ... &`, write progress to a file, and poll. Afterwards run `python -m libs._tools unstage` before trusting `git status`.
- **Verified by:** no command of the session ends with "timed out".
- **Added:** 2026-10-06. Root cause: a foreground verify was killed mid-build by the time limit and left staged copies in `libs/*/`.

### Rule 12: Process - stop after two failed attempts at the same idea
- **When:** a second variant of one approach fails for the same underlying reason.
- **Then:** stop, state the measured facts, and choose the mechanism that needs no assumption.
- **Verified by:** no removed helper is left behind in code or tests (`grep`).
- **Added:** 2026-10-06. Root cause: after "lowest-direct" picked a broken pre-release I tried "test the final release instead", assuming finals carry correct metadata; scikit-learn 1.3.0 final does not. The assumption was never measured before the code was written.

### Rule 14: Verification - a fix found while a run is in progress is written down, not applied
- **When:** a failure of a finished leg is being investigated while later legs still run.
- **Then:** reproduce it in a scratch environment outside the tree, write the fix to a scratch file, and apply it only after the run has ended; then rerun every leg.
- **Verified by:** no file under the repository has a modification time inside a run's start/end window.
- **Added:** 2026-10-06. Root cause: Rule 1 was read as "do not regenerate or stash"; a one-line test fix "while I was there" changed a file that the running leg was staging. Repeat of Rule 1, so the run was stopped and restarted from the first leg.

### Rule 22: Verification - nothing reads or rewrites the working tree while a run reads it
- **When:** a comparison with an earlier state is wanted (`git stash`, `git checkout`, an edit "just to see") and any run started from the tree is still alive.
- **Then:** make the earlier state somewhere else: `git worktree add <dir> <commit>`, or `git show <commit>:<path> > <scratch file>`. Check `ps` for the run before touching the tree at all.
- **Verified by:** the run writes its process id to a lock file and removes it on exit; every command that edits the tree is `guard.sh && { ...; }`, where `guard.sh` exits non-zero while that process is alive. The braces matter: after `guard.sh && a; b`, `b` runs whatever the guard said. The guard reads the lock, not command lines: its first version searched `ps` for the run's name and refused because the editing command's own text contained it (Rule 8).
- **Added:** 2026-10-07. Root cause: Rule 1 named "change"; a stash and pop was read as "no change" because the tree ends as it began. A run that reads files in between sees the other state. It happened a third time in the same round (a fix applied while the matrix was on its second Python): a rule that depends on remembering it is not a check, so it is now a command.

### Rule 23: Verification - a probe that stops at the first missing package proves nothing beyond it
- **When:** modules are imported with only the base requirements installed, and a module's import fails on an optional package.
- **Then:** import them again with the optional packages installed, and fail on anything that still does not import.
- **Verified by:** the harness row "every shipped module imports with its optional packages".
- **Added:** 2026-10-07. Root cause: `import yaml` failed first in a module that also could not be imported on Python 3.8 and 3.9 for another reason (`X | Y` between classes at import time); the first failure was classed as "optional package" and the second was never reached.

### Rule 31: Verification - run the suite the way the repository runs it, too
- **When:** a test is added or changed in a part.
- **Then:** besides the installed-wheel run, run it from a checkout with the repository's warning policy (`-W error`), and run the tests that are left out of the wheel run because they need the repository (the eight `test_ignore` modules of sphinx-ext).
- **Verified by:** `python -m pytest scikitplot/<part> --confcutdir scikitplot/<part> -W error` from a checkout, and the `test_ignore` modules from the same place, both recorded with counts in the round's results (round 5: `maintenances/_repo/_maintenance/history/r05_second_ci.md`).
- **Added:** 2026-10-07. Root cause: the full test job failed on two things the harness cannot see: a layout rule enforced by a repository-only test, and a `ResourceWarning` that only `pytest.ini` turns into an error.

### Rule 32: Verification - a child process is started with the environment's interpreter first on PATH
- **When:** a harness runs tests in a virtual environment without activating it.
- **Then:** put the environment's `bin` (`Scripts`) directory first on `PATH` for the test process.
- **Verified by:** `verify._interpreter_first` and its tests; a harness that needs `tomllib` fails here with Python 3.10 first on `PATH` and no fallback.
- **Added:** 2026-10-07. Root cause: a Node harness starts `python`; locally that was a 3.11 from the machine while the environment under test was 3.9, so the test passed here and failed in CI.

### Rule 34: Proof - a confirmation must be able to fail when the claim is true
- **When:** a second observation is used to confirm a diagnosis (here: "if the pipe is the cause, flushing fails again").
- **Then:** check the case in which the claim holds and the confirmation still succeeds, before relying on it; if there is one, say that the rule is an inference and what it can get wrong.
- **Verified by:** `test_a_closed_reader_on_windows_after_an_unbuffered_write`.
- **Added:** 2026-10-07. Root cause: a large write is not buffered, so after it fails there is nothing left to flush; the confirmation passed and the error was reported as internal.

### Rule 42: Evidence - an identifier is copied from the source it names
- **When:** a run number, a commit, a version or a count is written into code, a note or a report.
- **Then:** obtain it with a command from the artefact (`grep -o 'actions/runs/[0-9]*' <log>`), paste it, and search the tree for it afterwards.
- **Verified by:** `grep -rn <identifier>` in the delivered files and the same identifier in the log.
- **Added:** 2026-10-07. Root cause: a run number was typed from the shape of the previous one into five files; found only because a later step read the log header. Corrected before delivery.

### Rule 44: Verification - when the log is cut off, emulate the platform's rules one at a time
- **When:** a CI log names only part of a platform's failures and the platform is not available.
- **Then:** switch on one documented behaviour of the platform per run; first show that the switch reproduces the failures the log does name (Rule 28); then read every other test it fails; say which failures no switch produced.
- **Verified by:** the calibration table in the task notes (named on Windows / reproduced here), per suite.
- **Added:** 2026-10-07. Root cause: in round 5 some 60 Windows failures were unnamed and nothing could be done about them without another CI run.

### Rule 47: Verification - a silent emulation switch proves nothing until it has failed something
- **When:** a switch of the emulation produces no failure and that silence is about to be reported.
- **Then:** first give the switch a known positive: a three-line case the real platform is known to fail. If there is none, report the switch as untested, not as clean.
- **Verified by:** each switch has a calibration row in the task notes, or the word "untested" beside it.
- **Added:** 2026-10-08. Root cause: the `enc` switch replaced the default of the pipe reader; Python 3.11+ resolves the default before building the reader, so the switch did nothing for pipes, and round 6 reported its silence. Repeat occurrence of Rule 28.

### Rule 53: Diagnostics - "ready" is proved by running the thing on a fixed input
- **When:** a report says an optional component can run (an engine, a model, a data package, a driver).
- **Then:** decide it with the same code path the real run uses, on a small fixed input; a package's presence, or a file's presence under a guessed name, is not readiness. Keep one function for the report and the run.
- **Verified by:** a test with the component installed and its data absent (and with data under an outdated name) asserts that the report and the run fail alike, with the same remedy.
- **Added:** 2026-10-09. Root cause: `doctor` read the package (`CP-093`), then the data's path under either of two names (`CP-100`); the run needed the data, under the new name.

### Rule 56: Analysis - a static check is calibrated on known-bad, known-good and every shipped input
- **When:** writing a rule that classifies code or data (a regex risk check, a linter rule, a validator).
- **Then:** run it on a table of cases that must fire, a table that must not, and on *everything the project ships*; a finding on shipped input is a false positive until proved otherwise. Where the property is measurable (run time), measure it and compare with each verdict.
- **Verified by:** `TestBuiltinsAreClean` and `probe_round26.py` (measured slow == reported, for every case).
- **Added:** 2026-10-10. Root cause: the first pattern-risk rule flagged the core EMAIL pattern `(?:\.[...]+)+`; it read only the part after the inner repetition and missed that the group's mandatory leading `.` separates repetitions. Running it on the built-ins found it before anything shipped.

### Rule 59: Analysis - a static check of a measurable property is fuzzed against the measurement
- **When:** a check predicts behaviour that can be measured (run time, memory, output size).
- **Then:** generate random inputs from a fixed seed, run the check, and measure every input it passes in a killable child; a passed input that misbehaves is a miss. Report precision on the flagged side too, and let it decide the default policy.
- **Verified by:** `evidence/probe_round26_fuzz.py` ends `TOTAL FAILURES: 0`.
- **Added:** 2026-10-10. Root cause: after the review's fixes, three classes of exponential pattern still passed (nullable groups read as separators, optional parts trading characters); hand-written cases had not covered them.

### Rule 75: Gates - a version gate hides a test environment's limit, never a product defect
- **When:** adding to `test_gated`, `test_ignore`, a `skipif`, or an `xfail`.
- **Then:** state why the *code* is right on that version; if the reason is a defect in shipped code, fix the code and keep the test running.
- **Verified by:** review of every `test_gated` comment; the feedback gate is removed and its tests run on 3.9.
- **Added:** 2026-10-11. Root cause: round 4 gated `test_app.py`/`test_service.py` to >=3.10 for the loop-bound lock; the defect shipped, and a new ungated test (`test_network_policy.py`) hit it.

### Rule 76: Diagnostics - a check that fails prints what differs, not only that it differs
- **When:** a test compares digests, exit statuses, or opaque values.
- **Then:** put the evidence in the message: a unified diff for drifted copies, a child's stderr for a non-zero exit (`check=False` + assert), the leaked names for leaked state.
- **Verified by:** `test_a_duplicated_contract_has_not_drifted` prints the `__all__` reflow diff on the uploaded tree; `test_cloudflare_receiver.py` asserts with node's stderr.
- **Added:** 2026-10-11. Root cause: the CI line said only "drifted" (two hashes) and "CalledProcessError"; the first was a formatter's reflow, the second could not be diagnosed from the log at all.

### Rule 87: Verification - a background run verifies a snapshot, and is checked by its process id
- **When:** a long run (`libs._tools verify`, a suite) is started in the background while work continues.
- **Then:**
  - Run it on a copy of the tree (`cp -a work verify-tree`, or `git worktree add`), never on the tree being edited.
  - Record its id when it starts (`echo $! > run.pid`).
  - Decide whether it is alive with `kill -0 "$(cat run.pid)"`, never with `pgrep -f <pattern>`.
- **Verified by:** the round's notes name the snapshot directory and the pid file of every background run.
- **Added:** 2026-10-12.
- **Root cause:**
  - `mem_map.pyx` was edited while a verify run was alive. Its builds had finished and the tested distributions do not contain `memmap`, so the result stands, but only by the timing.
  - `pgrep -f "libs._tools verify"` matched the shell that ran it, so a finished run was reported as running for two hours.
  - Rules 1, 3, 8 and 22 all repeated. They were rules to remember; the snapshot and the pid file are mechanisms.

### Rule 91: Evidence - a claim about a third-party behaviour names the versions it was checked on
- **When:** a comment or docstring states how NumPy, Cython, CPython or another dependency behaves.
- **Then:** check it on the oldest and the newest version the project allows, and write the boundary into the text ("NumPy 1.23+ …; older NumPy, still allowed on Python 3.8, …").
- **Verified by:** a `bytearray` resize under `np.frombuffer` (an export blocks resizing): NumPy 1.21.6 holds no export; 1.23.5, 1.24.4 and 2.5.4 hold one.
- **Added:** 2026-10-12. Root cause: Run 1 wrote "`numpy.frombuffer(m)` does not hold an export" from one observation. It is false for every NumPy a Python 3.9+ install can have, and true only for NumPy before 1.23.

### Rule 99: Verification - a marker file and `find -newer` list every file edited while a background run used the tree
- **When:** a suite or harness runs in the background on a tree that may still be edited (Rule 87).
- **Then:** `touch` a marker when the run starts; before recording its result, `find <tree> -newer <marker> -type f` and re-run the modules those files belong to, or re-run on a snapshot taken after the last edit.
- **Verified by:** the round's notes name the listed files and the re-run (117 passed) for increment 2a.
- **Added:** 2026-10-12. Root cause: Rule 87 was applied to the harness but not to the plain suites; three files were edited during them. Rule 87's reminder became a listing step.

### Rule 101: Verification - a recorded fingerprint is refreshed after the last edit to what it covers, and the final snapshot re-runs its checker
- **When:** a maintenance record holds a hash of a tree (`runtime_tree_fingerprint`, TRACKER inventory) and that tree is edited later in the same round (a review fix, a test change).
- **Then:** refresh the record after the last edit, and run every maintenance checker against the final snapshot as a delivery gate; a fingerprint quoted in notes names the snapshot it was computed on.
- **Verified by:** `maintenances/levenshtein` tests (8) and `check_contract.py` (PASS/PASS) run on the delivered tree; the 2b notes record the refresh.
- **Added:** 2026-10-12. Root cause: 2a refreshed the levenshtein fingerprint, then the review fixes (LV-001/LV-002) changed `_core.py` and the tests; the delivered STATE/EVIDENCE were stale and `test_structural_contract_is_current` failed in the applied tree. A repeat of Rules 87 and 99 for records rather than test runs.

### Rule 123: Gates - a missing base makes every old finding look new
- **When:** `gates.sh <tree> <base>` reports "NEW black diff" or new ruff findings in a file the increment barely touched.
- **Then:** check that `<base>` exists first; rebuild it from the base commit (`git archive <commit> | tar -x -C <dir>`), never from memory of what it held.
- **Verified by:** the 4b gates ran against `r34/base4a`, rebuilt from `92d5e91`; `apis_reference.py` and the sync test then show as pre-existing.
- **Added:** 2026-10-16. Root cause: `snap4a` had been removed; `[ -f "$base/$f" ]` failed silently, so the gate labelled pre-existing black diffs new.

### Rule 131: Analysis - a cProfile record covers one copy of a module that ran twice
- **When:** reading the cumulative time of a function, or the calls of a fixture, from a profile of a process that may execute a module file more than once (a test module imported as two names, a source module put on `sys.path` and also imported as a package, `spec_from_file_location` helpers, a failed import retried).
- **Then:** count module executions first (the recorder's `module_exec` / `reexecuted`, or the `exec` audit event); where a file ran twice, use the callee-edge totals, read the analyzer's "records that do not add up" and "not attributed" lines, and profile the test in a run of its own file to see its callees.
- **Verified by:** `test_a_module_executed_twice_is_flagged` (`.github/scripts/tests/test_ci_test_perf.py`): two copies of one file, the record shows 20 calls where 40 ran, and `work` is listed as inconsistent; on the uploaded `_externals` profile, run168's `base` fixture shows 37.1 s not attributed.
- **Added:** 2026-10-23. Root cause: the first reading of the `_externals` profile ranked `source_tree_sha256` as a 196 s hot spot from its edges while its record said 0.01 s. Why: `cProfile.Profile.snapshot_stats` stores one record per `(file, line, name)` (the last code object wins) and merges the edges of all copies. Why two copies: the security tests insert `tests/` and `_hf_spaces_proxy/security/` into `sys.path` and import test and source modules as top-level names beside the package names pytest uses. Why it was read anyway: nothing checked that a function's edges fit inside its own time.

### Rule 133: Diagnostics - one profiler per process: check before enabling another
- **When:** code enables a profiler (`cProfile.Profile().enable()`, a sampling profiler) inside a process that may already be profiled, such as a test session started with `python -m cProfile`.
- **Then:** enable it only when `sys.getprofile()` is None and the `sys.monitoring` profiler slot is free (3.12+); otherwise skip and record why. Never rely on catching an error: on Python 3.11 a second `cProfile.Profile` silently replaces the first, which stops recording; 3.12+ raises `ValueError`.
- **Verified by:** `test_a_profiled_session_is_not_replaced` (`scikitplot/_testing/tests/test__perf_telemetry.py`): under `python -m cProfile` the selected test's line says why it was skipped and the outer profile still holds the later tests (probe on CPython 3.11.17, 3.12.3, 3.13.17).
- **Added:** 2026-10-23. Root cause: the first design of per-test profiling would have enabled a profiler inside `python -m cProfile -m pytest`, the workflow already in use; a probe showed that 3.11 keeps no error and no outer profile, only the inner one.

### Rule 134: Evidence - a long run writes its evidence as it goes, and CI keeps it from failed jobs
- **When:** a test run or job can end early: a failure, a cancellation, a time limit.
- **Then:** write one complete record per unit of work as it finishes (one JSON line per test, flushed), so a killed process keeps what it measured; upload the evidence with `if: always()`; the analysis names the last unit that finished.
- **Verified by:** `test_lines_written_before_a_kill_survive` (SIGKILL during the second test: the first test's line is intact) and the analyzer's `incomplete-session` finding (`test_a_truncated_last_line_is_tolerated_and_counted`); the coverage workflow's shard upload is `if: always()`.
- **Added:** 2026-10-23. Root cause: the shard upload was `if: success()`, so a failed or timed-out shard left no JUnit report, coverage or timing; its duration estimate was never refreshed, and a six-hour time-out (the reason sharding exists) would leave nothing to say which test was running.

### Rule 135: Evidence - a benchmark comparison is trusted only after an A/A run of the same method finds nothing
- **When:** comparing timings of two versions, environments or policies (benchmarks, test runs, GC policies) to claim a speed-up or a slowdown.
- **Then:** first compare the method with itself: the same code on both sides, the same machine. Use a rule that needs both a fixed factor and a separation of whole rounds (worker processes): every round's median on one side beyond every round's median on the other, with at least 4 rounds a side. Samples within one process are not independent, so a rank test on samples finds changes between two runs of the same code, and a comparison of the extreme samples misses a 2x slowdown behind one outlier. Alternate the two sides (base, head, head, base) instead of running one after the other, and pool samples from more than one process. Run nothing else on the machine while measuring. Quote the A/A result next to the claim.
- **Verified by:** `ci_bench.py pair` with the same interpreter on both sides reports `0 slower, 0 faster` over the whole suite (TP-1); `test_a_real_slowdown_is_found_and_no_change_is_not` checks both directions.
- **Added:** 2026-10-23. Root cause: two `ci_bench.py run`s of the same code, one after the other, marked two results 10-17% faster: the whole machine was faster during the second run, and within-run samples were tighter than that drift. The first fix (samples of both sides must not overlap) missed a 2x slowdown behind one outlier sample (independent review); the second (Mann-Whitney on all samples) marked 6 of 25 results changed in an A/A run, because samples of one process are correlated. Separately, test runs made while other work shared the machine showed tests "slower" that a quiet rerun did not.

### Rule 161: Evidence - a test run's exit status is read from pytest, not from a wrapper that may keep it
- **When:** pytest runs under another program (`python -m cProfile`, `coverage run`, a script) and its exit status is recorded or decides a result.
- **Then:** check that the wrapper returns pytest's status on a failing test before relying on it; for a profile, wrap `pytest.main` (`profile.runcall(pytest.main, args)`, then `sys.exit(int(status))`) instead of `python -m cProfile -m pytest`.
- **Verified by:** the recipe, run on a one-test suite that fails, records `1` (`make test-suite-evidence` with that test); a profile file is still written.
- **Added:** 2026-10-26. Root cause: the local `release-evidence.zip` of 2026-10-25 held `pytest-exit-code.txt` = 0 next to a `pytest.log` ending in `1 failed`. `python -m cProfile` catches the `SystemExit` of the program it runs and exits 0 (Python 3.8.20, 3.11.17, 3.12.3, 3.13.17: a script exiting 3 gives 0), so `PIPESTATUS[0]` was always 0.
