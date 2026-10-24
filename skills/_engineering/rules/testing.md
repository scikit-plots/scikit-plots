# Tests

Categories: Tests, Timing, Thresholds, Scanners, Shared defaults, Serialisation

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 13: Tests - a skipped module is not a passed module
- **When:** a test suite passes in a new environment.
- **Then:** compare the number of collected tests with a complete environment, and read the skip reasons (`-rs`), before calling it green.
- **Verified by:** the pass count in the report matches the count of the complete environment.
- **Added:** 2026-10-06. Root cause: the Sphinx extensions' suite "passed" with 5539 tests; 644 more were skipped by one module-level `importorskip("markdownify")`.

### Rule 15: Tests - a child interpreter that must import a copy runs without `site`
- **When:** a test starts `sys.executable` to import a package copy from a temporary directory.
- **Then:** pass `-I -S` and add the needed directories to `sys.path` in the child; never rely on `sys.path` order alone.
- **Verified by:** the test passes in an environment where the package is installed in editable mode (`pip install -e .`); locally, a `.pth` file that inserts a `sys.meta_path` finder for the package reproduces that environment.
- **Added:** 2026-10-07. Root cause: 21 tests passed in every environment I had and failed in CI, where the editable install's import hook answers `import scikitplot` before `sys.path` is consulted. I had never run them where the package was installed.

### Rule 16: Tests - assert a contract, not a cache
- **When:** an assertion uses `is` on objects produced by a standard-library factory (`typing.Literal[...]`, interned strings, small ints).
- **Then:** assert the property that matters (here: no `__doc__` stored on the alias) and compare by `==`.
- **Verified by:** the test passes after `for f in typing._cleanups: f()`.
- **Added:** 2026-10-07. Root cause: `Literal[...] is Literal[...]` is true only while a private cache holds the first object; the CI test session clears it.

### Rule 17: Tests - a "first result" assertion needs a strict winner
- **When:** a test asserts which element comes first by distance, score or order.
- **Then:** print the distances once and choose data where the expected first element wins by a margin; assert the margin too.
- **Verified by:** the test asserts `distances[0] != distances[1]`.
- **Added:** 2026-10-07. Root cause: all twelve angular distances from the zero vector were equal, so the asserted order was a tie-break that differed on macOS/arm64.

### Rule 28: Tests - an emulation must first be shown to refuse what the real platform refuses
- **When:** another platform's rule is applied by a fixture or a plugin.
- **Then:** write one test of the emulation itself (the forbidden operation fails, the allowed one works), and confirm the new tests fail on the old code.
- **Verified by:** `test_the_rule_is_applied_by_the_fixture`; 5 failed on the previous `_io.py`, 15 passed on the new one.
- **Added:** 2026-10-07. Root cause: an earlier plugin made *every* directory open fail, which is stricter than Windows, broke `tempfile`, and produced failures that were not Windows failures.

### Rule 30: Tests - a test does not write into the package it tests, and removes what it writes
- **When:** a test needs a file.
- **Then:** `tmp_path`; for anything large, remove it in `finally` (after releasing a mapping on it), because pytest keeps the last temporary directories.
- **Verified by:** `find <env> -name test_big.annoy` is empty after the run; the harness row "no residue".
- **Added:** 2026-10-07. Root cause: `f"{HERE}/test_big.annoy"`, 3.2 GB, in `site-packages`, never removed; the disk filled when three environments had run it. 16 more calls in the vendored suite save small files next to the tests (`grep -n 'HERE}/' scikitplot/annoy/tests/*.py`).

### Rule 39: Tests - the platform is an argument, never a patch of the process
- **When:** a function chooses by `os.name`, `sys.platform`, the environment or the home directory, and its branches are to be tested.
- **Then:** make those inputs keyword parameters that default to the real ones, and pass them in the test. Do not set `os.name` or `sys.platform` with `monkeypatch`: the standard library reads them too.
- **Verified by:** `grep -n 'setattr("os.name"' <tests>` names no test of that function; the Windows branches run on Linux and the POSIX branches on Windows.
- **Added:** 2026-10-07. Root cause: `monkeypatch.setattr("os.name", "posix")` on Windows made `pathlib` build a `PosixPath` ("cannot instantiate 'PosixPath' on your system").

### Rule 40: Tests - a test of a platform contract names what the platform may refuse
- **When:** a test runs readers and writers on one file at the same time.
- **Then:** state per platform what is allowed: on Windows a replace or an open may be refused while the other side holds the file; on POSIX nothing is refused. Assert what must hold everywhere (no partial content, no leftover file) and count refusals where they are allowed, forbid them where they are not.
- **Verified by:** the test body under the `held` switch: refusals occur, the assertions on content and leftovers hold.
- **Added:** 2026-10-07. Root cause: the test demanded that every write succeed while a thread held the file open in a loop; Windows does not grant that, and repetition cannot (14 of 40 writes still refused after 1.2 s each).

### Rule 41: Tests - a fixture built through a library holds what the library stored
- **When:** a test builds an input with a library constructor in order to feed a hostile value to the code under test.
- **Then:** assert, in the test, that the hostile value is in the artefact as stored (`orig_filename`, raw bytes) before asserting that the code refuses it.
- **Verified by:** `test_traversal_root_drive_backslash_and_ambiguous_segments_rejected` asserts the stored names.
- **Added:** 2026-10-07. Root cause: `zipfile.ZipInfo("a\\evil")` normalises the name on Windows; the test archive held `a/evil` and the test asked the product to refuse a valid name.

### Rule 55: Tests - an assertion about process-wide state uses a value no other test uses
- **When:** a test asserts something about state shared across a process (a log filter, a registry, a cache, an environment variable) that other tests also populate.
- **Then:** use a value unique to that test (or reset the state explicitly in a fixture); never rely on objects from earlier tests having been garbage-collected.
- **Verified by:** the full suite passes five times in a row under `-n 8` as well as serially.
- **Added:** 2026-10-10. Root cause: `test_the_cleaners_values_are_scrubbed_until_cleared` used `ann@example.com`, which other tests also encode; a cleaner waiting for garbage collection in the same xdist worker still held it in the shared scrub filter, so the value stayed scrubbed after `clear()` — 1 run in 3 failed, never serially.

### Rule 61: Scanners - test data that is dangerous on purpose is marked, and a gate keeps it marked
- **When:** a test needs input that a static analyser rightly flags in program code (a catastrophic regular expression, a credential shape).
- **Then:** build it through one documented helper (`regex_fixture`, as `I14` fragments do for credentials) and add a gate that fails on an unmarked fixture, using the subsystem's own detector.
- **Verified by:** `maintenances/cleanprompt/_maintenance/tests/test_regex_fixtures.py`; it fails on the round-26 tree at the line CodeQL reported.
- **Added:** 2026-10-10. Root cause: CodeQL `py/redos` alert 227 on PR 864 for `_RISKY` in `test__pattern_risk.py`; the fixture flowed into `re.compile` through `analyse_pattern`.

### Rule 62: Tests - an object that owns a resource is used as a context manager in tests
- **When:** a test creates an object with `close()` / `__exit__` that may own a temporary directory, file or thread pool.
- **Then:** use `with`, and assert the resource is gone after the block; never rely on garbage collection.
- **Verified by:** the corpus suite passes with `-W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning`.
- **Added:** 2026-10-10. Root cause: `test_downloader_factory_is_wired_to_builder_seam` created the builder's temporary directory and dropped the builder; the coverage job, which turns unraisable warnings into errors, failed with "Implicitly cleaning up <TemporaryDirectory>".

### Rule 64: Shared defaults - every site that consumes a shared default is tested against it
- **When:** a library ships a value bound to one deployment (a server allowlist default, a packaged data file carrying a `site_id`) that more than one site consumes.
- **Then:** name every known consumer in a test and assert its configuration agrees with the shared value; give each site a place for its own data (here: `feedback_aggregate_file` without a leading slash, beside `conf.py`) instead of one packaged file; never pin the shared value as a source-text string.
- **Verified by:** `test_page_feedback_integration_source.py::test_proxy_generic_feedback_site_authority_lists_both_public_sites` and `::test_library_site_feedback_is_served_by_the_proxy`, and `_sphinx_feedback/tests/test_feedback_site_builds.py` (any site builds; a packaged snapshot of another site fails closed).
- **Added:** 2026-10-10. Root cause: `_sphinx_feedback` was configured for scikit-plots-learn only. The proxy allowed `site_id` `scikit-plots-learn`, the one packaged snapshot carried `scikit-plots-learn`, and the integration test pinned both as text and ran only in the learn checkout, so https://scikit-plots.github.io/dev/ (`scikit-plots`) failed at build (`site_id does not match`) and at submit (`422 site_not_allowed`) with every check green.

### Rule 66: Tests - a test module basename is unique wherever pytest imports it as a top-level module
- **When:** adding a `test_*.py` in a directory without `__init__.py` (pytest's default import mode makes its basename the module name).
- **Then:** choose a basename no other such test file in `scikitplot/` uses; prefix it with the owner (`test_feedback_site_builds.py`, not `test_sphinx_build.py`).
- **Verified by:** the duplicate-basename scan of round 29b (`maintenances/_externals/_sphinx_ext/_sphinx_feedback/_maintenance/history/r29b_feedback_storage.md`) prints `{}`; a collision fails collection of the whole shard ("import file mismatch").
- **Added:** 2026-10-11. Root cause: round 29 added `_sphinx_feedback/tests/test_sphinx_build.py`; `_sphinx_ai_learn/tests/_integration/test_sphinx_build.py` already existed, so the full test shard stopped at collection and the sphinx-ext partial-distribution leg errored.

### Rule 69: Tests - a test that changes process-wide state restores it before the next test
- **When:** code under test configures the root logger (a CLI `main()`, importing a service app), or any other process-wide state.
- **Then:** an autouse function fixture snapshots and restores it (`_root_logging_restored_after_each_test`); never rely on test order.
- **Verified by:** running every test file of the package before `test_root_logging_isolation.py` (probe of round 30, `maintenances/_externals/_sphinx_ext/_sphinx_feedback/_maintenance/history/r30_feedback_forges.md`): no offender; three `-n 8` runs green.
- **Added:** 2026-10-11. Root cause: `deduplicate_dataset.main()` and a runtime import of the proxy replaced the root handlers inside tests; under xdist the isolation test failed whenever it was scheduled after either on the same worker (about one run in two), while serial runs passed because of alphabetical order.

### Rule 73: Tests - a Sphinx application in a test runs inside `docutils_namespace()` or is a `SphinxTestApp` that is cleaned up
- **When:** a test constructs `sphinx.application.Sphinx(...)` (or anything that builds one).
- **Then:** wrap construction and build in `with sphinx.util.docutils.docutils_namespace():`, as `sphinx-build` does; or use `SphinxTestApp` and call `cleanup()`. Never a bare application.
- **Verified by:** `_sphinx_ext/conftest.py::_sphinx_registries_restored_after_each_test` fails the leaking test at its own teardown (`additional_nodes` grew) and restores the state. Proof: with the round-31 `test_feedback_site_builds.py` put back, 13 teardown errors in that file and 0 failures elsewhere.
- **Added:** 2026-10-11. Root cause: a bare build in `_sphinx_feedback` left 47 nodes, 48 directives and 23 roles registered; the next real build (gallery grid, alphabetically next) warned "already registered" on every CI leg. Rule 69 covered this in words; this is its mechanical successor.

### Rule 79: Thresholds - a cutoff API is held to the uncut function at every threshold, 0 included
- **When:** a function takes a threshold, cutoff or maximum distance and exits early.
- **Then:** a randomised differential test asserts `criterion(a, b, t) == (distance(a, b) <= t)` for `t` from 0 upward, on odd and even lengths and on exact matches.
- **Verified by:** `test_criterion_agrees_with_the_distance_for_every_threshold` (4,500 cases) and `test_an_exact_match_is_within_threshold_zero` (lengths 0 to 8).
- **Added:** 2026-10-12. Root cause: the early exit read `d[i%1]`, which is always row 0, instead of `d[i&1]`. This made 91 of 20,000 random cases wrong, all exact matches rejected at threshold 0. No test used threshold 0.

### Rule 95: Serialisation - a round-trip test uses the format's full character repertoire and adversarial structure
- **When:** writing or reading JSON, JSON Lines, YAML or TOML.
- **Then:** fuzz strings over every code point class, including U+0085, U+2028, U+2029, lone surrogates and NUL; test aliases, deep nesting, huge and non-finite numbers, NumPy scalars and `None` in every container. Each format's limits are explicit errors (`ValueError` naming the value), never a silent rewrite.
- **Verified by:** `TestLineBreakCharacters`, `TestUntrustedInput` and `TestNumpyScalars` in `corpus/tests/test__serialize.py`.
- **Added:** 2026-10-12. Root cause: the first tests used ASCII and the happy path. YAML rewrote NEL as a space (598 of 3,000 fuzz trees), JSON Lines split under `str.splitlines()`, a 1,119-byte YAML file expanded to 1,166 MB through aliases.

### Rule 96: Timing - time the work, not the call
- **When:** measuring latency of a callable whose result may be lazy (a generator, an iterator, a lazy frame).
- **Then:** stop the clock only after the result is consumed into the form the caller uses.
- **Verified by:** `test_latency_includes_a_lazy_retriever` (150 ms of work in a generator).
- **Added:** 2026-10-12. Root cause: the clock stopped when the generator was created, so latency read 0.000001 s.

### Rule 156: Timing - a test that expects a timeout blocks far longer than the limit, until the limit stops it
- **When:** a test asserts that a deadline, timeout or watchdog fires.
- **Then:** make the blocked side unable to finish before the limit: wait on a `threading.Event` the test sets only afterwards (a thread), or sleep at least ten times the limit where the limit interrupts or kills it (a signal, a child process). Keep the limit itself small, so the test stays fast. Never shorten the blocked side to save time: the test then asks the limit to fire on work that has already finished.
- **Verified by:** in each changed test that expects a timeout, the blocked side is an `Event.wait()` or a sleep of at least ten times the limit (`grep -n "sleep(" <file>` beside the limit); `pytest --durations=0` shows the test near its limit, far below the blocked duration.
- **Added:** 2026-10-24. Root cause: verify run 40090107726 (PR 867) failed `DID NOT RAISE` in every Python leg for `cleanprompt/tests/test__bridge.py` (2 tests), `cleanprompt/tests/test__patterns.py` (1) and `cython/tests/test__budget.py` (1); `_utils/tests/test_timeout.py` (3, not part of verify) failed the same way. In each, the blocked side was `time.sleep(0.01)` against a limit of 0.1 to 1.0 s, one of them still commented "# far exceeds 1 second": the work ended before the limit could fire.

### Rule 157: Tests - a subtest's parameters are plain data
- **When:** a test calls `self.subTest(**params)` or the `subtests` fixture with parameters.
- **Then:** pass values execnet can send between processes: `str`, `int`, `float`, `bool`, `None`, `bytes` and lists, tuples and dicts of them; for a function, class or other object pass its name (`fn=fn.__name__`).
- **Verified by:** the changed test passes under `pytest -n 2` with pytest 9.0.x; `grep -n "subTest(" <file>` shows only plain-data values.
- **Added:** 2026-10-24. Root cause: `_utils/tests/test__export.py::TestExportAll::test_multiple_names_all_updated` passed alone and failed under pytest 9.0.3 with `-n auto` (`execnet.gateway_base.DumpError: can't serialize <class 'function'>`): it passed the functions themselves as `subTest(fn=fn)`, and pytest 9.0 sends subtest parameters from the xdist worker to the controller as data. pytest 9.1.1 passed it; the TF-1 evidence runs (9.0.x and 9.1.x, with and without `-n auto`) found it.
