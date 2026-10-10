# Lessons Learned

## Active Rules (Currently Applied)

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

### Rule 3: Shell - a process pattern must not match the command that searches for it
- **When:** using `pkill -f` / `pgrep -f`.
- **Then:** bracket the first letter of every alternative (`"[r]un-matrix"`), so the pattern text is not itself a match.
- **Verified by:** the command exits 0/1, not 143/144.
- **Added:** 2026-10-06. Root cause: the pattern occurred literally in the shell's own command line, so the shell killed itself.

### Rule 4: Packaging - a declared dependency floor is a claim; test it
- **When:** a distribution declares or inherits `pkg>=X`.
- **Then:** install it with the lowest versions the metadata allows (`uv pip install --resolution lowest-direct`) and run the import probe and the part's tests.
- **Verified by:** the `[lowest]` rows of the verify report.
- **Added:** 2026-10-06. Root cause: the root declares `scikit-learn>=1.3.0rc1`, but `scikitplot.annoy` imports `validate_data` (scikit-learn 1.6.0). Only the lowest-version run showed it; every default install resolves to the newest release and passes.

### Rule 5: Packaging - a declared Python floor is a claim; test it
- **When:** a distribution declares `requires-python`.
- **Then:** import every shipped module and run the part's tests on the lowest declared Python; where it fails, declare the measured floor instead of patching until the first import works.
- **Verified by:** the per-Python legs of the verify matrix, and the "installer refuses an unsupported Python" rows.
- **Added:** 2026-10-06. Root cause: the root says `>=3.8`; corpus (3.9), annoy and cython (3.10) and mlflow (3.11) do not import there. A first attempt to patch them for 3.8 uncovered further failures behind each fix.

### Rule 6: Tooling - generated files must be what the repository's formatter would write
- **When:** a generator emits a file that pre-commit hooks also process (`.py`, code blocks in `.md`).
- **Then:** emit formatter-stable output (exploded literals with trailing commas, double quotes) and keep a test that runs `ruff check` and `ruff format --check` on it.
- **Verified by:** `TestGeneratedFilesPassTheRepositoryLinter` in `libs/_tools/tests/test_generate.py`.
- **Added:** 2026-10-06. Root cause: a hook that rewrites a generated file makes the drift check (`python -m libs._tools check`) fail on every commit.

### Rule 7: Tooling - reading the source tree must not write into it
- **When:** build tooling needs a value from a module inside the package.
- **Then:** `compile()` + `exec()` the file's text instead of importing it, and run tools with `PYTHONDONTWRITEBYTECODE=1`.
- **Verified by:** the "build leaves nothing behind" row of the verify report.
- **Added:** 2026-10-06. Root cause: importing `scikitplot/_distributions.py` by path created `__pycache__` inside the tree, which the next build then staged.

### Rule 8: Shell - a process pattern must not occur anywhere else on the same command line
- **When:** a `pkill -f`/`pgrep -f` shares a command line with any other command.
- **Then:** run the kill as its own command; bracketing the first letter (Rule 3) protects only the pattern's own text, not a later command that contains the same words.
- **Verified by:** the kill command contains no other text that matches its pattern.
- **Added:** 2026-10-06. Root cause: `pkill -f "[l]ibs._tools verify"; ... python -m libs._tools verify ...` in one line: the shell's own command line contained the unbracketed words further on, so it matched and was killed.

### Rule 9: Verification - a long run goes to the background, never under the tool's time limit
- **When:** a build or verification can take longer than the command time limit (10 minutes).
- **Then:** start it with `nohup ... &`, write progress to a file, and poll. Afterwards run `python -m libs._tools unstage` before trusting `git status`.
- **Verified by:** no command of the session ends with "timed out".
- **Added:** 2026-10-06. Root cause: a foreground verify was killed mid-build by the time limit and left staged copies in `libs/*/`.

### Rule 10: Packaging - "lowest version" is only defined for a requirement that declares a floor
- **When:** testing a distribution at the low end of its dependency ranges.
- **Then:** resolve to the lowest version only for requirements that state `>=`/`~=`; leave version-free requirements at the newest release.
- **Verified by:** `TestDeclaresFloor` in `libs/_tools/tests/test_verify.py`; the `[lowest]` install row lists the versions used.
- **Added:** 2026-10-06. Root cause: a version-free `typing_extensions` resolved to a 2017 release that installs a `typing.py` shadowing the standard library; the failure said nothing about the project.

### Rule 11: Packaging - two floors must be installable *together*
- **When:** a project declares floors for two packages where one is built against the other (scikit-learn against NumPy).
- **Then:** install both at their floors and import; if that cannot work, record the measured lowest working pair where the test reads it (`lowest_constraints` in `libs/_tools/registry.py`) and report the disagreement.
- **Verified by:** the `[lowest]` rows of the verify report.
- **Added:** 2026-10-06. Root cause: root declares `numpy>=2.0.0` (Python >= 3.9) and `scikit-learn>=1.3.0rc1`; scikit-learn 1.3.0rc1 and 1.3.0 install beside NumPy 2.0.0 and cannot be imported, 1.3.2 to 1.4.1 are refused by the installer, 1.4.2 is the first that works.

### Rule 12: Process - stop after two failed attempts at the same idea
- **When:** a second variant of one approach fails for the same underlying reason.
- **Then:** stop, state the measured facts, and choose the mechanism that needs no assumption.
- **Verified by:** no removed helper is left behind in code or tests (`grep`).
- **Added:** 2026-10-06. Root cause: after "lowest-direct" picked a broken pre-release I tried "test the final release instead", assuming finals carry correct metadata; scikit-learn 1.3.0 final does not. The assumption was never measured before the code was written.

### Rule 13: Tests - a skipped module is not a passed module
- **When:** a test suite passes in a new environment.
- **Then:** compare the number of collected tests with a complete environment, and read the skip reasons (`-rs`), before calling it green.
- **Verified by:** the pass count in the report matches the count of the complete environment.
- **Added:** 2026-10-06. Root cause: the Sphinx extensions' suite "passed" with 5539 tests; 644 more were skipped by one module-level `importorskip("markdownify")`.

### Rule 14: Verification - a fix found while a run is in progress is written down, not applied
- **When:** a failure of a finished leg is being investigated while later legs still run.
- **Then:** reproduce it in a scratch environment outside the tree, write the fix to a scratch file, and apply it only after the run has ended; then rerun every leg.
- **Verified by:** no file under the repository has a modification time inside a run's start/end window.
- **Added:** 2026-10-06. Root cause: Rule 1 was read as "do not regenerate or stash"; a one-line test fix "while I was there" changed a file that the running leg was staging. Repeat of Rule 1, so the run was stopped and restarted from the first leg.

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

### Rule 18: Portability - emulate the other platform's condition before calling a failure "theirs"
- **When:** a failure appears only on macOS or Windows CI.
- **Then:** name the condition that differs (temporary directory behind a symbolic link, a config value that is already the "changed" value, a type of another size) and create it on Linux: `TMPDIR` pointing at a symlink, a preset `sysconfig` value, the same byte size with another type.
- **Verified by:** the failure is reproduced on Linux before the fix and gone after.
- **Added:** 2026-10-07. Root cause of the habit: the macOS int8/float128 failure looked untestable here; the differing condition was only "the float is 8 bytes", which int8 with float64 has on every platform.

### Rule 19: C++ - a count is not stored in the id type without a bound
- **When:** a quantity derived from sizes (`bytes / sizeof(S)`) is cast to a narrow integer type.
- **Then:** clamp to `std::numeric_limits<S>::max()` first, and test at dimensions that put the quantity just below and above each limit.
- **Verified by:** `test_small_index_types_return_full_results_at_every_dimension`.
- **Added:** 2026-10-07. Root cause: `static_cast<int8_t>(136)` is -120; one comparison used it as `size_t`, the other signed.

### Rule 20: Paths - a path recorded during a staged write is recorded relative to the entry
- **When:** metadata is written into a directory that is later renamed or can be moved.
- **Then:** store paths relative to that directory and join them at the point of use.
- **Verified by:** the recorded file exists after a fresh build, a forced rebuild and a cache hit, and after the entry directory is renamed.
- **Added:** 2026-10-07. Root cause: the annotation report was recorded as an absolute path inside `.staging-<key>-<random>`, which is renamed to the final entry a few lines later.

### Rule 21: Git - an ignore rule ending in `/` does not cover a symbolic link
- **When:** ignoring a path that may be a directory or a link to one.
- **Then:** write the rule without the trailing slash.
- **Verified by:** `git status --short --untracked-files=all` does not list the link.
- **Added:** 2026-10-07. Root cause: `/*/scikitplot/` ignored the staged directory but not the link of the earlier layout, which could then be committed.

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

### Rule 24: Portability - text is read and written with a named encoding
- **When:** `read_text`, `write_text` or `open` in text mode is written, in code or in a test.
- **Then:** pass `encoding="utf-8"` (or the encoding the format defines).
- **Verified by:** the harness row "text is read and written with a named encoding" (`verify._implicit_encodings`), which fails with file and line.
- **Added:** 2026-10-07. Root cause: the locale's encoding is UTF-8 on the machines the code was written on and cp1252 on the Windows runner; 831 lines relied on it.

### Rule 25: Portability - a mapped file pins itself and its directory on Windows
- **When:** code replaces, truncates, renames or removes a file, or a directory with files in it, that the same process may have memory-mapped.
- **Then:** release the mapping first, do the operation, map again; keep what is needed to restore the previous state if the operation fails. Give the order a compile-time or test-time switch so that it runs where the tests run.
- **Verified by:** `test_bundle_directory.py` (the rule applied by a fixture on any platform); a Linux build with `-DANNOY_SHRINK_REQUIRES_UNMAP=1 -DANNOY_REPLACE_REQUIRES_UNMAP=1` passing both annoy suites.
- **Added:** 2026-10-07. Root cause: POSIX allows all four operations on a mapped file, so the order was never a question on the platforms the code ran on.

### Rule 26: Errors - report the error of the API that failed
- **When:** a Win32 call fails (`MoveFileEx`, `SetEndOfFile`).
- **Then:** read `GetLastError()` immediately and put that code in the message; `errno` is not set by these calls.
- **Verified by:** the message of a failed replace on Windows contains "Windows error <n>", not "No error (0)".
- **Added:** 2026-10-07. Root cause: one error helper (`set_error_from_errno`) used after every call, whichever API the call belongs to.

### Rule 27: Build - a code path behind a macro no build defines is not compiled code
- **When:** a feature exists behind `#ifdef X` and no build file defines `X`.
- **Then:** build once with `X` before describing the feature as available, and keep a test that says which builds define it.
- **Verified by:** `test_no_meson_build_of_the_full_distribution_defines_the_macro`; the wheel test in `ci_wheels_conda_libs.yml` asserts the compiled-in state.
- **Added:** 2026-10-07. Root cause: `n_jobs` was documented and accepted; the code behind it had a missing include and had not been compiled.

### Rule 28: Tests - an emulation must first be shown to refuse what the real platform refuses
- **When:** another platform's rule is applied by a fixture or a plugin.
- **Then:** write one test of the emulation itself (the forbidden operation fails, the allowed one works), and confirm the new tests fail on the old code.
- **Verified by:** `test_the_rule_is_applied_by_the_fixture`; 5 failed on the previous `_io.py`, 15 passed on the new one.
- **Added:** 2026-10-07. Root cause: an earlier plugin made *every* directory open fail, which is stricter than Windows, broke `tempfile`, and produced failures that were not Windows failures.

### Rule 29: Packaging - "it imports" is not "it works" on a Python version
- **When:** a Python floor is defended by an import probe because the test suite cannot run there.
- **Then:** say exactly that in the registry comment, and search the code for API newer than the floor (`removeprefix`, `removesuffix`, `importlib.resources.files`, `ast.unparse`, built-in `anext`, `X | Y` evaluated at run time) before repeating the claim.
- **Verified by:** the suite of each touched extension run by hand on the floor version, with the counts in `tasks/todo.md`.
- **Added:** 2026-10-07. Root cause: "every module imports on 3.8" was written up as "the extensions are not affected on 3.8"; four of them called `str.removeprefix`.

### Rule 30: Tests - a test does not write into the package it tests, and removes what it writes
- **When:** a test needs a file.
- **Then:** `tmp_path`; for anything large, remove it in `finally` (after releasing a mapping on it), because pytest keeps the last temporary directories.
- **Verified by:** `find <env> -name test_big.annoy` is empty after the run; the harness row "no residue".
- **Added:** 2026-10-07. Root cause: `f"{HERE}/test_big.annoy"`, 3.2 GB, in `site-packages`, never removed; the disk filled when three environments had run it. 16 more calls in the vendored suite save small files next to the tests (`grep -n 'HERE}/' scikitplot/annoy/tests/*.py`).

### Rule 31: Verification - run the suite the way the repository runs it, too
- **When:** a test is added or changed in a part.
- **Then:** besides the installed-wheel run, run it from a checkout with the repository's warning policy (`-W error`), and run the tests that are left out of the wheel run because they need the repository (the eight `test_ignore` modules of sphinx-ext).
- **Verified by:** `python -m pytest scikitplot/<part> --confcutdir scikitplot/<part> -W error` from a checkout, and the `test_ignore` modules from the same place, both in `tasks/todo.md` with counts.
- **Added:** 2026-10-07. Root cause: the full test job failed on two things the harness cannot see: a layout rule enforced by a repository-only test, and a `ResourceWarning` that only `pytest.ini` turns into an error.

### Rule 32: Verification - a child process is started with the environment's interpreter first on PATH
- **When:** a harness runs tests in a virtual environment without activating it.
- **Then:** put the environment's `bin` (`Scripts`) directory first on `PATH` for the test process.
- **Verified by:** `verify._interpreter_first` and its tests; a harness that needs `tomllib` fails here with Python 3.10 first on `PATH` and no fallback.
- **Added:** 2026-10-07. Root cause: a Node harness starts `python`; locally that was a 3.11 from the machine while the environment under test was 3.9, so the test passed here and failed in CI.

### Rule 33: Parsing - read a tool's output by its grammar, not by where it happens to stop
- **When:** lines are taken from another program's report (pytest's short summary).
- **Then:** match the lines that have the form of an entry (`FAILED|ERROR`, one space, a node id ending in `.py`), over the whole section, and take the section from standard output only.
- **Verified by:** `TestFailureReporting`: a log record that starts with `ERROR`, a multi-line reason, and standard error that imitates a summary.
- **Added:** 2026-10-07. Root cause: two fixes in a row, each for the previous one's blind spot: first every `ERROR` line was an entry, then the list ended at the first line that was not one. On CI pytest prints whole messages there.

### Rule 34: Proof - a confirmation must be able to fail when the claim is true
- **When:** a second observation is used to confirm a diagnosis (here: "if the pipe is the cause, flushing fails again").
- **Then:** check the case in which the claim holds and the confirmation still succeeds, before relying on it; if there is one, say that the rule is an inference and what it can get wrong.
- **Verified by:** `test_a_closed_reader_on_windows_after_an_unbuffered_write`.
- **Added:** 2026-10-07. Root cause: a large write is not buffered, so after it fails there is nothing left to flush; the confirmation passed and the error was reported as internal.

### Rule 35: C++ - what is copied as bytes is initialised as bytes
- **When:** an object is built in uninitialised memory (`alloca`, `malloc`) and later copied whole (`memcpy`, `fwrite`).
- **Then:** zero the whole object first; padding is part of what is copied.
- **Verified by:** the saved file of one index is identical, byte for byte, from a wheel with threads and from one without (`test_one_thread_gives_the_same_file_however_it_was_asked_for`, and the cross-wheel measurement in `tasks/todo.md`).
- **Added:** 2026-10-07. Root cause: split nodes were assembled on the stack and copied into the index; four padding bytes per node carried stack contents into every saved file.

### Rule 36: Design - what a binary can do and what a run does are two switches
- **When:** a capability costs something at build time (a thread library, a platform that lacks it) and its use is a per-user choice.
- **Then:** one build switch that says whether the capability is compiled in, one run-time switch that says whether it is used, a function that reports both, and a default for the run-time switch that changes nothing for people who did not ask.
- **Verified by:** `threads_info()`; the table in `resolve_n_jobs`; both annoy suites on a wheel with threads in `auto` and in `multi`.
- **Added:** 2026-10-07. Root cause of the question: one switch had to answer both "may releases contain threads" and "do my builds use them".

### Rule 37: Data - a relocatable artefact records names, not locations
- **When:** a manifest is written into a directory that can be moved or published from a staging place.
- **Then:** record member names relative to the manifest and a format number; record no absolute path; validate names on reading so that the manifest cannot point outside its directory.
- **Verified by:** `test_the_manifest_records_no_location`, `test_a_manifest_cannot_name_a_file_outside_the_bundle`.
- **Added:** 2026-10-07. Root cause: the manifest was written while the index sat in the candidate directory and recorded that path; and that key, read back as metadata, configures an on-disk build at the path, which truncates the file.

### Rule 38: Portability - what is compared byte for byte is written as bytes
- **When:** a test fixture, a generated file that is committed, or a copy of a source file is written and something later compares its bytes, its lines or a digest of it.
- **Then:** write it with `write_bytes(text.encode("utf-8"))` (or `open(..., newline="")`). Text mode ends every line with CRLF on Windows.
- **Verified by:** the suite under the `nl` switch of the emulation (CRLF for every text-mode write) names no test; `test_compiling_twice_writes_the_same_bytes`.
- **Added:** 2026-10-07. Root cause: `Path.write_text` was used as "write this string"; on Windows it wrote another file. 29 mutation tests, 4 cleanprompt tests and one generated file (`_compiled.json`) depended on it.

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

### Rule 42: Evidence - an identifier is copied from the source it names
- **When:** a run number, a commit, a version or a count is written into code, a note or a report.
- **Then:** obtain it with a command from the artefact (`grep -o 'actions/runs/[0-9]*' <log>`), paste it, and search the tree for it afterwards.
- **Verified by:** `grep -rn <identifier>` in the delivered files and the same identifier in the log.
- **Added:** 2026-10-07. Root cause: a run number was typed from the shape of the previous one into five files; found only because a later step read the log header. Corrected before delivery.

### Rule 43: Delivery - run the repository's own hooks on the changed files before packaging
- **When:** a drop-in is about to be zipped.
- **Then:** run `precheck.sh`: one newline at the end of each file, no trailing blanks, codespell and black at the versions `.pre-commit-config.yaml` pins, on the files changed against the head.
- **Verified by:** `precheck status=0` in the results; the next commit on the branch does not touch the delivered files for formatting.
- **Added:** 2026-10-07. Root cause: round 5 was packaged without the hooks; your commit after it removed a blank last line from two files, parenthesised an expression black rewrites, and renamed a test value codespell read as a misspelling.

### Rule 44: Verification - when the log is cut off, emulate the platform's rules one at a time
- **When:** a CI log names only part of a platform's failures and the platform is not available.
- **Then:** switch on one documented behaviour of the platform per run; first show that the switch reproduces the failures the log does name (Rule 28); then read every other test it fails; say which failures no switch produced.
- **Verified by:** the calibration table in the task notes (named on Windows / reproduced here), per suite.
- **Added:** 2026-10-07. Root cause: in round 5 some 60 Windows failures were unnamed and nothing could be done about them without another CI run.

### Rule 45: Portability - a pipe read as text names its encoding
- **When:** `subprocess.run` / `Popen` is called with `text=True` (or `universal_newlines`) and the child is not a Python that shares the parent's locale by construction (Node, git, a compiler).
- **Then:** pass `encoding="utf-8"` (and `errors="replace"` where the output is only searched or shown). Without it Windows reads cp1252; one undecodable byte stops the reader thread and `stdout` is `None`.
- **Verified by:** the suite under the `enc` switch (cp1252 as the default) names no test; an AST count of calls with `text=True` and no `encoding` is recorded in the task notes.
- **Added:** 2026-10-08. Root cause: Rule 24 (text with a named encoding) was applied to files and not to pipes. Repeat occurrence of Rule 24.

### Rule 46: Messages - a path in a message is written in POSIX form
- **When:** an error or log message names a path relative to a build or data directory.
- **Then:** format it with `as_posix()`; tests compare against a literal with `/`. `str(path)` and `os.path.join` in an expected message make the message a property of the machine.
- **Verified by:** `test_verify_lists_stale_pages_relative_and_sorted` and `test_final_collection_asset_integrity_fails_closed_on_stale_output` expect the same literal form.
- **Added:** 2026-10-08. Root cause: two tests of one message disagreed (one `/`, one `os.path.join`); on Linux both passed, on Windows only one could.

### Rule 47: Verification - a silent emulation switch proves nothing until it has failed something
- **When:** a switch of the emulation produces no failure and that silence is about to be reported.
- **Then:** first give the switch a known positive: a three-line case the real platform is known to fail. If there is none, report the switch as untested, not as clean.
- **Verified by:** each switch has a calibration row in the task notes, or the word "untested" beside it.
- **Added:** 2026-10-08. Root cause: the `enc` switch replaced the default of the pipe reader; Python 3.11+ resolves the default before building the reader, so the switch did nothing for pipes, and round 6 reported its silence. Repeat occurrence of Rule 28.

### Rule 48: Docstrings - a backslash makes the docstring raw, and is then written once
- **When:** a docstring contains a backslash (a Windows path, a regular expression).
- **Then:** open it with `r"""` and write each backslash once; read the rendered text.
- **Verified by:** `python -c "import m; print(m.f.__doc__)"` shows the path as intended.
- **Added:** 2026-10-08. Root cause: a doubled backslash in a plain docstring was flagged by the linter; made raw by the maintainer, it then showed two backslashes.

### Rule 49: CI - a tool's placeholder is read in the tool's source, for the mode in use
- **When:** a workflow uses a placeholder or variable a build tool fills in (`{project}`, `{package}`, `{wheel}`), and especially when the tool's input is not a plain checkout (an sdist, an archive, a subdirectory).
- **Then:** read what the pinned release sets it to in that mode, and write the comment from that reading; prefer a path the workflow itself controls (`github.workspace`) when the file comes from outside the tool's input.
- **Verified by:** the comment beside the placeholder names the source line it relies on.
- **Added:** 2026-10-08. Root cause: `{project}/dist/libs` assumed `{project}` was the repository; built from an sdist, cibuildwheel sets it to the extracted sdist, which has no `dist/libs`.

### Rule 50: Guards - a false positive is fixed in the classifier, never by emptying its input
- **When:** a safety check refuses a configuration that is legitimate.
- **Then:** find the rule that misclassified it, narrow that rule to the contract (here: declared members, not a shared prefix), and add the legitimate configuration as a regression test next to the refusal tests.
- **Verified by:** the refusal tests still pass, and a test named after the legitimate configuration passes.
- **Added:** 2026-10-09. Root cause: `check_namespace` took any `_sphinx_ext.` name as a stack root; the docs' unrelated `_sphinx_ext.mpl_ext` helpers tripped it, and the workaround filtered every name away, which switched the guard off and failed 13 tests.

### Rule 51: Shell - text that may contain the heredoc delimiter goes in a file, not a heredoc
- **When:** a command embeds a document (a README, a transcript, a shell example) in a heredoc, and that document may itself contain the delimiter word on a line of its own (`END`, `EOF`).
- **Then:** write the content with the file-writing tool to a script in the scratchpad and run the script; never embed such content in `<<'END'`.
- **Verified by:** no heredoc in the session's commands carries a document that contains its delimiter; `grep -n "^END$" <content>` before embedding.
- **Added:** 2026-10-09. Root cause: a Python edit of the cleanprompt README was passed in `<<'END'`; the README's own `<<'END'` example ended the heredoc early, the edit did not run, and the shell executed the following lines, one of them a `pip install` (already satisfied; checked, nothing changed).

### Rule 52: Tooling - an escape written through a tool may arrive as the character it names
- **When:** text written through a tool parameter contains `\uXXXX` meant to stay an escape (invisible characters, confusables, test data for Unicode handling).
- **Then:** write the backslash doubled inside Python source, or generate the character with `chr()`; after writing, scan the file for categories `Cf`/`Zs`/confusable scripts and convert any literal one to an escape.
- **Verified by:** `python3 -c "import unicodedata,sys;t=open(sys.argv[1],encoding='utf-8').read();print([ascii(c) for c in t if ord(c)>127 and unicodedata.category(c) in ('Cf','Zs')])" FILE` prints `[]` for every file written in the round.
- **Added:** 2026-10-09. Root cause: zero-width spaces and a Cyrillic letter landed literally in tests, a probe, a gallery comment and two ledger notes — the exact disguises the round was fixing. Found by the scan, converted before delivery.

### Rule 53: Diagnostics - "ready" is proved by running the thing on a fixed input
- **When:** a report says an optional component can run (an engine, a model, a data package, a driver).
- **Then:** decide it with the same code path the real run uses, on a small fixed input; a package's presence, or a file's presence under a guessed name, is not readiness. Keep one function for the report and the run.
- **Verified by:** a test with the component installed and its data absent (and with data under an outdated name) asserts that the report and the run fail alike, with the same remedy.
- **Added:** 2026-10-09. Root cause: `doctor` read the package (`CP-093`), then the data's path under either of two names (`CP-100`); the run needed the data, under the new name.

### Rule 54: Documentation - every example is executed before it is published
- **When:** a guide page, README section or docstring shows a command or a call.
- **Then:** run each one against the tree being delivered and compare the output; add a mechanical check where the class of error can recur (options per command).
- **Verified by:** the round's notes list each page with "examples executed"; `test_documented_cli.py` passes.
- **Added:** 2026-10-09. Root cause: executing the rewritten guide found `encode --pack-file` (`CP-101`, an option that never existed), `encode_tree(source)` without its required target, and `batch --dry-run --out` (refused) — two of them copied from the previous guide.

## Pattern Analysis
- Pattern: a declaration (dependency floor, Python floor, licence, "pure Python") that nothing executes.
- Occurrences: 8 (scikit-learn floor, Python floors, missing extras named by a CLI hint, two root floors that cannot be combined, a test suite's Python floor; round 4: Python 3.8 for four Sphinx extensions, `n_jobs` behind a macro no build defined, a script that exits on import).
- Root Cause: metadata is written once by hand and only the newest environment is ever installed.

## Effectiveness Metrics
- Total lessons: 54
- Repeat occurrences: 8 (Rule 3 twice before it was written; once more in the form Rule 8 now covers; Rule 1 once, now Rule 14; Rule 1 twice more in round 4, now Rule 22 and a guard script; in round 5 a fix for one reporting mistake made the opposite one, now Rule 33; in round 6 Rule 24's subject, text files across platforms, returned as line endings instead of encodings, now Rule 38; in round 7 Rule 24 again, for pipes, now Rule 45, and Rule 28, for a switch that was never shown a positive, now Rule 47)
- Trend: Windows failures per run: 11 rows, then 5 rows (62 tests), then 1 row (2 tests). The two repeats of round 7 are both rules that were written and then applied too narrowly; their successors carry a mechanical check.
