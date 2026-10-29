# Packaging and dependencies

Categories: Packaging, Partial distributions, Optional dependencies, Optional SDKs, Build, Data, Paths

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

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

### Rule 20: Paths - a path recorded during a staged write is recorded relative to the entry
- **When:** metadata is written into a directory that is later renamed or can be moved.
- **Then:** store paths relative to that directory and join them at the point of use.
- **Verified by:** the recorded file exists after a fresh build, a forced rebuild and a cache hit, and after the entry directory is renamed.
- **Added:** 2026-10-07. Root cause: the annotation report was recorded as an absolute path inside `.staging-<key>-<random>`, which is renamed to the final entry a few lines later.

### Rule 27: Build - a code path behind a macro no build defines is not compiled code
- **When:** a feature exists behind `#ifdef X` and no build file defines `X`.
- **Then:** build once with `X` before describing the feature as available, and keep a test that says which builds define it.
- **Verified by:** `test_no_meson_build_of_the_full_distribution_defines_the_macro`; the wheel test in `ci_wheels_conda_libs.yml` asserts the compiled-in state.
- **Added:** 2026-10-07. Root cause: `n_jobs` was documented and accepted; the code behind it had a missing include and had not been compiled.

### Rule 29: Packaging - "it imports" is not "it works" on a Python version
- **When:** a Python floor is defended by an import probe because the test suite cannot run there.
- **Then:** say exactly that in the registry comment, and search the code for API newer than the floor (`removeprefix`, `removesuffix`, `importlib.resources.files`, `ast.unparse`, built-in `anext`, `X | Y` evaluated at run time) before repeating the claim.
- **Verified by:** the suite of each touched extension run by hand on the floor version, with the counts in the round's results (round 4: `maintenances/_repo/_maintenance/history/r04_windows_leg.md`).
- **Added:** 2026-10-07. Root cause: "every module imports on 3.8" was written up as "the extensions are not affected on 3.8"; four of them called `str.removeprefix`.

### Rule 37: Data - a relocatable artefact records names, not locations
- **When:** a manifest is written into a directory that can be moved or published from a staging place.
- **Then:** record member names relative to the manifest and a format number; record no absolute path; validate names on reading so that the manifest cannot point outside its directory.
- **Verified by:** `test_the_manifest_records_no_location`, `test_a_manifest_cannot_name_a_file_outside_the_bundle`.
- **Added:** 2026-10-07. Root cause: the manifest was written while the index sat in the candidate directory and recorded that path; and that key, read back as metadata, configures an on-disk build at the path, which truncates the file.

### Rule 60: Partial distributions - a part's test never reads a sibling part's files unconditionally
- **When:** a test in one subpackage (`scikitplot/_cli`) cross-checks another (`scikitplot/cleanprompt`) by importing it or reading its source.
- **Then:** assert what the part itself promises unconditionally; run the cross-check only where the sibling is installed, and skip with a reason naming the partial distribution otherwise.
- **Verified by:** `python -m libs._tools verify scikit-plots-skinny` (and `--python 3.8 --skip-build`) report "part's own tests pass".
- **Added:** 2026-10-10. Root cause: round 25's `_cli/tests/test_registry.py` read `cleanprompt/_capabilities.py` by path; `scikit-plots-skinny` ships `_cli` without cleanprompt, so every Verify leg (3.8–3.15, three OSes) failed with FileNotFoundError. The workflow existed and was not run locally before delivery.

### Rule 63: Paths - every tracked path fits the Windows checkout budget
- **When:** adding or renaming any file, above all deep maintenance notes and tests.
- **Then:** keep the repository path within the budget `tools/maint_tools/check_path_lengths.py budget` prints (131 today: 259 usable minus the longest pip clone directory); name notes as labels (identifier first, no repeated directory words, title in the heading).
- **Verified by:** `python tools/maint_tools/check_path_lengths.py check` on every pull request (`pr_check_path_lengths.yml`).
- **Added:** 2026-10-10. Root cause: a Git-based `pip install ...#subdirectory=libs/skinny` on Windows failed with "Filename too long" on two `_sphinx_ai_assistant` fresh-chat handoff notes (152 and 166 characters under a 108-character clone directory). pip clones the whole repository, so any long path anywhere breaks every Git-based install of every part.

### Rule 94: Optional dependencies - any exception while importing or probing an optional library is a recorded fallback, not a crash
- **When:** code chooses between an installed library's implementation and a vendored or pure-Python one.
- **Then:** catch `Exception` (not only `ImportError`) around the import, the subclassing and a contract probe; record the reason where the user can read it (`fallback_reason`); resolve once per process under a lock.
- **Verified by:** `test_a_broken_scikit_learn_falls_back_to_the_vendored_base[ValueError|AttributeError]` (fake `sklearn` packages) and `test_resolution_runs_once_under_racing_threads`.
- **Added:** 2026-10-12. Root cause: a NumPy ABI mismatch raises `ValueError` from `import sklearn`; only `ImportError` was caught, so every scorer raised although a vendored base was present.

### Rule 102: Optional SDKs - check request arguments against the installed client at construction, and test the oldest and newest SDK
- **When:** an adapter calls a third-party SDK whose keyword arguments change between versions (`max_completion_tokens` exists from openai 1.45).
- **Then:** read the client function's explicit signature when the model is made and refuse an argument it does not take (a `**kwargs` function is trusted); run the adapter tests against an old and the newest SDK, through the real SDK over HTTP to a local server.
- **Verified by:** `test_arguments_the_client_cannot_take_fail_at_construction`, `test_real_sdk_over_http` in venvs with openai 1.40/3.29, anthropic 0.34/1.13, google-genai 1.0/2.29.
- **Added:** 2026-10-12. Root cause: the adapter was written against the newest SDK only; on openai 1.40 every call would have raised `TypeError`.

### Rule 119: Optional dependencies - prove each called API exists in a real install, and test with the dependency present
- **When:** code calls into an optional package (`regex`, `jieba`, an analyser) inside `try/except Exception`, or behind an import probe, and CI does not install that package.
- **Then:** (1) check each called name against a real install of the package (`dir()`, the docs, a one-line call) before relying on it; (2) never wrap a call in a broad `except` that turns a missing attribute into a fallback value; (3) run the affected tests once with the package installed (a `pip install --target` directory on `PYTHONPATH` is enough) and record that run; (4) mark those tests `importorskip` so CI says they were skipped, not passed.
- **Verified by:** `TestScriptSegmenter` in `corpus/_chunkers/tests/test__notices_and_scripts.py`, run with `PYTHONPATH=r34/v3d_extra` (128 passed) and without (24 skipped).
- **Added:** 2026-10-16. Root cause: `ScriptSegmenter` called `regex.get_script`, which `regex` does not have; `except Exception: prop = None` made every character Common, so every text was one UNKNOWN span, and the only symptom was a log line about `'_unknown'`. No environment had `regex`, so nothing ever ran the code.

### Rule 125: Paths - a test of where a file sits is made on its repository-relative path
- **When:** code decides by a file's location (an archive, a test folder, a generated tree) whether to check, skip or own it.
- **Then:** compute the path relative to the repository root and test its parts (`("_maintenance", "history")` in consecutive parts), never a substring of the absolute path, which also contains the clone's location and folder names that only look alike.
- **Verified by:** `test_a_checkout_under_a_history_folder_is_still_checked` and `test_a_live_folder_named_history_is_not_an_archive` in `maintenances/_repo/_maintenance/tests/test_records.py`.
- **Added:** 2026-10-17. Root cause: the first fix for "archives are not link-checked" tested `"/history/" not in p.as_posix()` on absolute paths; in a checkout cloned below any folder named `history`, the link and budget checks silently passed everything (found by the 4R re-review).
