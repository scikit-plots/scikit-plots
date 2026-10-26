# Portability

Categories: Portability, Subprocess, Messages, Files, Async

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 18: Portability - emulate the other platform's condition before calling a failure "theirs"
- **When:** a failure appears only on macOS or Windows CI.
- **Then:** name the condition that differs (temporary directory behind a symbolic link, a config value that is already the "changed" value, a type of another size) and create it on Linux: `TMPDIR` pointing at a symlink, a preset `sysconfig` value, the same byte size with another type.
- **Verified by:** the failure is reproduced on Linux before the fix and gone after.
- **Added:** 2026-10-07. Root cause of the habit: the macOS int8/float128 failure looked untestable here; the differing condition was only "the float is 8 bytes", which int8 with float64 has on every platform.

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

### Rule 38: Portability - what is compared byte for byte is written as bytes
- **When:** a test fixture, a generated file that is committed, or a copy of a source file is written and something later compares its bytes, its lines or a digest of it.
- **Then:** write it with `write_bytes(text.encode("utf-8"))` (or `open(..., newline="")`). Text mode ends every line with CRLF on Windows.
- **Verified by:** the suite under the `nl` switch of the emulation (CRLF for every text-mode write) names no test; `test_compiling_twice_writes_the_same_bytes`.
- **Added:** 2026-10-07. Root cause: `Path.write_text` was used as "write this string"; on Windows it wrote another file. 29 mutation tests, 4 cleanprompt tests and one generated file (`_compiled.json`) depended on it.

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

### Rule 74: Async - no `asyncio` primitive is constructed outside a running loop in code that supports Python 3.9
- **When:** writing `asyncio.Lock/Event/Condition/Semaphore/Queue()` in `__init__`, at module level, or in a sync test body.
- **Then:** if the guarded section has no `await`, use `threading.Lock` (no loop binding, thread-safe). Otherwise create the primitive inside the coroutine that uses it.
- **Verified by:** the `_sphinx_feedback` suite on Python 3.9 (422 tests, incl. `test_app.py`, `test_service.py`); on the old code 22 fail at `app.py:266`.
- **Added:** 2026-10-11. Root cause: 3.9 binds these primitives to `get_event_loop()` at construction, which raises after any `asyncio.run()`; 3.10+ binds lazily, so only the 3.9 leg failed.

### Rule 77: Subprocess - on Windows a child's `env` starts from `os.environ`
- **When:** passing `env=` to `subprocess`.
- **Then:** `env = {**os.environ, ...}` minus named variables to withhold; never a hand-built minimal mapping. Python's `subprocess` documentation requires a valid `SystemRoot` on Windows.
- **Verified by:** `test_cloudflare_receiver.py` (Linux local; Windows leg on the next CI run).
- **Added:** 2026-10-11. Root cause: `env={"SECRET", "PATH"}` failed only on the Windows leg.

### Rule 98: Files - a published file gets the permissions `open()` would give it
- **When:** writing through a staging file and `os.replace` (atomic publish).
- **Then:** create the staging file with mode `0o666` (the umask applies) and copy a replaced target's mode; never inherit `mkstemp`'s `0o600`.
- **Verified by:** `TestPermissions` in `corpus/tests/test__atomic.py` (umask 022 fixture: a new file is `0o644`; a replaced `0o640` file keeps `0o640`).
- **Added:** 2026-10-12. Root cause: `tempfile.mkstemp` creates `0o600` by design; every report, export and cache file became private to its owner.
