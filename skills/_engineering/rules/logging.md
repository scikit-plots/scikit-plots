# Logging

Categories: Logging, Log capture

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)). Also apply Rule 69 (a test
that changes the root logger, or any process-wide logging state, restores
it).

## How the package logger works (read first)

Measured on the tree of 2026-10-23 (`scikitplot/logging/_logging.py`):

- **One configured logger, `"scikitplot"`.** `get_logger()` configures it
  once per process: the package's own handler (`AlwaysStdErrHandler`, to
  stdout in an interactive session), its level (from `SKPLT_LOGGING_LEVEL`,
  else `SKPLT_VERBOSE`), a `findCaller` that skips the helpers, and
  **`propagate = False`**, so the package never writes into an
  application's root handlers. The first call of any helper
  (`scikitplot.logging.debug`, `.warning`, ...) runs it; in practice
  `import scikitplot` does (`config/_citation.py` logs while importing).
- **Every module logs through a child of it.** `logging.getLogger(__name__)`
  in `scikitplot/annoy/_mixins/_io.py` is `"scikitplot.annoy._mixins._io"`;
  its records propagate to `"scikitplot"` and get the package's handler
  and level, with no import of `scikitplot.logging` and no configuration
  in the module. That is what makes a submodule work the same alone, in a
  partial distribution, or copied elsewhere: it names its logger by where
  it is, and the package decides what is shown.
- **`scikitplot.logging` is the only place that configures.** It also
  re-exports the standard library's names (`getLogger`, `WARNING`, ...),
  answers any other attribute from `logging` (`__getattr__`), and offers
  `sanitize_log_message` (not applied automatically) and the
  `log_every_n` / `log_first_n` helpers. Command-line entry points
  (`scikitplot/_cli/logging.py`) may set the level a user asked for.
- **Consequence for tests:** a record of a `scikitplot.*` logger stops at
  `"scikitplot"` and never reaches the root logger, where pytest's
  `caplog` handler is. Whether `caplog` sees it depends on the pytest
  version (Rule 139).

### Rule 138: Logging - a module's logger is `logging.getLogger(__name__)`, and the module configures nothing
- **When:** writing or changing any module of `scikitplot/` that logs (source, conftest helpers, tests that log themselves).
- **Then:** `logger = logging.getLogger(__name__)` at module level, and log through it. No hard-coded name (`getLogger("scikitplot.annoy._mixins._io")`), no `getLogger("scikitplot")` to borrow the package logger, and no `addHandler`, `setLevel`, `propagate =`, `basicConfig` or `logging.root.handlers = ...` in library code: only `scikitplot/logging/` configures, and a command-line entry point sets the level its user chose. A standalone app inside the package (a service with its own `main`) configures in its entry point only, never at import.
- **Verified by:** for the changed files, `grep -nE "getLogger\(\s*['\"]scikitplot|basicConfig\(|root\.handlers\s*=|\.propagate\s*=" <files>` prints nothing outside `scikitplot/logging/` and `scikitplot/_cli/logging.py`; the module's logger name equals its `__name__` (`python -c "import scikitplot.annoy._mixins._io as m; assert m.logger.name == m.__name__"`).
- **Added:** 2026-10-23. Root cause: the survey of 2026-10-23 found 23 hard-coded `scikitplot.*` logger names (3 in source, 20 in tests) and six modules of `_sphinx_ai_assistant` that replace `logging.root.handlers` (`_hf_spaces_proxy/app.py` at import, the others in a `main()` or setup function); Rule 69 exists because those replacements leaked into other tests.

### Rule 139: Log capture - a test captures a package logger through a handler on that logger, not with `caplog.at_level` alone
- **When:** a test asserts what a `scikitplot.*` module logged.
- **Then:** attach `caplog.handler` to the logger under test for the duration of the block and turn that logger's propagation off while it is attached, then restore both (the `captured_logger` fixture of `scikitplot/corpus/conftest.py` does exactly this; reuse it, or the shared helper once it exists, never a third copy). `caplog.at_level(level, logger=name)` only sets the level: it does not route the record to `caplog`.
- **Verified by:** the test passes under pytest 9.0.x and under 9.1 or later (`pip install "pytest==9.0.*"` in a throwaway environment). Probe of 2026-10-23: a `"scikitplot.annoy._mixins._io"` warning inside `caplog.at_level` alone was captured by pytest 9.1.1 and missed by 8.3.5 and 9.0.0; with the handler on the logger, captured once by all three.
- **Added:** 2026-10-23. Root cause: `"scikitplot"` has `propagate = False` (Rule 138's preamble). pytest up to 9.0 attaches its capture handler to the root logger only; 9.1 also attaches it to loggers that do not propagate at the moment capture starts. So `scikitplot/annoy/tests/test_bundle_directory.py:469` (`(record,) = ...`) and `:413` pass with pytest 9.1.1 and fail with 9.0.0: the test depended on the pytest release, not on the code. Turning propagation off during the capture also keeps 9.1 from counting a record twice (once at the module's logger, once at `"scikitplot"`), the double count the corpus helper's notes describe.

### Rule 140: Log capture - the logger name in a test comes from the module under test
- **When:** a test needs a logger name (for capture, for a level, for a filter).
- **Then:** import the module under test relatively and use its name: `from .._mixins import _io`, then `_io.__name__` (or `_io.logger.name` when the module defines `logger`). To capture a whole subpackage, use that package object's `__name__` the same way. Do not write the dotted name as a string, and do not derive it from the test's own `__name__` with `rsplit`: the number of parts to drop depends on how deep the test file sits (`tests/test_x.py` against `tests/_hf_spaces_proxy/security/test_x.py`), so it silently names another logger when a test moves.
- **Verified by:** `grep -nE "logger\s*=\s*['\"]scikitplot|caplog\.at_level\([^)]*['\"]scikitplot|__name__\.rsplit|^\s*['\"]scikitplot\.[A-Za-z0-9_.]+['\"]\s*[,)]" <changed test files>` prints nothing (the last pattern catches a name written on its own line inside a multi-line call); renaming or moving the module under test makes the test fail at import, not pass with no records.
- **Added:** 2026-10-23. Root cause: the annoy tests name `"scikitplot.annoy._mixins._io"` as a string, and 12 lines in seven test modules and conftests derive names with `__name__.rsplit(".", n)`, whose right `n` differs by directory depth. A string survives a rename of the module and then captures nothing; the import of the module object does not.

### Rule 141: Logging - a log record carries identifiers and sizes, never secrets or whole payloads
- **When:** logging anything that comes from a user, a request, a file or an environment variable.
- **Then:** log what identifies the event (an id, a path relative to its root, a count, a size, the exception type and message) and never a token, password, key, cookie, `Authorization` header, or a whole request/response body or document; when a value may hold a secret, log that it was present, not what it is. `sanitize_log_message` is a last-resort filter, not a substitute.
- **Verified by:** a test passes a sentinel secret (`"SENTINEL-TOKEN-7f3a"`) through the code path under the capture of Rule 139 and asserts it is absent from every captured message (as `_sphinx_ai_assistant/tests/_integration/test_logging_privacy.py` does).
- **Added:** 2026-10-23. Root cause: the package logs to stderr by default and CI keeps logs as artifacts; a secret in one record is a leaked credential (the coverage workflow's test job is kept free of repository secrets for the same reason, `.github/ci/README.md` section 9).

## Checklist

- [ ] `logger = logging.getLogger(__name__)`; no literal logger names (Rule 138).
- [ ] No handler, level, propagation or root configuration outside `scikitplot/logging/` and CLI entry points (Rule 138).
- [ ] Tests capture through a handler on the logger under test, restored afterwards (Rule 139, Rule 69).
- [ ] The logger name in a test is the imported module's `__name__` (Rule 140).
- [ ] No secrets or whole payloads in any message (Rule 141).
