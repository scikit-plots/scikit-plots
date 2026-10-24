# Logging and program output

Categories: Logging, Log capture, Output

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)). Also apply Rule 69 (a test
that changes the root logger, or any process-wide logging state, restores
it).

## Production modules sample

Each Python module or submodule derives its logger name from `__name__`,
never from a hard-coded package path (Rule 138):

```python
import logging

# oneline: __import__("logging").getLogger(__name__)
logger = logging.getLogger(__name__)
```

This names the logger by the module's actual import path. `scikitplot.logging`
is the project's one logging configuration; a module does not replace its
handlers, formatting, propagation or level.

## Output channel sample (which tool reports what)

The Python logging HOWTO ("When to use logging") assigns one tool to each
task; the package follows it in every `.py` file it ships (Rule 152):

| What the code does | Tool |
| --- | --- |
| Show the output a user asked for: a command's result, a function documented to print (`show_versions`) | `print(..., file=...)` / `sys.stdout.write` |
| Report an event of normal operation (status, progress, diagnostics) | `logger.info` / `logger.debug` |
| Warn about something the caller can fix (a deprecated argument, an inexact input) | `warnings.warn` |
| Note something the caller can do nothing about | `logger.warning` |
| Report an error | raise an exception |
| Report an error that is suppressed on purpose (a long-running service) | `logger.error` / `logger.exception` |

```python
# Not this: the result is printed, not returned, and the error is swallowed.
def fit_summary(data):
    try:
        mean = data.mean()
    except Exception as e:  # noqa: BLE001
        print(f"[fit ERROR] {e}")  # noqa: T201
        return None
    print(f"Fitted mean: {mean:.4f}")  # noqa: T201
    return mean


# This: the value is returned, the event is logged, the error propagates.
def fit_summary(data):
    mean = data.mean()
    logger.debug("fitted mean %.4f", mean)
    return mean
```

## Logging from tests sample

Tests usually live deeper in the package than the module under test. The
logger comes from the imported module (Rule 140), as in
`scikitplot/corpus/tests/test__atomic.py`:

```python
from .. import _atomic  # the module under test

logger = _atomic.logger  # == logging.getLogger(_atomic.__name__)
```

This uses the logger the module actually defines and needs no string. It
assumes `_atomic` defines `logger = logging.getLogger(__name__)` (Rule 138).

Not this:

```python
logger = logging.getLogger(__name__.rsplit(".", 2)[0])  # depends on the test's depth
logger = logging.getLogger("scikitplot.corpus._atomic")  # survives a rename, then captures nothing
```

`rsplit(".", 2)` names the module under test only while the test file sits
exactly two levels below it; moved one directory deeper it names another
logger and the test captures nothing. In a `conftest.py` that means "the
package I am in", use `__package__` (`scikitplot.corpus` for
`scikitplot/corpus/conftest.py`).

## Capturing in tests sample

One fixture, `captured_logger`, captures a logger through exactly one
handler path (Rule 139):

```python
def test_warns_once(captured_logger, caplog):
    with captured_logger(logging.WARNING, logger.name):
        _atomic.atomic_write_bytes(path, b"x")
    (record,) = caplog.records
    assert "fell back" in record.getMessage()
```

Not this (`_sphinx_youtube_gallery/tests/test_sync.py`,
`TestResolveSource::test_an_inexact_upload_tab_warns_and_returns_all_uploads`):
each test writes its own `logging.Handler` subclass, attaches it in a
`try`/`finally`, and wraps the call in a `caplog.at_level` that routes nothing.
The same file's `cli` fixture is a second hand-written copy, and the
neighbouring `test_an_exact_tab_does_not_warn` asserts "nothing logged"
through `caplog.at_level` alone, which passes under pytest 9.0 even when the
code warns (Rule 153).

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
  This is a deliberate departure from the logging HOWTO, which advises a
  library to add no handler but a `NullHandler`; the package does it so a
  user sees its messages without configuring anything.
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
  version (Rule 139), and an assertion that nothing was logged can pass
  without looking at anything (Rule 153).
- **Where tests can share code.** The root `scikitplot/conftest.py` and
  `scikitplot/_testing/` ship in no partial distribution; `scikitplot/logging/`
  ships in `scikit-plots-skinny`, which every partial distribution requires
  (`libs/*/pyproject.toml`). A helper every submodule's tests can use
  therefore lives in `scikitplot/logging/`, and must not import pytest at
  module level: `libs/_tools/verify.py` imports every shipped module.

### Rule 138: Logging - a module's logger is `logging.getLogger(__name__)`, and the module configures nothing
- **When:** writing or changing any module of `scikitplot/` that logs (source, conftest helpers, tests that log themselves).
- **Then:** `logger = logging.getLogger(__name__)` at module level, and log through it. No hard-coded name (`getLogger("scikitplot.annoy._mixins._io")`), no `getLogger("scikitplot")` to borrow the package logger, and no `addHandler`, `setLevel`, `propagate =`, `basicConfig` or `logging.root.handlers = ...` in library code: only `scikitplot/logging/` configures, and a command-line entry point sets the level its user chose. A standalone app inside the package (a service with its own `main`) configures in its entry point only, never at import.
- **Verified by:** for the changed files, `grep -nE "getLogger\(\s*['\"]scikitplot|basicConfig\(|root\.handlers\s*=|\.propagate\s*=" <files>` prints nothing outside `scikitplot/logging/` and `scikitplot/_cli/logging.py`; the module's logger name equals its `__name__` (`python -c "import scikitplot.annoy._mixins._io as m; assert m.logger.name == m.__name__"`).
- **Added:** 2026-10-23. Root cause: the survey of 2026-10-23 found 23 hard-coded `scikitplot.*` logger names (3 in source, 20 in tests) and six modules of `_sphinx_ai_assistant` that replace `logging.root.handlers` (`_hf_spaces_proxy/app.py` at import, the others in a `main()` or setup function); Rule 69 exists because those replacements leaked into other tests.

### Rule 139: Log capture - a test captures a package logger with the shared `captured_logger` fixture
- **When:** a test asserts what a `scikitplot.*` module logged, or that it logged nothing.
- **Then:** request the `captured_logger` fixture and capture inside `with captured_logger(level, logger_or_name):`; read `caplog.records`. The fixture attaches `caplog.handler` to that logger, turns its propagation off, sets the level on the logger and the handler, clears earlier records, and restores all of it on exit, also on failure. There is one implementation, `scikitplot/logging/_capture.py` (no pytest import, so it ships in `scikit-plots-skinny`); the root `scikitplot/conftest.py` and the top-level `conftest.py` of each submodule that ships as a partial distribution register it with one statement, `captured_logger = pytest.fixture(_capture.captured_logger, name="captured_logger")`, after `from ..logging import _capture` (one dot fewer in the root). The corpus fixture (`scikitplot/corpus/conftest.py`) becomes that statement; its signature `(level, logger)` is kept, so its 30 calls need no change. Do not write a `logging.Handler` subclass, an `addHandler`/`removeHandler` pair, or a `try`/`finally` around a capture in a test, and do not rely on `caplog.at_level(level, logger=name)`: it sets the level and routes nothing.
- **Verified by:** for the changed test files, `grep -nE "addHandler\(|removeHandler\(|class \w+\(logging\.Handler\)" <files>` prints nothing; the test passes under pytest 9.0.x and under 9.1 or later (`pip install "pytest==9.0.*"` in a throwaway environment). Probe of 2026-10-23: a `"scikitplot.annoy._mixins._io"` warning inside `caplog.at_level` alone was captured by pytest 9.1.1 and missed by 8.3.5 and 9.0.0; with the handler on the logger, captured once by all three.
- **Added:** 2026-10-23; amended 2026-10-24 (one shared fixture, its location and its registration). Root cause: `"scikitplot"` has `propagate = False` (preamble). pytest up to 9.0 attaches its capture handler to the root logger only; 9.1 also attaches it to loggers that do not propagate at the moment capture starts. So `scikitplot/annoy/tests/test_bundle_directory.py:469` (`(record,) = ...`) and `:413` pass with pytest 9.1.1 and fail with 9.0.0: the test depended on the pytest release, not on the code. Turning propagation off during the capture also keeps 9.1 from counting a record twice (once at the module's logger, once at `"scikitplot"`), the double count the corpus helper's notes describe. Without a shared fixture, 12 test files attach handlers by hand (21 `addHandler` lines), two of those lines in `_sphinx_youtube_gallery/tests/test_sync.py` alone; the root conftest cannot hold the fixture because it is not in any partial distribution (preamble).

### Rule 140: Log capture - the logger name in a test comes from the module under test
- **When:** a test or conftest needs a logger name (for capture, for a level, for a filter).
- **Then:** import the module under test relatively and use its logger: `from .. import _atomic`, then `_atomic.logger` (or `_atomic.__name__` when the module defines no `logger`). To capture a whole subpackage, use that package object's `__name__` the same way; in a `conftest.py` that means its own package, `__package__`. Do not write the dotted name as a string, and do not derive it from the test's own `__name__` with `rsplit`: the number of parts to drop depends on how deep the test file sits (`tests/test_x.py` against `tests/_hf_spaces_proxy/security/test_x.py`), so it silently names another logger when a test moves.
- **Verified by:** `grep -nE "logger\s*=\s*['\"]scikitplot|caplog\.at_level\([^)]*['\"]scikitplot|__name__\.rsplit|^\s*['\"]scikitplot\.[A-Za-z0-9_.]+['\"]\s*[,)]" <changed test files and conftests>` prints nothing (the last pattern catches a name written on its own line inside a multi-line call); renaming or moving the module under test makes the test fail at import, not pass with no records.
- **Added:** 2026-10-23; amended 2026-10-24 (`__package__` in a conftest). Root cause: the annoy tests name `"scikitplot.annoy._mixins._io"` as a string, and 12 lines in seven test modules and conftests derive names with `__name__.rsplit(".", n)`, whose right `n` differs by directory depth: `config/tests/test__distributor_init.py` uses `rsplit(".", 2)` at line 49 and `rsplit(".", 1)` at line 63, both commented "e.g. scikitplot.config", and the second is `scikitplot.config.tests`; in `scikitplot/corpus/conftest.py` the same expression is exactly `__package__`. A string survives a rename of the module and then captures nothing; the import of the module object does not.

### Rule 141: Logging - a log record carries identifiers and sizes, never secrets or whole payloads
- **When:** logging anything that comes from a user, a request, a file or an environment variable.
- **Then:** log what identifies the event (an id, a path relative to its root, a count, a size, the exception type and message) and never a token, password, key, cookie, `Authorization` header, or a whole request/response body or document; when a value may hold a secret, log that it was present, not what it is. `sanitize_log_message` is a last-resort filter, not a substitute.
- **Verified by:** a test passes a sentinel secret (`"SENTINEL-TOKEN-7f3a"`) through the code path under the capture of Rule 139, asserts that the expected record is there, and asserts the sentinel is absent from every captured message (as `_sphinx_feedback/tests/test_service_logging.py` does for `hunter2`, and `_sphinx_ai_assistant/tests/_integration/test_logging_privacy.py`); the presence assertion is what makes the absence one mean something (Rule 153).
- **Added:** 2026-10-23. Root cause: the package logs to stderr by default and CI keeps logs as artifacts; a secret in one record is a leaked credential (the coverage workflow's test job is kept free of repository secrets for the same reason, `.github/ci/README.md` section 9).

### Rule 152: Output - library code returns, logs, warns or raises; it prints only output a user asked for
- **When:** writing or changing any `.py` file of the package (library module, conftest, test, build helper), by hand or by an assistant.
- **Then:** choose the channel by the table of the output channel sample above: a computed value is returned, an event is logged through the module's `logger` (Rule 138), an avoidable problem of the caller is `warnings.warn` with a category and `stacklevel`, an error is raised. `print` and `sys.stdout.write`/`sys.stderr.write` are used only where the output is the product: a command-line entry point writing its result (data to stdout, diagnostics through the logger to stderr), or a function whose documented purpose is to display (its summary says so, it takes `file=` defaulting to `sys.stdout`, and its line carries `# noqa: T201 - <what is displayed>`). Never print an exception and return `None` instead of raising; never leave a debugging print behind; a test reports through its assertion message, not by printing. Vendored code (`externals/`, `cexternals/`, vendored `_externals/` trees) is left as upstream wrote it.
- **Verified by:** for the changed files, `ruff check --isolated --select T20 <files>` reports nothing, `grep -nE "noqa: *T20[13] *$" <files>` prints nothing (ruff honours a bare `# noqa: T201`, so the reason is checked separately: `utils/_serialize.py` passes ruff today with three bare ones), every remaining `# noqa: T201 - <reason>` is of the kind above, and `grep -nE "sys\.(stdout|stderr)\.write" <files>` prints only command-line entry points (`--isolated` matters: the project's ruff configuration excludes `**tests`, `**conftest.py`, 14 source directories and 2 files, so a plain `ruff check` does not look at most of these files).
- **Added:** 2026-10-24. Root cause: survey of 2026-10-24 (AST, calls of `print`, `pprint`, `sys.std*.write`): 132 in 32 library modules, 309 in tests, 141 in `_build_utils`, 104 in vendored code, 17 in command-line modules; `ruff` selects `T20` but reports 5, because of the exclusions. Three kinds are defects, not style: `_sphinx_youtube_gallery/sync.py:727-731` prints logger internals on every call of `resolve_source` (a debugging trail left from the capture problem of Rule 139); `utils/_serialize.py:177,205,237` print an exception and return `None`, so a caller cannot tell a failure from an empty result; `api/metrics/_regression/_residuals.py:216-270` print fitted parameters from a plotting function, where a caller can neither silence nor use them.

### Rule 153: Log capture - an assertion that nothing was logged is made inside a capture shown to see records
- **When:** a test asserts that a module logged nothing, or that a value is absent from its log.
- **Then:** use the same `captured_logger` capture as the matching positive test (Rule 139), and keep a positive test of that capture in the same file: one that makes the code log and asserts the record is there (the "warns" twin of a "does not warn" test). For an absence of a value (Rule 141), assert the expected record is present in the same block first. A negative assertion over `caplog.records` after `caplog.at_level` alone, or over a logger name typed as a string, is not a test.
- **Verified by:** temporarily make the code under test log in the negative case (a one-line `logger.warning("probe")` before the call, or `monkeypatch` of the condition) and the negative test fails; the positive twin passes under pytest 9.0.x and 9.1 or later.
- **Added:** 2026-10-24. Root cause: probe of 2026-10-24 (a module whose parent logger does not propagate, as `"scikitplot"`, that warns on every call): `assert [r for r in caplog.records if r.name == name] == []` after `caplog.at_level` alone passed under pytest 8.3.5 and 9.0.0 and failed only under 9.1.1; inside the handler-on-logger capture it failed under all three, as it must. The same pattern is in `_sphinx_youtube_gallery/tests/test_sync.py:827`, `annoy/tests/test__threads.py:157` and `tests/test__distributions.py:598` (the last at the level of `"scikitplot"` itself), so each passes under pytest 9.0 whatever the code logs.

## Checklist

- [ ] `logger = logging.getLogger(__name__)`; no literal logger names (Rule 138).
- [ ] No handler, level, propagation or root configuration outside `scikitplot/logging/` and CLI entry points (Rule 138).
- [ ] Tests capture with the shared `captured_logger` fixture; no hand-written handlers (Rules 139, 69).
- [ ] The logger in a test is the imported module's `logger` / `__name__`; `__package__` in a conftest (Rule 140).
- [ ] No secrets or whole payloads in any message (Rule 141).
- [ ] Values returned, events logged, caller problems warned, errors raised; `print` only for requested output, with `# noqa: T201 - <reason>` (Rule 152).
- [ ] Every "nothing logged" assertion has a positive twin in the same capture (Rule 153).
