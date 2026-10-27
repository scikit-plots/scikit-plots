# Imports, module identity and mocking

Categories: Imports, Mocking

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)). Also apply Rule 39 (the
platform is an argument, never a patch of the process), Rule 28 (an
emulation first refuses what the real thing refuses) and Rule 131 (a
module executed twice makes cProfile records wrong).

## Internal imports logic sample

A module or test of the package reaches every other module of the package
relatively (Rule 142). The number of dots is the number of package levels to
climb:

```python
from . import _reader                # a sibling module
from .. import _atomic               # a module one package level up (a test in tests/)
from .._mixins import _io            # a module in a sibling subpackage
mod = importlib.import_module("." + name, __name__)  # a lazy load by name
```

Not this, in code or tests (the user's view is the only exception, Rule 142):

```python
import scikitplot.corpus._atomic
from scikitplot import corpus
from scikitplot.corpus import _atomic
mock.patch("scikitplot.corpus._atomic.os.replace")
```

When a string name cannot be avoided (the `-c` code of a child process, a
`pytest -p` plugin, an entry point), build it from the imported module, as
`scikitplot/_testing/tests/test__perf_telemetry.py` does:

```python
from .. import _perf_telemetry as perf

CHILD_IMPORT = f"import importlib\n_perf_telemetry = importlib.import_module({perf.__name__!r})\n"
```

A relative import needs the file to be imported as part of its package: run
tests with `pytest` (or `python -m`), never as `python path/to/test_x.py`.

## Avoid cosmetic `sys.path` changes logic sample

Do not add `sys.path.insert(...)` to make a broken import work. It makes the
result depend on the working directory and the import order, hides packaging
errors, and loads one file under two module names, each with its own state
(Rule 143). Fix the layout instead: add the missing `__init__.py`, import the
helper relatively, or move a shared test helper into a `conftest.py` fixture
or a `_testing` module of the submodule. Path manipulation stays only in a
documented script entry point.

## Patch where the name is looked up logic sample

If the code under test does this:

```python
# _worker.py
from ._reader import read

def run(path):
    return read(path)
```

patch the name in `_worker`, the module that looks it up, with the real
signature (Rules 144, 145):

```python
from unittest.mock import patch

from .. import _worker

def test_run_returns_what_read_returns():
    with patch.object(_worker, "read", autospec=True, return_value="sample") as fake:
        assert _worker.run("input.txt") == "sample"
    fake.assert_called_once_with("input.txt")
```

If the code instead does this:

```python
from . import _reader

def run(path):
    return _reader.read(path)
```

patch the attribute that it looks up:

```python
with patch.object(_worker._reader, "read", autospec=True, return_value="sample"):
    assert _worker.run("input.txt") == "sample"
```

The target is decided by the lookup expression in the code under test, not
by where the function is defined.

## Absent optional dependency logic sample

```python
import sys

import pytest

from .. import _worker

def test_without_fancydep(monkeypatch):
    monkeypatch.setitem(sys.modules, "fancydep", None)
    with pytest.raises(ImportError, match="fancydep"):
        _worker.run_optional()   # does `import fancydep` inside the function
```

`import fancydep` then raises `ModuleNotFoundError: import of fancydep
halted; None in sys.modules`, for this test only (Rule 146).

The samples above were run (pytest 9.1.1, Python 3.11) before they were
published.

## Why (read first)

`scikitplot` is assembled from submodules that each work on their own: in
the full package, in a partial distribution (`libs/`), and copied into
another tree. A submodule keeps that property only if nothing in it, and
nothing in its tests, names the package from outside:

- a relative import (`from .._mixins import _io`) still resolves after the
  subtree is renamed, moved or vendored; `import scikitplot.annoy._mixins._io`
  does not;
- one module object per file: a `sys.path` entry that makes a package file
  importable under a second name executes it again, with its own state,
  its own registries and its own `logger` (measured: one file of
  `_sphinx_ai_assistant` executed 29 times in one session);
- the dots show the dependency: `from ...logging import` climbs out of the
  submodule and is therefore a dependency to own, not an accident.

Survey of 2026-10-23 (AST, `scikitplot/**/*.py`): absolute self-imports
112 in source and 624 in tests; `sys.path` edits 38 and 55; package files
loaded by path (`spec_from_file_location`) 20 and 34; string patch targets
`"scikitplot...."` 2 and 208. The largest areas: `corpus` (208 absolute
imports and 95 string patches in tests), `mlflow` (168), `_externals/_sphinx_ext/_sphinx_ai_assistant`
(36 `sys.path` edits and 32 loads by path in tests).

No linter enforces Rule 142: ruff's `banned-api` (`TID251`) resolves a
relative import to its absolute name before matching, so a ban on
`scikitplot` also flags `from .. import _io` (probe of 2026-10-24, ruff
0.16.9). The `grep` commands of each rule are the check.

### Rule 142: Imports - code and tests inside the package import package modules relatively
- **When:** a module or test under `scikitplot/` needs another module of the package.
- **Then:** import it relatively, as deep as the layout requires: `from . import _reader`, `from .. import _atomic`, `from .._mixins import _io`, and `importlib.import_module("." + name, __name__)` for a lazy load. The absolute name `scikitplot.<...>` is right only where a user's view is the subject: docstring examples (`>>> import scikitplot as sp`), user guides, a test whose subject is the public import path itself (what `import scikitplot` exposes, a lazy attribute, a `python -m scikitplot.<cli>` command), and a file executed as a script (Rule 143). Where a test must pass a module's name as a string (a child process, `pytest -p`), it builds the string from the imported module's `__name__` (the sample above), never types it.
- **Verified by:** for the changed files, `grep -nE "^\s*(import scikitplot|from scikitplot(\.| import))" <files>` prints only the exceptions above, each with a comment saying which; the tests pass when run as one file, as the directory, and with `-n auto`.
- **Added:** 2026-10-23. Root cause: the survey above (736 absolute self-imports); every one ties its file to the name `scikitplot` and to the subtree's current place, which breaks the copy-and-replace property the package is built on, and a test that imports the module absolutely while the module itself imports its siblings relatively can end up with two names for one file when a `sys.path` entry is also present (Rule 143).

### Rule 143: Imports - no `sys.path` edit and no loading of a package file by path, in code or in tests
- **When:** an import fails, a test cannot reach a helper, or a module "needs" a sibling directory on the path.
- **Then:** fix the layout instead: make the directory a package (`__init__.py`; 86 of 87 `tests` directories are), import the helper relatively, move a helper shared by several test packages into a module both can import relatively (a `conftest.py` fixture, or a `_testing` module of the submodule). `sys.path.insert`/`append` and `importlib.util.spec_from_file_location` on a file of the package are allowed only for a file that is executed as a script by design (the probes under `scikitplot/cython/_templates/probe/`, a standalone service's single entry point), and then once, at its top, with a comment saying why it cannot be imported as a module.
- **Verified by:** for the changed files, `grep -nE "sys\.path\.(insert|append|extend)|spec_from_file_location" <files>` prints only documented script entry points; a recorded run of the area (`SKPLT_TEST_PERF_DIR`, `.github/ci/README.md` section 12) has no `module-re-executed` finding for the area's files.
- **Added:** 2026-10-23. Root cause: the security tests of `_sphinx_ai_assistant` put `tests/` and `_hf_spaces_proxy/security/` on `sys.path` and import test and source modules as top-level names; `verify_attestation_lifecycle.py` ran 29 times in one session and 34 files ran more than once (`upcoming_changes/scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/security-tests-rebuild-their-chain.md`). Each copy has its own module state, a `sys.path` entry is never removed (Rule 69), and cProfile then reports one copy's times (Rule 131).

### Rule 144: Mocking - patch the module object, at the name the code under test looks up
- **When:** a test replaces a function, class or attribute used by the code under test.
- **Then:** import the module under test relatively and patch an attribute of an object: `patch.object(module, "name", ...)` or `monkeypatch.setattr(module, "name", value)`, never a dotted string (`patch("scikitplot.x.y.name")`). Patch where the name is looked up: when `_worker` does `from ._reader import read` and calls `read(...)`, patch `_worker.read`; when it does `from . import _reader` and calls `_reader.read(...)`, patch `_reader.read` (equivalently `_worker._reader.read`). Scope every patch with a context manager, a decorator or `monkeypatch`, so it is undone when the test ends, also on failure.
- **Verified by:** for the changed test files, `grep -nE "(patch|setattr|delattr|setitem)\(\s*['\"]scikitplot\.|^\s*['\"]scikitplot\.[A-Za-z0-9_.]+['\"]\s*[,)]" <files>` prints nothing (the second pattern catches a target written on the line after `patch(`, as in `scikitplot/_compat/tests/test__sklearn.py:105`); each patched test also asserts the behaviour the patch was for (the result or the error), not only that the mock was called, so a patch of the wrong name fails the test.
- **Added:** 2026-10-23. Root cause: 208 string targets in tests (95 in `corpus`, 39 in `_utils`, 28 in `datasets`). A string names the module by its absolute path and resolves it at patch time, so it keeps the test tied to the package's name and place, and a string that names where a function is defined instead of where it is looked up patches nothing the code calls.

### Rule 145: Mocking - a mock of a callable has the real signature
- **When:** a test replaces a function, method or class with a mock.
- **Then:** create it with `autospec=True` (`patch.object(module, "name", autospec=True)`) or `unittest.mock.create_autospec(real)`, so a call with arguments the real one does not accept fails as it would in production; for a replacement written by hand, give it the same parameters.
- **Verified by:** calling the patched name with an argument the real signature rejects raises `TypeError` in the test (one such assertion per newly mocked API is enough).
- **Added:** 2026-10-23. Root cause: no use of `autospec` anywhere in the package (survey of 2026-10-23). A plain `Mock` accepts any call, so a test keeps passing after the real function's signature changes, and the code under test then fails in production on a call no test checked.

### Rule 146: Mocking - an absent optional dependency is `sys.modules[name] = None`, set with `monkeypatch`
- **When:** a test checks how code behaves when an optional dependency (`torch`, `mlflow`, `sphinx`, ...) is not installed.
- **Then:** `monkeypatch.setitem(sys.modules, "name", None)` (and the same for its submodules the code imports); `import name` then raises `ImportError` for that test only and the entry is restored afterwards. The code under test imports the optional dependency where it is used (inside the function), so the test controls it; if the module under test imported it at import time, reload it under the patch and reload it again after (or test through a subprocess). Do not patch `builtins.__import__`.
- **Verified by:** for the changed test files, `grep -nE "builtins\.__import__|['\"]builtins\.__import__['\"]" <files>` prints nothing; after the test, `sys.modules` holds the same object for the dependency as before (assert it in the test or in a fixture).
- **Amended:** 2026-10-26. This holds for a test of "unavailable" behaviour too: the test makes the dependency absent itself, so it passes whether or not the package is installed, and a twin test (`pytest.importorskip`) covers the installed case. Root cause: `corpus/_similarity/tests/test__backends.py::TestSelectBackend::test_explicit_unavailable_raises` expected `select_backend("faiss")` to raise because faiss was not installed locally; CI installs `faiss-cpu` (`requirements/cpu.txt`), so it failed there with "DID NOT RAISE". `importlib.util.find_spec(name)` returns `None` for a `None` entry in `sys.modules`, so a backend's `is_installed` sees the package as absent.
- **Added:** 2026-10-23. Root cause: 55 places patch `builtins.__import__` (and 58 already use `sys.modules`). A patched `__import__` intercepts every import of the process while it is active, including lazy imports of unrelated modules and of pytest itself, so the test's result depends on what else imports during it, and a forgotten condition in the fake raises for modules the test never meant to hide.

## Checklist

- [ ] Package modules are imported relatively; absolute `scikitplot.` only in the listed exceptions, each commented (Rule 142).
- [ ] No `sys.path` edit or load by path, except a documented script entry point (Rule 143).
- [ ] Patches are `patch.object` / `monkeypatch.setattr` on the imported module, at the looked-up name, scoped (Rule 144).
- [ ] Mocks of callables have `autospec` (Rule 145).
- [ ] A missing dependency is `sys.modules[name] = None` via `monkeypatch` (Rule 146).
- [ ] The tests pass as one file, as a directory, and under `-n auto`.
