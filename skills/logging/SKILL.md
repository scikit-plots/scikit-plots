---
name: logging-maintainer
description: Maintain the scikitplot.logging package facade and private _logging core, including compatibility forwarding, logger lifecycle, handlers/formatters, CLI interoperability, concurrency, tests, and release evidence.
---

# `scikitplot.logging` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`.

The current runtime is a **package**, not the former single `scikitplot/logging.py` module:

```text
scikitplot/logging/
├── __init__.py          # public package facade
├── _logging.py          # implementation core
└── tests/
    ├── __init__.py
    ├── test__logging.py
    └── test__logging.sh
```

Do not collapse these planes back into a single-file mental model. Review the public facade and the private core separately.

## Read first

1. `maintenances/logging/MAINTAINING.md`
2. `maintenances/logging/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/logging/_maintenance/STATE.json`
4. `maintenances/logging/_maintenance/FAMILY.md`
5. `maintenances/logging/_maintenance/VERIFICATION.md`
6. `maintenances/logging/REVIEW.json`

Then run:

```sh
python -B maintenances/logging/_maintenance/check_trackers.py --json
python -B maintenances/logging/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/logging/_maintenance/tests -q -p no:cacheprovider
```

## Package move is a compatibility event

Moving `scikitplot/logging.py` to `scikitplot/logging/_logging.py` changes import mechanics even when the implementation bytes are otherwise similar.

The public package facade must deliberately decide which former module behaviors remain public:

- stable names in `_logging.__all__`;
- dynamic stdlib logging compatibility via `__getattr__`;
- enhanced `dir()` via `__dir__`;
- root aliases such as `scikitplot.logger`;
- private helpers intended only for focused tests.

Current **LOG-PKG-001** exists because `scikitplot.logging.__init__` re-exports `_logging.__all__` but does not forward `_logging.__getattr__` or `_logging.__dir__`. Therefore former public calls such as `scikitplot.logging.getLevelName` and `scikitplot.logging.getLogger` no longer resolve.

Do not fix this accidentally by `from ._logging import *`: Python star imports intentionally skip underscored names such as `__getattr__` and `__dir__`.

## Focused tests must follow the new package shape

Current **LOG-TEST-001** is a move-specific collection defect. `logging/tests/test__logging.py` still contains:

```python
from .. import logging as splog
```

From `scikitplot.logging.tests`, that asks the parent package for a `logging` child/name that no longer exists.

A harness-only correction to:

```python
from .. import _logging as splog
```

makes the focused suite run and currently yields 171/171 passing.

Never record that harness result as a native test PASS. The checked-in test must collect without source rewriting.

## Green tests can preserve broken behavior

Current **LOG-TEST-002** is important. The focused suite explicitly contains tests named like:

```text
test_error_log_is_noop
test_unknown_string_returns_none
```

Those tests encode current defects as expected behavior. A 171/171 result therefore proves implementation/test agreement, not that the public logging contract is correct.

When repairing a finding, first replace the corresponding defect-preserving test with a desired-contract regression, then modify runtime behavior.

## Keep facade tests and core tests separate

At minimum maintain two lanes:

1. public package tests against `import scikitplot.logging as logging`;
2. implementation tests against `scikitplot.logging._logging`.

The public lane must cover `__all__`, compatibility forwarding, import safety, root aliases in a complete package, and absence of accidental private exports.

The core lane may test helper functions such as `_default_logging_level`, `_coerce_level`, and internal caller utilities.

Do not make private helper availability part of the package API merely to simplify tests.

## `error_log` must log

Current **LOG-ERR-001** remains open. `_logging.error_log()` deletes its arguments and returns.

Its `level` parameter should be normalized consistently with `log()` and `setLevel`; formatting args and logging kwargs such as `exc_info`, `extra`, and `stacklevel` must survive.

Test emitted `LogRecord` content, not just return value.

## Environment policy must be wired into initialization

Current **LOG-ENV-001** remains open.

`_default_logging_level()` documents:

```text
SKPLT_LOGGING_LEVEL
SKPLT_VERBOSE
```

but `get_logger()` does not call it. `SKPLT_LOGGING_AUTO_CONFIG` also references a `configure()` path absent from the core.

Decide whether each environment variable is supported. Do not leave decorative environment contracts.

Test precedence, invalid levels, repeated initialization, explicit later `setLevel()`, and coexistence with application-owned logging configuration.

## Handler stream switching must obey stdlib semantics

Current **LOG-HDL-001** remains open.

`AlwaysStdErrHandler.setStream(sys.stdout)` currently reports the old stream but leaves future writes on stderr because the property setter never stores valid stdout/stderr assignments.

Preserve stdlib `StreamHandler.setStream` expectations: actual stream changes, prior stream flush, no closure of global streams, and correct redirected-stdio behavior.

## Shared logging and CLI policy are separate owners

`scikitplot._cli.logging` owns CLI `-v/-q` mapping and stderr purity.

`scikitplot.logging` owns the reusable named logger and common handlers.

Both touch `logging.getLogger("scikitplot")`, so initialization-order interoperability is mandatory.

Current **LOG-CLI-001** remains reproducible:

```text
_cli.logging.configure()
→ one handler

_logging.get_logger()
→ two handlers
```

A repair should identify/reuse compatible project-owned handlers rather than letting either subsystem silently seize the other's policy.

## Caller metadata must support wrappers and direct Logger use

Current **LOG-CALL-001** remains open.

The custom `findCaller` ignores its `stacklevel` argument and uses a fixed frame offset. Wrapper calls may appear correct while documented direct calls:

```python
get_logger().info("message")
```

resolve to `(unknown file):0`.

Capture `LogRecord.pathname`, `lineno`, and `funcName` for direct calls, wrappers, nested helpers, and explicit `stacklevel`.

## Dynamic compatibility lookup must be side-effect free

There are now two levels to consider.

The public package currently does not expose the dynamic compatibility lookup at all (**LOG-PKG-001**).

The private core still implements `__getattr__`, and current **LOG-ATTR-001** remains: a missing stdlib attribute calls `get_logger()`, installing a handler before raising `AttributeError`.

Introspection, Sphinx, IDEs, `hasattr`, and pickling must not configure logging.

## Formatter selection must be total

Current **LOG-FMT-001** remains open.

`_make_default_formatter("TOTALLY_UNKNOWN_FORMAT")` and `_make_default_formatter(None)` currently fall through to `None`; focused tests explicitly bless that behavior.

Choose an explicit policy: reject unsupported values with `ValueError`, or return a documented deterministic fallback. Never silently disable formatting because of a typo.

## Thread safety is more than singleton identity

The prior review demonstrated one logger identity across concurrent first initialization. Preserve that property after the package move.

Also assert one compatible internal handler, because a thread-safe singleton that duplicates handlers is still operationally wrong.

Rate-limit counters are separate mutable state and are documented as non-thread-safe; do not imply otherwise.

## Process and notebook behavior require distinct evidence

Release evidence should eventually cover:

- multiprocessing spawn;
- fork where supported;
- redirected stdout/stderr;
- notebook/Jupyter stream selection;
- interpreter shutdown;
- complete package aliases;
- supported Python/platform matrix.

A Linux single-process unit suite cannot certify these lanes.

## `test__logging.sh` is part of the compatibility contract

The shell probe still expects public `scikitplot.logging` to provide dynamic stdlib names such as `getLogger` and module-level `__getattr__`.

Treat that as useful historical intent, not as automatically authoritative. Reconcile it with the desired public package API and update code/tests together.

Do not leave shell probes testing an API that package tests no longer expose.

## Root aliases are a complete-package lane

The runtime docs still advertise `scikitplot.logger` as a compatibility alias.

This supplied snapshot has no root `scikitplot/__init__.py`, so that alias cannot be verified natively here. Keep it `UNAVAILABLE`; do not invent a root file and call that release evidence.

## Verification ladder

Keep these evidence lanes distinct:

1. maintenance static/mutation contract;
2. native focused test collection from the checked-in files;
3. harness-only focused tests when diagnosing move breakage;
4. public package facade compatibility;
5. direct negative probes for each open runtime finding;
6. thread singleton/handler lifecycle;
7. `_cli.logging` initialization-order and stderr purity;
8. process/notebook/stream lifecycle;
9. complete package root aliases and downstream consumers;
10. supported Python/platform matrix.

Missing lanes are `UNAVAILABLE`, never PASS.

## Repair workflow

For each repair:

1. reproduce the smallest failure;
2. decide whether the contract belongs to the facade or `_logging` core;
3. update the focused test so it states the desired behavior;
4. make one runtime change;
5. run native focused tests;
6. run public facade probes;
7. run CLI interoperability;
8. rerun maintenance mutation tests;
9. refresh `REVIEW.json`, `STATE.json`, and `EVIDENCE.json` only from executable evidence.

Never use `--update` to bless a structurally red runtime.

## Current mandatory findings

Do not close these from source inspection alone:

- **LOG-PKG-001**
- **LOG-TEST-001**
- **LOG-TEST-002**
- **LOG-ERR-001**
- **LOG-ENV-001**
- **LOG-HDL-001**
- **LOG-CLI-001**
- **LOG-CALL-001**
- **LOG-ATTR-001**
- **LOG-FMT-001**

Each requires a targeted regression or integration probe appropriate to its ownership layer.
