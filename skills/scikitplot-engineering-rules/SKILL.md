---
name: scikitplot-engineering-rules
description: The numbered engineering rules of scikit-plots (verification, delivery, tests, contracts, packaging, portability, native code, text). Use at the start of any increment, review or fix in this repository, and before adding a lesson after a correction.
---

# Engineering rules

The rules learned across the review rounds, each with a trigger
(**When**), an action (**Then**), a check (**Verified by**) and its origin
(**Added**). A rule keeps its number for good; other records cite rules by
number ("Rule 52").

## How to use

1. Read [`INDEX.md`](INDEX.md): one line per rule, generated from the files.
2. Always load the first three files below; load the others when the work
   touches their scope.
3. Recurring patterns behind the rules, and review counts per round, are in
   [`PATTERNS.md`](PATTERNS.md).

| File | Load for |
|---|---|
| [`rules/verification.md`](rules/verification.md) | every increment: runs, snapshots, evidence, gates |
| [`rules/delivery.md`](rules/delivery.md) | every increment: packaging a drop-in, CI, generated configs, labels, work records |
| [`rules/tooling.md`](rules/tooling.md) | every increment: tool and shell traps (escapes, heredocs) |
| [`rules/testing.md`](rules/testing.md) | writing or changing tests |
| [`rules/contracts.md`](rules/contracts.md) | public APIs, inputs, options, fallbacks, identities, loaders |
| [`rules/packaging.md`](rules/packaging.md) | dependencies, floors, partial distributions, paths |
| [`rules/portability.md`](rules/portability.md) | Windows, encodings, subprocesses, async |
| [`rules/state.md`](rules/state.md) | concurrency, caches, stores, SQLite |
| [`rules/numerics.md`](rules/numerics.md) | floats, scoring, native kernels |
| [`rules/native.md`](rules/native.md) | C++ and native memory |
| [`rules/text.md`](rules/text.md) | Unicode, spans, markup parsing |
| [`rules/documentation.md`](rules/documentation.md) | guides, docstrings, examples |
| [`rules/numpydocstr.md`](rules/numpydocstr.md) | writing or changing any docstring: sections, parameters, cross-references |
| [`rules/logging.md`](rules/logging.md) | code that logs, prints or writes to stdout/stderr, and tests that assert on logs |
| [`rules/module_submodule_actual_mock_patch.md`](rules/module_submodule_actual_mock_patch.md) | imports inside the package, `sys.path`, mocks and patches in tests |

## Adding a lesson (after any correction, failed check or review finding)

1. Take the next free number: one more than the last row of `INDEX.md`.
2. Write it in the file whose `Categories:` line names its category. A new
   category is added to exactly one file's `Categories:` line.
3. Use the four fields: **When** (trigger), **Then** (action), **Verified
   by** (a command or test that shows it was followed), **Added** (date and
   root cause, found with five whys).
4. A fact that only one submodule needs goes to that submodule's
   `maintenances/<sub>/todo/lessons.md` instead, without a number.
5. Run `python -B maintenances/_repo/_maintenance/tools/check_records.py
   --write-index`; it regenerates `INDEX.md` and checks numbering,
   categories and links.

## Where the other records live

See [`tasks/README.md`](../../tasks/README.md); the repository-wide
procedure is [`docs/maintenances/MAINTAINING.md`](../../docs/maintenances/MAINTAINING.md).
