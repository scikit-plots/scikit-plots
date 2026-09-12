# Local test-first debugging workflow

Use this workflow after Run 172 while the user supplies local compile/test
failures.

## 1. Capture evidence

Record the exact command, platform/Python/Node version when relevant, test node,
traceback, assertion values, and whether the failure reproduces alone.

## 2. Classify before editing

### Product/code defect
The executable behavior violates an intended contract.

### Test defect
The test is stale, globally scoped when it should be branch-scoped, vacuous, or
reimplements the logic it claims to test. Fixture builders also belong here when
they emit identifiers that disagree with the canonical namespace consumed by the
owning test family. Fix the fixture generator; do not broaden production parsing
to accept a malformed test-only namespace.
If a shared helper returns a positional tuple, verify the return order at the
helper before editing consumers. Bind path/state slots explicitly; do not infer
object attributes on an adjacent list or adapter collection just to make one test
pass.
If a setup helper creates an authoritative output directory and returns it, reuse
that returned path in transition tests. Do not discard it and hard-code a visually
similar sibling path; otherwise prerequisite validation can mask the invariant the
test claims to exercise.

### Environment/isolation
The code/test passes alone but fails because of missing optional dependencies,
process state, inherited descriptors, locale/time, interpreter lookup, or fixture contamination.

When a test mutates a module-level singleton on the **canonical imported module** (logger, cache,
registry, provider map, etc.), scope that mutation with pytest `monkeypatch` or an explicit
`try/finally`. A direct assignment can survive into unrelated later owners even when the test
itself passes. Distinguish canonical imports from private `spec_from_file_location` module copies;
only the former share state with later canonical tests.

For subprocess resource failures, the adapter owns both process reaping and every pipe endpoint it opens. Reader threads must finish before overflow/parse classification; otherwise cleanup can be correct while the result classification races.

For hermetic child-process tests, separate **command discovery** from **environment
isolation**. If production intentionally pins `PATH=os.defpath`, invoke a Python
fixture with `sys.executable` when the contract accepts argv. If the contract is
intentionally path-only, give the fixture an absolute interpreter shebang rather
than `#!/usr/bin/env python3`. For portability regressions, force the effective
child PATH to an empty directory so a host `/usr/bin/python3` cannot hide the
contract mistake.

### Race/broken pipe
The failure depends on cancellation, supersede, delayed awaits, stream reader
ownership, late finally blocks, or stale conversation generation.

### Path/layout
The code is correct but a file moved, a relative path is stale, or packaging
included/omitted non-authoritative files. For deny-by-default Docker build
contexts, treat `Dockerfile COPY` sources and `.dockerignore` re-inclusions as one
contract: every local COPY source must exist and survive the ignore rules.

## 3. Reproduce narrowly

Prefer one node ID. For Node harnesses, run the exact `.mjs`. For cryptographic
release tests, use a fresh process if the full module stalls.

## 4. Fix the owner

Change the smallest surface that owns the invariant. Do not change runtime code
for a stale test, and do not change a test to excuse a real product defect.

## 5. Prove the fix can fail

For important guards, use the existing mutation catalogue or a targeted
positive-control mutation. A green test that cannot detect removal of the guard
is not strong evidence.

## 6. Expand verification

Order:

```text
exact failure
  -> same module / adjacent contract
  -> cross-surface regression
  -> browser/mutation gate when relevant
  -> broader ordinary suite
```

## 7. Record reusable lessons

Add a lesson only when the root cause generalizes. Include context, root cause,
prevention rule, and how to verify the prevention rule.
## Canonical test ownership while repairing failures

Before creating or moving a Python test, resolve its runtime owner. Use exactly
`test_<module>.py` (`test___init__.py` for `__init__.py`). If that canonical file
is too large, add a non-collected fragment under `_cases/<module>/` and load it
through `_case_loader.py`; do not create another collected owner. Integration
and architecture tests must remain explicit cross-module planes. After any move,
run collection first so stale imports/paths fail before behavioral debugging.
