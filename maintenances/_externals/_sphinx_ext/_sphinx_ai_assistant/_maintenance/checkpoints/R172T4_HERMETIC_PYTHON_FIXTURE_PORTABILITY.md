# R172T4 — hermetic Python fixture portability local repair

## Trigger

The user's Python 3.11.14 / pytest 9.0.2 rerun from Fix 3 stopped after
937 passes and 4 skips at
`test_archive_native_status_evidence.py::test_run160_command_adapter_bounds_output_while_produced`.
The child exited 127 with `env: 'python3': No such file or directory`, so the
test observed `NATIVE_ARCHIVE_ADAPTER_FAILED` before it could exercise the
intended `OUTPUT_TOO_LARGE` contract.

## Classification

**Test environment/isolation mismatch.** Production hermeticity is intentional:
release/security adapters give children only `PATH=os.defpath`, `LC_ALL=C`, and
`LANG=C`. The failing fixture passed a temporary Python script as the executable
and relied on `#!/usr/bin/env python3`, coupling an output-bound test to whether
the host has `python3` under the hermetic PATH.

No production adapter behavior was relaxed or changed for this repair.

## Fix

The fixture contract now separates executable identity from child environment:

- argv-capable Run 160 and Run 151 command-adapter fixtures invoke Python with
  `sys.executable` explicitly;
- the Run 159 path-only vendor-verifier fixture uses an absolute interpreter
  shebang derived from `sys.executable` because that production interface
  intentionally accepts one executable path rather than argv;
- focused tests force the production `os.defpath` value to an empty directory,
  proving that interpreter discovery does not depend on child PATH;
- strict production child environment, output bounds, error classification,
  process reaping, and pipe ownership remain unchanged.

## Verification

```text
exact reported Run160 node                         1/1 passed
portable hermetic fixture probes                   3/3 passed
Run151 finalize_publication owner                 14/14 passed
Run159 native-status owner                        33/33 passed
Run160 native-archive owner                       34/34 passed
all above with ResourceWarning promoted to error
```

## Next authority

The user's next `pytest -x -vv` run against the packaged R172T4 / Fix 4
workspace.

## Packaged-byte candidate replay

An independently extracted deterministic candidate reproduced the four focused
hermetic portability nodes **4/4**, test-layout architecture **9/9**, full
collection **2321 / 0 errors**, and maintenance drift **GREEN**. Final archive
SHA-256 remains external so the maintained evidence does not self-reference the
archive whose bytes it describes.
