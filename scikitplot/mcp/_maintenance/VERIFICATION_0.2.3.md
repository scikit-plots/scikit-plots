# Verification — 0.2.3

Validated in the sandbox without the optional MCP SDK installed:

- 102 passed, 2 intentional skips.
- Two consecutive ordinary test runs.
- Reverse-order execution.
- Root-level nested ZIP does not alter the source manifest.
- Source fixture ZIP remains eligible for manifest tracking.
- Manifest `--write` is a no-op when current.
- Python `compileall` passes.
- Shell scripts pass `bash -n`.
- Deterministic ZIP generation and clean extraction tests pass.

The real SDK live acceptance test must be rerun in Docker after rebuilding the
0.2.3 server. The expected formerly failing case is `extra argument`, which
should now return a tool validation error.
