# Verification and release boundaries

All commands below run from the repository root; absolute script paths also
work from another current directory. Tools do not import the installed package.

```sh
python -B maintenances/cexternals/_annoy/_maintenance/check_trackers.py
python -B maintenances/cexternals/_annoy/_maintenance/dependency_map.py
python -B maintenances/cexternals/_annoy/_maintenance/review_subsystem.py --json
python -B -m unittest discover -s maintenances/cexternals/_annoy/_maintenance/tests -v
```

After reviewing deliberate inventory/dependency changes:

```sh
python -B maintenances/cexternals/_annoy/_maintenance/check_trackers.py --update
```

This refreshes derived metadata, not test results. Record real evidence in
[EVIDENCE.json](EVIDENCE.json) with the review's current `input_digest`, a concise
scope statement, and the SHA-256 of the captured log. Result logs belong under
`_maintenance/evidence/`. Missing tools are UNAVAILABLE; failed assertions or
compilation are FAIL. Never record a skipped platform branch as platform PASS.

```sh
python -B maintenances/cexternals/_annoy/_maintenance/review_subsystem.py --release
```

Exit codes: 0 passes the requested gate; 1 means failed checks, unavailable
release evidence, or a closed output pipe; 2 means invalid metadata/input. JSON
output reports maintenance and release independently. Metadata cannot register
shell commands. The native verifier has its own explicit, fixed commands and
writes binaries only into a temporary directory.

| Gate | What it establishes |
|---|---|
| `maintenance_tests` | Positive/negative behavior of the maintenance contract and CLI |
| `native_cpp` | Existing standalone C++ regression executables on this host; not a full platform matrix |
| `native_extension` | Standalone native CPython extension smoke, when available; not the Meson build |
| `consumer_build` | Clean native/Cython builds of `_annoy`, `annoy`, `random`, `memmap` with real project tooling |
| `downstream_runtime` | Impute index behavior, both Corpus backends/persistence, and MCP-to-Corpus integration |
| `windows_runtime` | Real Windows mmap/file lifecycle and extension integration; host mocks are insufficient |

The latest full ZIP includes the root build configuration. The current host
lacks Meson/Ninja/Cython/pytest; see the ledger for actual results. Windows tests
and optional hardware/platform branches require their respective hosts.

Exact inventory hashes detect any runtime change, including one byte or a new
file. Syntax errors in inspected runtime sources fail instead of disappearing
from the graph. Comments/docstrings are not Python imports or Cython externs.
Static scanning cannot certify computed imports, Cython signature parity, linker
behavior, numerical correctness or thread/file lifecycle safety. Clean builds
and appropriate behavior tests remain required for native release claims.

This maintenance-and-skill-only slice does not run native tests. The optional
`tools/verify_native.py` uses temporary output and never marks compile-only or
unavailable probes as native runtime PASS. Runtime caches are observed in state
and preserved; hygiene checks apply to maintained tooling and skill output.
