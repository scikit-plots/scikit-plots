# Verification policy

A release-quality `_cli` review needs all of the following lanes current for the
same runtime fingerprint:

1. Maintenance contract regression tests.
2. SDK/backend-independent CLI core tests: import optionality, delegation, doctor,
   error taxonomy, and stream/output contracts.
3. Full `_cli` test suite in a complete project checkout where `scikitplot.config`
   and `scikitplot.utils` are integrated through the real top-level package API
   (`scikitplot.__init__`, logger/show_config exports, `_testing`, exceptions) and
   declared test extras/writers are installed.
4. Full argparse/click parity matrix when click is installed.
5. Installed console-script and `python -m scikitplot._cli` smoke checks from a
   built wheel/sdist environment.
6. At least one live delegated command path (currently MCP), verifying verbatim
   argv ownership and semantic exit propagation.

`UNAVAILABLE` means the lane could not be proven in this snapshot/environment. It
must never be silently converted to GREEN.
