# Keep useful helper scripts in the repository

Apply whenever you write a script to analyse, sweep, measure, benchmark or
sample the repository, its submodules or real data.

1. A helper whose output becomes evidence (a pull request, a ledger note, a
   baseline) or that would be run again belongs in the repository, not in a
   scratch folder: `tools/maint_tools/` for a repository-wide helper,
   `maintenances/<area>/_maintenance/tools/` for one area's.
2. Extend an existing helper before adding one.
3. Meet the folder's standard: NumPyDoc module docstring, `argparse` and
   `main()`, `--json` output, tests that need no build, no import of the
   package at import time, tests run on the Python floor it claims.

Full rule: engineering Rule 173
(`plugins/_shared/skills/scikitplot-engineering-rules/rules/tooling.md`).
