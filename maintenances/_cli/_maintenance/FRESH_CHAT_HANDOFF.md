# Fresh-chat handoff — `scikitplot._cli`

`_cli` owns shell composition, not backend semantics. Preserve one neutral command
IR projected into argparse and optional click. Argparse is the deterministic
default. Handler imports stay lazy; top-level help must not import command handlers
or project-heavy subsystems.

MCP owns its subcommand parser and protocol behavior after `_cli` delegates to
`scikitplot.mcp.__main__:main`. Corpus owns retrieval semantics. Annoy owns vector
and native mechanics. `_cli` must not acquire module-scope dependencies on those
subsystems merely to expose them.

Keep stdout as the result channel and stderr as the diagnostic/logging channel.
Structured stdout must remain parseable. Missing optional serializers/backends
must fail with semantic exit/error behavior rather than raw dependency tracebacks
where the CLI contract promises recovery.

Current review findings are in `STATE.json`. In particular, do not assume the
`CommandSpec.capabilities` field is enforced: the supplied runtime declares it but
no dispatch path consumes it. Source also references a missing `EXTENDING.md`.
These are runtime findings, not permission to weaken the maintenance gate.

Run:

```sh
python -B maintenances/_cli/_maintenance/check_trackers.py --json
python -B maintenances/_cli/_maintenance/review_subsystem.py --json
python -m unittest discover -s maintenances/_cli/_maintenance/tests -p 'test_*.py'
```

Maintenance PASS, runtime structural PASS, and release PASS are separate facts.
The updated snapshot now includes `scikitplot.config` and `scikitplot.utils`, but
the surrounding top-level package integration is still incomplete (`scikitplot`
root init/exports, `_testing`, exceptions) and no TOML writer is installed. The
full suite/parity evidence therefore remains unavailable for release, not because
config/utils are absent and not because `_cli` should absorb their ownership.
