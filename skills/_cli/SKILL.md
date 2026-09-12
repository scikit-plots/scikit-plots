---
name: cli-maintainer
description: Maintain scikitplot._cli as the project-wide shell adapter. Use for neutral command/parameter specs, argparse/click parity, lazy handler loading, delegated subcommands, stdout/stderr contracts, output formats, semantic exit codes, optional CLI dependencies, module runners, and CLI release verification.
---

# `_cli` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and
`skills/`.

## Read first

1. `maintenances/_cli/MAINTAINING.md`
2. `maintenances/_cli/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/_cli/_maintenance/STATE.json`
4. `maintenances/_cli/_maintenance/FAMILY.md`
5. `maintenances/_cli/_maintenance/VERIFICATION.md`

Then run:

```sh
python -B maintenances/_cli/_maintenance/check_trackers.py --json
python -B maintenances/_cli/_maintenance/review_subsystem.py --json
```

## Preserve one command model

`Param` and `CommandSpec` are the neutral source of truth. Argparse and click are
projections of that model, not separate command definitions. Any parameter or
command change must be checked in both frontends and in the parity matrix.

Argparse is the deterministic default and must remain usable without click. Do
not import click from the bootstrap path merely to share convenience helpers.

## Keep loading lazy

Top-level help and registry import must not import handlers. Native command
handlers may lazily import project APIs inside `run()`. Optional serializers and
frontends stay optional at package import time.

Do not add module-scope MCP, Corpus, or Annoy dependencies. Shell exposure is not
ownership of backend behavior.

## Delegation is an ownership transfer

A delegated command receives trailing argv verbatim. `_cli` selects the command,
forwards argv, normalizes import/SystemExit behavior, and preserves process exit
semantics. The target submodule owns its own parser, help, options, protocol, and
business logic.

For MCP specifically, `_cli` delegates to `scikitplot.mcp.__main__:main`; never
mirror the MCP parser inside `_cli`.

## Protect stdout/stderr and exit codes

Stdout is the result channel. Stderr is diagnostics, warnings, usage text, and
logging. Structured JSON/YAML/TOML stdout must remain machine-parseable with no
log contamination.

Keep semantic exit codes centralized in `exit_codes.py`. Optional dependency or
capability failures should be actionable and mapped deliberately, not swallowed
or accidentally converted to success.

## Treat metadata as executable contract

Do not add fields to `CommandSpec` that claim runtime behavior unless a dispatch
path actually enforces them. The current snapshot has an open finding because
`capabilities` is documented as pre-dispatch enforcement but is not consumed.
Resolve the implementation/metadata mismatch rather than teaching maintenance to
ignore it.

Source references to extension documentation must resolve to a live document.
The current `EXTENDING.md` references are an open finding.

## Evidence discipline

Maintenance PASS, runtime structural PASS, and release PASS are separate facts.
A partial archive that lacks `scikitplot.config` or `scikitplot.utils` cannot
prove the full command suite. Missing optional TOML/click dependencies likewise
make their lanes unavailable unless the test environment installs them.

Before release, require current full-suite, frontend-parity, installed-entrypoint,
and live delegated-command evidence for the same runtime fingerprint.
