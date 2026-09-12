# Maintaining `scikitplot._cli`

`scikitplot._cli` is the shell-facing composition layer. It owns command metadata,
frontend projection, dispatch/delegation, stream discipline, and CLI-specific exit
semantics. It does **not** own the behavior of MCP, Corpus, Annoy, or project APIs
that commands wrap.

Start every review with:

```sh
python -B maintenances/_cli/_maintenance/check_trackers.py --json
python -B maintenances/_cli/_maintenance/review_subsystem.py --json
```

Then read `_maintenance/FRESH_CHAT_HANDOFF.md` and `_maintenance/STATE.json`.
Do not mark a release green from static checks alone; frontend parity, packaging
entry-point behavior, optional writers, top-level package integration for wrapped
config/utils APIs, and delegated subcommands need executable evidence in a
sufficiently complete checkout/environment.
