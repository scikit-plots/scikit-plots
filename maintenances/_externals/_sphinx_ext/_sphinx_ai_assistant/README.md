# `_sphinx_ai_assistant` maintenance plane

This directory mirrors the runtime module path without becoming a runtime
dependency.

Start with [`MAINTAINING.md`](./MAINTAINING.md). A fresh chat should then read
`_maintenance/FRESH_CHAT_HANDOFF.md` and `_maintenance/STATE.json`.

```text
_backup/                 maintenance-only retained backups
_maintenance/            current contracts, schemas, checkpoints, tools, history
MAINTAINING.md            human/fresh-chat entry point
README.md                 this index
todo/                     current work + reusable lessons
```

Production code must never import this tree.
