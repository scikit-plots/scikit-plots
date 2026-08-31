# `_sphinx_ai_assistant` maintenance control plane

This directory is the durable maintenance control plane for the runtime module:

```text
scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/
```

It intentionally lives at:

```text
maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/_maintenance/
```

rather than inside production source.

It separates current source truth from desired architecture across four runtime
planes:

1. Sphinx build-time configuration/injection;
2. browser UI/state/request logic;
3. proxy/model/edge service authority and persistence;
4. build-time static representation produced by the extension.

Historical research and prototypes may remain useful, but `STATE.json`, the
live trackers, registry, active runbooks, checkpoints, and verification contract
take precedence for fresh work.

For proxy/chat failures start with `APP_STREAMING_RUNBOOK.md`, not historical
chat transcripts or archived prototypes.
