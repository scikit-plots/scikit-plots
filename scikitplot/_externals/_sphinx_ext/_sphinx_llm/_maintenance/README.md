# `_sphinx_llm` maintenance control plane

This directory is the durable source of maintenance truth. It separates:

- **logical contracts** — what the subsystem promises;
- **physical inventory** — what exists on disk and what may grow;
- **campaign state** — what is being changed now;
- **verification evidence** — what has actually been proved;
- **upstream provenance** — exactly what came from NVIDIA and how it is synced;
- **historical rationale** — useful completed context that must not override
  current source truth.

A fresh session should not need a previous chat transcript to recover current
architecture, open risks, or the next exact action.
