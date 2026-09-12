# Maintenance model

The checker reports maintenance health separately from runtime health. A maintenance PASS means the guardrails, skill, metadata and evidence are coherent; it does **not** mean dataset runtime is releasable. `--update` refuses to bless a runtime that still violates the structural contract.
