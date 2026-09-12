# Corpus maintenance model

**Why:** prevent plausible-but-unverified Corpus behavior and prevent stale campaign prose from masquerading as current evidence.

**When:** run the structural review before semantic work, after source restoration, after contract movement, and before any release claim.

**Where:** runtime under `scikitplot/corpus/`; maintenance under `maintenances/corpus/`; onboarding skill under `skills/corpus/`.

**Which:** current source and tests outrank historical trackers/checkpoints. Cross-module defects are edited at the owning boundary.

**How many:** one live state file, one live registry, one evidence ledger, one skill. Historical narratives stay in `history/`.

**How much:** static gates prove structure only. Runtime semantics, examples, optional dependencies, and consumer integration require execution evidence.
