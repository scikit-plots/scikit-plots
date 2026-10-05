# Maintaining `scikitplot.cleanprompt`

The runtime is a package at `scikitplot/cleanprompt/`. `_engine.py` is the
algorithmic core, `__init__.py` is the public, lazily-resolving facade, and
`_patterns.py` is the curated detection library. `_api.py` is the surface that
code sending prompts to a model actually touches — `encode`, `decode` and
`Session` — and `_engines.py` decides which of the two entity engines, spaCy
or NLTK, runs behind it.

Start with `_maintenance/DESIGN.md` — it states the invariants, the failure
modes and the reasoning behind every structural choice — then
`_maintenance/FRESH_CHAT_HANDOFF.md` and `REVIEW.json`.

Five properties are load-bearing and must never be traded away for
convenience:

1. **`import scikitplot.cleanprompt` imports no third-party package.** Not even
   transitively, and not even one that happens to be installed.
2. **No detector ever sees rewritten text.** Composition happens over spans.
3. **The vault is structurally separate from the redacted text**, and nothing
   serializes it implicitly.
4. **Entity labels are canonicalised before a span leaves the detector.** spaCy
   says `ORG` where NLTK says `ORGANIZATION`; a raw label makes the placeholder
   depend on which engine was installed, and a vault written under one engine
   stops restoring under the other.
5. **No removed value reaches a log record.** Every call site logs counts,
   kinds, labels and durations, never surfaces; `SecretFilter` is defence in
   depth, not the mechanism.

One further rule follows from `CP-024`: a default may degrade quietly, a
request may not. An explicit `--ner` that cannot
be met raises rather than running no engine and exiting 0.

Run the checker, the reviewer and the focused suite before touching evidence —
1521 tests pass and 6 are skipped as of this round, alongside 44 on the
maintenance plane — and run
`_maintenance/evidence/probe_engines.py` for anything touching entity
detection, since it is the only lane that exercises a real spaCy model and a
real NLTK. Runtime changes and maintenance-plane updates are separate commits.
