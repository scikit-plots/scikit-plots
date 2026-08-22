# `_sphinx_ai_assistant` maintenance control plane

This directory separates current source truth from desired architecture. It
covers four runtime planes:

1. Sphinx build-time configuration/injection;
2. browser UI/state/request logic;
3. proxy/model/edge service authority and persistence;
4. integration with the sibling `_sphinx_llm` static representation producer.

Historical research and prototypes may remain useful, but the live trackers,
registry, state file, and verification contract take precedence for fresh work.
