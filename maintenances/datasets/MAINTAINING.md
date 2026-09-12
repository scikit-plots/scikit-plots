# Maintaining `scikitplot.datasets`

Start with `_maintenance/FRESH_CHAT_HANDOFF.md`, then run the contract checker, subsystem reviewer and maintenance regression suite. Keep the three surfaces (curated datasets, export, general loader) independent in evidence and ownership.

The maintenance layer may inspect runtime source but runtime code must never import `maintenances` or `skills`. Runtime edits require focused regressions and updated evidence.
