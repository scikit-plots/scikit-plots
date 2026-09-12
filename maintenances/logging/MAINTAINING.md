# Maintaining `scikitplot.logging`

The runtime is now a package at `scikitplot/logging/`; `_logging.py` is the implementation core and `__init__.py` is the public facade. Start with `_maintenance/FRESH_CHAT_HANDOFF.md` and `REVIEW.json`. Do not use the old `scikitplot/logging.py` path.

Run the checker, reviewer, and maintenance tests before changing evidence. Runtime fixes are separate from maintenance-plane updates.
