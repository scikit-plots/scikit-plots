# Maintaining `scikitplot._brand`

`scikitplot._brand` owns deterministic Matplotlib logo generation and terminal-banner generation. It does not own Matplotlib, NumPy, the system `figlet` executable, remote font hosting, or root package entrypoint configuration.

Start with `_maintenance/FRESH_CHAT_HANDOFF.md`, run the maintenance contract/review tools, then run the focused logo suite. Keep logo rendering and banner generation as separate evidence lanes.
