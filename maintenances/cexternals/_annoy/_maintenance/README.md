# _annoy maintenance entry

Read [MAINTAINING.md](../MAINTAINING.md) and
[FRESH_CHAT_HANDOFF.md](FRESH_CHAT_HANDOFF.md). These are the current continuation
routes. [MAINTENANCE.json](../MAINTENANCE.json) is the ownership contract;
[REVIEW.json](../REVIEW.json) declares the fixed check lanes.

The checker is standalone Python standard library code. It locates the wide
repository from its script path and never imports `scikitplot`. It observes six
consumer modules without changing their maintenance or skill directories.

`history/` and `../_backup/` preserve old rationale and upstream provenance.
They are not current source, instructions, verification, or inventory authority.
