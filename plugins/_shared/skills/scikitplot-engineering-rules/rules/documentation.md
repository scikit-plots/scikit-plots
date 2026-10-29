# Documentation

Categories: Documentation, Docstrings

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 48: Docstrings - a backslash makes the docstring raw, and is then written once
- **When:** a docstring contains a backslash (a Windows path, a regular expression).
- **Then:** open it with `r"""` and write each backslash once; read the rendered text.
- **Verified by:** `python -c "import m; print(m.f.__doc__)"` shows the path as intended.
- **Added:** 2026-10-08. Root cause: a doubled backslash in a plain docstring was flagged by the linter; made raw by the maintainer, it then showed two backslashes.

### Rule 54: Documentation - every example is executed before it is published
- **When:** a guide page, README section or docstring shows a command or a call.
- **Then:** run each one against the tree being delivered and compare the output; add a mechanical check where the class of error can recur (options per command).
- **Verified by:** the round's notes list each page with "examples executed"; `test_documented_cli.py` passes.
- **Added:** 2026-10-09. Root cause: executing the rewritten guide found `encode --pack-file` (`CP-101`, an option that never existed), `encode_tree(source)` without its required target, and `batch --dry-run --out` (refused) — two of them copied from the previous guide.

### Rule 126: Documentation - a towncrier fragment holds one top-level bullet, which an increment continues
- **When:** an increment adds to an existing `docs/source/whats_new/upcoming_changes/<section>/<PR>.<type>.rst` (all of round 33 shares PR 865).
- **Then:** continue the fragment's single `- ` bullet with an indented paragraph (two spaces, no new `- `); a new top-level bullet is a second entry the generator refuses.
- **Verified by:** the gates' "towncrier sections" step prints no `exactly one top-level '- ' bullet` line; `grep -c '^- '` on each changed fragment prints `1`.
- **Added:** 2026-10-17. Root cause: 4c appended its entries to `865.feature.rst` and `865.fix.rst` as new bullets; the gates report was read only by its last lines (`tail -25`), which cut off the towncrier step, so it was first seen after the re-review.
