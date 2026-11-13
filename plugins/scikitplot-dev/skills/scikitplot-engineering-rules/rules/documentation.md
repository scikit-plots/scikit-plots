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
- **Then:** run each one against the tree being delivered and compare the output; add a mechanical check where the class of error can recur (options per command). A docstring's `Examples` hold Python that doctest can run; a shell command goes in a `.. code-block:: bash`, never on a doctest line.
- **Verified by:** the round's notes list each page with "examples executed"; `test_documented_cli.py` passes.
- **Added:** 2026-10-09. Root cause: executing the rewritten guide found `encode --pack-file` (`CP-101`, an option that never existed), `encode_tree(source)` without its required target, and `batch --dry-run --out` (refused) — two of them copied from the previous guide. Amended 2026-11-09 (CI-9): the `scikitplot.mlflow` docstring wrote its command-line example as a doctest line, which could never run; it reached the published `scikitplot-mlflow` skill as Python and was found when that skill's example was compiled (Rule 170).

### Rule 126: Documentation - a towncrier fragment holds one top-level bullet, which an increment continues
- **When:** an increment adds to an existing `docs/source/whats_new/upcoming_changes/<section>/<PR>.<type>.rst` (all of round 33 shares PR 865).
- **Then:** continue the fragment's single `- ` bullet with an indented paragraph (two spaces, no new `- `); a new top-level bullet is a second entry the generator refuses.
- **Verified by:** the gates' "towncrier sections" step prints no `exactly one top-level '- ' bullet` line; `grep -c '^- '` on each changed fragment prints `1`.
- **Added:** 2026-10-17. Root cause: 4c appended its entries to `865.feature.rst` and `865.fix.rst` as new bullets; the gates report was read only by its last lines (`tail -25`), which cut off the towncrier step, so it was first seen after the re-review.

### Rule 172: Documentation - a public name added, renamed or removed is added, renamed or removed in the API reference template in the same change
- **When:** a change adds a name to a module's `__all__` (a class, function, constant or exception), renames one, or removes one; or moves a name to another module.
- **Then:** edit `scikitplot/_build_utils/generate_apis_reference/apis_reference.py.in`, never `docs/source/apis_reference.py`: put an added name in the `autosummary` list of the section it belongs to (a new section only when no existing one fits), under the comment that groups its source file, in that group's order; rename it where it is listed, `classes` (inheritance diagram) included; delete a removed name from every list. Then run `python -m scikitplot._build_utils.generate_apis_reference rebuild --apply` and commit the template and the generated file together. A name that is public on purpose but not documented is left out with a commented line saying why. The module's usage skill (`tools/maint_tools/generate_usage_skills.py sync`) and a towncrier fragment follow the same change.
- **Verified by:** `python -m scikitplot._build_utils.generate_apis_reference rebuild --check` prints "clean"; `... plan --json` (with a built package, or `--allow-uninstalled` from outside the source tree) no longer lists the name under the module's `missing`, `stale` or `ambiguous`; the pre-push hook `check-apis-reference` (`imports`) passes.
- **Added:** 2026-11-13. Root cause: `OfficeReader` was added to `scikitplot.corpus.__all__` over three increments with its usage skill, release note and tests, and the API reference was not touched: the reference is checked against the package only by a pre-push hook that needs a built package, so nothing in the increment's own gates named the gap; `plan` listed the name as `ambiguous`.
