---
title: "Synchronize CleanPrompt README command inventory"
status: open
kind: "docs-contract"
area: "scikitplot/cleanprompt"
discovered_during: "source-grounded CleanPrompt user-guide synchronization"
release_note: "not-required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "other"
towncrier_fragment: ""
---

# Synchronize CleanPrompt README command inventory

## Summary

``scikitplot/cleanprompt/README.md`` says that the command line has "Ten
subcommands", but ``scikitplot/cleanprompt/_cli.py`` currently defines 18
canonical commands.

## Why it matters

The README is a contributor/user source of truth and is also richer than the
current user guide. A stale command count makes later documentation work easy
to copy incorrectly and hides newer surfaces such as ``batch``, ``ask``,
``mcp``, ``plan``, ``skill``, and ``docker``.

## Current evidence

- ``scikitplot/cleanprompt/README.md``: the Command line section states "Ten
  subcommands".
- ``scikitplot/cleanprompt/_cli.py``: ``COMMANDS`` contains 18 canonical
  ``Command`` entries: ``redact``, ``decode``, ``inspect``, ``encode``,
  ``forget``, ``roundtrip``, ``scan``, ``doctor``, ``kinds``, ``cli``,
  ``flask``, ``packs``, ``batch``, ``ask``, ``mcp``, ``plan``, ``skill``, and
  ``docker``.
- ``python -m scikitplot.cleanprompt --help`` exposes those commands plus their
  aliases.

## Root cause / current understanding

The README command overview predates later CleanPrompt workflow additions and
its prose count was not derived from the canonical ``COMMANDS`` schema.

## Expected behavior

The README should describe the current command groups without a hand-maintained
numeric count, or the count should be generated/verified from ``COMMANDS``.

## Affected paths and ownership

- ``scikitplot/cleanprompt/README.md``
- ``scikitplot/cleanprompt/_cli.py``
- ``docs/source/user_guide/cleanprompt/index.rst``

## Constraints and non-goals

Do not rename commands as part of the documentation fix. Preserve aliases and
the single ``COMMANDS`` definition used by both CLI frontends.

## Edge cases to cover

- canonical commands versus aliases;
- future additions/removals;
- rendered Click and argparse command inventories.

## Proposed direction

Replace the fixed numeric wording with grouped task-oriented wording and add a
small static documentation check that derives canonical command names from the
``COMMANDS`` AST.

## Verification / acceptance criteria

- README no longer claims an obsolete command count;
- every canonical command is represented or intentionally grouped;
- command aliases are not counted as additional canonical commands.

## Documentation impact

Keep the CleanPrompt user guide and gallery README synchronized with the same
canonical command surface.

## Release-note promotion

No Towncrier fragment is required for documentation-only synchronization.
