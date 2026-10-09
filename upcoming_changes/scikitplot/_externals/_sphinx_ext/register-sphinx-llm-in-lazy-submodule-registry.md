---
title: "Register _sphinx_llm in the _sphinx_ext lazy submodule registry"
status: open
kind: "bug"
area: "scikitplot/_externals/_sphinx_ext"
discovered_during: "source-grounded Sphinx extension user-guide synchronization"
release_note: "required"
towncrier_section: "scikitplot._externals"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Register _sphinx_llm in the _sphinx_ext lazy submodule registry

## Summary

The package-level documentation says child submodules are lazily reachable, but
``_sphinx_llm`` exists as a child package and is omitted from both
``_CORE_PRIVATE_SUBMODULES`` and ``_OPTIONAL_PRIVATE_SUBMODULES``.

## Why it matters

Direct import of the child package can work while attribute-style lazy access
through ``scikitplot._externals._sphinx_ext`` raises ``AttributeError`` and
``dir()`` omits the installed child.  That makes namespace introspection
inconsistent with the package's stated lazy-loading contract.

## Current evidence

- physical package:
  ``scikitplot/_externals/_sphinx_ext/_sphinx_llm/__init__.py``;
- lazy registry:
  ``scikitplot/_externals/_sphinx_ext/__init__.py`` lists the core and optional
  private submodules but does not include ``_sphinx_llm``;
- ``__getattr__`` only loads names present in ``_PRIVATE_SUBMODULES``.

## Root cause / current understanding

The lazy registry predates the newer ``_sphinx_llm`` package and was not
synchronized when that child was added.

## Expected behavior

If ``_sphinx_llm`` remains a supported bundled child, the root namespace should
expose it consistently with the other optional/private Sphinx extensions while
retaining lazy imports.

## Affected paths and ownership

- ``scikitplot/_externals/_sphinx_ext/__init__.py``
- root namespace tests
- Sphinx-extension user-guide index

## Constraints and non-goals

Do not eagerly import Sphinx or ``_sphinx_llm`` at package import time.  Preserve
``__all__ = []`` and lazy-loading behavior.

## Edge cases to cover

- environments where the child is present;
- standalone replacement archives where it is absent;
- ``dir()`` and ``__getattr__``;
- no eager optional-dependency import.

## Proposed direction

Add ``_sphinx_llm`` to the optional lazy registry so ``find_spec`` continues to
control exposure when the child is actually present.

## Verification / acceptance criteria

- ``dir(scikitplot._externals._sphinx_ext)`` includes ``_sphinx_llm`` when
  installed;
- attribute access lazy-imports the child;
- importing the root still does not import Sphinx or the LLM child eagerly;
- standalone trees without the child remain valid.

## Documentation impact

No user-guide content change should be required beyond removing any temporary
caveat if one is added.

## Release-note promotion

Promote as a ``fix`` for the Sphinx-extension namespace when implemented.
