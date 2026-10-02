# Private Sphinx extension stack

## Scope

Package root:

`docs/source/scikitplot/_externals/_sphinx_ext/`

The same private extension tree is designed to work in two deployment layouts:

- development/test: standalone `_sphinx_ext` package from the documentation checkout;
- stable/release: `scikitplot._externals._sphinx_ext` from the installed Scikit-plots library.

Internal sibling imports are relative, so module implementation does not change when the outer package namespace changes.

## One authority per Sphinx process

The documentation bootstrap chooses the extension authority once in `docs/source/conf.py` using `SCIKITPLOT_SPHINX_EXT_MODE`.

Supported modes:

### `local`

Uses:

`docs/source/scikitplot/_externals/_sphinx_ext`

as standalone package `_sphinx_ext`.

This is the default used by `docs/Makefile` and `docs/make.bat` for development.

### `installed`

Uses:

`scikitplot._externals._sphinx_ext`

from the active Python environment. Stable/release documentation should select this explicitly after the matching extension stack has been published with Scikit-plots.

### `auto`

Selects automatically only when exactly one authority is available. If both local and installed stacks are visible, it fails closed rather than guessing.

## Build commands

Development:

```bash
cd docs
make html
# or explicitly:
make SPHINX_EXT_MODE=local html
```

Stable/release:

```bash
cd docs
make SPHINX_EXT_MODE=installed html
```

Windows development uses `make.bat html`; stable/release sets `SCIKITPLOT_SPHINX_EXT_MODE=installed` before invoking `make.bat`.

## Compatibility handshake

`_sphinx_ext/__init__.py` exposes:

`SPHINX_EXT_STACK_API = 1`

The docs bootstrap checks this API before loading children. Cross-extension incompatible changes should bump the stack API and the whole private extension stack should be released together.

Narrower subsystems can expose their own contracts. The shared collection browser currently exposes:

`COLLECTION_UI_CONTRACT = "controls-status-results-v4"`

Stable docs must not silently use an installed extension stack with an incompatible expected contract.

## Relative-import rule

Inside `_sphinx_ext`, sibling dependencies must use package-relative imports, for example:

```python
from .._sphinx_collection import apply_selection
from .model import VideoRecord
```

Do not introduce absolute sibling imports such as:

```python
from scikitplot._externals._sphinx_ext._sphinx_collection import ...
```

or:

```python
from _sphinx_ext._sphinx_collection import ...
```

inside the module implementation. Absolute names are selected only at the outer bootstrap/extension registration boundary.

This prevents a process that already imported another `scikitplot` package from anchoring only part of the extension graph to a different source tree.

## Namespace consistency

`_extension_setup.check_namespace()` rejects mixed private-extension roots in one Sphinx application. It also detects obsolete private package names that have been replaced by the current ownership split.

Do not add per-module `try installed / except local` fallbacks. Fallbacks at individual imports can mix versions in a single process and are harder to diagnose than an explicit authority error.

## Current top-level modules

- `_pydata_component_list` — PyData theme component inventory directive.
- `_sphinx_collection` — shared collection/query/browser engine.
- `_sphinx_gallery_grid` — generic gallery grid directive.
- `_sphinx_youtube_core` — YouTube provider grammar/options.
- `_sphinx_youtube_gallery` — YouTube gallery adapter.
- `_sphinxcontrib_youtube` — standalone media/player directives.
- `_sphinx_ai_assistant` — AI Assistant and service assets.
- `_sphinx_feedback` — generic page feedback.
- `_sphinx_ai_learn` — JSON-first Learn materializer/pages/publication.

The namespace initializer is lazy so importing `_sphinx_ext` itself does not eagerly pull Sphinx or other heavy optional dependencies.

## Promotion workflow

Use one code tree through development and release:

1. edit/test the checkout-local `_sphinx_ext` tree in `local` mode;
2. release the complete compatible tree under `scikitplot/_externals/_sphinx_ext`;
3. install that Scikit-plots release in stable documentation CI;
4. build stable docs with `installed` mode;
5. let the stack API and subsystem contracts fail clearly if docs and library versions do not match.

Do not infer release authority from a Git branch, editable-install path, or filesystem prefix. Mode selection is explicit.

## Safe editing rules

1. Select authority once in `conf.py` before Sphinx imports children.
2. Keep sibling imports relative.
3. Never mix `_sphinx_ext.*` and `scikitplot._externals._sphinx_ext.*` in one Sphinx application.
4. Release incompatible sibling changes as one extension stack.
5. Keep package initializers lazy where practical.
6. Use explicit compatibility contracts instead of filesystem-path guesses.
7. Keep this file current-state only and focused on the active contract.
