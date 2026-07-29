# scikitplot.cython Mermaid RST user-guide package

This package contains reusable reStructuredText Mermaid diagrams for the
`scikitplot.cython` user guide.

## Contents

- `index.rst`: consolidated user-guide page
- `architecture.rst`: architecture-focused page
- `workflows.rst`: build and recovery workflows
- `lifecycle.rst`: cache, lock, GC, and pin lifecycle
- `_includes/diagrams/*.rst`: one reusable diagram per file
- `conf_snippet.py`: Sphinx extension configuration example

## Installation

1. Copy this directory into `docs/user_guide/cython/`.
2. Add `sphinxcontrib.mermaid` to `extensions` in `docs/conf.py`.
3. Add `index` to the appropriate parent `toctree`.
4. Build the documentation and inspect both HTML and link-check output.

## Include pattern

```rst
.. include:: _includes/diagrams/architecture_overview.rst
```

The include fragments intentionally contain no title and no `:orphan:` marker.
Use `:orphan:` only on standalone wrapper pages that are intentionally excluded
from all toctrees.
