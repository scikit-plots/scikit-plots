# scikitplot/datasets/__init__.py

"""
Utilities to load popular datasets and artificial data generators.

::

    scikitplot.datasets
    ├── _load_dataset.py
    │   └── named remote datasets / cache / normalization
    │
    ├── _data_export.py + .pyi
    │   └── deterministic sampling / streaming / profiles / manifests / CLI
    │
    ├── _data_loader.py
    │   └── files / streams / URLs / ZIP / uploads / DBs / optional backends
    │
    └── _autoscout24_tasks.py
        └── examples only
"""

from . import _load_dataset
from ._load_dataset import *  # noqa: F403

__all__ = []
__all__ += _load_dataset.__all__
