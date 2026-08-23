"""
Maintenance tracking helper files like pytest `tests` logic structure.

Main paclage and each submodule (nested) structure sync with
library::

    tests ~ maintenances ~ module
    ├── _backup          → test_ ~ module specific backup files
    ├── _maintenance     → test_ ~ module specific maintenance md files
    |   ├── checkpoints  → test_ ~ module specific checkpoints md files
    |   └── history      → test_ ~ module specific historical md files
    ├── MAINTAINING.md   → test_ ~ module specific main maintenance md file
    └── __init__.py
"""
