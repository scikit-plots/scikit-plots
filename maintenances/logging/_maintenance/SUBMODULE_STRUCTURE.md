# Submodule structure

```text
scikitplot/logging/
├── __init__.py
├── _logging.py
└── tests/
    ├── __init__.py
    ├── test__logging.py
    └── test__logging.sh
```

`__init__.py` is the public API boundary. `_logging.py` contains implementation and private helpers. Tests must name which layer they target.
