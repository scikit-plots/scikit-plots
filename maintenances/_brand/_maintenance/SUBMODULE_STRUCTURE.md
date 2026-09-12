# Submodule structure

```text
scikitplot._brand
├── __init__.py
├── _logo.py
├── _banner.py
└── tests/test__logo.py
```

`_logo.py` and `_banner.py` are independently executable implementation modules. Package initialization must not preload them in a way that makes `python -m` execution noisy or ambiguous.
