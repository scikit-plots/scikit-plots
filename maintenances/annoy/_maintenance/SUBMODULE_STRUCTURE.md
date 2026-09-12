# Submodule structure

```text
scikitplot/annoy/
  __init__.py / __init__.pyi       public exports; native Annoy + high-level Index
  _base.py / _base.pyi             high-level Index inheriting cexternals Annoy
  _mixins/                          Python behavior mixed into high-level Index
  _annoy/
    annoylib.pyx.in                 authoritative Cython implementation template
    annoylib.pxd.in                 authoritative Cython C++ declaration template
    meson.build                     Tempita + Cython extension pipeline
    annoymodule.cpp                 inactive checked-in legacy pybind11 source
    tests/                          focused Cython/backend contracts
  tests/                            inherited/upstream-style Annoy tests

maintenances/annoy/                 maintenance plane only
skills/annoy/SKILL.md               fresh-chat routing only
```

Do not add maintenance imports to runtime code. Do not vendor upstream Annoy
headers into this tree. Do not edit build-generated `.pyx`, `.pxd`, or C++ as a
permanent fix.
