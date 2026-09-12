# Dependency map

Production `scikitplot.cython` has no sibling-submodule dependency by design.
Its unusual dependencies are a **toolchain** and **caller-supplied source**:

```text
caller source -> scikitplot.cython -> Cython/setuptools -> C/C++ compiler
                   |       |
                   |       +-> native artifact -> transactional loader
                   +-> cache / locks / pins / GC
```

A new `scikitplot.<sibling>` import is an architectural change and must not be
normalized as a convenience dependency.
