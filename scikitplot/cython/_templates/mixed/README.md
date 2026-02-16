# scikitplot.cython templates

This directory contains **20 packaged `.pyx` templates** intended as copy/paste
starting points and micro-demos. Each template is **strictly unique** and
focuses on a different Cython concept (typed args, memoryviews, NumPy ndarrays,
compiler directives, structs/enums, fused types, cdef classes, etc.).

## How to list templates

```python
from scikitplot.cython import list_templates
print(list_templates())
```

## How to compile and run a template

```python
from scikitplot.cython import compile_template
m = compile_template("t01_square_int")
print(m.f(10))
```

## Notes
- Templates are packaged under `scikitplot.cython._templates` (private).
- Use the public helper `compile_template()` to compile them deterministically.
- Some templates require NumPy (`numpy_support=True` is the default).
