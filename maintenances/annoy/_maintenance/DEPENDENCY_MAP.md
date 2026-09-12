# Dependency map

```text
scikitplot/cexternals/_annoy/src/annoylib.h -----------+
scikitplot/cexternals/_annoy/src/kissrandom.h ----------+--> annoy/_annoy Cython extension
scikitplot/cexternals/_annoy/src/annoy_type_support.h --+

scikitplot/cexternals/_annoy.Annoy ------------------------> annoy/_base.Index

annoylib.pyx.in + annoylib.pxd.in
          -> Tempita generator
          -> build-generated annoylib.pyx/.pxd
          -> Cython-generated C++
          -> compiled scikitplot.annoy._annoy.annoylib
```

The first edge is source/ABI coupling; the second is Python extension-type
coupling. They require different tests and neither makes `annoy` the owner of
`cexternals/_annoy`.
