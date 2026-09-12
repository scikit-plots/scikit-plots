# Dependency map — `scikitplot.memmap`

```text
scikitplot.cexternals._annoy
  mman.h
    -> scikitplot.memmap
              -> scikitplot/memmap/_memmap/mem_map.pxd
       -> scikitplot/memmap/_memmap/mem_map.pyx
```

The relative Cython extern edge is checked against the exact repository path. This proves file ownership/path resolution only; it does not prove ABI or behavior.
