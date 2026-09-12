# Logical tracker — `scikitplot.memmap`

- Runtime owner: `scikitplot/memmap`.
- Shared upstream: `scikitplot/cexternals/_annoy/src/mman.h`.
- Public implementation/stub pair: `scikitplot/memmap/_memmap/mem_map.pyx` / `scikitplot/memmap/_memmap/mem_map.pyi`.
- Build owner: `scikitplot/memmap/_memmap/meson.build` (`mem_map`, C++ Cython).
- Tests: `scikitplot/memmap/_memmap/tests/test_mman.py`.
- Runtime must not import maintenance or skill code.
- Maintenance pass does not imply native/runtime release readiness.
