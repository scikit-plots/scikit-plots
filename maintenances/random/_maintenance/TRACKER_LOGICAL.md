# Logical tracker — `scikitplot.random`

- Runtime owner: `scikitplot/random`.
- Shared upstream: `scikitplot/cexternals/_annoy/src/kissrandom.h`.
- Public implementation/stub pair: `scikitplot/random/_kiss/kiss_random.pyx` / `scikitplot/random/_kiss/kiss_random.pyi`.
- Build owner: `scikitplot/random/_kiss/meson.build` (`kiss_random`, C++ Cython).
- Tests: `scikitplot/random/_kiss/tests/test_kiss_random.py`, `scikitplot/random/_kiss/tests/test_kiss_random_unit_interval.py`, `scikitplot/random/_kiss/tests/test_kiss_state_continuation.py`.
- Runtime must not import maintenance or skill code.
- Maintenance pass does not imply native/runtime release readiness.
