# Verification

Release verification is intentionally layered.

1. `python -B -m pytest maintenances/cython/_maintenance/tests -q -p no:cacheprovider`
2. `python -B maintenances/cython/_maintenance/check_trackers.py --json`
3. Run the runtime suite from a **complete** `scikitplot` package root and inspect
   skips with `-rs`; do not accept CYTHON-CON-002 self-skips as compile evidence.
4. Run at least one real `compile_and_load` smoke test with a fresh cache.
5. Re-run the three real-compile tests that self-skip in the current full-suite
   order until the order-dependence is fixed.
6. Run the standalone probe battery. Remember `repro_con001.py` has inverted
   success polarity.
7. Validate source/wheel contents contain no `__pycache__`/`.pyc`.
8. For release claims, exercise supported platform/toolchain combinations; a
   Linux GCC/Clang host does not prove Windows/MSVC behavior.

The current archive has strong Linux runtime evidence but is not release-green.
