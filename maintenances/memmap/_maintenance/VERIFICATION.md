# Verification — `scikitplot.memmap`

Run structural maintenance checks from the wide repository:

```sh
python -B maintenances/memmap/_maintenance/check_trackers.py --json
python -B maintenances/memmap/_maintenance/review_subsystem.py --json
python -B maintenances/memmap/_maintenance/tests/test_contract.py
```

The contract checks the upstream header edge, required public symbols in implementation/stub, C++ Cython build intent, runtime/maintenance plane separation, fresh-chat files, inventory fingerprint and evidence hashes. It never imports `scikitplot`. Metadata is declarative; no command from JSON is executed.

For a runtime change, also perform a clean Meson/Cython build and run:

```sh
python -m pytest scikitplot/memmap/_memmap/tests/test_mman.py -q -p no:cacheprovider
```

For `mman.h` changes, use the `cexternals/_annoy` maintenance owner and verify all affected consumers. For release claims, `review_subsystem.py --release` requires every gate named in `REVIEW.json` to be current `GREEN`; missing toolchains, unbuilt extensions and untested platforms are `UNAVAILABLE`, not `PASS`.
