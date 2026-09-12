# Fresh-chat handoff — `_build_utils`

## Current truth

Maintenance tooling is healthy. Runtime/build-tool structural status is FAIL because six owned findings are open. The focused shipped suite is 25/25, but its Git-version tests do not call the production APIs they import. Direct `tempita.py` generation of the Annoy `.pxd.in` template succeeds; the broader `cython_generate.py --validate` command then rejects the generated `.pyx` because it mistakes legitimate doubled braces in embedded CSS/JavaScript for Tempita residue.

`mesonbuild` is not installed in the evidence environment, so the custom `_meson_features` package was not copied into or executed inside a live Meson installation. Do not mark that lane green from static imports.

## First commands

```bash
python maintenances/_build_utils/_maintenance/tools/check_contract.py --json
python -m pytest -q maintenances/_build_utils/_maintenance/tests
python -m pytest -q scikitplot/_build_utils/tests
```

Then read `REVIEW.json` and the evidence logs. Do not repair unrelated consumer submodules while working this domain. In particular, the current archive now contains the Tempita helper previously missing from the Annoy snapshot; that changes Annoy evidence and should be refreshed in the Annoy campaign, not silently rewritten here.
