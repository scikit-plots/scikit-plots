# Fresh-chat handoff — `_build_utils`

## Current truth

Maintenance tooling is healthy and the runtime/build-tool structural status is PASS: the seven owned findings were closed on 2026-11-04 (`REVIEW.json` resolutions, `evidence/repair-2026-11-04.log`). The shipped suite is 80 tests and calls the production APIs. Release stays BLOCKED on the integration lanes only: the feature installer was verified live on meson 1.12.1, but no full Meson build or cross-platform run has been recorded.

`mesonbuild` is not installed in the evidence environment, so the custom `_meson_features` package was not copied into or executed inside a live Meson installation. Do not mark that lane green from static imports.


The domain also contains the `generate_apis_reference/` maintenance package. Its adjacent `apis_reference.py.in` file is the durable source of truth for generated `docs/source/apis_reference.py`; deleting the generated docs file is recoverable with `python -m scikitplot._build_utils.generate_apis_reference rebuild --apply`. The helper parses the template/generated configuration with AST rather than executing it, while runtime API verification remains a separate installed-distribution lane. The focused rebuild/generator tests are self-contained; installed-package comparison still requires an installed Scikit-Plots distribution.

## First commands

```bash
python maintenances/_build_utils/_maintenance/tools/check_contract.py --json
python -m pytest -q maintenances/_build_utils/_maintenance/tests
python -m pytest -q scikitplot/_build_utils/tests
```

Then read `REVIEW.json` and the evidence logs. Do not repair unrelated consumer submodules while working this domain. In particular, the current archive now contains the Tempita helper previously missing from the Annoy snapshot; that changes Annoy evidence and should be refreshed in the Annoy campaign, not silently rewritten here.
