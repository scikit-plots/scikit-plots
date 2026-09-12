---
name: datasets-maintainer
description: Maintain scikitplot.datasets curated downloads/cache/postprocessing, deterministic dataset export/provenance, general file/URL/upload/database loading, tests and release evidence.
---

# `scikitplot.datasets` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. This skill owns **three independent runtime surfaces**: curated example datasets, deterministic export tooling, and the general-purpose loader. A green result from one surface never certifies the others.

## Read first

1. `maintenances/datasets/MAINTAINING.md`
2. `maintenances/datasets/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/datasets/_maintenance/STATE.json`
4. `maintenances/datasets/_maintenance/FAMILY.md`
5. `maintenances/datasets/_maintenance/VERIFICATION.md`
6. `maintenances/datasets/REVIEW.json`

Then run:

```sh
python -B maintenances/datasets/_maintenance/check_trackers.py --json
python -B maintenances/datasets/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/datasets/_maintenance/tests -q -p no:cacheprovider
```

## Keep the three surfaces separate

`_load_dataset.py` owns the small named example-dataset experience. `_data_export.py` owns reproducible subset generation and provenance. `_data_loader.py` is a much broader ingestion dispatcher over local files, streams, URLs, archives, uploads and databases. Do not use `_data_export`'s large passing test suite as evidence for `_data_loader`. Current finding **DSET-TEST-001** exists because `_data_loader.py` has no focused tests at all.

## Curated dataset normalization must preserve source values

Dataset-specific categorical ordering is public behavior. The allowed-category list must contain the real source labels. Current **DSET-TIPS-001** uses `"Their"` instead of `"Thur"`, silently converting a valid Thursday value to missing. Add regressions that assert values as well as dtype/category metadata.

## Cache/download correctness is transactional work

Cache hits, first download, interrupted download, corrupt/partial files, concurrent creators and offline reuse are separate states. Prefer staged downloads plus atomic replacement when changing cache behavior. Do not infer a valid cache merely from final-path existence. Record live network/cache evidence separately from mocked URL tests.

## Archive loading must operate on the archive member

For ZIP input, explicitly select and return/read a member. Current **DSET-ZIP-001** calls `get_file_from_zip()` and discards its result, then reopens the ZIP container; a normal zipped CSV fails. Test empty archives, multiple members, unsupported members and lifecycle/closure behavior. Avoid extraction paths vulnerable to traversal if extraction is ever introduced.

## Upload lifecycle flags must compose coherently

`return_file=True` means the returned path must still exist. `clean_tmp=True` cannot delete it before return; current **DSET-UPL-002** violates that. For `return_file=False`, ordinary non-database uploads must load with the default `query=None`; current **DSET-UPL-001** calls a string-only SQL cleaner on `None` and swallows the resulting failure. Broad exception-to-`None` behavior must not hide programming errors.

## Optional database queries must really be optional

The public default is `query=None`. For database sources, choose the documented default query without first calling `clean_sql(None)`. Current **DSET-DB-001** makes default SQLite/DuckDB loading fail before their own `SELECT 1` defaults can be used. Ensure every opened connection/engine is deterministically closed/disposed and pass documented kwargs through consistently.

## `clean_sql` is not a security sanitizer

It removes comments. It does not prove a query is read-only, prevent injection, or make arbitrary SQL safe. Do not document it as a trust boundary. SQL execution must follow the database API's actual safety/transaction contract. Test strings containing comment-like tokens inside SQL literals before changing this helper.

## Serialized-object loaders are explicitly unsafe for untrusted inputs

`pickle`, `joblib` and `cloudpickle` can execute attacker-controlled code during deserialization. Preserve clear trust-boundary documentation. Never add automatic loading of such formats from untrusted URLs/uploads without an explicit safety decision.

## URL loading and dataset downloads are network trust boundaries

Keep URL parsing, redirects, timeouts, errors and cache behavior observable. Tests with mocked `urlopen`/`urlretrieve` are not live network evidence. Network failures are not equivalent to unknown dataset names.

## Preserve deterministic export guarantees

For `_data_export.py`, keep stable-hash selection deterministic across processes/platforms; random sampling seed-controlled; stratified allocation exact and auditable; requested sizes validated against prepared rows; manifests sufficient to reproduce parameters; streaming and full-data hash selections equivalent where promised. Test output values/order, not only row counts.

## Keep type/runtime surfaces aligned

`_data_export.pyi` is a public typing contract. Any callable/signature change in `_data_export.py` requires stub review. Likewise, helper annotations such as `detect_file_type` must describe what the function actually returns; unused public parameters should be implemented or removed deliberately.

## CLI examples must be executable

For `python -m`, module names do not end in `.py`. Current **DSET-CLI-001** appears in exporter/AutoScout examples. Exercise help and one tiny CSV export whenever CLI docs change. Recipe scripts are examples, not authority for core API semantics.

## Evidence ladder

Keep these lanes distinct:

1. maintenance static/mutation contract;
2. curated `_load_dataset` focused tests;
3. `_data_export` unit/subtests and tiny real CSV CLI export;
4. focused `_data_loader` file/archive/upload/database tests;
5. Parquet/Excel/Feather/DB-driver optional-backend integration;
6. live network/cache integration;
7. complete installed-package and supported-platform evidence.

Missing optional engines are `UNAVAILABLE`, not PASS. Isolation stubs for the missing package root/appdirs/logger/safe_import are diagnostic only.

## Runtime changes require adversarial regressions

For every repair, add the smallest failing-before test and then rerun the whole datasets suite plus maintenance mutation suite. Include probes for `Thur`, ZIP CSV, upload `query=None`, cleanup/return interactions, database `query=None`, and `python -m` examples before calling the runtime green.
