# Structure

```text
scikitplot/datasets/
├── __init__.py                 # curated loader exports only
├── _load_dataset.py            # named example datasets + cache/postprocess
├── _data_export.py/.pyi        # deterministic export + CLI
├── _data_loader.py             # generic file/URL/upload/DB dispatcher
├── _autoscout24_tasks.py       # example ML recipes
├── data_export_recipes_autoscout24.md
└── tests/
    ├── test__load_dataset.py
    └── test__data_export.py
```

The absence of a focused `_data_loader` test module is a tracked maintenance gap.
