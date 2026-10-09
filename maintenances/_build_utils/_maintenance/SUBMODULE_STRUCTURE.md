# Submodule structure

```text
scikitplot/_build_utils
├── tempita.py / cython_generate.py      Tempita generation
├── conv_template.py / process_src_template.py  NumPy-style .src generation
├── cythoner.py / gcc_build_bitness.py   compiler wrappers
├── gitversion.py / version.py           version provenance
├── fix_submodule_import*.py             vendored-import rewriting
├── copyfiles.py                         filesystem copy/archive helper
├── generate_apis_reference/             docs API generation package
│   ├── __init__.py                       parser, rebuild/customization API, runtime verification
│   ├── __main__.py                       `python -m ...generate_apis_reference` CLI launcher
│   └── apis_reference.py.in              canonical editable API-reference template
├── install_meson_features.py
│   └── _meson_features/*                code copied into mesonbuild.modules.features
└── tests/*                              focused local tests
```

The submodule is build-time infrastructure. Production modules outside `_build_utils` should not import it at runtime.
