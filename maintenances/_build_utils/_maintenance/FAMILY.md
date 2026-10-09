# `_build_utils` ownership family

The domain contains several distinct tools; one green lane cannot bless another.

- **Template generation:** `tempita.py`, `cython_generate.py`, `conv_template.py`, `process_src_template.py`.
- **Cython/compiler wrappers:** `cythoner.py`, `gcc_build_bitness.py`.
- **Version provenance:** `gitversion.py`, legacy `version.py`, `link-version-pyinit.map`.
- **Vendoring/refactoring:** `fix_submodule_import.py`, `fix_submodule_import_v2.py`.
- **Filesystem support:** `copyfiles.py`.
- **Meson extension:** `install_meson_features.py` + `_meson_features/*`.
- **Configuration helpers:** `system_info.py`.
- **Documentation API reference maintenance:** `generate_apis_reference/` owns the generator code while `generate_apis_reference/apis_reference.py.in` owns the durable editorial/layout template for deterministic regeneration of `docs/source/apis_reference.py`. The package statically parses template/generated source and keeps installed-package symbol verification as a separate explicit lane.

Consumer modules such as `annoy` own their templates and compiled API. `_build_utils` owns the correctness and safety of the generator/wrapper invoked on those templates, not the consumer's ABI semantics. Stock Meson owns its interpreter/compiler APIs; this repository owns compatibility of its copied `features` module with supported Meson versions.
