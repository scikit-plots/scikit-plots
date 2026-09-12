# scikitplot/_build_utils/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_build_utils.
"""

# The new maintenance model separates:
#     scikitplot._build_utils
#     ├── template generation
#     │   ├── tempita.py
#     │   ├── cython_generate.py
#     │   ├── conv_template.py
#     │   └── process_src_template.py
#     ├── compiler wrappers
#     │   ├── cythoner.py
#     │   └── gcc_build_bitness.py
#     ├── version provenance
#     │   ├── gitversion.py
#     │   └── version.py
#     ├── vendoring / import rewriting
#     │   ├── fix_submodule_import.py
#     │   └── fix_submodule_import_v2.py
#     ├── filesystem support
#     │   └── copyfiles.py
#     └── Meson integration
#         ├── install_meson_features.py
#         └── _meson_features/*
