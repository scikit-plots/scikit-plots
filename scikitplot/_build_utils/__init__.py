# scikitplot/_build_utils/__init__.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation. Use in setup.py::
#
#   config.add_extension('_name', sources=['source_fname'], **numpy_nodepr_api)
#
# numpy_nodepr_api = dict(
#     define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")]
# )
#
# def import_file(folder, module_name):
#     """
#     Import a file directly, avoiding importing scipy
#     """
#     import importlib
#     import pathlib

#     fname = pathlib.Path(folder) / f'{module_name}.py'
#     spec = importlib.util.spec_from_file_location(module_name, str(fname))
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module

from . import (
    copyfiles as copyfiles,
)
from . import (
    cythoner as cythoner,
)
from . import (
    gcc_build_bitness as gcc_build_bitness,
)
from . import (
    gitversion as gitversion,
)
from . import (
    system_info as system_info,
)
from . import (
    tempita as tempita,
)
from . import (
    version as version,
)
