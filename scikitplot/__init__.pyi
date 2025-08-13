# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# https://abseil.io/docs/python/quickstart.html
# https://abseil.io/docs/python/guides/logging
# import logging
# from absl import logging
# logger = logging.getLogger(__name__)

from scikitplot import sp_logging as logger
from scikitplot import logger, get_logger

# logger = get_logger()
logger.setLevel(level=logger.INFO)  # default WARNING
logger.warning("scikitplot warning!!!")
logger.info("This is a info message from the sp logger.")

# Only imports when type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Heavy import, only for type checking
    # Only imports when type checking, not at runtime
    from typing import Final, LiteralString
    from typing_extensions import LiteralString

    pass  # Usage as str type 'tf'

__numpy_version__: Final[LiteralString]
_BUILT_WITH_MESON: Final[bool]

######################################################################
## scikit-plots version
## PEP0440 compatible formatted version, see:
## https://peps.python.org/pep-0440/#version-scheme
## https://www.python.org/dev/peps/pep-0440/
## Version scheme: [N!]N(.N)*[{a|b|rc}N][.postN][.devN]
## Generic release markers (.devN, aN, bN, rcN, <no suffix>, .postN):
#   X.Y.devN             # 'Development release' 1.0.dev1
#   X.YrcN.devM          # Developmental release of a release candidate
#   X.Y.postN.devM       # Developmental release of a post-release
#   X.YrcN.postN.devN    # Developmental release of a post-release of a release candidate
#   X.Y{a|b|rc|c}N       # 'Pre-release' 1.0a1
#   X.Y==X.Y.0==N(.N)*   # For first 'Release' after an increment in Y
#   X.Y{post|rev|r}N     # 'Post-release' 1.0.post1
#   X.YrcN.postM         # Post-release of a release candidate
#   X.Y.N                # 'Bug fixes' 1.0.1
## setuptools-scm extracts Python package versions
######################################################################

# https://packaging.python.org/en/latest/discussions/versioning/#valid-version-numbers
# Admissible pre-release markers:
#   Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
#   X.Y.dev0     # is the canonical version of 'X.Y.dev'
#   X.Y.ZaN      # Alpha release
#   X.Y.ZbN      # Beta release
#   X.Y.ZrcN     # Release Candidate
#   X.Y.Z        # Final release
#   X.Y.Z.postM  # Post release
## https://libraries.io/pypi/scikit-plots
__version__: Final[LiteralString]
__git_hash__: Final[LiteralString]

######################################################################
## Public Interface define explicitly `__all__`
######################################################################

# Avoiding heavy imports top level module unless actually used
# from .utils.lazy_load import LazyLoader
# Lazily load scikitplot flavors to avoid excessive dependencies.
# from ._compat.optional_deps import LazyImport, nested_import

# Without __all__: All public names (not starting with _) are importedto supmodule.
# {'_base', '_core', '_docstrings', '_orig_rc_params', '_statistics', '_stats'}

## Define __all__ to control what gets imported with 'from module import *'.
## If __all__ is not defined, Python falls back to using the module's global namespace
## (as returned by dir() or globals().keys()) for 'from module import *'.
## This means that all names in the global namespace, except those starting with '_',
## will be imported by default.
## Reference: https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
# Scope: dir() with no arguments returns the all defined names in the current local scope.
# Behavior: Includes functions, variables, classes, modules, __builtins__ that are defined so far.
# Scope: globals() returns a dict of all symbols defined at the module level.
# Behavior: Almost equivalent to dir() for a module context. No builtins included
# More explicit about scope—only includes actual global symbols.
# Keeps out internal modules, helpers, and unwanted globals
# def build_all():
#     import types
#     return [
#         name for name, val in globals().items()
#         if not name.startswith("_")
#         and not isinstance(val, types.ModuleType)
#     ]
# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
# __all__ = tuple(
#     sorted(
#         [
#             name
#             for name in (
#                 {*globals().keys()}.union(_submodules).difference({"_submodules"})
#             )
#             ## Exclude private/internal names (those starting with '_') placeholder
#             if not (name.startswith("...") and not name.endswith("..."))
#         ]
#     )
# )

######################################################################
## globally seeding
######################################################################

# def setup_module(module) -> None:
#     """
#     Fixture to seed random number generators for reproducible tests.

#     This function ensures a globally controllable random seed for Python's built-in `random`,
#     and NumPy's RNG, optionally configurable via the `SKPLT_SEED` environment variable.

#     Parameters
#     ----------
#     module : Any
#         The test module passed by the testing framework (e.g., pytest). This parameter
#         is required by the `setup_module` hook but is not directly used in this function.

#     Notes
#     -----
#     - If `SKPLT_SEED` is not set in the environment, a random seed is generated.
#     - This function supports both legacy and newer NumPy random APIs.
#     """
#     try:
#         import os
#         import random
#         import numpy as np

#         # Get seed from environment variable or generate one
#         seed_env = os.environ.get("SKPLT_SEED")
#         if seed_env is not None:
#             seed = int(seed_env)
#         else:
#             seed = int(np.random.uniform() * np.iinfo(np.int32).max)  # noqa: NPY002

#         print(f"I: Seeding RNGs with {seed}")  # noqa: T201, UP031

#         # Seed both NumPy and Python RNG
#         # np.random.Generator
#         # Legacy NumPy seeding (safe across versions)
#         np.random.seed(seed)  # noqa: NPY002
#         random.seed(seed)

#     except Exception as e:
#         print(f"Warning: RNG seeding failed: {e}")  # noqa: T201

######################################################################
##
######################################################################
