# scikitplot/_utils/tests/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Test package for :mod:`scikitplot._utils`.

This ``__init__.py`` installs a minimal ``scikitplot`` stub into
``sys.modules`` when the test package is imported outside of the real
scikitplot installation (e.g. ``python -m unittest`` run from the repo
root or an extracted archive).  If the real ``scikitplot`` package is
already present the stub is a no-op.

Developer note
--------------
The stub satisfies the relative imports inside ``_utils``:

* ``from .. import logger``                     → ``scikitplot.logger``
* ``from ..exceptions import ScikitplotException``
* ``from ..environment_variables import SKPLT_LOGGING_LEVEL``
* ``from .. import logging``                    → thin wrapper around stdlib
"""

from __future__ import annotations

# import logging as _logging
# import logging.config as _logging_config
# import sys as _sys
# import types as _types


# def _install_scikitplot_stub() -> None:
#     """Register a minimal scikitplot package in sys.modules if absent.

#     Safe to call multiple times; subsequent calls are no-ops.
#     """
#     if "scikitplot" in _sys.modules:
#         return  # real package or stub already present

#     # -- root package -------------------------------------------------------
#     pkg = _types.ModuleType("scikitplot")
#     pkg.__path__ = []           # marks it as a package
#     pkg.__package__ = "scikitplot"
#     pkg.logger = _logging.getLogger("scikitplot")

#     # -- scikitplot.exceptions ----------------------------------------------
#     exc_mod = _types.ModuleType("scikitplot.exceptions")

#     class ScikitplotException(Exception):
#         """Minimal stand-in for the real ScikitplotException."""

#         def __init__(self, message: str = "", error_code: int = 0) -> None:
#             super().__init__(message)
#             self.error_code = error_code

#     class MissingConfigException(ScikitplotException):
#         """Minimal stand-in for MissingConfigException."""

#     exc_mod.ScikitplotException = ScikitplotException
#     exc_mod.MissingConfigException = MissingConfigException
#     pkg.exceptions = exc_mod

#     # -- scikitplot.environment_variables -----------------------------------
#     env_mod = _types.ModuleType("scikitplot.environment_variables")

#     class _EnvVar:
#         def get(self) -> str:
#             return "INFO"

#     env_mod.SKPLT_LOGGING_LEVEL = _EnvVar()
#     pkg.environment_variables = env_mod

#     # -- scikitplot.logging (thin wrapper around stdlib logging) ------------
#     log_mod = _types.ModuleType("scikitplot.logging")
#     log_mod.getLogger = _logging.getLogger
#     log_mod.Filter = _logging.Filter
#     log_mod.Formatter = _logging.Formatter
#     log_mod.StreamHandler = _logging.StreamHandler
#     log_mod.config = _logging_config
#     pkg.logging = log_mod

#     _sys.modules["scikitplot"] = pkg
#     _sys.modules["scikitplot.exceptions"] = exc_mod
#     _sys.modules["scikitplot.environment_variables"] = env_mod
#     _sys.modules["scikitplot.logging"] = log_mod


# _install_scikitplot_stub()
