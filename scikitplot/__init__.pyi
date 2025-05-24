# pylint: skip-file
# ruff: noqa: PGH004
# ruff: noqa
# flake8: noqa
# type: ignore
# mypy: ignore-errors

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# https://abseil.io/docs/python/quickstart.html
# https://abseil.io/docs/python/guides/logging
import logging

from absl import logging

_log = logging.getLogger(__name__)
del logging

# Only imports when type checking, not at runtime
from typing import TYPE_CHECKING, Final, LiteralString

# from typing_extensions import LiteralString

if TYPE_CHECKING:
    # Heavy import, only for type checking
    pass  # Usage as str type 'tf'

from scikitplot import get_logger

logger = get_logger()  # python logger, not have direct log level

from scikitplot import sp_logging as logging  # class instance logger

logger.setLevel(level=logging.INFO)  # default WARNING
logger.warning("scikitplot warning!!!")
logger.info("This is a info message from the sp logger.")

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
__array_api_version__: Final[LiteralString]

__bibtex__: Final[LiteralString]
__citation__: Final[LiteralString]

_BUILT_WITH_MESON: Final[bool]

# Without __all__: All public names (not starting with _) are importedto supmodule.
# {'_base', '_core', '_docstrings', '_orig_rc_params', '_statistics', '_stats'}
