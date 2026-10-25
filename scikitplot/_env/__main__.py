# scikitplot/_env/__main__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""``python -m scikitplot._env``: the ``scikitplot env`` command line."""

import sys

from ._cli import main

if __name__ == "__main__":
    sys.exit(main())
