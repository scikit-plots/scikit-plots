# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Platform requirements a test can state, each with its reason in one place.

A test that cannot run on a platform says which fact of that platform stops
it, so that a skip is a statement about the platform and not about the test.

Notes
-----
**Developer.** Every marker here was introduced for a failure measured in the
Windows job of CI run 37668804532, not for a platform nobody ran. A marker is
for a test whose *subject* does not exist on the platform. A test that only
compares a path as text, or writes a fixture in text mode, is repaired
instead: it has no platform requirement.
"""

from __future__ import annotations

import os

import pytest

__all__ = ["POSIX_RELEASE_GATE", "RUNS_A_SCRIPT_BY_ITS_FIRST_LINE"]

_ON_WINDOWS = os.name == "nt"

#: The release gate records permission bits (``0o755`` in a patch, in an
#: archive and in a tree digest) and finds Git through ``os.defpath``.
#: Windows has one read-only flag and no executable bit, so such a bit can be
#: neither set nor compared there; and its ``os.defpath`` is ``.;C:\bin``,
#: where no Git is installed, so the gate stops with ``GIT_REQUIRED``.
POSIX_RELEASE_GATE = pytest.mark.skipif(
    _ON_WINDOWS,
    reason=(
        "the release gate records POSIX permission bits and finds Git "
        "through os.defpath; Windows has no executable bit, and its "
        "os.defpath ('.;C:\\bin') holds no Git"
    ),
)

#: The test starts a file as a program, relying on its ``#!`` line and its
#: executable bit. Windows starts a file by its extension and reads neither.
RUNS_A_SCRIPT_BY_ITS_FIRST_LINE = pytest.mark.skipif(
    _ON_WINDOWS,
    reason=(
        "starts a script through its #! line and executable bit; Windows "
        "reads neither"
    ),
)
