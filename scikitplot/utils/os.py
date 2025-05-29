"""os.py."""

import os


def is_windows():
    """
    Return true if the local system/OS name is Windows.

    Returns
    -------
    True if the local system/OS name is Windows.

    """
    return os.name == "nt"
