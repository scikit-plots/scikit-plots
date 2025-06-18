"""os.py."""

import os as _os


def is_windows():
    """
    Return true if the local system/OS name is Windows.

    Returns
    -------
    True if the local system/OS name is Windows.

    """
    # return sys.platform.startswith("win")
    return _os.name == "nt"
