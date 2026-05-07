# scikitplot/_build_utils/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause


def __getattr__(name: str) -> object:
    """
    Lazy submodule loader.

    Parameters
    ----------
    name : str
        Attribute name requested on this package.

    Returns
    -------
    object
        The requested lazily-loaded submodule, cached in ``globals()``
        for subsequent accesses.

    Examples
    --------
    >>> from scikitplot._build_utils import _meson_features
    >>> _meson_features = None
    """
    if name in ["_meson_features"]:
        return None
