# scikitplot/_env/_backends/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Backend registry: one table, modules imported only when a backend is used.

==============  ==================  =========================================
Name            Installs from       Notes
==============  ==================  =========================================
``uv``          PyPI                any CPython (downloads it); bootstrappable
``venv``        PyPI                always available; running Python only
``pipenv``      PyPI (Pipfile.lock) exact lock install into a venv
``micromamba``  Conda + PyPI        single binary; cross-builds; bootstrappable
``mamba``       Conda + PyPI        mamba 2.x, micromamba command line
``conda``       Conda + PyPI        classic conda
==============  ==================  =========================================

Notes
-----
**Developer notes.** Add a backend by writing a :py:class:`~._base.Backend`
subclass and one line in :py:data:`BACKENDS`. Order matters: within one
ecosystem, the automatic choice takes the first available entry of
:py:data:`PREFERENCE`.
"""

from __future__ import annotations

import importlib
from types import MappingProxyType

from .._core._errors import ConfigError

__all__ = ["BACKENDS", "PREFERENCE", "get_backend"]

#: ``name -> "module:Class"`` of every backend.
BACKENDS = MappingProxyType(
    {
        "uv": "._python:UvBackend",
        "venv": "._python:VenvBackend",
        "pipenv": "._pipenv:PipenvBackend",
        "micromamba": "._conda:MicromambaBackend",
        "mamba": "._conda:MambaBackend",
        "conda": "._conda:CondaBackend",
    }
)

#: Automatic choice order per ecosystem need.
PREFERENCE = MappingProxyType(
    {
        "pypi": ("uv", "venv"),
        "conda": ("micromamba", "mamba", "conda"),
    }
)

_CACHE: dict = {}


def get_backend(name: str):
    """
    Return the backend registered as ``name``.

    Parameters
    ----------
    name : str
        A key of :py:data:`BACKENDS`.

    Returns
    -------
    Backend
        A shared instance (backends hold no state).

    Raises
    ------
    ConfigError
        For an unknown name.

    Examples
    --------
    >>> from scikitplot._env._backends import get_backend
    >>> get_backend("uv").ecosystems == {"pypi"}
    True
    """
    if name not in BACKENDS:
        raise ConfigError(
            f"unknown backend {name!r}",
            hint=f"choose one of {sorted(BACKENDS)} or 'auto'",
        )
    if name not in _CACHE:
        module, _, attr = BACKENDS[name].partition(":")
        _CACHE[name] = getattr(importlib.import_module(module, __name__), attr)()
    return _CACHE[name]
