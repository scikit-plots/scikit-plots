# scikitplot/cython/tests/test__cache_fingerprint.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-CACHE-003.

The cache fingerprint omitted toolchain/ABI inputs (compiler identity, extension
ABI tag, pointer width, free-threaded flag), so an artifact built with one
compiler or ABI could be wrongly reused under an incompatible one.  The
fingerprint now includes those inputs, so any of them differing yields a
different cache key.
"""
from __future__ import annotations

import sysconfig

import pytest

from .._cache import make_cache_key, runtime_fingerprint


TOOLCHAIN_KEYS = {
    "cc",
    "cxx",
    "ext_suffix",
    "soabi",
    "pointer_size",
    "gil_disabled",
    "sysconfig_platform",
}


def test_fingerprint_includes_toolchain_and_abi() -> None:
    fp = runtime_fingerprint(cython_version="3.2.9", numpy_version=None)
    assert TOOLCHAIN_KEYS <= set(fp), f"missing toolchain keys: {TOOLCHAIN_KEYS - set(fp)}"
    # Original inputs still present (backward compatible superset).
    assert {"python", "cython", "numpy", "abi"} <= set(fp)


@pytest.mark.parametrize(
    "var,newval",
    [
        ("CC", "clang"),
        ("CXX", "clang++"),
        ("EXT_SUFFIX", ".cpython-999-fake.so"),
        ("SOABI", "cpython-999-fake"),
        ("SIZEOF_VOID_P", "4"),
    ],
)
def test_toolchain_change_changes_key(var, newval, monkeypatch) -> None:
    """Any toolchain/ABI change must produce a different cache key."""
    base_fp = runtime_fingerprint(cython_version="3.2.9", numpy_version=None)
    base_key = make_cache_key({"source": "x", "fp": base_fp})

    cfg = sysconfig.get_config_vars()
    monkeypatch.setitem(cfg, var, newval)
    new_fp = runtime_fingerprint(cython_version="3.2.9", numpy_version=None)
    new_key = make_cache_key({"source": "x", "fp": new_fp})

    assert new_key != base_key, f"changing {var} did not change the cache key"


def test_gil_disabled_flag_present_and_boolean() -> None:
    fp = runtime_fingerprint(cython_version="3.2.9", numpy_version=None)
    assert isinstance(fp["gil_disabled"], bool)


def test_same_environment_is_stable() -> None:
    """The fingerprint is deterministic within one environment."""
    a = runtime_fingerprint(cython_version="3.2.9", numpy_version="2.0")
    b = runtime_fingerprint(cython_version="3.2.9", numpy_version="2.0")
    assert a == b
    assert make_cache_key({"fp": a}) == make_cache_key({"fp": b})
