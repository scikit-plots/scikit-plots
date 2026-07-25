# scikitplot/cython/tests/test__thread_safety.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-CON-002.

Two pieces of global build/import state were mutated without synchronisation:
the lazy ``_SETUPTOOLS_CACHE`` singleton in ``_builder`` and the
``CompilerRegistry`` dict in ``_custom_compiler`` (whose docstring even declared
itself "not thread-safe").  Both are now guarded by locks: the setuptools
singleton uses double-checked locking so concurrent first-time callers
initialise exactly once, and the registry guards every dict operation with an
internal re-entrant lock.
"""
from __future__ import annotations

import threading

import pytest

from .. import _builder as _b
from .._custom_compiler import CompilerRegistry, CustomCompilerProtocol


class _FakeCompiler:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args, **kwargs):  # pragma: no cover - not invoked
        return None

    # Minimal surface so isinstance(CustomCompilerProtocol) is satisfied.
    def compile(self, *args, **kwargs):  # pragma: no cover
        return None


class TestSetuptoolsSingletonThreadSafe:
    def test_concurrent_first_call_initialises_once(self) -> None:
        """16 threads racing the first call must get one identical class pair."""
        saved = _b._SETUPTOOLS_CACHE
        _b._SETUPTOOLS_CACHE = None
        results: list[tuple[int, int]] = []
        barrier = threading.Barrier(16)

        def worker() -> None:
            barrier.wait()  # maximise contention on the first call
            ext, dist = _b._import_setuptools()
            results.append((id(ext), id(dist)))

        threads = [threading.Thread(target=worker) for _ in range(16)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            _b._SETUPTOOLS_CACHE = saved

        assert len(results) == 16
        assert len(set(results)) == 1, "concurrent init produced divergent identities"

    def test_lock_object_exists(self) -> None:
        assert isinstance(_b._SETUPTOOLS_LOCK, type(threading.Lock()))


class TestCompilerRegistryThreadSafe:
    def test_registry_has_lock(self) -> None:
        reg = CompilerRegistry()
        assert hasattr(reg, "_lock")

    def test_concurrent_register_unregister_no_corruption(self) -> None:
        reg = CompilerRegistry()
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                c = _FakeCompiler(f"custom_c{i}")
                if isinstance(c, CustomCompilerProtocol):
                    reg.register(c, overwrite=True)
                    reg.list()
                    reg.unregister(f"custom_c{i}")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(64)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"concurrent registry ops raised: {errors[:3]}"

    def test_concurrent_readers_and_writers(self) -> None:
        """Interleaved list() reads and register()/unregister() writes are safe."""
        reg = CompilerRegistry()
        stop = threading.Event()
        errors: list[Exception] = []

        def reader() -> None:
            try:
                while not stop.is_set():
                    reg.list()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def writer(i: int) -> None:
            try:
                c = _FakeCompiler(f"custom_w{i}")
                if isinstance(c, CustomCompilerProtocol):
                    for _ in range(20):
                        reg.register(c, overwrite=True)
                        reg.unregister(c.name)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        readers = [threading.Thread(target=reader) for _ in range(4)]
        writers = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in readers:
            t.start()
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join()

        assert errors == [], f"reader/writer race raised: {errors[:3]}"
