# scikitplot/lean/_parallel.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Resolve a :class:`~scikitplot.lean.LeanIndex`'s names in worker processes.

Name resolution reads the whole index (names, namespaces, what each file
sees) and writes one sorted array of edges per declaration; one
declaration's edges never depend on another's. Forked workers start from
the index as it is in memory, so nothing is sent to them; each resolves a run
of consecutive declarations and sends back its edges, which are taken in
order. The edges are therefore the same for any number of workers.

Notes
-----
**User.** Used on Linux when the index has ``workers > 1`` and enough
declarations to pay for starting the processes; elsewhere, and whenever
forking would not be safe (other threads running, inside a daemonic
process), names are resolved here, with the same result.

**Developer.** Memory: a worker copies the pages it reads, since Python's
reference counts write to the objects, and builds its own lookup cache.
What a worker reads most is kept flat so that it copies none of it: each
declaration's reference numbers in arrays, and (:class:`FlatNames`, built
just before forking) each reference and context as text in one buffer,
decoded privately. On mathlib4 a worker resolving every name makes 62 MB
its own, against 160 MB with Python objects; what remains comes from the
name tables' values (dictionaries of Python objects), which a flat table
would remove at about twice the lookup time. ``gc.freeze()`` before
forking saved under 100 MB, so it is not used.

``fork`` is requested explicitly: Python 3.14 no longer
uses it by default on Linux, and other start methods would have to pickle
the index. It is not used on macOS, where forking a process that has used
system frameworks is unsafe.
"""

from __future__ import annotations

import multiprocessing
import sys
import threading
from array import array
from typing import Any

__all__: list[str] = []

#: Fewest declarations to resolve that make starting worker processes pay.
MIN_NODES = 20_000

#: The index the workers read: set in the parent just before forking.
_INDEX: Any = None


def usable(workers: int, nodes: int) -> bool:
    """Return whether *nodes* declarations are resolved in *workers* forked processes."""
    return (
        workers > 1
        and nodes >= MIN_NODES
        and sys.platform.startswith("linux")
        and "fork" in multiprocessing.get_all_start_methods()
        and threading.active_count() == 1
        and not multiprocessing.current_process().daemon
    )


class FlatNames:
    """
    References and contexts as UTF-8 text in two buffers, for forked workers.

    Built in the parent just before forking, read in the workers: a buffer
    is one object, so decoding a name from it writes no reference count on
    the shared string objects, and the worker copies no page of them. Each
    decoded name or context is the worker's own.
    """

    def __init__(
        self, names: list[str], contexts: list[tuple[str, tuple[str, ...]]]
    ) -> None:
        self._names, self._name_at = _pack(names)
        self._contexts, self._context_at = _pack(
            "\x00".join((namespace, *opens)) for namespace, opens in contexts
        )
        self._decoded: dict[int, tuple[str, tuple[str, ...]]] = {}

    def name(self, ref_id: int) -> str:
        """Return reference *ref_id*, decoded."""
        start, stop = self._name_at[ref_id], self._name_at[ref_id + 1]
        return self._names[start:stop].decode("utf-8")

    def context(self, ctx_id: int) -> tuple[str, tuple[str, ...]]:
        """Return context *ctx_id* as ``(namespace, opens)``, decoded once."""
        found = self._decoded.get(ctx_id)
        if found is None:
            start, stop = self._context_at[ctx_id], self._context_at[ctx_id + 1]
            namespace, *opens = self._contexts[start:stop].decode("utf-8").split("\x00")
            found = self._decoded[ctx_id] = (namespace, tuple(opens))
        return found


def _pack(texts: Any) -> tuple[bytes, array]:
    """Return *texts* as one UTF-8 buffer and the offset of each (and the end)."""
    buffer = bytearray()
    offsets = array("q", [0])
    for text in texts:
        buffer += text.encode("utf-8")
        offsets.append(len(buffer))
    return bytes(buffer), offsets


def _enter(flat: FlatNames) -> None:
    """Pool initializer (in each worker): read names from the shared text."""
    _INDEX._flat = flat


def _resolve(bounds: tuple[int, int]) -> tuple[int, bytes, bytes]:
    """Resolve declarations ``start .. stop - 1`` (in a worker): lengths and edges."""
    start, stop = bounds
    lengths = array("q")
    edges = array("l")
    for node in range(start, stop):
        found = _INDEX._compute_edges(node)
        lengths.append(len(found))
        edges.extend(found)
    return start, lengths.tobytes(), edges.tobytes()


def resolve(
    index: Any, nodes: list[int], workers: int, flat: FlatNames | None = None
) -> dict[int, array]:
    """
    Return the edges of *nodes* (ascending), resolved in *workers* processes.

    The caller has checked :func:`usable` and warmed what every worker
    reads (each file's visibility, the short names), so that work is done
    once, before forking, instead of once per worker. With *flat*, each
    worker reads reference names and contexts from it (:class:`FlatNames`).
    """
    global _INDEX  # noqa: PLW0603
    runs = _runs(nodes, max(1, len(nodes) // (workers * 32)))
    found: dict[int, array] = {}
    _INDEX = index
    try:
        context = multiprocessing.get_context("fork")
        initializer = (_enter, (flat,)) if flat is not None else (None, ())
        with context.Pool(workers, initializer[0], initializer[1]) as pool:
            for start, raw_lengths, raw_edges in pool.imap(_resolve, runs):
                lengths = array("q")
                lengths.frombytes(raw_lengths)
                edges = array("l")
                edges.frombytes(raw_edges)
                at = 0
                for offset, length in enumerate(lengths):
                    found[start + offset] = edges[at : at + length]
                    at += length
    finally:
        _INDEX = None
    return found


def _runs(nodes: list[int], size: int) -> list[tuple[int, int]]:
    """Split ascending *nodes* into runs of consecutive numbers, at most *size* each."""
    runs: list[tuple[int, int]] = []
    start = previous = None
    for node in nodes:
        if start is not None and node == previous + 1 and node - start < size:
            previous = node
            continue
        if start is not None:
            runs.append((start, previous + 1))
        start = previous = node
    if start is not None:
        runs.append((start, previous + 1))
    return runs
