# scikitplot/annoy/_annoy/tests/test_concurrency_policy.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Concurrency policy guard (CY-009).

The documented policy (Policy A) is: an index is NOT thread-safe for mutation.
Only two things are supported concurrently, and this test exercises exactly those
— it deliberately does NOT assert that same-instance mutation races are safe:

1. Independent instances used from independent threads.
2. Multiple threads issuing read-only queries against a single FULLY-BUILT index
   that no thread mutates.

Kept small (few threads, tiny indices) to stay deterministic and avoid OOM.
"""
from concurrent.futures import ThreadPoolExecutor

from scikitplot.annoy._annoy import annoylib as A

DIM = 8


def _build_own(seed):
    idx = A.Index(DIM, "euclidean")
    for i in range(64):
        idx.add_item(i, [float((i + seed) % 7)] * DIM)
    idx.build(5)
    return idx.get_nns_by_item(0, 5)


def test_independent_instances_are_safe_in_parallel():
    # supported: each thread builds and queries its OWN index
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_build_own, range(8)))
    assert len(results) == 8
    assert all(len(r) == 5 and r[0] == 0 for r in results)


def test_concurrent_reads_on_a_built_index_are_safe():
    # supported: one fully-built, non-mutated index queried from many threads
    shared = A.Index(DIM, "euclidean")
    for i in range(128):
        shared.add_item(i, [float(i % 11), float(i % 5)] + [0.0] * (DIM - 2))
    shared.build(10)
    baseline = {i: shared.get_nns_by_item(i, 5) for i in range(0, 128, 8)}

    def query(i):
        return i, shared.get_nns_by_item(i, 5)

    with ThreadPoolExecutor(max_workers=8) as ex:
        out = dict(ex.map(query, list(baseline) * 4))
    # concurrent reads must match the single-threaded baseline (deterministic)
    for i, expected in baseline.items():
        assert out[i] == expected


def test_policy_is_documented_not_thread_safe_for_mutation():
    # the class docstring must state the Policy-A concurrency contract
    doc = A.Index.__doc__ or ""
    base_doc = ""
    for cls in type(A.Index(DIM, "euclidean")).__mro__:
        base_doc += (cls.__doc__ or "")
    combined = doc + base_doc
    assert "Concurrency" in combined
    assert "not thread-safe" in combined.lower()
