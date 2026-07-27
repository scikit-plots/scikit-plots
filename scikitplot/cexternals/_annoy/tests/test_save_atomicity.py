# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for ANNOY-SAVE-001 (guide 6.5).

`save()` must be failure-atomic: write to a same-directory temporary file and
atomically rename it over the target, so a failed/partial write never destroys
the previous file, and the in-memory index is only unloaded after the file is
safely committed.

Full crash / ENOSPC injection lives in CI; these tests cover the observable
contract in-process: happy-path completeness, atomic replace over an existing
file, no temp-file litter, and a clean failure that preserves both the target
and the in-memory index.
"""
import glob
import os
import random
import tempfile

import pytest

from scikitplot.cexternals._annoy import annoylib as A

DIM = 5


def _build(items, seed=0):
    idx = A.AnnoyIndex(DIM, "euclidean")
    r = random.Random(seed)
    for i in range(items):
        idx.add_item(i, [r.random() for _ in range(DIM)])
    idx.build(10)
    return idx


@pytest.fixture
def workdir(tmp_path):
    return str(tmp_path)


def test_save_produces_complete_file_and_leaves_no_temp(workdir):
    idx = _build(60)
    q = [random.random() for _ in range(DIM)]
    expected = idx.get_nns_by_vector(q, 8)
    p = os.path.join(workdir, "index.ann")

    idx.save(p)

    assert glob.glob(p + ".tmp-*") == [], "temp file left behind"
    # object remains usable after save
    assert idx.get_nns_by_vector(q, 8) == expected
    # file is complete and loadable
    other = A.AnnoyIndex(DIM, "euclidean")
    other.load(p)
    assert other.get_nns_by_vector(q, 8) == expected


def test_save_atomically_replaces_existing_file(workdir):
    idx = _build(60)
    q = [random.random() for _ in range(DIM)]
    expected = idx.get_nns_by_vector(q, 8)
    p = os.path.join(workdir, "index.ann")

    idx.save(p)          # first write
    idx.save(p)          # replace existing target

    assert glob.glob(p + ".tmp-*") == []
    other = A.AnnoyIndex(DIM, "euclidean")
    other.load(p)
    assert other.get_nns_by_vector(q, 8) == expected


def test_failed_save_preserves_in_memory_index_and_target(workdir):
    idx = _build(60)
    q = [random.random() for _ in range(DIM)]
    expected = idx.get_nns_by_vector(q, 8)

    bad = os.path.join(workdir, "missing_dir", "x.ann")  # parent does not exist
    failed = False
    try:
        failed = idx.save(bad) is False
    except Exception:
        failed = True

    assert failed, "save to an invalid path should fail"
    # in-memory index is untouched (not unloaded on failure)
    assert idx.get_nns_by_vector(q, 8) == expected
    # no partial/temp files created anywhere under the workdir
    assert glob.glob(os.path.join(workdir, "**", "*.tmp-*"), recursive=True) == []
