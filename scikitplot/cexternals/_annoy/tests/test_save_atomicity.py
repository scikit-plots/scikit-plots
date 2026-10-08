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


def _read(path):
    """Return a file's bytes and close it (an unclosed file is an error here)."""
    with open(path, "rb") as handle:
        return handle.read()


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


def test_failed_replace_keeps_a_loaded_index_usable(workdir):
    """ANNOY-WIN-002: an index mapped from a file survives a failed replace.

    Where a mapped file cannot be replaced (Windows), ``save`` releases the
    mapping before the replace. If the replace then fails, the index must be
    mapped again from the file it came from: same answers, source untouched,
    no temporary file. Elsewhere the mapping is never released, and the same
    assertions hold. The replace is made to fail by naming a directory as the
    target, which no platform replaces with a file.
    """
    q = [random.random() for _ in range(DIM)]
    source = os.path.join(workdir, "source.ann")
    _build(60).save(source)
    before = _read(source)

    idx = A.AnnoyIndex(DIM, "euclidean")
    idx.load(source)
    expected = idx.get_nns_by_vector(q, 8)

    blocked = os.path.join(workdir, "a_directory")
    os.mkdir(blocked)
    with pytest.raises(OSError, match="replace target file"):
        idx.save(blocked)

    assert idx.get_n_items() == 60
    assert idx.get_nns_by_vector(q, 8) == expected
    assert _read(source) == before
    assert os.path.isdir(blocked) and os.listdir(blocked) == []
    assert glob.glob(os.path.join(workdir, "*.tmp-*")) == []
    # The index is still mapped from a file it can be saved over.
    idx.save(source)
    assert idx.get_nns_by_vector(q, 8) == expected


def test_save_over_the_file_another_index_was_loaded_from(workdir):
    """A second index mapped from the target does not corrupt either one.

    On POSIX the replace succeeds and the reader keeps the old contents. Where
    a mapped file cannot be replaced, the save fails cleanly instead: the
    target and both indexes are as they were.
    """
    q = [random.random() for _ in range(DIM)]
    p = os.path.join(workdir, "shared.ann")
    _build(60, seed=1).save(p)
    reader = A.AnnoyIndex(DIM, "euclidean")
    reader.load(p)
    reader_expected = reader.get_nns_by_vector(q, 8)

    writer = _build(40, seed=2)
    writer_expected = writer.get_nns_by_vector(q, 8)
    try:
        writer.save(p)
    except OSError:
        replaced = False
    else:
        replaced = True

    assert writer.get_nns_by_vector(q, 8) == writer_expected
    assert reader.get_nns_by_vector(q, 8) == reader_expected
    assert glob.glob(p + ".tmp-*") == []
    fresh = A.AnnoyIndex(DIM, "euclidean")
    fresh.load(p)
    assert fresh.get_n_items() == (40 if replaced else 60)
