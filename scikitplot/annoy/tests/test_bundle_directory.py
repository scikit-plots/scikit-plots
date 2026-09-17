"""
Bundle regressions for :mod:`scikitplot.annoy` (slices S-37, S-38).

``save_bundle`` documented a directory bundle but took no directory, so default
arguments wrote both members into the process working directory. ``load_bundle``
never called ``load_index``: the call was commented out and both its
``index_filename`` and ``prefault`` arguments were marked unused, so the index
came back only as a side effect of the manifest carrying an absolute path
recorded at save time. A bundle could not be moved, and naming a different index
file had no effect.

See Also
--------
scikitplot.annoy._mixins._io.IndexIOMixin.save_bundle
scikitplot.annoy._mixins._io.IndexIOMixin.load_bundle
"""

import os
from pathlib import Path

import pytest

from scikitplot import annoy

INDEX = next(c for c in (annoy.Index, annoy.Annoy) if hasattr(c, "save_bundle"))


def _built(dimension=3, count=5):
    index = INDEX(dimension, "angular")
    for position in range(count):
        index.add_item(position, [float(position), 1.0, 2.0])
    index.build(4)
    return index


def test_a_bundle_is_a_directory(tmp_path):
    """The documented unit exists: one directory holding both members."""
    bundle = tmp_path / "mybundle"
    _built().save_bundle(bundle)
    assert sorted(p.name for p in bundle.iterdir()) == ["index.ann", "manifest.json"]


def test_nothing_is_written_to_the_working_directory(tmp_path, monkeypatch):
    """Defaults resolve against the bundle, never against the process cwd."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    _built().save_bundle(tmp_path / "bundle")
    assert list(work.iterdir()) == []


def test_a_bundle_can_be_moved(tmp_path):
    """Members resolve relative to the manifest, not to a path recorded at save."""
    original = tmp_path / "original"
    _built().save_bundle(original)
    moved = tmp_path / "moved"
    original.rename(moved)
    reloaded = INDEX.load_bundle(moved)
    assert reloaded.get_nns_by_item(0, 2)


def test_a_bundle_can_be_copied_and_both_copies_load(tmp_path):
    """Two copies of one bundle are two usable bundles."""
    import shutil

    first = tmp_path / "first"
    _built().save_bundle(first)
    second = tmp_path / "second"
    shutil.copytree(first, second)
    assert INDEX.load_bundle(first).get_nns_by_item(0, 1)
    assert INDEX.load_bundle(second).get_nns_by_item(0, 1)


def test_load_bundle_returns_a_queryable_index(tmp_path):
    """The load path loads the index, rather than returning an empty shell."""
    bundle = tmp_path / "bundle"
    built = _built()
    built.save_bundle(bundle)
    reloaded = INDEX.load_bundle(bundle)
    assert reloaded.get_nns_by_item(0, 3) == built.get_nns_by_item(0, 3)


def test_a_missing_index_member_is_refused(tmp_path):
    """An unloadable bundle raises instead of returning something unusable."""
    bundle = tmp_path / "bundle"
    _built().save_bundle(bundle)
    (bundle / "index.ann").unlink()
    with pytest.raises((OSError, ValueError)):
        INDEX.load_bundle(bundle)


def test_a_custom_index_filename_is_honoured_on_both_sides(tmp_path):
    """The argument naming the index actually names the index."""
    bundle = tmp_path / "bundle"
    _built().save_bundle(bundle, index_filename="vectors.ann")
    assert (bundle / "vectors.ann").is_file()
    assert INDEX.load_bundle(bundle, index_filename="vectors.ann").get_nns_by_item(0, 1)


def test_a_failed_save_leaves_no_partial_bundle(tmp_path, monkeypatch):
    """Publication commits or leaves nothing; it never accumulates."""
    bundle = tmp_path / "bundle"

    def boom(self, path, *args, **kwargs):
        raise OSError("injected: manifest write failed")

    monkeypatch.setattr(type(_built()), "to_json", boom, raising=False)
    with pytest.raises(OSError):
        _built().save_bundle(bundle)
    assert not bundle.exists()
    assert list(tmp_path.iterdir()) == []


def test_saving_over_an_existing_bundle_preserves_it_on_failure(tmp_path):
    """A failed replacement leaves the previous bundle exactly as it was."""
    bundle = tmp_path / "bundle"
    _built(count=5).save_bundle(bundle)
    before = (bundle / "index.ann").read_bytes()

    replacement = _built(count=7)

    def boom(self, path, *args, **kwargs):
        raise OSError("injected")

    original = type(replacement).to_json
    type(replacement).to_json = boom
    try:
        with pytest.raises(OSError):
            replacement.save_bundle(bundle)
    finally:
        type(replacement).to_json = original
    assert (bundle / "index.ann").read_bytes() == before
    assert INDEX.load_bundle(bundle).get_nns_by_item(0, 1)
