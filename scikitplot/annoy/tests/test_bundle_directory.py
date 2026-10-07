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


# ---------------------------------------------------------------------------
# Publication where a mapped file pins its directory (Windows)
# ---------------------------------------------------------------------------
#
# The backend's ``save`` and ``load`` memory-map the index file. POSIX lets a
# directory be renamed or removed while a file in it is mapped; Windows refuses
# both. ``_PinnedByMapping`` applies the Windows rule on any platform, so the
# order of operations in ``save_bundle`` is tested where the tests run:
#
# * it records which file the index is mapped from, by observing the four
#   backend calls that change it;
# * ``os.replace`` of a directory that holds the mapped file raises the error
#   Windows raises, and ``shutil.rmtree(..., ignore_errors=True)`` of such a
#   directory leaves it in place, as it does on Windows.


class _Tracked(INDEX):
    """An index that records which file it is memory-mapped from."""

    mapped_from = None

    def save(self, fn, *args, **kwargs):
        result = super().save(fn, *args, **kwargs)
        self.mapped_from = Path(fn).resolve()
        return result

    def load(self, fn, *args, **kwargs):
        result = super().load(fn, *args, **kwargs)
        self.mapped_from = Path(fn).resolve()
        return result

    def unload(self, *args, **kwargs):
        result = super().unload(*args, **kwargs)
        self.mapped_from = None
        return result

    def deserialize(self, *args, **kwargs):
        result = super().deserialize(*args, **kwargs)
        self.mapped_from = None
        return result


def _tracked(count=5):
    index = _Tracked(3, "angular")
    for position in range(count):
        index.add_item(position, [float(position), 1.0, 2.0])
    index.build(4)
    return index


@pytest.fixture
def pinned_by_mapping(monkeypatch):
    """Apply the Windows rule for the indexes registered with the fixture."""
    import shutil

    from scikitplot.annoy._mixins import _io

    indexes = []
    real_replace, real_rmtree = os.replace, shutil.rmtree

    def pins(directory):
        directory = Path(directory).resolve()
        return any(
            index.mapped_from is not None and directory in index.mapped_from.parents
            for index in indexes
        )

    def replace(source, destination):
        if Path(source).is_dir() and pins(source):
            raise PermissionError(5, "Access is denied", os.fspath(source))
        return real_replace(source, destination)

    def rmtree(path, ignore_errors=False, **kwargs):
        if pins(path):
            if ignore_errors:
                return None
            raise PermissionError(5, "Access is denied", os.fspath(path))
        return real_rmtree(path, ignore_errors=ignore_errors, **kwargs)

    monkeypatch.setattr(_io.os, "replace", replace)
    monkeypatch.setattr(_io.shutil, "rmtree", rmtree)
    return indexes


def test_the_rule_is_applied_by_the_fixture(tmp_path, pinned_by_mapping):
    """The emulation refuses what Windows refuses; otherwise it proves nothing."""
    from scikitplot.annoy._mixins import _io

    index = _tracked()
    pinned_by_mapping.append(index)
    holder = tmp_path / "holder"
    holder.mkdir()
    index.save(os.fspath(holder / "index.ann"))
    with pytest.raises(PermissionError):
        _io.os.replace(holder, tmp_path / "elsewhere")
    _io.shutil.rmtree(holder, ignore_errors=True)
    assert holder.is_dir()
    index.unload()
    _io.os.replace(holder, tmp_path / "elsewhere")
    assert (tmp_path / "elsewhere" / "index.ann").is_file()


def test_a_bundle_is_published_where_a_mapping_pins_its_directory(
    tmp_path, pinned_by_mapping
):
    index = _tracked()
    pinned_by_mapping.append(index)
    expected = index.get_nns_by_item(0, 3)
    bundle = tmp_path / "bundle"
    members = index.save_bundle(bundle)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle"]
    assert [Path(member).name for member in members] == ["manifest.json", "index.ann"]
    # The index is usable, and is backed by the published member.
    assert index.get_nns_by_item(0, 3) == expected
    assert index.mapped_from == (bundle / "index.ann").resolve()
    assert INDEX.load_bundle(bundle).get_nns_by_item(0, 3) == expected


def test_a_bundle_is_replaced_where_a_mapping_pins_its_directory(
    tmp_path, pinned_by_mapping
):
    bundle = tmp_path / "bundle"
    index = _tracked(count=5)
    pinned_by_mapping.append(index)
    index.save_bundle(bundle)
    # Saving again: the index is mapped from the bundle it is about to replace.
    index.save_bundle(bundle)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle"]
    assert INDEX.load_bundle(bundle).get_n_items() == 5

    larger = _tracked(count=7)
    pinned_by_mapping.append(larger)
    index.unload()  # the reader of the old bundle lets go, as it must on Windows
    larger.save_bundle(bundle)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle"]
    assert INDEX.load_bundle(bundle).get_n_items() == 7


def test_a_failed_manifest_leaves_nothing_and_keeps_the_index(
    tmp_path, pinned_by_mapping, monkeypatch
):
    index = _tracked()
    pinned_by_mapping.append(index)
    expected = index.get_nns_by_item(0, 3)

    def boom(self, path, *args, **kwargs):
        raise OSError("injected: manifest write failed")

    monkeypatch.setattr(_Tracked, "to_json", boom, raising=False)
    with pytest.raises(OSError, match="injected"):
        index.save_bundle(tmp_path / "bundle")
    assert list(tmp_path.iterdir()) == []
    assert index.mapped_from is None
    assert index.get_nns_by_item(0, 3) == expected
    assert index.get_n_items() == 5 and index.get_n_trees() == 4


def test_a_failed_swap_restores_the_previous_bundle_and_keeps_the_index(
    tmp_path, pinned_by_mapping
):
    bundle = tmp_path / "bundle"
    _built(count=5).save_bundle(bundle)
    before = (bundle / "index.ann").read_bytes()
    reader = _Tracked.load_bundle(bundle)  # another index holds the old bundle
    pinned_by_mapping.append(reader)

    replacement = _tracked(count=7)
    pinned_by_mapping.append(replacement)
    expected = replacement.get_nns_by_item(0, 3)
    with pytest.raises(PermissionError):
        replacement.save_bundle(bundle)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle"]
    assert (bundle / "index.ann").read_bytes() == before
    assert replacement.mapped_from is None
    assert replacement.get_n_items() == 7
    assert replacement.get_nns_by_item(0, 3) == expected
    assert reader.get_n_items() == 5


def test_a_backend_that_cannot_be_released_is_refused_before_anything_is_written(
    tmp_path, monkeypatch
):
    index = _tracked()
    monkeypatch.setattr(_Tracked, "unload", None)
    with pytest.raises(TypeError, match=r"unload\(\)"):
        index.save_bundle(tmp_path / "bundle")
    assert list(tmp_path.iterdir()) == []
