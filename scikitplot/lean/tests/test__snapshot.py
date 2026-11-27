# lean/tests/test__snapshot.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
A saved index loads back to the same reports, and a bad file is refused
=======================================================================

``LeanIndex.save`` writes the index's state (``lean.index.snapshot/1``);
``LeanIndex.load`` reads it back. The same files give the same bytes
whatever order they were added in; a damaged or altered file raises
``ValueError`` (checksum, offsets, numbers out of range), even when its
checksum was made to match; without ``root`` no file is read, and with it
only files inside it.
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from .. import _snapshot
from .._index import LeanIndex
from .test__order import _CACHES, _write


def _reports(index):
    return json.dumps(
        [
            index.summary(max_items=50),
            index.discover(top=50),
            [index.dossier(n, max_items=50) for n in ("top", "top2", "viaL", "zed")],
        ],
        sort_keys=True,
        default=str,
    )


@pytest.mark.parametrize("graph", ["python", "numpy"])
def test_a_saved_index_gives_the_same_reports(tmp_path, graph):
    if graph == "numpy":
        pytest.importorskip("numpy")
    _write(tmp_path)
    index = LeanIndex.from_folder(tmp_path)
    want = _reports(index)
    path = index.save(tmp_path / "out" / "index.leanidx")
    loaded = LeanIndex.load(path, root=tmp_path, graph=graph)
    assert _reports(loaded) == want


def test_the_same_files_give_the_same_bytes_in_any_order(tmp_path):
    files = _write(tmp_path)
    first = _snapshot.dumps(LeanIndex.from_folder(tmp_path))
    assert _snapshot.dumps(LeanIndex.from_folder(tmp_path)) == first
    index = LeanIndex()
    for path, label in random.Random(1).sample(files, len(files)):
        index.add_file(path, label=label)
    index.summary()  # resolution caches must not change the bytes
    assert _snapshot.dumps(index) == first
    reloaded = LeanIndex.load(index.save(tmp_path / "a.leanidx"))
    assert _snapshot.dumps(reloaded) == first  # idempotent through a load


def test_without_root_no_file_is_read(tmp_path):
    _write(tmp_path)
    path = LeanIndex.from_folder(tmp_path).save(tmp_path / "i.leanidx")
    loaded = LeanIndex.load(path)
    assert all(p is None for p in loaded._paths)
    report = loaded.dossier("top")
    assert report["target"]["statement"] is None
    assert report["flags_found"] == LeanIndex.from_folder(tmp_path).dossier("top")["flags_found"]


def _resign(blob, section=None, replace=None, header_change=None):
    """Return *blob* with a section replaced and the checksum made to match."""
    size = int.from_bytes(blob[8:16], "little")
    header = json.loads(blob[16 : 16 + size])
    body = bytearray(blob[16 + size :])
    if section is not None:
        place = header["sections"][section]
        old = place["length"]
        body[place["offset"] : place["offset"] + old] = replace
        shift = len(replace) - old
        for other in header["sections"].values():
            if other["offset"] > place["offset"]:
                other["offset"] += shift
        place["length"] = len(replace)
    if header_change:
        header_change(header)
    header["sha256"] = hashlib.sha256(bytes(body)).hexdigest()
    encoded = json.dumps(header).encode()
    return b"LEANIDX\x00" + len(encoded).to_bytes(8, "little") + encoded + bytes(body)


def test_a_bad_snapshot_is_refused(tmp_path):
    _write(tmp_path)
    blob = _snapshot.dumps(LeanIndex.from_folder(tmp_path))
    # Each damaged file, and the message it must be refused with.
    cases = {
        "not a LeanIndex snapshot": b"PK\x03\x04" + blob[4:],
        "truncated": blob[:12],
        "checksum does not match": blob[:-1] + bytes([blob[-1] ^ 1]),
        "is not 'lean.index.snapshot/1'": _resign(
            blob, header_change=lambda h: h.update(format="lean.index.snapshot/9")
        ),
        "reference number is out of range": _resign(
            blob, "ref_data", (2**30).to_bytes(4, "little") * (len(_section(blob, "ref_data")) // 4)
        ),
        "file number is out of range": _resign(
            blob, "file_of", (999).to_bytes(8, "little") * (len(_section(blob, "file_of")) // 8)
        ),
        "reference offsets": _resign(blob, "ref_ptr", _section(blob, "ref_ptr")[::-1]),
        "sections missing: shapes": _resign(
            blob, header_change=lambda h: h["sections"].pop("shapes")
        ),
    }
    for number, (expected, bad) in enumerate(cases.items()):
        path = tmp_path / f"bad{number}.leanidx"
        path.write_bytes(bad)
        with pytest.raises(ValueError, match="LeanIndex.load") as caught:
            LeanIndex.load(path)
        assert expected in str(caught.value), (expected, str(caught.value))


def _section(blob, name):
    size = int.from_bytes(blob[8:16], "little")
    header = json.loads(blob[16 : 16 + size])
    place = header["sections"][name]
    body = blob[16 + size :]
    return body[place["offset"] : place["offset"] + place["length"]]


def test_a_label_leading_outside_root_is_not_read(tmp_path):
    secret = tmp_path / "secret.lean"
    secret.write_text("theorem s : True := trivial\n", encoding="utf-8")
    project = tmp_path / "proj"
    (project / "P").mkdir(parents=True)
    (project / "P" / "A.lean").write_text("theorem t : True := trivial\n", encoding="utf-8")
    blob = _snapshot.dumps(LeanIndex.from_folder(project))
    meta = json.loads(_section(blob, "meta"))  # labels live in the metadata
    meta["labels"] = ["../secret.lean"]
    bad = _resign(blob, "meta", json.dumps(meta).encode())
    path = tmp_path / "bad.leanidx"
    path.write_bytes(bad)
    loaded = LeanIndex.load(path, root=project)
    assert loaded._paths == [None]  # outside the root: never read
    assert loaded.dossier("t")["target"]["statement"] is None


def test_an_empty_index_and_a_folder_path(tmp_path):
    empty = LeanIndex()
    loaded = LeanIndex.load(empty.save(tmp_path / "e.leanidx"))
    assert loaded.summary()["declarations"] == 0
    with pytest.raises(ValueError, match="is a folder"):
        empty.save(tmp_path)


def test_load_takes_the_options_the_index_was_built_with(tmp_path):
    _write(tmp_path)
    index = LeanIndex.from_folder(tmp_path, visibility="all", suffix="unique")
    loaded = LeanIndex.load(index.save(tmp_path / "o.leanidx"))
    assert (loaded._visibility, loaded._suffix) == ("all", "unique")


def test_a_snapshot_decides_for_every_attribute(tmp_path):
    _write(tmp_path)
    index = LeanIndex.from_folder(tmp_path)
    index.summary()
    index.discover()
    index.dossier("top")
    placed = (
        _snapshot.SAVED | _snapshot.REBUILT | _snapshot.CHOSEN_WHEN_LOADING | _CACHES
    )
    assert set(vars(index)) <= placed, sorted(set(vars(index)) - placed)
    assert not _snapshot.SAVED & _snapshot.REBUILT


def test_the_command_line_saves_and_loads(tmp_path):
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    from .._index import main  # noqa: PLC0415

    _write(tmp_path)
    snapshot = tmp_path / "s.leanidx"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        built = main([str(tmp_path), "--save", str(snapshot), "--json"])
    first = out.getvalue()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        loaded = main([str(tmp_path), "--load", str(snapshot), "--json"])
    assert built == loaded and snapshot.is_file()
    assert json.loads(out.getvalue()) == json.loads(first)
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as caught:
        main([str(tmp_path), "--load", str(tmp_path / "missing.leanidx")])
    assert caught.value.code == 2  # a missing snapshot is a usage error
