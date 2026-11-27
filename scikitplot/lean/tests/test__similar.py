# lean/tests/test__similar.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Near-duplicate statements are found exactly
===========================================

Every pair of results whose statement shapes share at least the threshold
of their adjacent token pairs is found, and none below: the prefix-filtered
search gives the pairs comparing every two sets gives. Results of one shape
are left to discovery; layers choose the two sides.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import textwrap
from array import array
from fractions import Fraction

import pytest

from .._index import LeanIndex, main
from .._similar import similar_pairs

SOURCE = {
    "P/A.lean": """
        theorem le_sum (a b c : Nat) (h : a ≤ b) (k : b ≤ c) : a + b ≤ c + c := by omega
        theorem lt_sum (a b c : Nat) (h : a ≤ b) (k : b ≤ c) : a + b < c + c + 1 := by omega
        theorem same_shape (x y z : Nat) (h : x ≤ y) (k : y ≤ z) : x + y ≤ z + z := by omega
        theorem other (n : Nat) : n * n = n ^ 2 := by ring
        theorem tiny : True := trivial
    """,
}


def _write(root, files):
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return root


def _brute(raw, threshold):
    return sorted(
        (i, j, len(raw[i] & raw[j]), len(raw[i] | raw[j]))
        for i in range(len(raw))
        for j in range(i + 1, len(raw))
        if Fraction(len(raw[i] & raw[j]), len(raw[i] | raw[j])) >= threshold
    )


@pytest.mark.parametrize("seed", range(8))
def test_prefix_filtering_finds_what_comparing_every_pair_finds(seed):
    rng = random.Random(seed)
    for _ in range(10):
        vocabulary = rng.randint(3, 30)
        raw = [
            set(rng.sample(range(vocabulary), rng.randint(1, min(vocabulary, 12))))
            for _ in range(rng.randint(2, 70))
        ]
        counts = {}
        for found in raw:
            for token in found:
                counts[token] = counts.get(token, 0) + 1
        rank = {t: r for r, t in enumerate(sorted(counts, key=lambda t: (counts[t], t)))}
        sets = [array("i", sorted(rank[t] for t in found)) for found in raw]
        for threshold in (Fraction(1, 3), Fraction(1, 2), Fraction(4, 5), Fraction(1)):
            assert sorted(similar_pairs(sets, threshold)) == _brute(raw, threshold)


def test_near_statements_are_paired_and_one_shape_left_to_discovery(tmp_path):
    index = LeanIndex.from_folder(_write(tmp_path / "p", SOURCE))
    report = index.near_duplicates(threshold=0.5)
    found = {
        (p["a"]["declaration"], p["b"]["declaration"]): (p["shared_pairs"], p["pairs_in_either"])
        for p in report["pairs"]["items"]
    }
    assert found == {("le_sum", "lt_sum"): (25, 30), ("lt_sum", "same_shape"): (25, 30)}
    assert report["pairs"]["items"][0]["similarity"] == 0.8333
    assert report["statements_compared"] == 4  # tiny has too few token pairs
    assert report["pairs"]["items"][0]["a"]["statement"].startswith("theorem le_sum")
    assert index.near_duplicates(threshold=0.9)["pairs"]["count"] == 0
    assert index.near_duplicates(threshold=0.5, min_pairs=1)["statements_compared"] == 5
    text = LeanIndex.render(report)
    assert "## 83%: `le_sum` and `lt_sum`" in text and "P/A.lean:2" in text


def test_the_report_is_the_same_for_any_number_of_workers(tmp_path):
    root = _write(tmp_path / "p", SOURCE)
    for k in range(6):
        _write(root, {f"P/More{k}.lean": SOURCE["P/A.lean"].replace("le_sum", f"le_sum{k}")})
    one = LeanIndex.from_folder(root).near_duplicates(threshold=0.5)
    two = LeanIndex.from_folder(root, workers=2).near_duplicates(threshold=0.5)
    assert one == two and one["pairs"]["count"] > 2


def test_layers_choose_the_two_sides(tmp_path):
    base = _write(tmp_path / "base", {"B/A.lean": SOURCE["P/A.lean"].split("\n", 2)[1] + "\n"})
    saved = LeanIndex.from_folder(base).save(tmp_path / "b.leanidx")
    index = LeanIndex.load(saved, root=base)
    index.add_folder(_write(tmp_path / "proj", SOURCE), name="p")
    across = index.near_duplicates(threshold=0.5, layer="p", against="")
    pairs = {(p["a"]["file"], p["b"]["file"]) for p in across["pairs"]["items"]}
    assert pairs == {("B/A.lean", "p:P/A.lean")}
    assert [p["b"]["declaration"] for p in across["pairs"]["items"]] == ["lt_sum"]
    within = index.near_duplicates(threshold=0.5, layer="p", against="p")
    assert within["pairs"]["count"] == 2


def test_files_without_a_path_are_listed_not_read(tmp_path):
    root = _write(tmp_path / "p", SOURCE)
    saved = LeanIndex.from_folder(root).save(tmp_path / "a.leanidx")
    report = LeanIndex.load(saved).near_duplicates(threshold=0.5)
    assert report["files_not_read"]["items"] == ["P/A.lean"]
    assert report["pairs"]["count"] == 0
    assert LeanIndex.load(saved, root=root).near_duplicates(threshold=0.5)["pairs"]["count"] == 2


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("threshold", 0, "threshold must be a number in"),
        ("threshold", 1.5, "threshold must be a number in"),
        ("threshold", True, "threshold must be a number in"),
        ("threshold", "0.8", "threshold must be a number in"),
        ("min_pairs", 0, "min_pairs must be an int > 0"),
        ("top", -1, "top must be an int > 0"),
        ("layer", "q", "layer must be None or one of"),
        ("against", "q", "layer must be None or one of"),
    ],
)
def test_options_are_checked(tmp_path, option, value, message):
    index = LeanIndex.from_folder(_write(tmp_path / "p", SOURCE))
    with pytest.raises(ValueError, match=message):
        index.near_duplicates(**{option: value})


def test_the_command_line_lists_near_duplicates(tmp_path):
    root = _write(tmp_path / "p", SOURCE)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = main([str(root), "--near-duplicates", "--threshold", "0.5", "--json"])
    assert status == 0
    report = json.loads(out.getvalue())
    assert (report["report"], report["threshold"], report["pairs"]["count"]) == (
        "near_duplicates", 0.5, 2,
    )  # fmt: skip


# -- shapes kept when read ------------------------------------------------------


def _many(root):
    for k in range(4):
        _write(root, {f"P/More{k}.lean": SOURCE["P/A.lean"].replace("le_sum", f"le_sum{k}")})
    return root


@pytest.mark.parametrize("workers", [1, 2])
def test_kept_shapes_give_the_report_reading_gives(tmp_path, workers):
    root = _many(_write(tmp_path / "p", SOURCE))
    read = LeanIndex.from_folder(root).near_duplicates(threshold=0.5)
    kept = LeanIndex.from_folder(root, keep_shapes=True, workers=workers)
    assert kept.near_duplicates(threshold=0.5) == read
    assert read["pairs"]["count"] > 2


def test_kept_shapes_need_no_file(tmp_path):
    root = _many(_write(tmp_path / "p", SOURCE))
    saved = LeanIndex.from_folder(root, keep_shapes=True).save(tmp_path / "k.leanidx")
    blind = LeanIndex.load(saved).near_duplicates(threshold=0.5)
    assert blind["files_not_read"]["count"] == 0
    seeing = LeanIndex.load(saved, root=root).near_duplicates(threshold=0.5)
    assert [p["similarity"] for p in blind["pairs"]["items"]] == [
        p["similarity"] for p in seeing["pairs"]["items"]
    ]
    assert blind["pairs"]["items"][0]["a"]["statement"] is None  # not quoted
    assert seeing["pairs"]["items"][0]["a"]["statement"].startswith("theorem")


def test_kept_shapes_survive_saving_refreshing_and_adding_in_any_order(tmp_path):
    root = _many(_write(tmp_path / "p", SOURCE))
    index = LeanIndex.from_folder(root, keep_shapes=True)
    _write(root, {"P/More0.lean": "theorem changed (n : Nat) : n + 0 = n := rfl\n"})
    (root / "P" / "More1.lean").unlink()
    index.refresh()
    fresh = LeanIndex.from_folder(root, keep_shapes=True)
    assert index.save(tmp_path / "a").read_bytes() == fresh.save(tmp_path / "b").read_bytes()
    shuffled = LeanIndex(keep_shapes=True)
    for path in sorted(root.rglob("*.lean"), reverse=True):
        shuffled.add_file(path, label=path.relative_to(root).as_posix())
    assert shuffled.save(tmp_path / "c").read_bytes() == fresh.save(tmp_path / "d").read_bytes()
    loaded = LeanIndex.load(tmp_path / "b")
    assert loaded._keep_shapes and loaded._shape_tokens_of(0) == fresh._shape_tokens_of(0)


def test_keeping_shapes_changes_nothing_else(tmp_path):
    root = _many(_write(tmp_path / "p", SOURCE))
    plain, kept = LeanIndex.from_folder(root), LeanIndex.from_folder(root, keep_shapes=True)
    assert plain.summary() == kept.summary()
    assert plain.discover() == kept.discover()
    assert plain._shape_of == kept._shape_of


def test_keep_shapes_is_checked(tmp_path):
    with pytest.raises(TypeError, match="keep_shapes must be True or False"):
        LeanIndex(keep_shapes=1)


def test_kept_shapes_in_a_snapshot_are_checked(tmp_path):
    from .test__snapshot import _resign, _section

    root = _many(_write(tmp_path / "p", SOURCE))
    saved = LeanIndex.from_folder(root, keep_shapes=True).save(tmp_path / "k.leanidx")
    blob = saved.read_bytes()
    tokens = bytearray(_section(blob, "shape_tokens"))
    tokens[0:4], tokens[4:8] = tokens[4:8], tokens[0:4]  # two tokens swapped
    bad = {
        "do not match its shape": _resign(blob, "shape_tokens", bytes(tokens)),
        "sections are missing": _resign(
            blob, header_change=lambda h: h["sections"].pop("shape_ptr")
        ),
    }
    for message, changed in bad.items():
        saved.write_bytes(changed)
        with pytest.raises(ValueError, match=message):
            LeanIndex.load(saved)


def test_the_command_line_keeps_shapes(tmp_path):
    root = _many(_write(tmp_path / "p", SOURCE))
    saved = tmp_path / "k.leanidx"
    with contextlib.redirect_stdout(io.StringIO()):
        assert main([str(root), "--keep-shapes", "--save", str(saved)]) == 0
    assert LeanIndex.load(saved)._keep_shapes
