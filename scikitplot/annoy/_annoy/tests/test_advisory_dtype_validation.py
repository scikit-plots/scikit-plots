# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CY-013 (guide 39).

``wrapper_dtype`` and ``random_dtype`` are advisory metadata (not used for
concrete dispatch), but they previously accepted arbitrary strings silently while
``index_dtype``/``dtype`` were validated. They are now validated against
``{"uint32", "uint64"}`` at construction so the reported metadata is meaningful.
"""
import pytest

from scikitplot.annoy._annoy import annoylib as A


@pytest.mark.parametrize("param", ["wrapper_dtype", "random_dtype"])
@pytest.mark.parametrize("bad", ["xyz", "nope", "float64", "int32", ""])
def test_invalid_advisory_dtype_rejected(param, bad):
    with pytest.raises(ValueError, match=f"Invalid {param}"):
        A.Index(4, "euclidean", **{param: bad})


@pytest.mark.parametrize("param", ["wrapper_dtype", "random_dtype"])
@pytest.mark.parametrize("good", ["uint32", "uint64"])
def test_valid_advisory_dtype_accepted_and_reported(param, good):
    idx = A.Index(4, "euclidean", **{param: good})
    assert idx.get_params()[param] == good     # honest, round-trips in metadata


def test_defaults_unchanged():
    idx = A.Index(4, "euclidean")
    assert idx.get_params()["wrapper_dtype"] == "uint64"
    assert idx.get_params()["random_dtype"] == "uint64"


def test_advisory_dtype_does_not_alter_behaviour():
    # advisory only: same data + metric -> identical query results regardless of
    # the advisory dtype label chosen.
    def build(wd):
        idx = A.Index(4, "euclidean", wrapper_dtype=wd)
        for i in range(8):
            idx.add_item(i, [float(i)] * 4)
        idx.build(5)
        return idx.get_nns_by_item(0, 4)
    assert build("uint32") == build("uint64")


def test_docstring_marks_them_advisory():
    doc = A.Index.__init__.__doc__ or ""
    assert "Advisory metadata only" in doc
