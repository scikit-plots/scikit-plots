# scikitplot/corpus/tests/test__filter_capability.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression gate for F-R07-01 — filters must never be silently ignored.

Review run R07 measured that ``StorageQuery.full_text`` was documented as
*"ignored by InMemoryStorage and JSONLStorage"*, and that on a three-document
corpus where exactly one matched, both backends returned **all three** with no
exception, warning or status.

That direction matters: an unsupported filter *over*-reports, which looks like
success, and any downstream ``top_k`` or threshold then operates on the wrong
candidate set.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from .._schema import CorpusDocument
from .._storage._storage import (
    FilterSupport,
    InMemoryStorage,
    JSONLStorage,
    SQLiteStorage,
    StorageQuery,
)

__all__: "list[str]" = [
    "TestFilterIsNeverIgnored",
    "TestFilterSupportReporting",
]


def _corpus():
    """Three documents; exactly one contains 'quantum'."""
    texts = ["alpha quantum", "beta ordinary", "gamma ordinary"]
    return [
        CorpusDocument.create(input_path=f"f{i}.txt", chunk_index=i, text=text)
        for i, text in enumerate(texts)
    ]


@pytest.fixture(name="backends")
def _backends(tmp_path: pathlib.Path):
    """One instance of each storage backend, populated identically."""
    made = {
        "memory": InMemoryStorage(),
        "jsonl": JSONLStorage(tmp_path / "c.jsonl"),
        "sqlite": SQLiteStorage(tmp_path / "c.db"),
    }
    for store in made.values():
        store.save_batch(_corpus())
    return made


class TestFilterIsNeverIgnored:
    """The defect itself."""

    @pytest.mark.parametrize("name", ["memory", "jsonl", "sqlite"])
    def test_full_text_actually_filters(self, backends, name: str) -> None:
        """Every backend must apply the filter, not return the whole corpus."""
        result = backends[name].query(StorageQuery(full_text="quantum", limit=100))
        assert len(result.documents) == 1, (
            f"{name} returned {len(result.documents)} of 3 documents for a "
            "filter matching exactly one -- F-R07-01 regression"
        )
        assert "quantum" in result.documents[0].text

    @pytest.mark.parametrize("name", ["memory", "jsonl", "sqlite"])
    def test_full_text_support_is_declared(self, backends, name: str) -> None:
        """A backend must say how it handled the filter -- never stay silent."""
        result = backends[name].query(StorageQuery(full_text="quantum", limit=100))
        support = result.filter_support["full_text"]
        assert support in (FilterSupport.SUPPORTED, FilterSupport.EMULATED)
        # Silently ignoring is not one of the options.
        assert support is not None

    def test_scanning_backends_declare_emulation_not_native_support(
        self, backends
    ) -> None:
        """EMULATED is a first-class state, not a hidden detail.

        A substring scan and an FTS5 index both *apply* the filter but differ in
        ranking quality and cost. Claiming SUPPORTED for a scan would be the
        unverified capability claim this contract exists to prevent.
        """
        for name in ("memory", "jsonl"):
            result = backends[name].query(StorageQuery(full_text="quantum"))
            assert result.filter_support["full_text"] is FilterSupport.EMULATED
            assert result.emulated_filters == ["full_text"]


class TestFilterSupportReporting:
    """Shape of the capability response."""

    def test_unrequested_filters_are_omitted(self, backends) -> None:
        """A caller must distinguish 'not asked for' from 'asked for and emulated'."""
        result = backends["memory"].query(StorageQuery(language="en"))
        assert "full_text" not in result.filter_support

    def test_attribute_filters_report_native_support(self, backends) -> None:
        result = backends["memory"].query(StorageQuery(input_path="f0.txt"))
        assert result.filter_support["input_path"] is FilterSupport.SUPPORTED

    def test_require_native_raises_on_emulation(self, backends) -> None:
        """Callers for whom emulation is unacceptable can say so."""
        result = backends["memory"].query(StorageQuery(full_text="quantum"))
        with pytest.raises(RuntimeError, match="emulated"):
            result.require_native()

    def test_require_native_passes_when_nothing_emulated(self, backends) -> None:
        result = backends["memory"].query(StorageQuery(input_path="f0.txt"))
        result.require_native()

    def test_sqlite_reports_truthfully_about_fts5(self, backends) -> None:
        """FTS5 is a compile-time option, so support is probed (unknown U-10).

        On a build with FTS5 this reports SUPPORTED; on one without, the same
        code path reports EMULATED rather than claiming a capability the build
        does not have.
        """
        from .._storage._storage import _sqlite_has_fts5

        result = backends["sqlite"].query(StorageQuery(full_text="quantum"))
        expected = (
            FilterSupport.SUPPORTED if _sqlite_has_fts5() else FilterSupport.EMULATED
        )
        assert result.filter_support["full_text"] is expected
