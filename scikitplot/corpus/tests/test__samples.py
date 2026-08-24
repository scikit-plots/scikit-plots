# scikitplot/corpus/tests/test__samples.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import scikitplot.corpus as corpus


def test_hamlet_text_is_public_nonempty_and_useful() -> None:
    assert corpus.HAMLET_TEXT.startswith("THE TRAGEDY OF HAMLET")
    assert "To be, or not to be" in corpus.HAMLET_TEXT
    assert "Alas, poor Yorick" in corpus.HAMLET_TEXT
    assert len(corpus.HAMLET_TEXT) > 1000


def test_new_helpers_are_top_level_public_exports() -> None:
    expected = {
        "HAMLET_TEXT",
        "HashEmbedder",
        "SimpleEnricherSpec",
        "SimpleFrequencyEnricher",
    }
    assert expected <= set(corpus.__all__)
    for name in expected:
        assert hasattr(corpus, name)
