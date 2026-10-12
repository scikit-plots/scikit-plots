# scikitplot/corpus/_metrics/_sklearn_scorer.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The corpus scorer on scikit-learn's own ``_BaseScorer``.

Imported only by :func:`scikitplot.corpus._metrics._scorer._resolve`, on first
use of a scorer, because importing this module imports scikit-learn.

Notes
-----
**Developer note.** scikit-learn 1.3 constructs ``_BaseScorer(score_func,
sign, kwargs)``; 1.4 and later add ``response_method``. The signature is read
once, here, and the base is called with what it accepts; ``_response_method``
is then set in both cases, which is what 1.4+ does itself. Checked on 1.3.2,
1.4.2, 1.5.2, 1.6.1, 1.7.2, 1.8.0 and 1.9.1.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from sklearn.metrics._scorer import _BaseScorer

from ._scorer import CorpusScorerMixin

__all__: list[str] = []

_BASE_ACCEPTS_RESPONSE_METHOD = (
    "response_method" in inspect.signature(_BaseScorer.__init__).parameters
)


class CorpusScorer(CorpusScorerMixin, _BaseScorer):
    """
    Corpus scorer whose base is scikit-learn's ``_BaseScorer``.

    Parameters
    ----------
    score_func : callable
        ``score_func(y_true, y_pred, **kwargs)``.
    sign : {1, -1}
        ``-1`` negates a loss.
    kwargs : dict
        Fixed arguments of ``score_func``.
    response_method : {"predict", "search"}, default="predict"
        How predictions are obtained.
    id_getter : callable or None, default=None
        Keyword-only. Maps a search result to its identifier.
    on_degraded : {"raise", "ignore"}, default="raise"
        Keyword-only. What a degraded search response does.

    Notes
    -----
    Construct it through :func:`scikitplot.corpus.make_corpus_scorer`, which
    validates the arguments.
    """

    def __init__(
        self,
        score_func: Callable[..., Any],
        sign: int,
        kwargs: dict[str, Any],
        response_method: str = "predict",
        *,
        id_getter: Callable[[Any], Any] | None = None,
        on_degraded: str = "raise",
    ) -> None:
        if _BASE_ACCEPTS_RESPONSE_METHOD:
            super().__init__(score_func, sign, kwargs, response_method)
        else:  # scikit-learn 1.3
            super().__init__(score_func, sign, kwargs)
        self._response_method = response_method
        self._id_getter = id_getter
        self._on_degraded = on_degraded
