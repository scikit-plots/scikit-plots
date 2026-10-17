# scikitplot/corpus/_metrics/_sklearn_compat.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# ``_BaseScorer`` below is adapted from scikit-learn's
# ``sklearn/metrics/_scorer.py`` at commit
# afe5bcd80ce17fd8ca88c33beb17534bd36423bc
# (https://github.com/scikit-learn/scikit-learn/blob/afe5bcd80ce17fd8ca88c33beb17534bd36423bc/sklearn/metrics/_scorer.py),
# whose notice follows (the COPYING file at that commit). The adaptation keeps
# the constructor, ``__call__`` (including the deprecation of a positional
# ``sample_weight``), ``_warn_overlap`` and ``_accept_sample_weight`` contracts
# and replaces metadata routing, which needs scikit-learn, by an explicit error.
#
# BSD 3-Clause License
#
# Copyright (c) 2007-2026 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Vendored scorer base, used when scikit-learn's cannot be.

:func:`scikitplot.corpus._metrics._scorer._resolve` selects this base when
scikit-learn is not importable (a ``scikit-plots-corpus`` install without it),
or when the installed ``_BaseScorer`` fails the contract probe.

Notes
-----
**User note.** A scorer on this base behaves like one on scikit-learn's for a
direct call, ``sample_weight`` included. Metadata routing is scikit-learn's
mechanism, so ``set_score_request`` raises here, as do extra keyword
arguments at call time (scikit-learn raises for them too unless routing is
enabled).

**Developer note.** Keep the signatures in step with scikit-learn's base: the
contract probe and ``test__sklearn_compat.py`` run both bases on the same
inputs.
"""

from __future__ import annotations

import warnings
from inspect import signature
from typing import Any, Callable

from ._scorer import CorpusScorerMixin

__all__: list[str] = []


def _call_method(
    estimator: Any,
    response_method: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call ``estimator.<response_method>(*args, **kwargs)`` (no cache)."""
    return getattr(estimator, response_method)(*args, **kwargs)


class _BaseScorer:
    """
    Base scorer used as ``scorer(estimator, X, y_true)``.

    Parameters
    ----------
    score_func : callable
        Called as ``score_func(y_true, y_pred, **kwargs)``.
    sign : int
        ``1`` or ``-1``; the score is ``sign * score_func(...)``.
    kwargs : dict
        Additional parameters of the score function.
    response_method : str, default="predict"
        The estimator method that produces the predictions.
    """

    def __init__(
        self,
        score_func: Callable[..., Any],
        sign: int,
        kwargs: dict[str, Any],
        response_method: str = "predict",
    ) -> None:
        self._score_func = score_func
        self._sign = sign
        self._kwargs = kwargs
        self._response_method = response_method

    def _accept_sample_weight(self) -> bool:
        return "sample_weight" in signature(self._score_func).parameters

    def _routing_repr(self) -> str:
        return repr(self)

    def __call__(
        self,
        estimator: Any,
        X: Any,
        y_true: Any,
        *args: Any,
        sample_weight: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Evaluate the predictions of ``estimator`` on ``X`` against ``y_true``.

        Parameters
        ----------
        estimator : object
            The model or index to score.
        X : sequence
            Queries.
        y_true : sequence
            Ground truth per query.
        *args : tuple
            A positional ``sample_weight``: accepted with a ``FutureWarning``,
            as scikit-learn deprecates it (removal in its 1.11).
        sample_weight : array-like or None, default=None
            Per-query weights. Pass it by keyword.
        **kwargs : dict
            Not supported without scikit-learn's metadata routing.

        Returns
        -------
        float

        Raises
        ------
        TypeError
            For more than one extra positional argument, or ``sample_weight``
            given both ways.
        ValueError
            If ``kwargs`` is not empty.
        """
        if args:
            if len(args) > 1 or sample_weight is not None:
                raise TypeError(
                    "__call__ takes at most one extra positional argument, "
                    "sample_weight, and not together with sample_weight=."
                )
            warnings.warn(
                "Pass sample_weight=... as a keyword argument; passing it "
                "positionally is deprecated, as in scikit-learn (removal in its 1.11).",
                FutureWarning,
                stacklevel=2,
            )
            sample_weight = args[0]
        if kwargs:
            raise ValueError(
                f"Passing extra keyword arguments ({sorted(kwargs)}) to a scorer "
                "is only supported with scikit-learn's metadata routing, which "
                "is not available; pass them to make_corpus_scorer instead."
            )
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        return self._score(_call_method, estimator, X, y_true, **kwargs)

    def _warn_overlap(self, message: str, kwargs: dict[str, Any]) -> None:
        """Warn if there is any overlap between ``self._kwargs`` and ``kwargs``."""
        _kwargs = set() if self._kwargs is None else set(self._kwargs.keys())
        overlap = _kwargs.intersection(kwargs.keys())
        if overlap:
            warnings.warn(
                f"{message} Overlapping parameters are: {overlap}",
                UserWarning,
                stacklevel=3,
            )

    def set_score_request(self, **kwargs: Any) -> Any:
        """
        Refuse: metadata routing needs scikit-learn.

        Raises
        ------
        RuntimeError
            Always.
        """
        try:
            import sklearn  # noqa: F401, PLC0415
        except Exception:  # noqa: BLE001 - any import failure: not installed or broken
            advice = (
                "install scikit-learn >= 1.4 and enable it with "
                "sklearn.set_config(enable_metadata_routing=True)"
            )
        else:
            advice = (
                "scikit-learn is installed, but this scorer uses the vendored base; "
                "scorer_backend_info().reason says why"
            )
        raise RuntimeError(
            f"set_score_request needs scikit-learn's metadata routing: {advice}."
        )


class CorpusScorer(CorpusScorerMixin, _BaseScorer):
    """
    Corpus scorer on the vendored base.

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
        super().__init__(score_func, sign, kwargs, response_method)
        self._id_getter = id_getter
        self._on_degraded = on_degraded
