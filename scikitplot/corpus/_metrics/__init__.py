# scikitplot/corpus/_metrics/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluation of corpus retrieval and answers.

Three layers, each usable on its own:

Metric functions
    Ranking metrics (:func:`precision_at_k`, :func:`recall_at_k`,
    :func:`hit_rate_at_k`, :func:`mean_reciprocal_rank`,
    :func:`mean_average_precision`, :func:`r_precision`, :func:`ndcg_at_k`,
    or several at once with :func:`ranking_metrics`) and answer metrics
    (:func:`exact_match_score`, :func:`token_f1_score`,
    :func:`edit_similarity_score`, :func:`anls_score`), all with the
    scikit-learn signature ``metric(y_true, y_pred, ...)``.
scikit-learn scorers
    :func:`make_corpus_scorer` and :func:`get_corpus_scorer` turn a metric into
    a scorer accepted by ``cross_validate``, ``GridSearchCV`` and
    ``check_scoring``, and able to score a corpus index directly
    (``response_method="search"``).
Harness
    :func:`evaluate_retrieval` runs a retriever over a query set and returns a
    :class:`RetrievalEvaluation`: per-query and aggregate metrics, rankings,
    latency and response statuses, written to JSON, JSON Lines, YAML or TOML,
    and compared with a baseline by :meth:`RetrievalEvaluation.compare`.
LLM judges
    :func:`make_llm` puts OpenAI, Anthropic, Gemini or any callable behind one
    call; :class:`LLMJudge` scores faithfulness, answer relevance, context
    precision and recall, answer correctness, a custom rubric and pairwise
    preference from strictly parsed verdicts; :func:`evaluate_answers` and
    :func:`evaluate_preference` run judges (one or a panel) over a dataset,
    and :func:`krippendorff_alpha`, :func:`fleiss_kappa` and
    :func:`cohen_kappa` measure how far the judges agree.

Notes
-----
**User note.** Importing this package does not import scikit-learn; a scorer
resolves its base class on first use (:func:`scorer_backend_info`).

**Developer note.** All metric paths share :mod:`._inputs` (validation) and
the per-query formulas of :mod:`._ranking`, so a metric has one definition.
"""

from __future__ import annotations

from . import (
    _agreement,
    _answers,
    _evaluate,
    _inputs,
    _judge,
    _llm,
    _ranking,
    _scorer,
    _text,
)
from ._agreement import *  # noqa: F403
from ._answers import *  # noqa: F403
from ._evaluate import *  # noqa: F403
from ._inputs import *  # noqa: F403
from ._judge import *  # noqa: F403
from ._llm import *  # noqa: F403
from ._ranking import *  # noqa: F403
from ._scorer import *  # noqa: F403
from ._text import *  # noqa: F403

__all__: list[str] = []
__all__ += _agreement.__all__
__all__ += _answers.__all__
__all__ += _evaluate.__all__
__all__ += _inputs.__all__
__all__ += _judge.__all__
__all__ += _llm.__all__
__all__ += _ranking.__all__
__all__ += _scorer.__all__
__all__ += _text.__all__
