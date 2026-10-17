# scikitplot/corpus/_metrics/_answers.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Answer evaluation harness: a question set, a system, judges, one report.

:func:`evaluate_answers` asks a system (your model, a RAG pipeline, or stored
answers) every question of a dataset, scores each answer with deterministic
text metrics against the references and with one or more
:class:`~._judge.LLMJudge` judges, and returns an :class:`AnswerEvaluation`:
per item and mean scores, each judge's scores and the panel mean, the
judges' agreement, token usage, latency, and the settings, written to JSON,
JSON Lines, YAML or TOML and compared with another evaluation by
:meth:`AnswerEvaluation.compare`.

:func:`evaluate_preference` compares two systems' answers head to head with
the judges' order-swapped :meth:`~._judge.LLMJudge.preference`.

Metric names in a report:

* ``exact_match``, ``token_f1``: deterministic, against the references;
* ``<metric>/<judge>``: one judge's score (``faithfulness/gpt``);
* ``<metric>``: the panel score, the mean of the judges' defined scores;
* ``criteria:<name>``: a caller's rubric (``criteria={"name": "rubric"}``).

Notes
-----
**User note.** A dataset is a sequence of mappings with ``"question"``
(required), and optionally ``"id"``, ``"reference"`` (a ``str``, or a
sequence of acceptable ``str`` for the text metrics), ``"contexts"`` and
``"answer"``. With ``system=None`` the stored ``"answer"`` of each item is
evaluated: this is how to score answers your model, GPT, Claude or Gemini
produced elsewhere. A metric whose input an item lacks (no reference, no
contexts) is undefined (``NaN``) for that item and costs no call. Judges
see a reference only when the item has exactly one; an item with several
acceptable references gives them to the text metrics, and
``context_recall`` and ``answer_correctness`` are undefined for it.

Judges are models too: their scores vary between runs and between judges.
Report the judge identities (recorded automatically), use more than one
judge when the decision matters, and read the agreement: an alpha well below
1 means the judges do not measure the same thing.

**Developer note.** Results are in dataset order whatever ``n_jobs`` is:
work is mapped over a thread pool and collected in input order. The paired
comparison is :func:`._evaluate.paired_comparison`, shared with
:class:`~._evaluate.RetrievalEvaluation`.
"""

from __future__ import annotations

import math
import numbers
import re
import time
import warnings
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np

from .. import _serialize
from ._agreement import krippendorff_alpha
from ._evaluate import EvaluationComparison, paired_comparison
from ._judge import JUDGE_METRICS, JudgeResult, LLMJudge, context_texts
from ._llm import LLM, Completion
from ._text import exact_match_score, token_f1_score

__all__ = [
    "AnswerEvaluation",
    "AnswerItem",
    "PreferenceEvaluation",
    "evaluate_answers",
    "evaluate_preference",
]

SCHEMA = "scikitplot.corpus.answer_evaluation"
PREFERENCE_SCHEMA = "scikitplot.corpus.preference_evaluation"
SCHEMA_VERSION = 1

#: Deterministic metrics against the references.
TEXT_METRICS: Mapping[str, Callable[..., Any]] = {
    "exact_match": exact_match_score,
    "token_f1": token_f1_score,
}

#: Judge metrics :func:`evaluate_answers` runs by default.
DEFAULT_JUDGE_METRICS: tuple[str, ...] = ("faithfulness", "answer_relevance")

#: What each judge metric needs from an item.
_NEEDS: Mapping[str, tuple[str, ...]] = {
    "faithfulness": ("answer", "contexts"),
    "answer_relevance": ("answer",),
    "context_precision": ("contexts",),
    "context_recall": ("contexts", "reference"),
    "answer_correctness": ("answer", "reference"),
    "criteria": ("answer",),
}

_NAME = re.compile(r"[A-Za-z0-9_.-]+")
_MIN_PANEL = 2  # judges needed for an agreement statistic
_TWO_SYSTEMS = 2
_ON_ERROR = ("raise", "record")
_ITEM_KEYS = frozenset({"id", "question", "reference", "contexts", "answer"})


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _check_name(name: Any, what: str) -> str:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(
            f"{what} {name!r} must match [A-Za-z0-9_.-]+ (it becomes part of metric names)."
        )
    return name


def _check_judges(judges: Any, on_error: str) -> dict[str, LLMJudge]:
    if judges is None:
        return {}
    if isinstance(judges, (LLMJudge, LLM)) or (
        callable(judges) and not isinstance(judges, Mapping)
    ):
        judges = {"judge": judges}
    if not isinstance(judges, Mapping):
        raise TypeError(
            "judges must be a mapping name -> judge, a single judge, or None."
        )
    out: dict[str, LLMJudge] = {}
    for name, judge in judges.items():
        _check_name(name, "Judge name")
        llm = judge.llm if isinstance(judge, LLMJudge) else judge
        out[name] = LLMJudge(llm, on_error=on_error)
    return out


def _check_common(
    on_error: Any, n_jobs: Any, metadata: Any
) -> tuple[int, dict[str, Any]]:
    """Validate the options both harnesses share; return the worker count and metadata."""
    if on_error not in _ON_ERROR:
        raise ValueError(f"on_error must be one of {_ON_ERROR}, got {on_error!r}.")
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, numbers.Integral)
        or n_jobs < 1
    ):
        raise ValueError(f"n_jobs must be an int >= 1, got {n_jobs!r}.")
    meta = {} if metadata is None else metadata
    if not isinstance(meta, Mapping):
        raise TypeError("metadata must be a mapping or None.")
    _serialize.to_builtin(dict(meta))
    return int(n_jobs), dict(meta)


def _references(value: Any, label: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping) or not isinstance(value, Iterable):
        raise TypeError(
            f"{label}: reference must be a str or a sequence of str, got {type(value).__name__!r}."
        )
    refs = tuple(value)
    if not refs or not all(isinstance(ref, str) for ref in refs):
        raise TypeError(f"{label}: reference must be a non-empty sequence of str.")
    return refs


def _items(dataset: Any) -> list[dict[str, Any]]:
    """Validate the dataset; return normalised items with unique ids."""
    if isinstance(dataset, (str, bytes, Mapping)) or not isinstance(dataset, Iterable):
        raise TypeError(
            f"dataset must be a sequence of mappings, got {type(dataset).__name__!r}."
        )
    items = []
    seen: set = set()
    for i, row in enumerate(dataset):
        label = f"dataset[{i}]"
        if not isinstance(row, Mapping):
            raise TypeError(f"{label} must be a mapping, got {type(row).__name__!r}.")
        unknown = sorted(set(row) - _ITEM_KEYS)
        if unknown:
            raise ValueError(
                f"{label} has unknown keys {unknown}; allowed: {sorted(_ITEM_KEYS)}."
            )
        if not isinstance(row.get("question"), str):
            raise TypeError(f"{label}['question'] must be a str.")
        item_id = row.get("id", i)
        if isinstance(item_id, bool) or not isinstance(
            item_id, (str, numbers.Integral)
        ):
            raise TypeError(
                f"{label}['id'] must be a str or an int, got {type(item_id).__name__!r}."
            )
        item_id = item_id if isinstance(item_id, str) else int(item_id)
        if item_id in seen:
            raise ValueError(f"{label}: duplicate id {item_id!r}.")
        seen.add(item_id)
        answer = row.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise TypeError(f"{label}['answer'] must be a str or absent.")
        contexts = row.get("contexts")
        items.append(
            {
                "id": item_id,
                "question": row["question"],
                "references": _references(row.get("reference"), label),
                "contexts": (
                    None
                    if contexts is None
                    else context_texts(contexts, label=f"{label}['contexts']")
                ),
                "answer": answer,
            }
        )
    if not items:
        raise ValueError("dataset is empty.")
    ids = {type(item["id"]) for item in items}
    if len(ids) > 1:
        raise TypeError("dataset ids mix str and int; use one type.")
    return items


def _check_metrics(
    metrics: Any, criteria: Any, text_metrics: Any
) -> tuple[list[str], dict[str, str], list[str]]:
    if isinstance(metrics, str) or not isinstance(metrics, Iterable):
        raise TypeError("metrics must be a sequence of judge metric names.")
    names = list(metrics)
    for name in names:
        if name not in JUDGE_METRICS or name == "criteria":
            raise ValueError(
                f"Unknown judge metric {name!r}; choose from "
                f"{[m for m in JUDGE_METRICS if m != 'criteria']} (rubrics go in criteria=)."
            )
    if len(set(names)) != len(names):
        raise ValueError("metrics must not repeat a name.")
    rubrics = {} if criteria is None else criteria
    if not isinstance(rubrics, Mapping):
        raise TypeError("criteria must be a mapping name -> rubric text, or None.")
    for key, text in rubrics.items():
        _check_name(key, "Criteria name")
        if not isinstance(text, str) or not text.strip():
            raise TypeError(f"criteria[{key!r}] must be a non-empty str.")
    if isinstance(text_metrics, str) or not isinstance(text_metrics, Iterable):
        raise TypeError("text_metrics must be a sequence of names.")
    texts = list(text_metrics)
    for name in texts:
        if name not in TEXT_METRICS:
            raise ValueError(
                f"Unknown text metric {name!r}; choose from {list(TEXT_METRICS)}."
            )
    if len(set(texts)) != len(texts):
        raise ValueError("text_metrics must not repeat a name.")
    return names, dict(rubrics), texts


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _nan_or(value: Any) -> float:
    return math.nan if value is None else float(value)


@dataclass(frozen=True)
class AnswerItem:
    """
    One item of an :class:`AnswerEvaluation`.

    Parameters
    ----------
    item_id : str or int
        Item identifier.
    question : str
        The question.
    answer : str or None
        The evaluated answer; ``None`` when the system failed.
    scores : mapping
        Metric name to score (``NaN`` when undefined or failed).
    n_contexts : int or None, default=None
        Contexts the judges saw (``None``: no contexts).
    latency_s : float, default=0.0
        Seconds the system took (``0.0`` for stored answers).
    error : str or None, default=None
        Why the system produced no answer.
    judgments : mapping, default={}
        ``{metric: {judge: JudgeResult}}``, when verdicts are kept.
    """

    item_id: Any
    question: str
    answer: str | None
    scores: Mapping[str, float]
    n_contexts: int | None = None
    latency_s: float = 0.0
    error: str | None = None
    judgments: Mapping[str, Mapping[str, JudgeResult]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a mapping in the report schema."""
        return {
            "item_id": self.item_id,
            "question": self.question,
            "answer": self.answer,
            "scores": dict(self.scores),
            "n_contexts": self.n_contexts,
            "latency_s": self.latency_s,
            "error": self.error,
            "judgments": {
                metric: {judge: result.to_dict() for judge, result in by_judge.items()}
                for metric, by_judge in self.judgments.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnswerItem:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            item_id=data["item_id"],
            question=data["question"],
            answer=data.get("answer"),
            scores={k: _nan_or(v) for k, v in data["scores"].items()},
            n_contexts=data.get("n_contexts"),
            latency_s=float(data.get("latency_s", 0.0)),
            error=data.get("error"),
            judgments={
                metric: {
                    judge: JudgeResult.from_dict(r) for judge, r in by_judge.items()
                }
                for metric, by_judge in data.get("judgments", {}).items()
            },
        )


class _Report:
    """Serialisation shared by the two report classes."""

    _schema: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_rows(self) -> list[dict[str, Any]]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _payload(self, fmt: str) -> Any:
        if fmt == "toml" and _has_none(self.metadata):
            raise ValueError(
                "metadata holds None, which TOML cannot represent and would drop; "
                "remove those entries or write json or yaml."
            )
        return self.to_rows() if fmt == "jsonl" else self.to_dict()

    def dumps(self, format: str = "json") -> str:  # noqa: A002
        """Serialise to ``"json"``, ``"jsonl"`` (one row per item), ``"yaml"`` or ``"toml"``."""
        return _serialize.dumps(self._payload(format), format)

    def write(
        self,
        path: Any,
        format: str | None = None,
    ) -> Any:  # noqa: A002
        """Write atomically; the format defaults to the path suffix."""
        fmt = _serialize.resolve_format(path, format)
        return _serialize.write(path, self._payload(fmt), fmt)

    @classmethod
    def _read_dict(
        cls,
        path: Any,
        format: str | None,
    ) -> Mapping[str, Any]:  # noqa: A002
        fmt = _serialize.resolve_format(path, format)
        if fmt == "jsonl":
            raise ValueError(
                "A JSON Lines file holds per-item rows, not a full report; write json, yaml or toml."
            )
        data = _serialize.read(path, fmt)
        if not isinstance(data, Mapping) or data.get("schema") != cls._schema:
            raise ValueError(f"Not a {cls._schema} report.")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Report schema version {data.get('schema_version')!r} is not {SCHEMA_VERSION}."
            )
        return data


def _has_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Mapping):
        return any(_has_none(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_none(v) for v in value)
    return False


def _mean(values: Sequence[float]) -> float:
    defined = [v for v in values if not math.isnan(v)]
    return math.fsum(defined) / len(defined) if defined else math.nan


@dataclass(frozen=True)
class AnswerEvaluation(_Report):
    """
    Result of :func:`evaluate_answers`.

    Parameters
    ----------
    metrics : mapping
        Metric name to the mean over items where it is defined.
    items : tuple of AnswerItem
        Per-item results, in dataset order.
    judges : mapping
        Judge name to :meth:`~._judge.LLMJudge.identity`.
    agreement : mapping
        Judge metric to Krippendorff's alpha (interval) between the judges,
        when there are at least two; ``NaN`` when undefined.
    usage : mapping
        Judge name to ``responses`` (model responses received),
        ``cached_responses``, ``input_tokens`` and ``output_tokens`` (summed
        over responses that were not cached, so a cached rerun costs 0) and
        ``responses_without_token_counts``.
    errors : mapping
        Failures by kind: ``"system"`` (items without an answer), then one
        count per judge (failed judgements), whether or not verdicts are kept.
    settings : mapping
        The options the evaluation ran with.
    metadata : mapping, default={}
        Free-form, JSON-compatible description supplied by the caller.
    """

    _schema = SCHEMA

    metrics: Mapping[str, float]
    items: tuple
    judges: Mapping[str, Any]
    agreement: Mapping[str, float]
    usage: Mapping[str, Mapping[str, int]]
    errors: Mapping[str, int]
    settings: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        """Number of evaluated items."""
        return len(self.items)

    @property
    def coverage(self) -> dict[str, int]:
        """Metric name to the number of items where it is defined."""
        return {
            name: sum(
                1
                for item in self.items
                if not math.isnan(item.scores.get(name, math.nan))
            )
            for name in self.metrics
        }

    def per_item(self, metric: str) -> np.ndarray:
        """
        Return one metric's per-item values, in dataset order.

        Raises
        ------
        KeyError
            If the evaluation did not compute ``metric``.
        """
        if metric not in self.metrics:
            raise KeyError(f"{metric!r} was not evaluated; have {list(self.metrics)}.")
        return np.array(
            [item.scores.get(metric, math.nan) for item in self.items], dtype=np.float64
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the full report as a mapping (schema versioned)."""
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "n_items": self.n_items,
            "metrics": dict(self.metrics),
            "coverage": self.coverage,
            "agreement": dict(self.agreement),
            "errors": dict(self.errors),
            "usage": {k: dict(v) for k, v in self.usage.items()},
            "judges": dict(self.judges),
            "settings": dict(self.settings),
            "metadata": dict(self.metadata),
            "items": [item.to_dict() for item in self.items],
        }

    def to_rows(self) -> list[dict[str, Any]]:
        """One flat row per item: identifier, answer, each score, latency, error."""
        rows = []
        for item in self.items:
            row: dict[str, Any] = {
                "item_id": item.item_id,
                "question": item.question,
                "answer": item.answer,
            }
            row.update(item.scores)
            row.update(
                n_contexts=item.n_contexts, latency_s=item.latency_s, error=item.error
            )
            rows.append(row)
        return rows

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnswerEvaluation:
        """
        Rebuild a report from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If the schema or its version is not this one.
        """
        if data.get("schema") != SCHEMA:
            raise ValueError(
                f"Not an answer evaluation report: schema {data.get('schema')!r}."
            )
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Report schema version {data.get('schema_version')!r} is not {SCHEMA_VERSION}."
            )
        return cls(
            metrics={k: _nan_or(v) for k, v in data["metrics"].items()},
            items=tuple(AnswerItem.from_dict(item) for item in data["items"]),
            judges=dict(data.get("judges", {})),
            agreement={k: _nan_or(v) for k, v in data.get("agreement", {}).items()},
            usage={k: dict(v) for k, v in data.get("usage", {}).items()},
            errors=dict(data.get("errors", {})),
            settings=dict(data.get("settings", {})),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def read(
        cls,
        path: Any,
        format: str | None = None,
    ) -> AnswerEvaluation:  # noqa: A002
        """Read a report written by :meth:`write` in JSON, YAML or TOML."""
        return cls.from_dict(cls._read_dict(path, format))

    def compare(
        self, baseline: AnswerEvaluation, *, n_resamples: int = 10_000, seed: int = 0
    ) -> EvaluationComparison:
        """
        Compare with ``baseline`` over the same items (paired sign-flip test).

        Parameters
        ----------
        baseline : AnswerEvaluation
            Evaluation of the same item identifiers (another system, or
            another prompt of the same model).
        n_resamples, seed : int
            As in :meth:`RetrievalEvaluation.compare`.

        Returns
        -------
        EvaluationComparison

        Raises
        ------
        TypeError
            If ``baseline`` is not an :class:`AnswerEvaluation`.
        ValueError
            If the item sets differ or no metric is shared.

        Notes
        -----
        Compare evaluations scored by the same judges: a judge identity that
        differs between the two reports makes the difference partly the
        judges'. :meth:`compare` does not refuse it; check ``judges``.
        """
        if not isinstance(baseline, AnswerEvaluation):
            raise TypeError("baseline must be an AnswerEvaluation.")
        return paired_comparison(
            [(item.item_id, item.scores) for item in self.items],
            [(item.item_id, item.scores) for item in baseline.items],
            list(self.metrics),
            list(baseline.metrics),
            n_resamples=n_resamples,
            seed=seed,
            unit="items",
        )


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def _map(func: Callable[[Any], Any], work: list, n_jobs: int) -> list:
    if n_jobs == 1 or len(work) <= 1:
        return [func(w) for w in work]
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(func, work))


def _ask_system(
    system: Any, item: dict, on_error: str, clock: Callable[[], float]
) -> dict:
    """Return answer, contexts, latency and error for one item."""
    if system is None:
        if item["answer"] is None:
            raise ValueError(f"item {item['id']!r} has no 'answer' and system is None.")
        return {
            "answer": item["answer"],
            "contexts": item["contexts"],
            "latency_s": 0.0,
            "error": None,
        }
    start = clock()
    try:
        out = system(item["question"])
        if isinstance(out, Completion):
            out = out.text
        if isinstance(out, str):
            answer, contexts = out, item["contexts"]
        elif isinstance(out, Mapping):
            answer = out.get("answer")
            contexts = (
                item["contexts"]
                if out.get("contexts") is None
                else context_texts(out["contexts"])
            )
        else:
            answer = getattr(out, "answer", None)
            got = getattr(out, "contexts", None)
            contexts = item["contexts"] if got is None else context_texts(got)
        if not isinstance(answer, str):
            raise TypeError(
                "system must return a str, a mapping with 'answer' (and optionally "
                f"'contexts'), or an object with .answer; got {type(out).__name__!r}."
            )
    except Exception as exc:  # noqa: BLE001 - the system is the caller's code
        if on_error == "raise":
            raise
        return {
            "answer": None,
            "contexts": item["contexts"],
            "latency_s": clock() - start,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "answer": answer,
        "contexts": contexts,
        "latency_s": clock() - start,
        "error": None,
    }


def _judge_one(
    judge: LLMJudge, metric: str, item: dict, run: dict, rubric: str | None
) -> JudgeResult:
    """One judge metric for one item, or an undefined result naming the missing input."""
    available = {
        "answer": run["answer"] is not None,
        "contexts": run["contexts"] is not None,
        "reference": item["references"] is not None,
    }
    base = "criteria" if rubric is not None else metric
    missing = [need for need in _NEEDS[base] if not available[need]]
    if missing:
        return JudgeResult(metric, math.nan, undefined=f"no {' or '.join(missing)}")
    refs = item["references"]
    if base in ("context_recall", "answer_correctness") and len(refs) != 1:
        return JudgeResult(
            metric,
            math.nan,
            undefined=f"{len(refs)} references; this judge metric takes one",
        )
    reference = None if refs is None or len(refs) != 1 else refs[0]
    q, a, ctx = item["question"], run["answer"], run["contexts"]
    if base == "faithfulness":
        result = judge.faithfulness(q, a, ctx)
    elif base == "answer_relevance":
        result = judge.answer_relevance(q, a)
    elif base == "context_precision":
        result = judge.context_precision(q, ctx, reference)
    elif base == "context_recall":
        result = judge.context_recall(q, ctx, reference)
    elif base == "answer_correctness":
        result = judge.answer_correctness(q, a, reference)
    else:
        result = judge.criteria_score(q, a, rubric, reference=reference, contexts=ctx)
    return result if result.metric == metric else replace(result, metric=metric)


def _usage(judges: Mapping[str, LLMJudge], judged: list) -> dict[str, dict[str, int]]:
    """Return calls and tokens per judge over every item's results."""
    usage = {}
    for name in judges:
        responses = cached = tokens_in = tokens_out = unreported = 0
        for by_metric in judged:
            for by_judge in by_metric.values():
                result = by_judge.get(name)
                if result is None:
                    continue
                for completion in result.completions:
                    responses += 1
                    if completion.cached:
                        cached += 1
                    elif (
                        completion.input_tokens is None
                        or completion.output_tokens is None
                    ):
                        unreported += 1
                    else:
                        tokens_in += completion.input_tokens
                        tokens_out += completion.output_tokens
        usage[name] = {
            "responses": responses,
            "cached_responses": cached,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "responses_without_token_counts": unreported,
        }
    return usage


def _errors(judges: Mapping[str, LLMJudge], runs: list, judged: list) -> dict[str, int]:
    """Return failures by kind: ``"system"``, then one count per judge."""
    out = {"system": sum(run["error"] is not None for run in runs)}
    for name in judges:
        out[name] = sum(
            1
            for by_metric in judged
            for by_judge in by_metric.values()
            if name in by_judge and by_judge[name].error is not None
        )
    return out


def _answer_item(
    item: dict,
    run: dict,
    by_metric: dict,
    *,
    texts: list[str],
    judge_metrics: list[str],
    judges: list[str],
    keep_verdicts: bool,
) -> AnswerItem:
    """Return one item's scores: text metrics, then each judge and the panel mean per metric."""
    scores: dict[str, float] = {}
    for name in texts:
        if item["references"] is None or run["answer"] is None:
            scores[name] = math.nan
        else:
            scores[name] = float(
                TEXT_METRICS[name]([list(item["references"])], [run["answer"]])
            )
    for metric in judge_metrics:
        per_judge = [by_metric[metric][judge].score for judge in judges]
        for judge, value in zip(judges, per_judge):
            scores[f"{metric}/{judge}"] = value
        scores[metric] = _mean(per_judge)
    return AnswerItem(
        item_id=item["id"],
        question=item["question"],
        answer=run["answer"],
        scores=scores,
        n_contexts=None if run["contexts"] is None else len(run["contexts"]),
        latency_s=float(run["latency_s"]),
        error=run["error"],
        judgments=by_metric if keep_verdicts else {},
    )


def evaluate_answers(
    system: Callable[[str], Any] | None,
    dataset: Sequence[Mapping[str, Any]],
    *,
    judges: Any = None,
    metrics: Sequence[str] = DEFAULT_JUDGE_METRICS,
    criteria: Mapping[str, str] | None = None,
    text_metrics: Sequence[str] = ("exact_match", "token_f1"),
    on_error: str = "record",
    n_jobs: int = 1,
    keep_verdicts: bool = True,
    metadata: Mapping[str, Any] | None = None,
    clock: Callable[[], float] | None = None,
) -> AnswerEvaluation:
    """
    Evaluate a system's answers with text metrics and LLM judges.

    Parameters
    ----------
    system : callable or None
        ``system(question)`` returning the answer (``str``, a
        :class:`~._llm.Completion`, a mapping with ``"answer"`` and optional
        ``"contexts"``, or an object with ``.answer`` and optional
        ``.contexts``). ``None`` evaluates each item's stored ``"answer"``.
    dataset : sequence of mapping
        Items with ``"question"`` and optional ``"id"``, ``"reference"``,
        ``"contexts"``, ``"answer"`` (see the module notes).
    judges : mapping, LLMJudge, LLM, callable or None, default=None
        Judges by name (``{"gpt": make_llm("openai", ...), "claude": ...}``);
        a single judge is named ``"judge"``. ``None`` runs the text metrics
        only.
    metrics : sequence of str, default=("faithfulness", "answer_relevance")
        Judge metrics: any of ``faithfulness``, ``answer_relevance``,
        ``context_precision``, ``context_recall``, ``answer_correctness``.
    criteria : mapping or None, default=None
        Rubrics by name, each reported as ``criteria:<name>``.
    text_metrics : sequence of str, default=("exact_match", "token_f1")
        Deterministic metrics against the references (SQuAD normalisation).
    on_error : {"record", "raise"}, default="record"
        What a failing system call or judge does: record it in the item
        (``NaN`` scores) or raise.
    n_jobs : int, default=1
        Threads for system and judge calls; results stay in dataset order.
    keep_verdicts : bool, default=True
        Keep every :class:`~._judge.JudgeResult` (verdicts and completions)
        in the report.
    metadata : mapping or None, default=None
        Free-form, JSON-compatible description stored in the report.
    clock : callable or None, default=None
        ``clock() -> float`` seconds for system latency.

    Returns
    -------
    AnswerEvaluation

    Raises
    ------
    TypeError, ValueError
        For a malformed dataset, unknown metric names, invalid options, or an
        item without an answer when ``system`` is ``None``.

    See Also
    --------
    LLMJudge, make_llm, evaluate_preference

    Examples
    --------
    >>> data = [{"id": "q1", "question": "Capital of France?", "reference": "Paris"}]
    >>> report = evaluate_answers(lambda q: "Paris", data, judges=None)
    >>> report.metrics
    {'exact_match': 1.0, 'token_f1': 1.0}
    """
    workers, meta = _check_common(on_error, n_jobs, metadata)
    for value, name in ((system, "system"), (clock, "clock")):
        if value is not None and not callable(value):
            raise TypeError(f"{name} must be callable or None.")
    if not isinstance(keep_verdicts, bool):
        raise TypeError("keep_verdicts must be a bool.")
    judge_metrics, rubrics, texts = _check_metrics(metrics, criteria, text_metrics)
    panel = _check_judges(judges, on_error)
    items = _items(dataset)
    timer = time.perf_counter if clock is None else clock

    runs = _map(lambda item: _ask_system(system, item, on_error, timer), items, workers)
    tasks = (
        [(name, None) for name in judge_metrics]
        + [(f"criteria:{key}", text) for key, text in rubrics.items()]
        if panel
        else []
    )
    work = [
        (i, metric, rubric, judge_name)
        for i in range(len(items))
        for metric, rubric in tasks
        for judge_name in panel
    ]
    results = _map(
        lambda w: _judge_one(panel[w[3]], w[1], items[w[0]], runs[w[0]], w[2]),
        work,
        workers,
    )
    judged: list[dict[str, dict[str, JudgeResult]]] = [{} for _ in items]
    for (i, metric, _, judge_name), result in zip(work, results):
        judged[i].setdefault(metric, {})[judge_name] = result

    metric_names = list(texts)
    for metric, _ in tasks:
        metric_names += [f"{metric}/{judge}" for judge in panel] + [metric]
    out_items = [
        _answer_item(
            item,
            run,
            by_metric,
            texts=texts,
            judge_metrics=[metric for metric, _ in tasks],
            judges=list(panel),
            keep_verdicts=keep_verdicts,
        )
        for item, run, by_metric in zip(items, runs, judged)
    ]
    agreement = {}
    if len(panel) >= _MIN_PANEL:
        for metric, _ in tasks:
            matrix = [
                [item.scores[f"{metric}/{judge}"] for item in out_items]
                for judge in panel
            ]
            agreement[metric] = _alpha(matrix, "interval")
    return AnswerEvaluation(
        metrics={
            name: _mean([item.scores[name] for item in out_items])
            for name in metric_names
        },
        items=tuple(out_items),
        judges={name: judge.identity() for name, judge in panel.items()},
        agreement=agreement,
        usage=_usage(panel, judged),
        errors=_errors(panel, runs, judged),
        settings={
            "system": (
                "stored answers"
                if system is None
                else getattr(system, "__qualname__", type(system).__name__)
            ),
            "metrics": judge_metrics,
            "criteria": rubrics,
            "text_metrics": texts,
            "on_error": on_error,
            "n_jobs": workers,
            "keep_verdicts": keep_verdicts,
        },
        metadata=dict(meta),
    )


def _alpha(matrix: list, level: str) -> float:
    """Krippendorff's alpha between judges, ``NaN`` (no warning) when undefined."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return krippendorff_alpha(matrix, level=level)


# --------------------------------------------------------------------------
# Head to head
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreferenceEvaluation(_Report):
    """
    Result of :func:`evaluate_preference`.

    Parameters
    ----------
    names : tuple of str
        The two systems, ``(a, b)``.
    summary : mapping
        Per judge and for the panel: ``wins``, ``ties``, ``losses`` (for
        ``a``), ``win_rate`` (``(wins + ties / 2) / judged``),
        ``position_consistency`` and ``judged``.
    items : tuple of mapping
        Per item: ``item_id``, ``question``, the panel ``score`` for ``a`` and,
        per judge, the :class:`~._judge.JudgeResult` as a mapping.
    judges : mapping
        Judge identities.
    agreement : float
        Krippendorff's alpha (nominal) over the judges' winners; ``NaN`` with
        fewer than two judges or when undefined.
    settings, metadata : mapping
    """

    _schema = PREFERENCE_SCHEMA

    names: tuple
    summary: Mapping[str, Mapping[str, Any]]
    items: tuple
    judges: Mapping[str, Any]
    agreement: float
    settings: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the full report as a mapping (schema versioned)."""
        return {
            "schema": PREFERENCE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "names": list(self.names),
            "summary": {k: dict(v) for k, v in self.summary.items()},
            "agreement": self.agreement,
            "judges": dict(self.judges),
            "settings": dict(self.settings),
            "metadata": dict(self.metadata),
            "items": [dict(item) for item in self.items],
        }

    def to_rows(self) -> list[dict[str, Any]]:
        """One row per item: identifier, panel score for ``a``, each judge's winner."""
        rows = []
        for item in self.items:
            row = {"item_id": item["item_id"], "score": item["score"]}
            for judge, result in item["judges"].items():
                row[f"winner/{judge}"] = result["details"].get("winner")
            rows.append(row)
        return rows

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreferenceEvaluation:
        """Rebuild a report from :meth:`to_dict` output."""
        if (
            data.get("schema") != PREFERENCE_SCHEMA
            or data.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError(
                "Not a preference evaluation report of this schema version."
            )
        items = [
            {
                "item_id": item["item_id"],
                "question": item["question"],
                "score": _nan_or(item.get("score")),
                "judges": {
                    judge: JudgeResult.from_dict(result).to_dict()
                    for judge, result in item["judges"].items()
                },
            }
            for item in data["items"]
        ]
        return cls(
            names=tuple(data["names"]),
            summary={k: dict(v) for k, v in data["summary"].items()},
            items=tuple(items),
            judges=dict(data.get("judges", {})),
            agreement=_nan_or(data.get("agreement")),
            settings=dict(data.get("settings", {})),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def read(
        cls,
        path: Any,
        format: str | None = None,
    ) -> PreferenceEvaluation:  # noqa: A002
        """Read a report written by :meth:`write` in JSON, YAML or TOML."""
        return cls.from_dict(cls._read_dict(path, format))


def _summary(results: list[JudgeResult]) -> dict[str, Any]:
    judged = [r for r in results if r.error is None and not math.isnan(r.score)]
    wins = sum(r.details["winner"] == "a" for r in judged)
    losses = sum(r.details["winner"] == "b" for r in judged)
    ties = len(judged) - wins - losses
    consistent = sum(bool(r.details["position_consistent"]) for r in judged)
    n = len(judged)
    return {
        "judged": n,
        "failed": len(results) - n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": (wins + ties / 2) / n if n else math.nan,
        "position_consistency": consistent / n if n else math.nan,
    }


def evaluate_preference(
    dataset: Sequence[Mapping[str, Any]],
    answers_a: Sequence[str] | Mapping[Any, str],
    answers_b: Sequence[str] | Mapping[Any, str],
    *,
    judges: Any,
    names: tuple[str, str] = ("a", "b"),
    on_error: str = "record",
    n_jobs: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> PreferenceEvaluation:
    """
    Judge two systems' answers head to head, in both orders.

    Parameters
    ----------
    dataset : sequence of mapping
        Items with ``"question"`` and optional ``"id"`` and ``"reference"``,
        shown to the judges when the item has exactly one.
    answers_a, answers_b : sequence of str or mapping
        Each system's answer per item: a sequence in dataset order, or a
        mapping from item id.
    judges : mapping, LLMJudge, LLM or callable
        At least one judge (see :func:`evaluate_answers`).
    names : tuple of str, default=("a", "b")
        Names of the two systems in the report.
    on_error : {"record", "raise"}, default="record"
        What a failing judge does: record it (``NaN`` score) or raise.
    n_jobs : int, default=1
        Threads for judge calls; results stay in dataset order.
    metadata : mapping or None, default=None
        Free-form, JSON-compatible description stored in the report.

    Returns
    -------
    PreferenceEvaluation

    Raises
    ------
    TypeError, ValueError
        For malformed inputs, a missing answer, or no judge.

    Notes
    -----
    Each judge is asked twice per item, once in each order (see
    :meth:`~._judge.LLMJudge.preference`); a judge that changes its mind
    with the order contributes a tie and lowers ``position_consistency``.
    """
    workers, meta = _check_common(on_error, n_jobs, metadata)
    if not (isinstance(names, tuple) and len(names) == _TWO_SYSTEMS):
        raise TypeError("names must be a tuple of two str.")
    for name in names:
        _check_name(name, "System name")
    if names[0] == names[1]:
        raise ValueError("names must differ.")
    panel = _check_judges(judges, on_error)
    if not panel:
        raise ValueError("evaluate_preference needs at least one judge.")
    items = _items(dataset)

    def answers(source: Any, label: str) -> list[str]:
        if isinstance(source, Mapping):
            missing = [item["id"] for item in items if item["id"] not in source]
            if missing:
                raise ValueError(f"{label} has no answer for ids {missing[:5]}.")
            got = [source[item["id"]] for item in items]
        elif isinstance(source, (str, bytes)) or not isinstance(source, Iterable):
            raise TypeError(f"{label} must be a sequence or a mapping of str.")
        else:
            got = list(source)
            if len(got) != len(items):
                raise ValueError(
                    f"{label} has {len(got)} answers for {len(items)} items."
                )
        if not all(isinstance(a, str) for a in got):
            raise TypeError(f"{label} answers must be str.")
        return got

    first, second = answers(answers_a, "answers_a"), answers(answers_b, "answers_b")
    work = [(i, judge) for i in range(len(items)) for judge in panel]

    def run(w: tuple) -> JudgeResult:
        i, judge = w
        refs = items[i]["references"]
        return panel[judge].preference(
            items[i]["question"],
            first[i],
            second[i],
            reference=refs[0] if refs is not None and len(refs) == 1 else None,
        )

    results = _map(run, work, workers)
    by_item: list[dict[str, JudgeResult]] = [{} for _ in items]
    for (i, judge), result in zip(work, results):
        by_item[i][judge] = result
    out_items = []
    for item, by_judge in zip(items, by_item):
        out_items.append(
            {
                "item_id": item["id"],
                "question": item["question"],
                "score": _mean([r.score for r in by_judge.values()]),
                "judges": {judge: r.to_dict() for judge, r in by_judge.items()},
            }
        )
    summary = {
        judge: _summary([by_item[i][judge] for i in range(len(items))])
        for judge in panel
    }
    scores = [item["score"] for item in out_items]
    defined = [s for s in scores if not math.isnan(s)]
    summary["panel"] = {
        "judged": len(defined),
        "failed": len(scores) - len(defined),
        "win_rate": math.fsum(defined) / len(defined) if defined else math.nan,
    }
    agreement = math.nan
    if len(panel) >= _MIN_PANEL:
        codes = {"a": 1, "tie": 2, "b": 3}
        matrix = [
            [
                (
                    math.nan
                    if r.error is not None or math.isnan(r.score)
                    else codes[r.details["winner"]]
                )
                for r in (by_item[i][judge] for i in range(len(items)))
            ]
            for judge in panel
        ]
        agreement = _alpha(matrix, "nominal")
    return PreferenceEvaluation(
        names=tuple(names),
        summary=summary,
        items=tuple(out_items),
        judges={name: judge.identity() for name, judge in panel.items()},
        agreement=agreement,
        settings={"on_error": on_error, "n_jobs": workers},
        metadata=dict(meta),
    )
