# scikitplot/feature_selection/_rename.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Frozen rename plans: mapping, list, callable or template, decided once.

A :class:`RenamePlan` is the *result* of applying a naming rule to a list of
column names. It is computed once (at fit time inside
:class:`~scikitplot.feature_selection.FeatureRenamer`) and then applied
unchanged to every later frame, so a callable rule is never re-run on test
data and cannot rename it differently.

Rules
-----
mapping
    ``{"old": "new"}``; unlisted columns keep their names. Keys that are not
    columns are refused unless ``on_missing="ignore"``.
list
    One new name per column, in column order.
callable
    ``rule(name) -> str``, called once per column.
template
    A :meth:`str.format` string with the fields ``{name}`` and ``{index}``
    (``index = start + position``): ``"f{index}"``, ``"col-{index:03d}"``,
    ``"raw__{name}"``. Any other field is refused.

Notes
-----
**User note.** Duplicate results are refused by default
(``on_collision="raise"``). ``on_collision="suffix"`` keeps the first
occurrence and appends ``_1``, ``_2`` ... to later ones, skipping any suffix
already taken, so the plan is deterministic.

**Developer note.** The plan stores complete ``(old, new)`` pairs for every
input column, so it is invertible (:meth:`RenamePlan.inverse`) and
serialisable without the rule that produced it.
"""

from __future__ import annotations

import dataclasses
import string
from typing import Any, Mapping, Sequence

import pandas as pd

__all__ = ["RenamePlan"]

_TEMPLATE_FIELDS = frozenset({"name", "index"})


def _template_fields(template: str) -> set[str]:
    fields = set()
    for _, field, _, _ in string.Formatter().parse(template):
        if field is None:
            continue
        root = field.split(".", 1)[0].split("[", 1)[0]
        if root != field:
            raise ValueError(
                f"template field {{{field}}} uses attribute or index access; only"
                " {name} and {index} (with a format spec) are allowed"
            )
        fields.add(root)
    return fields


@dataclasses.dataclass(frozen=True)
class RenamePlan:
    """
    A complete, frozen ``old -> new`` name mapping.

    Parameters
    ----------
    pairs : tuple of (str, str)
        One pair per input column, in column order.

    See Also
    --------
    FeatureRenamer : Applies a plan inside a pipeline.
    FeatureSchema.rename : Renames a fitted schema, keeping aliases.

    Examples
    --------
    >>> from scikitplot.feature_selection import RenamePlan
    >>> RenamePlan.build(["a", "b", "c"], "col-{index}", start=1).names_out
    ['col-1', 'col-2', 'col-3']
    >>> RenamePlan.build(["a", "b"], {"a": "alpha"}).names_out
    ['alpha', 'b']
    >>> RenamePlan.build(["x", "y"], str.upper).mapping
    {'x': 'X', 'y': 'Y'}
    """

    pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:  # noqa: D105
        pairs = tuple((str(a), str(b)) for a, b in self.pairs)
        object.__setattr__(self, "pairs", pairs)
        olds = [a for a, _ in pairs]
        news = [b for _, b in pairs]
        if len(set(olds)) != len(olds):
            raise ValueError("a rename plan cannot have duplicate input names")
        if len(set(news)) != len(news):
            dup = sorted({n for n in news if news.count(n) > 1})
            raise ValueError(f"a rename plan cannot produce duplicate names {dup}")

    @classmethod
    def build(
        cls,
        names: Sequence[str],
        rule: Any,
        *,
        start: int = 0,
        on_collision: str = "raise",
        on_missing: str = "raise",
    ) -> RenamePlan:
        """
        Apply ``rule`` to ``names`` once and freeze the result.

        Parameters
        ----------
        names : sequence of str
            Input column names (unique).
        rule : mapping, sequence of str, callable, str or None
            See the module documentation. None keeps every name.
        start : int, default=0
            First ``{index}`` of a template.
        on_collision : {"raise", "suffix"}, default="raise"
            Handling of duplicate new names.
        on_missing : {"raise", "ignore"}, default="raise"
            Handling of mapping keys that are not in ``names``.

        Returns
        -------
        RenamePlan
            The plan.

        Raises
        ------
        ValueError
            For a list of the wrong length, an unknown template field, an
            unknown mapping key, an empty new name or a refused collision.
        TypeError
            For a rule of an unsupported type or a callable returning a
            non-string.
        """
        names = [str(n) for n in names]
        if len(set(names)) != len(names):
            raise ValueError("input names must be unique")
        if on_collision not in ("raise", "suffix"):
            raise ValueError(
                f"on_collision={on_collision!r} is not 'raise' or 'suffix'"
            )
        if on_missing not in ("raise", "ignore"):
            raise ValueError(f"on_missing={on_missing!r} is not 'raise' or 'ignore'")
        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError(f"start must be an int, got {start!r}")
        new = cls._apply_rule(names, rule, start, on_missing)
        for old, value in zip(names, new):
            if not isinstance(value, str):
                raise TypeError(
                    f"the new name for {old!r} is {type(value).__name__}, not str"
                )
            if not value:
                raise ValueError(f"the new name for {old!r} is empty")
        if on_collision == "suffix":
            new = cls._suffix(new)
        else:
            seen: dict[str, str] = {}
            for old, value in zip(names, new):
                if value in seen:
                    raise ValueError(
                        f"{seen[value]!r} and {old!r} would both be named {value!r};"
                        " change the rule or pass on_collision='suffix'"
                    )
                seen[value] = old
        return cls(tuple(zip(names, new)))

    @staticmethod
    def _apply_rule(
        names: list[str], rule: Any, start: int, on_missing: str
    ) -> list[Any]:
        if rule is None:
            return list(names)
        if isinstance(rule, str):
            fields = _template_fields(rule)
            unknown = fields - _TEMPLATE_FIELDS
            if unknown:
                raise ValueError(
                    f"template fields {sorted(unknown)} are not allowed; use {{name}}"
                    " and {index}"
                )
            if not fields and len(names) > 1:
                raise ValueError(
                    f"template {rule!r} has no field, so every column would get the"
                    " same name"
                )
            return [rule.format(name=n, index=start + i) for i, n in enumerate(names)]
        if isinstance(rule, Mapping):
            missing = [k for k in rule if k not in set(names)]
            if missing and on_missing == "raise":
                raise ValueError(
                    f"rename mapping names column(s) {missing} that are not present;"
                    " pass on_missing='ignore' to allow it"
                )
            return [rule.get(n, n) for n in names]
        if callable(rule):
            return [rule(n) for n in names]
        if isinstance(rule, (list, tuple, pd.Index)):
            values = list(rule)
            if len(values) != len(names):
                raise ValueError(
                    f"a list rule needs one name per column: got {len(values)} names"
                    f" for {len(names)} columns"
                )
            return values
        raise TypeError(
            "rule must be a mapping, a list of names, a callable, a template string"
            f" or None; got {type(rule).__name__}"
        )

    @staticmethod
    def _suffix(values: list[str]) -> list[str]:
        taken = set(values)
        used: set[str] = set()
        out = []
        for value in values:
            if value not in used:
                used.add(value)
                out.append(value)
                continue
            k = 1
            while f"{value}_{k}" in taken or f"{value}_{k}" in used:
                k += 1
            candidate = f"{value}_{k}"
            used.add(candidate)
            out.append(candidate)
        return out

    @property
    def names_in(self) -> list[str]:
        """List of str: Input names, in order."""
        return [a for a, _ in self.pairs]

    @property
    def names_out(self) -> list[str]:
        """List of str: Output names, in order."""
        return [b for _, b in self.pairs]

    @property
    def mapping(self) -> dict[str, str]:
        """dict: Old name to new name, every column included."""
        return dict(self.pairs)

    @property
    def changed(self) -> dict[str, str]:
        """dict: Only the columns whose name changes."""
        return {a: b for a, b in self.pairs if a != b}

    def inverse(self) -> RenamePlan:
        """
        Return the plan that undoes this one.

        Returns
        -------
        RenamePlan
            ``new -> old`` for every column.
        """
        return RenamePlan(tuple((b, a) for a, b in self.pairs))

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Rename the columns of ``X``, which must equal :attr:`names_in` in order.

        Parameters
        ----------
        X : pandas.DataFrame
            Frame to rename.

        Returns
        -------
        pandas.DataFrame
            A new frame with renamed columns (data not copied where pandas
            avoids it).

        Raises
        ------
        ValueError
            If the columns differ from the plan's inputs.
        """
        if list(X.columns) != self.names_in:
            raise ValueError(
                "the frame's columns differ from the rename plan's inputs; a plan"
                " applies to exactly the columns it was built for"
            )
        out = X.copy(deep=False)
        out.columns = self.names_out
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"pairs": [[old, new], ...]}``."""
        return {"pairs": [list(p) for p in self.pairs]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RenamePlan:
        """
        Rebuild a plan from :meth:`to_dict` output.

        Parameters
        ----------
        data : mapping
            Stored form.

        Returns
        -------
        RenamePlan
            The plan.
        """
        if set(data) != {"pairs"} or not isinstance(data["pairs"], list):
            raise ValueError("a rename plan record is {'pairs': [[old, new], ...]}")
        pairs = []
        len_item = 2
        for pair in data["pairs"]:
            if (
                not isinstance(pair, list)
                or len(pair) != len_item
                or not all(isinstance(p, str) for p in pair)
            ):
                raise ValueError("each rename pair must be [old, new] strings")
            pairs.append(tuple(pair))
        return cls(tuple(pairs))
