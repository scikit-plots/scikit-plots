"""
Role-preserving stand-ins for a dataset's schema.

A data-science artefact discloses through its *identifiers* as much as through
its values. ``customer_ssn`` holds no social-security number; it discloses that
the dataset does. ``internal_risk_score_v3`` is a proprietary feature set named
out loud.

Notes
-----
**User notes.** You rarely call this module directly. It is what turns

.. code-block:: python

    df[["customer_ssn", "acct_balance_usd", "signup_date", "region_code"]]

into

.. code-block:: python

    df[["id_1", "amount_1", "date_1", "category_1"]]

so that a model can still give correct advice — log-transform the amount,
one-hot the category, drop the identifier — while learning nothing about your
data.

**Developer notes — why a bracket placeholder is the wrong stand-in here.**

Everywhere else in this submodule, the stand-in is deliberately unmistakable:
``[EMAIL-1]`` cannot be confused with a real address. That property is worth
its cost in prose, and is the wrong trade in source code, for two reasons.

The first is mechanical: ``df[['[COLUMN-1]']]`` is noise, and a model asked to
edit it returns something that no longer parses.

The second matters more. The reason to send a notebook to a model is to get
help with the code, and that help depends on the schema's semantics — that this
column is numeric and skewed, that one is categorical, that one is a date and
therefore a leakage risk. Replace all of them with ``[COLUMN-n]`` and the
information the advice depends on is exactly the information that was removed.
The redaction succeeds and the task fails.

So a stand-in here keeps the **role** and discards the **name**. It is a valid
Python identifier, it is stable across every cell of a notebook, and it carries
forward the one property the model needs.

**Developer notes — how a role is decided, and what happens when it cannot be.**

Rule 4 of this project forbids heuristics in logic, and a name is not evidence
of a type: ``count`` could be anything and ``date_added`` could be a string. So
:func:`role_for` is never called with a guess. Roles arrive from one of three
places, and the third is an admission rather than an answer:

1. *declared* — the caller says so, and is never second-guessed;
2. *observed* — the artefact contains evidence, such as a dtype in a
   ``df.info()`` output or a ``parse_dates=`` argument;
3. *neutral* — :data:`NEUTRAL_ROLE`, reported as unestablished.

Name-based classification lives in :func:`infer_role`, is **off by default**,
and is labelled inferred wherever it is used. That split is what keeps the
module honest: hiding a column is deductive and unconditional, while choosing
*which* stand-in it gets is evidential. An unestablished role costs the model
some context; it never costs the user a leak.

See Also
--------
scikitplot.cleanprompt._surrogates : The same idea for people and places.
scikitplot.cleanprompt._code : Finds the identifiers this module renames.
"""

from __future__ import annotations

import keyword
import re
from typing import Iterable

from ._exceptions import PolicyError

__all__ = [
    "NEUTRAL_ROLE",
    "ROLES",
    "SCHEMA_KINDS",
    "infer_role",
    "is_safe_to_rename",
    "path_surrogate",
    "role_for",
    "schema_surrogate",
]

#: Semantic roles a column can carry into a stand-in.
#:
#: Each maps to the stem of the identifier a column of that role is renamed to.
#: The set is deliberately small: it exists to preserve the distinctions that
#: change a model's advice, not to be a type system.
ROLES: dict[str, str] = {
    "id": "id",  # an identifier; never a feature
    "amount": "amount",  # money or a measured quantity
    "count": "count",  # a non-negative integer tally
    "score": "score",  # a bounded model or risk output
    "rate": "rate",  # a ratio or percentage
    "date": "date",  # a calendar date
    "datetime": "timestamp",  # a date with a time
    "category": "category",  # a low-cardinality label
    "text": "text",  # free text
    "flag": "flag",  # a boolean
    "target": "target",  # the thing being predicted
    "field": "field",  # role not established
}

#: The role assigned when nothing established one.
NEUTRAL_ROLE = "field"

#: Kinds this module issues, in the vocabulary the vault records.
SCHEMA_KINDS = ("COLUMN", "DATASET", "TABLE", "LEVEL", "PATH")

#: Python identifiers a column name may not be rewritten *from*, because
#: rewriting every occurrence would change code that has nothing to do with the
#: dataset. A single-letter name is excluded for the same reason.
_UNSAFE_NAMES = frozenset(
    set(keyword.kwlist)
    | {
        "df",
        "data",
        "index",
        "values",
        "columns",
        "name",
        "type",
        "id",
        "list",
        "dict",
        "set",
        "sum",
        "min",
        "max",
        "len",
        "str",
        "int",
        "float",
        "bool",
        "object",
        "all",
        "any",
        "filter",
        "map",
        "next",
        "range",
        "round",
        "sorted",
        "format",
        "input",
        "open",
        "print",
        "self",
    }
    # Attributes a pandas object already has. A column renamed to one of these,
    # or rewritten *from* one, would collide with a method call: rewriting
    # `count` would change every `df.count()` in the notebook. pandas is not a
    # dependency of this tier, so the list is curated here rather than read
    # from `dir(pd.DataFrame)`; it covers the names that actually occur as
    # column names, which is a much smaller set than the full surface.
    | {
        "apply",
        "axes",
        "columns",
        "copy",
        "count",
        "describe",
        "drop",
        "dtypes",
        "empty",
        "fillna",
        "groupby",
        "head",
        "iloc",
        "index",
        "info",
        "items",
        "join",
        "keys",
        "loc",
        "mask",
        "max",
        "mean",
        "median",
        "merge",
        "min",
        "mode",
        "ndim",
        "pivot",
        "pop",
        "query",
        "rank",
        "rename",
        "replace",
        "reset",
        "sample",
        "shape",
        "shift",
        "size",
        "sort",
        "std",
        "sum",
        "tail",
        "take",
        "value",
        "values",
        "var",
        "where",
    }
)

#: A name that can be rewritten safely: a Python identifier of at least two
#: characters that is not reserved.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Where a stand-in path is rooted. Absolute, short, and obviously not a real
#: tree, so a reader is never misled into thinking it is the original.
_STANDIN_ROOT = "/data"

#: Name fragments that *suggest* a role. Consulted only by :func:`infer_role`,
#: which is opt-in and whose output is always reported as inferred.
#:
#: Order matters: the first match wins, so the more specific fragments come
#: first. ``_id`` before ``id`` prevents ``valid`` from matching.
_ROLE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("_ssn_", "_nino_", "_tckn_", "_uuid_", "_guid_", "_id_", "_key_", "_ref_"), "id"),
    (("_at_", "_timestamp_", "_ts_", "_datetime_"), "datetime"),
    (("_date_", "_day_", "_dob_", "_birthday_"), "date"),
    (
        (
            "_count_",
            "_counts_",
            "_n_",
            "_num_",
            "_qty_",
            "_quantity_",
            "_tenure_",
            "_months_",
            "_days_",
            "_years_",
            "_age_",
        ),
        "count",
    ),
    (
        (
            "_amount_",
            "_balance_",
            "_price_",
            "_cost_",
            "_revenue_",
            "_salary_",
            "_usd_",
            "_eur_",
            "_gbp_",
            "_total_",
            "_sum_",
            "_log_",
        ),
        "amount",
    ),
    (("_score_", "_prob_", "_probability_", "_likelihood_", "_risk_"), "score"),
    (("_rate_", "_ratio_", "_pct_", "_percent_", "_share_"), "rate"),
    (("_is_", "_has_", "_flag_", "_churned_", "_bool_", "_active_"), "flag"),
    (
        (
            "_code_",
            "_region_",
            "_segment_",
            "_category_",
            "_type_",
            "_status_",
            "_tier_",
            "_group_",
            "_class_",
            "_label_",
            "_channel_",
        ),
        "category",
    ),
    (
        (
            "_comment_",
            "_note_",
            "_notes_",
            "_description_",
            "_text_",
            "_message_",
            "_body_",
        ),
        "text",
    ),
)


def is_safe_to_rename(name: str) -> tuple[bool, str]:
    """
    Report whether a column name can be rewritten without changing other code.

    Parameters
    ----------
    name : str
        The column name as it appears in the artefact.

    Returns
    -------
    tuple of (bool, str)
        Whether the name may be rewritten, and — when it may not — the reason,
        phrased for a person reading a report.

    Notes
    -----
    **User notes.** A refusal is not a failure of detection. The column *is*
    recognised; it is the rewrite that is unsafe, and the report names it so
    you can rename the column in your own code or hide it explicitly.

    **Developer notes.** This is the ``CP-001`` problem in a new setting.
    Rewriting every occurrence of a bare ``id``, ``type`` or ``x`` in a
    notebook would change list comprehensions, keyword arguments and unrelated
    variables, producing a file that no longer runs — a worse outcome than a
    named refusal, and a much harder one to diagnose, because the damage is
    silent and spread over the whole document.

    Examples
    --------
    >>> is_safe_to_rename("acct_balance_usd")
    (True, '')
    >>> is_safe_to_rename("type")[0]
    False
    >>> is_safe_to_rename("x")[0]
    False
    """
    if not isinstance(name, str) or not name:
        return False, "the column name is empty"
    if not _IDENTIFIER.match(name):
        return (
            False,
            (
                f"{name!r} is not a plain identifier, so it can only appear as a "
                "quoted string; it is hidden there but attribute access to it "
                "cannot be rewritten"
            ),
        )
    if len(name) < 2:  # ruff: ignore[magic-value-comparison]
        return (
            False,
            (
                f"{name!r} is a single character and occurs in unrelated code; "
                "rename the column or hide it explicitly with --hide"
            ),
        )
    if name in _UNSAFE_NAMES:
        return (
            False,
            (
                f"{name!r} is a Python keyword or a common builtin; rewriting every "
                "occurrence would change code unrelated to the dataset"
            ),
        )
    return True, ""


def role_for(
    name: str,
    declared: dict[str, str] | None = None,
    observed: dict[str, str] | None = None,
) -> tuple[str, str]:
    """
    Return the role of a column and how it was established.

    Parameters
    ----------
    name : str
        The column name.
    declared : dict, optional
        Roles the caller stated, keyed by column name. Never second-guessed.
    observed : dict, optional
        Roles read out of the artefact itself — a dtype in an ``info()``
        output, a ``parse_dates=`` argument, a value in a rendered table.

    Returns
    -------
    tuple of (str, str)
        The role, and its provenance: ``'declared'``, ``'observed'`` or
        ``'none'``.

    Raises
    ------
    PolicyError
        If a declared or observed role is not one of :data:`ROLES`.

    Notes
    -----
    **Developer notes.** The provenance is returned rather than discarded
    because the report has to distinguish "this is a date" from "nobody
    established what this is". Collapsing the two would turn an admission into
    a claim, which is the failure ``CP-023`` exists to prevent.

    Examples
    --------
    >>> role_for("signup_date", declared={"signup_date": "date"})
    ('date', 'declared')
    >>> role_for("mystery_column")
    ('field', 'none')
    """
    for source, mapping in (("declared", declared), ("observed", observed)):
        if not mapping:
            continue
        role = mapping.get(name)
        if role is None:
            continue
        if role not in ROLES:
            msg = "{} role {!r} for column {!r} is unknown; choose from {}".format(
                source,
                role,
                name,
                ", ".join(sorted(ROLES)),
            )
            raise PolicyError(msg)
        return role, source
    return NEUTRAL_ROLE, "none"


def infer_role(name: str) -> str | None:
    """
    Suggest a role from the column's name, or ``None`` when nothing matches.

    Parameters
    ----------
    name : str
        The column name.

    Returns
    -------
    str or None
        A key of :data:`ROLES`, or ``None``.

    Notes
    -----
    **User notes.** This is a *suggestion*, the same kind
    :func:`~scikitplot.cleanprompt.suggest_terms` makes: something to confirm,
    not something acted upon. It is off unless you ask for it, and wherever its
    output is used the report says the role was inferred.

    **Developer notes.** A name is not evidence of a type, so this function is
    firewalled away from :func:`role_for` rather than folded into it as a final
    fallback. Folding it in would make every report say "date" where it means
    "the word *date* appears in the name", and there would be no way left to
    tell the two apart.

    Examples
    --------
    >>> infer_role("acct_balance_usd")
    'amount'
    >>> infer_role("signup_date")
    'date'
    >>> infer_role("customer_ssn")
    'id'
    >>> infer_role("wibble") is None
    True
    """
    # Underscore-delimited so a fragment matches a whole token: '_date_'
    # finds `signup_date` and not `date_of_birth_string`, and '_months_' does
    # not make `tenure_months` look like a date.
    padded = "_{}_".format(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower())
    for fragments, role in _ROLE_HINTS:
        if any(fragment in padded for fragment in fragments):
            return role
    return None


def schema_surrogate(
    role: str,
    ordinal: int,
    avoid: Iterable[str] | None = None,
) -> str:
    """
    Return the identifier a column of this role is renamed to.

    Parameters
    ----------
    role : str
        A key of :data:`ROLES`.
    ordinal : int
        1-based index within the role. Two columns of the same role must never
        share an ordinal.
    avoid : iterable of str, optional
        Identifiers already in use — other stand-ins, and every name the
        artefact itself uses. A collision is stepped past rather than accepted.

    Returns
    -------
    str
        A valid Python identifier.

    Raises
    ------
    PolicyError
        If ``role`` is unknown or ``ordinal`` is below 1.

    Notes
    -----
    **Developer notes.** The collision check takes the *document's* identifiers
    as well as the issued ones, because a stand-in that happens to equal a
    variable already in the notebook would be rewritten on the way back and
    corrupt code the user wrote. That is the same reasoning as the source-text
    check in :mod:`~scikitplot.cleanprompt._surrogates`, and the same failure
    it prevents.

    Examples
    --------
    >>> schema_surrogate("amount", 1)
    'amount_1'
    >>> schema_surrogate("target", 1)
    'target'
    >>> schema_surrogate("amount", 1, avoid={"amount_1"})
    'amount_2'
    """
    if role not in ROLES:
        msg = "unknown column role {!r}; choose from {}".format(
            role,
            ", ".join(sorted(ROLES)),
        )
        raise PolicyError(msg)
    if ordinal < 1:
        raise PolicyError(f"ordinal must be at least 1, got {ordinal!r}")

    taken = {str(one) for one in avoid or ()}
    stem = ROLES[role]
    # There is normally one target, so it reads better without a suffix; a
    # second one falls back to the numbered form rather than colliding.
    candidates = ([stem] if role == "target" else []) + [
        f"{stem}_{ordinal + step}" for step in range(1000)
    ]
    for candidate in candidates:
        if candidate not in taken:
            return candidate
    raise PolicyError(
        f"could not find a free stand-in for role {role!r}: the artefact already "
        "uses every candidate"
    )


def path_surrogate(
    original: str,
    ordinal: int,
    avoid: Iterable[str] | None = None,
) -> str:
    """
    Return a stand-in path that keeps the shape and discards everything else.

    Parameters
    ----------
    original : str
        The path or URI as it appears in the artefact.
    ordinal : int
        1-based index among the paths in this document.
    avoid : iterable of str, optional
        Stand-ins already issued.

    Returns
    -------
    str
        A path of the same kind: same URI scheme if there was one, same
        absolute-or-relative shape, same file extension.

    Raises
    ------
    PolicyError
        If ``ordinal`` is below 1.

    Notes
    -----
    **User notes.** The extension is kept on purpose. ``read_parquet`` and
    ``read_csv`` take different advice, and a model that cannot see which one
    you are using will guess.

    **Developer notes.** Everything else goes, and the parts that go are the
    parts that identify: the organisation in a bucket name, the environment in
    ``/mnt/prod``, the quarter in a filename, and the person in a home
    directory. ``/home/marion.holt/work/acme-churn/`` names an individual and a
    client project in nine tokens, none of which is a value any pattern in this
    submodule would match.

    Examples
    --------
    >>> path_surrogate("/mnt/prod/exports/2026_q1_acme_pii.parquet", 1)
    '/data/dataset_1.parquet'
    >>> path_surrogate("s3://acme-internal/models/churn.pkl", 2)
    's3://bucket/dataset_2.pkl'
    >>> path_surrogate("reports/fig_drivers.png", 3)
    'dataset_3.png'
    >>> path_surrogate("/home/marion/work/features.py:18", 4)
    '/data/dataset_4.py'
    """
    if ordinal < 1:
        raise PolicyError(f"ordinal must be at least 1, got {ordinal!r}")

    taken = {str(one) for one in avoid or ()}
    scheme = ""
    remainder = original
    if "://" in original:
        scheme, _, remainder = original.partition("://")
        scheme = scheme + "://"

    absolute = remainder.startswith("/")
    suffix = ""
    tail = remainder.rsplit("/", 1)[-1]
    # A traceback writes `features.py:18` and `features.py:18:4`. The line
    # reference is not part of the name, and dropping it is what keeps the
    # extension — which is the one part worth preserving.
    while ":" in tail and tail.rsplit(":", 1)[-1].isdigit():
        tail = tail.rsplit(":", 1)[0]
    if "." in tail:
        candidate_suffix = tail[tail.rindex(".") :]
        # A suffix is a file extension only when it is short and alphanumeric;
        # a dotted directory name is not one.
        _len = 2 <= len(candidate_suffix) <= 12  # ruff: ignore[magic-value-comparison]
        if _len and candidate_suffix[1:].isalnum():
            suffix = candidate_suffix.lower()

    for step in range(1000):
        stem = f"dataset_{ordinal + step}"
        if scheme:
            candidate = f"{scheme}bucket/{stem}{suffix}"
        elif absolute:
            candidate = f"{_STANDIN_ROOT}/{stem}{suffix}"
        else:
            candidate = f"{stem}{suffix}"
        if candidate not in taken:
            return candidate
    raise PolicyError(f"could not find a free stand-in path for {original!r}")
