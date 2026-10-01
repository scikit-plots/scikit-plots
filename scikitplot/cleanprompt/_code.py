"""
Find the schema inside Python source, by syntax rather than by vocabulary.

Notes
-----
**User notes.** This is what lets ``cleanprompt`` hide the *names* of your
columns as well as the values in them. It reads the code the way the
interpreter does, so it finds ``df['customer_ssn']`` and
``pd.read_csv(..., usecols=['acct_balance_usd'])`` without being told what your
columns are called, and without ever guessing from a word list.

**Developer notes — why an AST and not a regular expression.**

A column name is an ordinary string. There is nothing about ``'region_code'``
that distinguishes it from a log message, a dictionary key or an English word,
so no pattern over the text can find columns without also finding everything
else. What *is* distinctive is the **position**: a string inside
``df[...]``, or bound to ``usecols=``, occupies a place where a column name is
the only sensible reading.

That makes discovery a question about syntax, and syntax is decidable. The
walk below visits exactly the positions where the language guarantees the
meaning, and it stops there. Every site in :data:`COLUMN_KEYWORDS` and
:data:`COLUMN_METHODS` is one a reader could defend line by line.

**Developer notes — why attribute access is not a discovery site.**

``df.region_code`` is how half of all pandas code is written, and it is
deliberately **not** used to discover a column, because it cannot be told from
a method call without knowing the runtime type of ``df``. Treating it as a
discovery site would make ``df.shape``, ``df.copy`` and ``model.coef_`` into
columns.

The asymmetry is the point: discovery is conservative and works only from
unambiguous positions, while *rewriting* — done elsewhere, once the name is
known — covers every occurrence including attribute access. Being sure a name
is a column is a separate question from finding where that name occurs, and
collapsing them would either miss columns or corrupt code.

**Developer notes — malformed input is reported, not absorbed.**

A notebook cell is frequently not valid Python: ``%matplotlib inline``,
``!pip install``, and a cell that was never finished all raise
:class:`SyntaxError`. Those are ordinary, so :func:`discover` returns what it
found and *names* the cells it could not parse, rather than raising or — worse
— silently returning an empty set, which would read as "this cell has no
columns in it".

See Also
--------
scikitplot.cleanprompt._schema : Decides what a discovered column is renamed to.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable

__all__ = [
    "COLUMN_KEYWORDS",
    "COLUMN_METHODS",
    "Discovery",
    "discover",
]

#: Keyword arguments whose string values name columns. Every entry is a
#: documented pandas/scikit-learn parameter that takes a column label or a
#: sequence of them.
COLUMN_KEYWORDS = frozenset(
    {
        "by",
        "columns",
        "id_vars",
        "index_col",
        "left_on",
        "on",
        "parse_dates",
        "right_on",
        "subset",
        "usecols",
        "value_vars",
        "x",
        "y",
        "hue",
        "target",
        "features",
        "feature_names",
    }
)

#: Methods whose **first positional argument** names a column or columns.
COLUMN_METHODS = frozenset(
    {
        "groupby",
        "sort_values",
        "set_index",
        "drop_duplicates",
        "nlargest",
        "nsmallest",
        "explode",
        "value_counts",
    }
)

#: Keyword arguments that establish a column's role as evidence rather than as
#: a guess: the artefact itself says these columns are dates.
_DATE_KEYWORDS = frozenset({"parse_dates", "infer_datetime_format"})

#: pandas dtype spellings mapped to the roles in
#: :mod:`~scikitplot.cleanprompt._schema`. Used when the artefact states a
#: dtype, for example in ``astype({...})`` or ``dtype={...}``.
_DTYPE_ROLES = {
    "int": "count",
    "int8": "count",
    "int16": "count",
    "int32": "count",
    "int64": "count",
    "uint8": "count",
    "float": "amount",
    "float16": "amount",
    "float32": "amount",
    "float64": "amount",
    "bool": "flag",
    "boolean": "flag",
    "category": "category",
    "object": "text",
    "string": "text",
    "datetime64[ns]": "datetime",
    "datetime64": "datetime",
}


@dataclass
class Discovery:
    """
    What a walk over one artefact's code established.

    Parameters
    ----------
    columns : dict
        Column name to the sorted set of sites it was seen in, for example
        ``{'region_code': ('subscript', 'keyword:usecols')}``. The sites are
        carried so a report can say *why* something is believed to be a column.
    observed_roles : dict
        Column name to a role the artefact itself stated. Evidence, never a
        guess; see :func:`~scikitplot.cleanprompt._schema.role_for`.
    identifiers : set of str
        Every name bound or referenced anywhere in the source. Used to keep a
        generated stand-in from colliding with something the user already
        wrote.
    unparsed : tuple of str
        A short description of each fragment that could not be parsed, in the
        order encountered.
    bindings : dict
        Names bound to a literal list of strings, for example
        ``ID_COLUMNS = ["a", "b"]``. Kept so that a later
        ``df.drop(columns=ID_COLUMNS)`` — possibly in another cell — can be
        resolved.
    deferred : set of str
        Names used *as* a column selector whose value is not yet known.
        Resolved against ``bindings`` by :func:`discover`.

    Notes
    -----
    **Developer notes.** ``unparsed`` is a field rather than a log line because
    a caller has to be able to *report* it. A cell that did not parse is a cell
    whose columns were not discovered, and the difference between "no columns
    here" and "this was not read" is exactly the distinction ``CP-023``
    exists to preserve.
    """

    columns: dict[str, tuple[str, ...]] = field(default_factory=dict)
    observed_roles: dict[str, str] = field(default_factory=dict)
    identifiers: set[str] = field(default_factory=set)
    unparsed: tuple[str, ...] = ()
    bindings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    deferred: set[str] = field(default_factory=set)

    def merge(self, other: Discovery) -> Discovery:
        """
        Return the union of this discovery and another.

        Parameters
        ----------
        other : Discovery
            The discovery to fold in.

        Returns
        -------
        Discovery
            A new object; neither input is modified.

        Notes
        -----
        **Developer notes.** A notebook is many fragments and one schema, so
        merging is the normal case rather than an extra. Sites accumulate
        because a column seen in two different positions is better evidence
        than one seen once, and a report should be able to say so.
        """
        columns: dict[str, tuple[str, ...]] = {}
        for source in (self.columns, other.columns):
            for name, sites in source.items():
                columns[name] = tuple(sorted(set(columns.get(name, ())) | set(sites)))
        roles = dict(self.observed_roles)
        roles.update(other.observed_roles)
        bindings = dict(self.bindings)
        bindings.update(other.bindings)
        return Discovery(
            columns=columns,
            observed_roles=roles,
            identifiers=set(self.identifiers) | set(other.identifiers),
            unparsed=tuple(self.unparsed) + tuple(other.unparsed),
            bindings=bindings,
            deferred=set(self.deferred) | set(other.deferred),
        )


def _string_constants(node: ast.AST | None) -> list[str]:
    """
    Return the string constants directly inside a node.

    Notes
    -----
    **Developer notes.** Directly is meant literally: a bare constant, or the
    elements of a list, tuple or set. It does not recurse into arbitrary
    expressions, because ``df[some_call("x")]`` gives no guarantee that ``"x"``
    is a column, and a discovery site that is only usually right is not one.
    """
    if node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
    return []


def _subscript_value(node: ast.Subscript) -> ast.AST | None:
    """
    Return the expression inside ``x[...]``, across Python versions.

    Notes
    -----
    **Developer notes.** Python 3.8 wraps a simple subscript in
    :class:`ast.Index`; 3.9 removed the wrapper. This submodule targets 3.8, so
    both shapes are handled here rather than at each call site.
    """
    inner = node.slice
    wrapper = getattr(ast, "Index", None)
    if wrapper is not None and isinstance(inner, wrapper):  # pragma: no cover
        return getattr(inner, "value", None)
    extended = getattr(ast, "ExtSlice", None)
    if extended is not None and isinstance(inner, extended):  # pragma: no cover
        # Python 3.8 spells `df.loc[:, "col"]` as ExtSlice([Slice, Index]);
        # 3.9+ spells it as a Tuple. Rebuild the Tuple so one code path reads
        # both — measured on 3.8.20, where the column was otherwise missed.
        parts = [
            getattr(dim, "value", dim) if isinstance(dim, wrapper) else dim
            for dim in inner.dims
        ]
        return ast.Tuple(elts=parts, ctx=ast.Load())
    if isinstance(inner, ast.Slice):
        # `df.loc[:, "col"]` arrives as a Tuple containing the slice; a bare
        # slice on its own carries no column name.
        return None
    return inner


def _tuple_lists(node: ast.AST | None) -> list[str]:
    """
    Return strings from string lists that sit directly inside tuples.

    Notes
    -----
    **Developer notes.** ``ColumnTransformer(transformers=[("num", scaler,
    ["age", "income"])])`` names its columns one level down, inside a tuple.
    Only a *list of strings* in that position is taken: the bare ``"num"``
    beside it is a step name, and a step name is a string, not a list, so the
    rule cannot mistake one for a column.
    """
    found: list[str] = []
    tuples = []
    if isinstance(node, ast.Tuple):
        tuples = [node]
    elif isinstance(node, (ast.List, ast.Tuple)):
        tuples = [item for item in node.elts if isinstance(item, ast.Tuple)]
    for item in tuples:
        for element in item.elts:
            if isinstance(element, (ast.List, ast.Tuple)) and element.elts:
                values = _string_constants(element)
                if len(values) == len(element.elts):
                    found.extend(values)
    return found


class _Walker(ast.NodeVisitor):
    """Collect column names, stated roles and identifiers from one module."""

    def __init__(
        self,
        keywords: frozenset[str] = COLUMN_KEYWORDS,
        methods: frozenset[str] = COLUMN_METHODS,
        dtype_roles: dict[str, str] | None = None,
    ) -> None:
        self.keywords = keywords
        self.methods = methods
        self.dtype_roles = dtype_roles if dtype_roles is not None else _DTYPE_ROLES
        self.columns: dict[str, set[str]] = {}
        self.roles: dict[str, str] = {}
        self.identifiers: set[str] = set()
        self.bindings: dict[str, tuple[str, ...]] = {}
        self.deferred: set[str] = set()

    def _defer(self, node: ast.AST | None) -> None:
        """Note a name used as a column selector whose value is not yet known."""
        if isinstance(node, ast.Name):
            self.deferred.add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - ast API
        """
        Record ``NAME = ["a", "b"]`` so a later use of NAME can be resolved.

        Notes
        -----
        **Developer notes.** This is constant propagation, deliberately
        confined to one shape: a single name bound to a literal sequence whose
        elements are all string constants. That is decidable — nothing about
        it is a guess — and it is the shape that actually occurs, because
        ``ID_COLUMNS = [...]`` at the top of a module and
        ``df.drop(columns=ID_COLUMNS)` two hundred lines later is how people
        write this. Without it the conservative rule would miss the columns a
        careful author took the trouble to name.
        """
        values = _string_constants(node.value)
        if values and isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
            if len(values) == len(node.value.elts):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.bindings[target.id] = tuple(values)
        elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.bindings[target.id] = (node.value.value,)
        self.generic_visit(node)

    def _record(self, names: Iterable[str], site: str) -> None:
        for name in names:
            if name:
                self.columns.setdefault(name, set()).add(site)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802 - ast API
        self.identifiers.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802 - ast API
        # Recorded as an identifier so a stand-in cannot collide with it, but
        # deliberately not recorded as a column: see the module docstring.
        self.identifiers.add(node.attr)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:  # noqa: N802 - ast API
        self.identifiers.add(node.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node) -> None:  # noqa: N802 - ast API
        self.identifiers.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - ast API
        self.identifiers.add(node.name)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802 - ast API
        inner = _subscript_value(node)
        self._record(_string_constants(inner), "subscript")
        # `df[TARGET]` where TARGET = "churned": the selector is a name, and
        # resolving it is the same constant propagation `drop(columns=COLS)`
        # gets. Without this, a module that names its target once at the top
        # leaks it, which is the shape most feature modules are written in.
        self._defer(inner)
        if isinstance(inner, ast.Tuple) and inner.elts:
            # `df.loc[:, "col"]` and `df.loc[mask, ["a", "b"]]`: the column
            # selector is the last element, and only when the first is a slice
            # or a name, which is what makes this a label-based lookup.
            self._record(_string_constants(inner.elts[-1]), "loc")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        method = ""
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
        elif isinstance(node.func, ast.Name):
            method = node.func.id

        if method in self.methods and node.args:
            self._record(_string_constants(node.args[0]), f"method:{method}")
            self._defer(node.args[0])
            for argument in node.args:
                self._record(_tuple_lists(argument), f"method:{method}")

        for keyword_node in node.keywords:
            name = keyword_node.arg
            if name is None:
                continue
            if name in self.keywords:
                found = _string_constants(keyword_node.value)
                self._record(found, f"keyword:{name}")
                self._record(_tuple_lists(keyword_node.value), f"keyword:{name}")
                self._defer(keyword_node.value)
                if name in _DATE_KEYWORDS:
                    for column in found:
                        self.roles.setdefault(column, "date")
            if name in ("columns", "dtype") and isinstance(
                keyword_node.value, ast.Dict
            ):
                self._record_mapping(keyword_node.value, f"keyword:{name}")

        if method == "astype" and node.args and isinstance(node.args[0], ast.Dict):
            self._record_mapping(node.args[0], "method:astype")

        self.generic_visit(node)

    def _record_mapping(self, node: ast.Dict, site: str) -> None:
        """
        Record a ``{column: something}`` mapping.

        Notes
        -----
        **Developer notes.** Both halves of a ``rename(columns=...)`` are
        columns: the key is the name now and the value is the name afterwards,
        and a notebook goes on to use the second. Recording only the key would
        hide the old name and leak the new one, which is the more current of
        the two.
        """
        for key, value in zip(node.keys, node.values):
            names = _string_constants(key)
            self._record(names, site)
            self._record(_string_constants(value), site)
            for column in names:
                role = self._role_from_dtype(value)
                if role is not None:
                    self.roles.setdefault(column, role)

    def _role_from_dtype(self, node: ast.AST | None) -> str | None:
        """Return the role a stated dtype establishes, if it states one."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self.dtype_roles.get(node.value.strip().lower())
        if isinstance(node, ast.Attribute):
            return self.dtype_roles.get(node.attr.strip().lower())
        if isinstance(node, ast.Name):
            return self.dtype_roles.get(node.id.strip().lower())
        return None


def discover(
    sources: Iterable[tuple[str, str]],
    vocabulary: object | None = None,
) -> Discovery:
    """
    Walk one or more Python fragments and report the schema they reveal.

    Parameters
    ----------
    sources : iterable of (str, str)
        ``(label, source)`` pairs. The label names the fragment in the
        ``unparsed`` report — a cell number, a file name.
    vocabulary : CodeSpec, optional
        Extra column-naming sites from the selected packs
        (:class:`~scikitplot.cleanprompt._packs.CodeSpec`). They *extend* the
        core sites above; a pack can widen discovery and never narrow it.

    Returns
    -------
    Discovery
        Columns with their discovery sites, roles the code stated, every
        identifier seen, and the labels of fragments that did not parse.

    Notes
    -----
    **User notes.** Nothing here is a guess. A name appears in ``columns`` only
    because it was used in a position where a column label is the only
    reading — inside ``df[...]``, or bound to a parameter documented to take
    one.

    **Developer notes.** Fragments are walked independently and merged, so one
    unparsable cell costs only that cell. That matters for notebooks, where a
    single ``!pip install`` line would otherwise discard the schema of the
    whole document.

    Examples
    --------
    >>> found = discover([("cell 1", "x = df['region_code']")])
    >>> sorted(found.columns)
    ['region_code']
    >>> found.columns["region_code"]
    ('subscript',)
    >>> discover([("cell 2", "%matplotlib inline")]).unparsed
    ('cell 2',)

    A stated dtype is evidence, and is carried as an observed role:

    >>> stated = discover([("c", "df = df.astype({'region_code': 'category'})")])
    >>> stated.observed_roles
    {'region_code': 'category'}
    """
    keywords = COLUMN_KEYWORDS | frozenset(
        getattr(vocabulary, "column_keywords", ()) or ()
    )
    methods = COLUMN_METHODS | frozenset(
        getattr(vocabulary, "column_methods", ()) or ()
    )
    dtype_roles = dict(_DTYPE_ROLES)
    for dtype, role in getattr(vocabulary, "dtype_roles", ()) or ():
        dtype_roles.setdefault(dtype, role)
    result = Discovery()
    for label, source in sources:
        if not isinstance(source, str) or not source.strip():
            continue
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            # Not exceptional in a notebook: magics, shell escapes and
            # half-written cells all land here. Named rather than swallowed.
            result = result.merge(Discovery(unparsed=(str(label),)))
            continue
        walker = _Walker(keywords, methods, dtype_roles)
        walker.visit(tree)
        result = result.merge(
            Discovery(
                columns={
                    name: tuple(sorted(sites)) for name, sites in walker.columns.items()
                },
                observed_roles=dict(walker.roles),
                identifiers=set(walker.identifiers),
                bindings=dict(walker.bindings),
                deferred=set(walker.deferred),
            )
        )

    # Resolve names used as selectors against the literal lists they were bound
    # to, possibly in an earlier fragment. Anything still unresolved stays
    # unresolved: a name whose value was computed is not a column list.
    columns = dict(result.columns)
    for name in sorted(result.deferred):
        for column in result.bindings.get(name, ()):
            sites = set(columns.get(column, ())) | {f"binding:{name}"}
            columns[column] = tuple(sorted(sites))
    result.columns = columns
    return result
