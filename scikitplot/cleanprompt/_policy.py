"""
Declarative configuration for a redaction pass.

Everything that changes *what* the pipeline does lives here, as data. No stage
of the engine reads an environment variable, a module global, or a constructor
flag that is not part of a :class:`RedactionPolicy`.

Notes
-----
**User notes.** Start from :data:`DEFAULT_POLICY` and adjust with
:meth:`RedactionPolicy.evolve`, which returns a new policy rather than mutating
the one you have::

    policy = DEFAULT_POLICY.evolve(kinds=("EMAIL", "URL"), case_insensitive=True)

**Developer notes.** The policy is a frozen, hashable dataclass, and
:attr:`RedactionPolicy.fingerprint` is a stable digest of it. That fingerprint is
recorded in every :class:`~scikitplot.cleanprompt._types.RedactionResult` and in
every :class:`~scikitplot.cleanprompt._vault.Vault`, which is what lets
restoration refuse a vault that was issued under a different placeholder
grammar. The digest must therefore depend on every field that can change the
placeholder text or the detection set, and must not depend on anything else.

The fingerprint is computed from a canonical JSON rendering, not from
:func:`hash`, because :func:`hash` is randomised per process for ``str`` and
would make the digest unstable across runs.

See Also
--------
scikitplot.cleanprompt._engine : Consumes a policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable

from ._exceptions import PolicyError

__all__ = [
    "DEFAULT_LIMITS",
    "DEFAULT_POLICY",
    "DEFAULT_TAG_STYLE",
    "Limits",
    "OverlapStrategy",
    "RedactionPolicy",
    "TagStyle",
]

#: Characters that may not appear in a placeholder delimiter or separator,
#: because they carry meaning in the regular expression that recognises
#: placeholders during restoration and idempotence masking.
_FORBIDDEN_IN_KIND = set("[]{}()<>|\\^$*+?.")


class OverlapStrategy(str, Enum):
    """
    How to arbitrate two detections that share characters.

    Attributes
    ----------
    LONGEST_WINS : str
        Keep the longer span; break ties by higher priority, then by detector
        name. The default, and the safe choice: covering more of the text
        cannot disclose more than covering less.
    PRIORITY_WINS : str
        Keep the higher-priority span; break ties by longer, then by detector
        name. Use when a curated detector must beat a broad one even on a
        shorter match.
    STRICT : str
        Raise :class:`~scikitplot.cleanprompt._exceptions.OverlapError`. Use in
        tests and in pipelines that must prove their detector set is disjoint.

    Notes
    -----
    **Developer notes.** There is deliberately no "first wins" strategy. It
    would make the outcome depend on detector registration order, which is
    exactly the kind of hidden positional coupling that makes a redaction result
    unreproducible.
    """

    LONGEST_WINS = "LONGEST_WINS"
    PRIORITY_WINS = "PRIORITY_WINS"
    STRICT = "STRICT"


@dataclass(frozen=True)
class TagStyle:
    """
    The placeholder grammar.

    Parameters
    ----------
    prefix : str, default='['
        Opening delimiter.
    suffix : str, default=']'
        Closing delimiter.
    separator : str, default='-'
        Separator between the category and the ordinal.
    uppercase_kind : bool, default=True
        Whether to upper-case the category in the rendered label.
    style : {'placeholder', 'surrogate'}, default='placeholder'
        What a removed value is replaced *with*. ``placeholder`` renders a
        bracket label; ``surrogate`` substitutes an invented but ordinary-
        looking value for the kinds that have one, falling back to a label for
        the rest. See :mod:`scikitplot.cleanprompt._surrogates`.

        This belongs to the grammar rather than to the wider policy because it
        changes how a stand-in is spelled, and restoration checks the grammar
        digest before reading a vault. A vault written in one style therefore
        cannot be silently read as the other.
    surrogates : str, optional
        The identity (``name@version#digest16``) of the custom surrogate set
        whose names this grammar issues. Filled in from ``surrogate_set``
        when that is given; recorded in a vault, so ``decode`` needs no set
        file. Requires ``style='surrogate'``.
    surrogate_set : SurrogateSet, optional
        The set itself (:mod:`~scikitplot.cleanprompt._surrogate_sets`),
        needed to *issue* stand-ins. Not compared, not serialised: restoring
        reads stand-ins from the vault and needs only ``surrogates``.

    Raises
    ------
    PolicyError
        If any part is empty, or if the suffix could begin inside a label and
        make the grammar ambiguous.

    Notes
    -----
    **Developer notes.** A non-empty ``suffix`` is mandatory, and this is
    load-bearing rather than cosmetic. With a closing delimiter, ``[EMAIL-1]``
    is not a prefix of ``[EMAIL-11]``, so a naive scan cannot confuse the two.
    Remove the suffix and ordinals above nine begin to corrupt each other. The
    upstream project happened to be safe here; making the suffix mandatory turns
    that accident into a guarantee, and
    ``test__engine.py::test_ordinal_ten_does_not_collide`` keeps it honest.

    Examples
    --------
    >>> TagStyle().render("email", 3)
    '[EMAIL-3]'
    >>> TagStyle(prefix="<<", suffix=">>", separator="_").render("email", 3)
    '<<EMAIL_3>>'
    """

    prefix: str = "["
    suffix: str = "]"
    separator: str = "-"
    uppercase_kind: bool = True
    style: str = "placeholder"
    surrogates: str | None = None
    surrogate_set: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("prefix", "suffix", "separator"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PolicyError(
                    f"TagStyle.{name} must be a non-empty string, got {value!r}"
                )
        if self.separator in self.prefix or self.separator in self.suffix:
            raise PolicyError(
                f"TagStyle.separator {self.separator!r} must not occur in the delimiters; "
                "the placeholder grammar would be ambiguous"
            )
        from ._surrogates import STYLES  # ruff: ignore[import-outside-top-level]

        if self.style not in STYLES:
            msg = "TagStyle.style {!r} is unknown; choose from {}".format(
                self.style, ", ".join(STYLES)
            )
            raise PolicyError(msg)
        self._check_surrogates()

    def _check_surrogates(self) -> None:
        """
        Tie ``surrogates`` to ``surrogate_set`` and to the surrogate style.

        Notes
        -----
        **Developer notes.** The identity is what is recorded and compared;
        the set object is what issues names. When both are given they must
        agree, or the vault would record a set other than the one that wrote
        it. A set never switches the style on by itself: asking for custom
        names under placeholders is a contradiction, refused here.
        """
        if self.surrogate_set is not None:
            identity = getattr(self.surrogate_set, "identity", None)
            if not isinstance(identity, str) or not identity:
                raise PolicyError("TagStyle.surrogate_set must be a SurrogateSet")
            if self.surrogates is None:
                object.__setattr__(self, "surrogates", identity)
            elif self.surrogates != identity:
                msg = (
                    f"TagStyle.surrogates {self.surrogates!r} names a different set "
                    f"than surrogate_set ({identity!r})"
                )
                raise PolicyError(msg)
        if self.surrogates is None:
            return
        if not isinstance(self.surrogates, str) or not self.surrogates:
            raise PolicyError("TagStyle.surrogates must be a non-empty string or None")
        if self.style != "surrogate":
            msg = (
                f"a surrogate set ({self.surrogates}) needs style='surrogate'; "
                f"this grammar is {self.style!r}"
            )
            raise PolicyError(msg)

    def render(self, kind: str, ordinal: int) -> str:
        """
        Render one placeholder label.

        Parameters
        ----------
        kind : str
            Category, for example ``"email"``.
        ordinal : int
            1-based index within the category.

        Returns
        -------
        str
            The label, including delimiters.

        Raises
        ------
        PolicyError
            If ``kind`` is empty, contains a character reserved by the
            placeholder grammar, or ``ordinal`` is not positive.
        """
        if not kind:
            raise PolicyError("kind must be a non-empty string")
        if ordinal < 1:
            raise PolicyError(f"ordinal must be >= 1, got {ordinal!r}")
        bad = sorted(set(kind) & _FORBIDDEN_IN_KIND)
        if bad:
            msg = (
                "kind {!r} contains characters reserved by the placeholder "
                "grammar: {}".format(kind, "".join(bad))
            )
            raise PolicyError(msg)
        if self.separator in kind:
            raise PolicyError(
                f"kind {kind!r} must not contain the separator {self.separator!r}"
            )
        rendered = kind.upper() if self.uppercase_kind else kind
        return f"{self.prefix}{rendered}{self.separator}{ordinal}{self.suffix}"

    def pattern(self) -> re.Pattern:
        """
        Return the compiled recogniser for labels of this grammar.

        Returns
        -------
        re.Pattern
            A pattern with two named groups, ``kind`` and ``ordinal``, matching
            any label this style can render.

        Notes
        -----
        **Developer notes.** Restoration scans with this pattern and then does a
        dictionary lookup, instead of replacing each known label in turn. That
        is a single left-to-right pass whose cost is independent of the number
        of entries, and — more importantly — it cannot let one label's
        replacement text be re-matched as another label.
        """
        return re.compile(
            f"{re.escape(self.prefix)}(?P<kind>[A-Za-z0-9_]+){re.escape(self.separator)}(?P<ordinal>[0-9]+){re.escape(self.suffix)}"
        )

    def lenient_pattern(self) -> re.Pattern:
        r"""
        Return a recogniser that also matches labels a model has rewritten.

        Returns
        -------
        re.Pattern
            A superset of :meth:`pattern`, with the same ``kind`` and
            ``ordinal`` groups.

        Notes
        -----
        **Developer notes — the channel is not lossless.**

        Every other invariant in this submodule holds across parts it controls.
        Restoration does not: between redaction and restoration the text passes
        through a language model, which rewrites tokens. Measured against real
        replies, a label comes back in these shapes::

            [EMAIL-1]       as issued
            [email-1]       lower-cased
            [EMAIL_1]       separator changed
            [EMAIL 1]       separator changed to a space
            [EMAIL<U+2011>1]  ASCII hyphen replaced with a Unicode dash
            \[EMAIL-1\]     brackets escaped for Markdown
            [EMAIL\_1]      separator escaped for Markdown
            [EMAIL-<NL>1]     wrapped across a line

        The table is written literally: every row above is the exact text a
        recogniser has to accept, except for two characters a source file
        cannot show inline -- ``<U+2011>`` is one non-breaking hyphen and
        ``<NL>`` is one line break. ``CP-044`` exists because an
        earlier version of this table was written in doubled escapes and was
        therefore easier to check against itself than against the pattern.

        Exact matching restores the first and misses the rest, which is how a
        reply arrives reporting "0 restored" while plainly full of
        placeholders.

        The escaped separator is the composition of two shapes already in that
        list rather than a new one: a model writing ``[EMAIL_1]`` inside
        Markdown escapes the underscore, because an unescaped one would open
        emphasis. It is admitted for the same reason the escaped brackets are,
        and it was found by ``CP-044`` — by checking the list above against the
        implementation rather than against itself.

        Admitting it widens the *match* set by exactly the escaped spellings and
        widens the prose match set by nothing, because the surrounding structure
        — delimiter, letter-initial category, digits, delimiter — is untouched.
        That was measured rather than reasoned about.

        This pattern accepts that bounded set — case, the separator family,
        internal whitespace, and backslash escapes — and nothing else. It does
        not accept a missing delimiter or an invented category, because those
        cannot be told apart from ordinary prose. The category must start with a
        letter, so a footnote marker like ``[1]`` is never a candidate.

        Matching is only half of it. A lenient match is acted on **only** when
        the label it normalises to is in the vault, and the caller reports every
        one it acted on; see :func:`~scikitplot.cleanprompt.restore`.

        Examples
        --------
        >>> style = TagStyle()
        >>> bool(style.lenient_pattern().search("see [email_1] there"))
        True
        >>> bool(style.lenient_pattern().search(r"see [email\_1] there"))
        True
        >>> bool(style.lenient_pattern().search("see [1] there"))
        False
        >>> style.lenient_pattern().search(r"a\\[EMAIL-1]").group()
        '[EMAIL-1]'
        """
        prefix = re.escape(self.prefix)
        suffix = re.escape(self.suffix)
        # The separator as issued, plus what a model substitutes for it: the
        # other ASCII connectors and the Unicode dash family.
        separators = re.escape(self.separator) + r"_\-\u2010-\u2015\u2212"
        # A leading backslash is an escape only when it is not itself escaped:
        # in JSON, ``a\\[EMAIL-1]`` is a literal backslash followed by the
        # label, and consuming it would drop a character from the restored
        # text (found by the JSON round-trip tests of ``_runtime.py``).
        return re.compile(
            r"(?:(?<!\\)\\)?"
            + prefix
            + r"\s*(?P<kind>[A-Za-z][A-Za-z0-9_]*)\s*"
            # A Markdown-escaped separator, for the same reason the delimiters
            # allow one: an unescaped '_' would open emphasis (CP-044).
            + r"\\?"
            + "["
            + separators
            + r"\s]\s*"
            + r"(?P<ordinal>[0-9]+)\s*"
            + r"\\?"
            + suffix
        )

    def normalize(self, candidate: str) -> str | None:
        r"""
        Return the canonical label that a rewritten one was meant to be.

        Parameters
        ----------
        candidate : str
            Text matched by :meth:`lenient_pattern`.

        Returns
        -------
        str or None
            The label as this grammar renders it, or ``None`` when the text is
            not a rewritten label at all.

        Examples
        --------
        >>> TagStyle().normalize("[email_1]")
        '[EMAIL-1]'
        >>> TagStyle().normalize(r"[email\_1]")
        '[EMAIL-1]'
        >>> TagStyle().normalize("not a label") is None
        True

        Notes
        -----
        **Developer notes.** The canonical label is *rebuilt* from the ``kind``
        and ``ordinal`` groups rather than repaired in place, so whatever a
        model did to the separator and the delimiters cannot reach the result.
        That is why admitting a new rewrite shape to
        :meth:`lenient_pattern` needs no change here.
        """
        match = self.lenient_pattern().fullmatch(candidate)
        if match is None:
            return None
        return self.render(match.group("kind"), int(match.group("ordinal")))

    def as_dict(self) -> dict[str, Any]:
        """
        Return a JSON-safe dictionary of this style.

        Returns
        -------
        dict
            Field names to values.
        """
        return {
            "prefix": self.prefix,
            "suffix": self.suffix,
            "separator": self.separator,
            "uppercase_kind": self.uppercase_kind,
            "style": self.style,
            **({} if self.surrogates is None else {"surrogates": self.surrogates}),
        }

    def _fingerprint_payload(self) -> dict[str, Any]:
        """
        Return the fields that decide how a stand-in is spelled.

        Notes
        -----
        **Developer notes.** This is :meth:`as_dict` minus a default-valued
        ``style``, and the omission is deliberate rather than an oversight.

        A grammar in the default style spells labels exactly as every earlier
        version of this submodule did. Its digest must therefore stay the same,
        or adding the field would make every vault written before it
        unreadable — restoration compares the stored digest with the current
        one and refuses a mismatch, which is the right behaviour applied to the
        wrong question.

        A non-default style *does* change the spelling, so it enters the digest
        and a vault written under it cannot be read as the other.

        ``surrogates`` follows the same rule from the other side: absent from
        :meth:`as_dict` while it is ``None``, so it changes no existing digest,
        and present — as the set's content-derived identity — when a custom
        set issues the names (``GENERATOR_DESIGN.md`` G6).
        """
        payload = self.as_dict()
        if payload.get("style") == "placeholder":
            del payload["style"]
        return payload

    @property
    def fingerprint(self) -> str:
        """
        str: Stable 16-character digest of this *grammar*.

        Notes
        -----
        **Developer notes.** This is deliberately narrower than
        :attr:`RedactionPolicy.fingerprint`, and the two answer different
        questions.

        Restoration cares about exactly one thing: were these labels written in
        the grammar I am about to read them with? It does not care which kinds
        were detected, what the limits were, or how overlaps were arbitrated —
        none of those can change how ``[EMAIL-1]`` is spelled. Checking the full
        policy digest at restoration time would reject a perfectly valid vault
        merely because the caller had narrowed ``kinds``, which is a false
        alarm on the common path.

        The full policy digest remains on
        :class:`~scikitplot.cleanprompt._types.RedactionResult`, where it serves
        its own purpose: reproducing a pass exactly.
        """
        payload = json.dumps(
            self._fingerprint_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Limits:
    """
    Hard bounds on one redaction pass.

    Parameters
    ----------
    max_input_chars : int, default=1_000_000
        Longest text accepted. Exceeding it raises
        :class:`~scikitplot.cleanprompt._exceptions.LimitExceededError`.
    max_spans : int, default=100_000
        Most detections accepted before overlap resolution.
    max_entries : int, default=50_000
        Most distinct values newly detected in one text. Values carried in
        from earlier turns or pieces (a ``seed``) do not count.
    max_literal_terms : int, default=10_000
        Most user-supplied literal terms accepted.
    max_literal_length : int, default=4_096
        Longest single literal term accepted.

    Raises
    ------
    PolicyError
        If any bound is not a positive integer.

    Notes
    -----
    **Developer notes.** Bounds exist so that adversarial input fails loudly and
    early rather than consuming the process. They are policy, not constants, so
    a caller who genuinely needs a larger document can raise them explicitly and
    own that decision. Nothing in this submodule silently truncates: the
    ``truncated`` field on
    :class:`~scikitplot.cleanprompt._types.RedactionResult` is always ``False``
    and exists so a caller can assert that.

    The default ``max_input_chars`` matches the default document ceiling of the
    optional spaCy tier, so the base tier and the NER tier fail at the same
    size rather than at two different, surprising ones.
    """

    max_input_chars: int = 1_000_000
    max_spans: int = 100_000
    max_entries: int = 50_000
    max_literal_terms: int = 10_000
    max_literal_length: int = 4_096

    def __post_init__(self) -> None:
        for name in (
            "max_input_chars",
            "max_spans",
            "max_entries",
            "max_literal_terms",
            "max_literal_length",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise PolicyError(
                    f"Limits.{name} must be a positive int, got {value!r}"
                )

    def as_dict(self) -> dict[str, Any]:
        """
        Return a JSON-safe dictionary of these limits.

        Returns
        -------
        dict
            Field names to values.
        """
        return {
            "max_input_chars": self.max_input_chars,
            "max_spans": self.max_spans,
            "max_entries": self.max_entries,
            "max_literal_terms": self.max_literal_terms,
            "max_literal_length": self.max_literal_length,
        }


@dataclass(frozen=True)
class RedactionPolicy:
    """
    Complete, declarative configuration for a redaction pass.

    Parameters
    ----------
    kinds : tuple of str, optional
        Categories to detect, naming entries of the pattern library. ``None``
        means "every pattern enabled by default".
    tag_style : TagStyle, optional
        Placeholder grammar. Defaults to :data:`DEFAULT_TAG_STYLE`.
    overlap : OverlapStrategy, default=OverlapStrategy.LONGEST_WINS
        Arbitration rule for overlapping detections.
    limits : Limits, optional
        Hard bounds. Defaults to :data:`DEFAULT_LIMITS`.
    case_insensitive : bool, default=False
        Whether two surfaces that differ only in case share one label.
    min_confidence : float, default=0.0
        Detections below this confidence are discarded before resolution.
    allow : tuple of str, default=()
        Surfaces that must never be redacted, compared exactly against a
        detection's own text. An allowlist is the necessary complement to a
        denylist: without it the only way to stop a false positive is to
        disable the whole pattern, which silently removes cover for everything
        else that pattern would have caught.
    preserve_placeholders : bool, default=True
        Whether text already matching the placeholder grammar is protected from
        re-detection, making redaction idempotent.

    Raises
    ------
    PolicyError
        If ``kinds`` is empty, contains a non-string or a duplicate, or if
        ``min_confidence`` is outside ``[0.0, 1.0]``.

    Notes
    -----
    **User notes.** ``case_insensitive=True`` is the right choice for names that
    appear in mixed case ("Acme" and "ACME" become one label). Leave it off for
    identifiers where case is meaningful, such as API keys.

    **Developer notes.** ``preserve_placeholders`` is what gives invariant I7.
    Before detection runs, the engine finds every substring matching the active
    placeholder grammar and treats those ranges as reserved; detectors may still
    report spans there, but resolution drops them. Without this, feeding an
    already-redacted text back through the pipeline lets a detector tag the
    inside of a placeholder and destroy it — which is the reproduced upstream
    defect ``CP-006``.

    Examples
    --------
    >>> policy = DEFAULT_POLICY.evolve(case_insensitive=True)
    >>> policy.case_insensitive
    True
    >>> policy is DEFAULT_POLICY
    False
    >>> len(policy.fingerprint)
    16
    """

    kinds: tuple[str, ...] | None = None
    allow: tuple[str, ...] = ()
    tag_style: TagStyle = TagStyle()
    overlap: OverlapStrategy = OverlapStrategy.LONGEST_WINS
    limits: Limits = Limits()
    case_insensitive: bool = False
    min_confidence: float = 0.0
    preserve_placeholders: bool = True

    def __post_init__(self) -> None:
        if self.kinds is not None:
            if not isinstance(self.kinds, tuple):
                raise PolicyError(
                    "RedactionPolicy.kinds must be a tuple or None, got "
                    f"{type(self.kinds).__name__!r}"
                )
            if not self.kinds:
                raise PolicyError(
                    "RedactionPolicy.kinds must not be empty; pass None to "
                    "select the default pattern set"
                )
            seen = set()
            for kind in self.kinds:
                if not isinstance(kind, str) or not kind:
                    raise PolicyError(
                        "RedactionPolicy.kinds entries must be non-empty "
                        f"strings, got {kind!r}"
                    )
                if kind in seen:
                    raise PolicyError(
                        f"RedactionPolicy.kinds contains a duplicate: {kind!r}"
                    )
                seen.add(kind)
        if not isinstance(self.allow, tuple):
            raise PolicyError(
                f"RedactionPolicy.allow must be a tuple, got {type(self.allow).__name__!r}"
            )
        for term in self.allow:
            if not isinstance(term, str) or not term.strip():
                raise PolicyError(
                    "RedactionPolicy.allow entries must be non-empty strings, "
                    f"got {term!r}"
                )
        if not isinstance(self.overlap, OverlapStrategy):
            raise PolicyError(
                "RedactionPolicy.overlap must be an OverlapStrategy, got "
                f"{self.overlap!r}"
            )
        if not 0.0 <= self.min_confidence <= 1.0:
            raise PolicyError(
                "RedactionPolicy.min_confidence must be in [0.0, 1.0], got "
                f"{self.min_confidence!r}"
            )

    def evolve(self, **changes: Any) -> RedactionPolicy:
        """
        Return a copy of this policy with ``changes`` applied.

        Parameters
        ----------
        **changes
            Field names to new values. ``kinds`` accepts any iterable of strings
            and is normalised to a tuple.

        Returns
        -------
        RedactionPolicy
            A new, validated policy. This policy is unchanged.

        Raises
        ------
        PolicyError
            If a name is not a field of this class, or the result is invalid.

        Examples
        --------
        >>> DEFAULT_POLICY.evolve(kinds=["EMAIL"]).kinds
        ('EMAIL',)
        """
        known = set(self.__dataclass_fields__)
        unknown = sorted(set(changes) - known)
        if unknown:
            msg = "unknown RedactionPolicy field(s): {}; known fields are {}".format(
                ", ".join(unknown), ", ".join(sorted(known))
            )
            raise PolicyError(msg)
        if "allow" in changes and changes["allow"] is not None:
            allowed = changes["allow"]
            if isinstance(allowed, str):
                raise PolicyError(
                    "allow must be an iterable of strings, not a single string"
                )
            changes["allow"] = tuple(allowed)
        if "kinds" in changes and changes["kinds"] is not None:
            value: Iterable[str] = changes["kinds"]
            if isinstance(value, str):
                raise PolicyError(
                    "kinds must be an iterable of strings, not a single string"
                )
            changes["kinds"] = tuple(value)
        return replace(self, **changes)

    def selected_kinds(self, available: Iterable[str]) -> frozenset[str]:
        """
        Resolve :attr:`kinds` against the kinds a registry offers.

        Parameters
        ----------
        available : iterable of str
            Kinds the registry can provide.

        Returns
        -------
        frozenset of str
            The kinds to run.

        Raises
        ------
        PolicyError
            If a requested kind is not available. The message lists what is.

        Notes
        -----
        **Developer notes.** Requesting an unknown kind raises rather than being
        ignored. A silently dropped category is a redaction that did not happen,
        and the caller would have no way to notice before transmitting.
        """
        offered = frozenset(available)
        if self.kinds is None:
            return offered
        missing = sorted(set(self.kinds) - offered)
        if missing:
            msg = "unknown detection kind(s): {}; available kinds are {}".format(
                ", ".join(missing), ", ".join(sorted(offered))
            )
            raise PolicyError(msg)
        return frozenset(self.kinds)

    def as_dict(self) -> dict[str, Any]:
        """
        Return a JSON-safe dictionary of this policy.

        Returns
        -------
        dict
            Field names to JSON-safe values, with ``kinds`` sorted so that two
            policies differing only in the order they were written produce the
            same rendering.
        """
        return {
            "kinds": None if self.kinds is None else sorted(self.kinds),
            "allow": sorted(self.allow),
            "tag_style": self.tag_style.as_dict(),
            "overlap": self.overlap.value,
            "limits": self.limits.as_dict(),
            "case_insensitive": self.case_insensitive,
            "min_confidence": self.min_confidence,
            "preserve_placeholders": self.preserve_placeholders,
        }

    @property
    def fingerprint(self) -> str:
        """
        str: Stable 16-character digest of this policy.

        Notes
        -----
        **Developer notes.** Computed from canonical JSON with sorted keys, so
        it is identical across processes, platforms and Python versions.
        :func:`hash` is unsuitable: ``PYTHONHASHSEED`` randomises string hashing
        per process, so a hash-derived fingerprint would fail to match a vault
        written by an earlier run.
        """
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


#: Default placeholder grammar: ``[EMAIL-1]``.
DEFAULT_TAG_STYLE = TagStyle()

#: Default bounds.
DEFAULT_LIMITS = Limits()

#: Default policy: every default pattern, longest-wins arbitration, idempotent.
DEFAULT_POLICY = RedactionPolicy()


def allowed_surfaces(policy: RedactionPolicy) -> frozenset[str]:
    """
    Return the allowlist, folded to match the policy's identity rule.

    Parameters
    ----------
    policy : RedactionPolicy
        The active policy.

    Returns
    -------
    frozenset of str
        Surfaces to leave alone, case-folded when the policy folds case, so the
        allowlist and the value-identity rule agree. An allowlist that matched
        case-sensitively under a case-insensitive policy would be a surprise in
        the dangerous direction: the user would believe a term was exempt and
        find it redacted anyway.
    """
    if policy.case_insensitive:
        return frozenset(term.casefold() for term in policy.allow)
    return frozenset(policy.allow)


#: Sentinel meaning "resolve this at call time from the pattern library".
_STRICT_KINDS = "<default-plus-title-case>"

#: Named policy bundles, so a team can agree on one word instead of six flags.
#:
#: Each maps to keyword arguments for :meth:`RedactionPolicy.evolve`. ``None``
#: for ``kinds`` means "every pattern enabled by default"; an explicit tuple
#: narrows or widens it.
PROFILES: dict[str, dict[str, Any]] = {
    "minimal": {
        "kinds": ("EMAIL", "PHONE", "URL"),
        "case_insensitive": False,
    },
    "balanced": {
        "kinds": None,
        "case_insensitive": False,
    },
    # ``kinds`` is resolved at call time: "strict" means "everything enabled by
    # default, plus TITLE_CASE", and the default set lives in the pattern
    # library. Freezing a literal list here would silently stop tracking it.
    "strict": {
        "kinds": _STRICT_KINDS,
        "case_insensitive": True,
    },
}


def profile(name: str, base: RedactionPolicy | None = None) -> RedactionPolicy:
    """
    Return the policy for a named profile.

    Parameters
    ----------
    name : str
        Profile name; one of the keys of :data:`PROFILES`.
    base : RedactionPolicy, optional
        Policy to evolve from. Defaults to :data:`DEFAULT_POLICY`.

    Returns
    -------
    RedactionPolicy
        The configured policy.

    Raises
    ------
    PolicyError
        If ``name`` is not a known profile. The message lists the known ones.

    Notes
    -----
    **User notes.** ``minimal`` covers contact details only. ``balanced`` is the
    default: every high-precision structural pattern. ``strict`` adds
    capitalised-name detection and folds case, which redacts more and will
    produce some false positives — that is the trade it exists to make.

    Examples
    --------
    >>> profile("minimal").kinds
    ('EMAIL', 'PHONE', 'URL')
    """
    if name not in PROFILES:
        msg = "unknown profile {!r}; known profiles are {}".format(
            name, ", ".join(sorted(PROFILES))
        )
        raise PolicyError(msg)
    changes = dict(PROFILES[name])
    if changes.get("kinds") is _STRICT_KINDS:
        from ._patterns import (  # ruff: ignore[import-outside-top-level]
            PATTERNS,
            default_patterns,
        )

        enabled = {spec.kind for spec in default_patterns()}
        if "TITLE_CASE" in PATTERNS:
            enabled.add("TITLE_CASE")
        changes["kinds"] = tuple(sorted(enabled))
    return (base if base is not None else DEFAULT_POLICY).evolve(**changes)
