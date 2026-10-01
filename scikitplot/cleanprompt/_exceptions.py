"""
Typed exception hierarchy for :mod:`scikitplot.cleanprompt`.

Every failure mode of the redaction pipeline is represented by a distinct
exception type carrying the machine-readable context a caller needs to react,
rather than a bare :class:`ValueError` with a prose message.

Notes
-----
**User notes.** Catch :class:`CleanPromptError` to catch everything this
submodule raises. Catch a leaf type when the reaction differs: a
:class:`CapabilityError` means *install something*, a
:class:`LimitExceededError` means *send less text*, and a
:class:`RestorationError` means *the reply refers to a placeholder this vault
never issued*.

**Developer notes.** No code path in this submodule may degrade silently. If a
stage cannot honour its contract it raises. This is deliberate: the subject
matter is personal data, and a partially applied redaction that returns
successfully is worse than a loud failure, because the caller will transmit it.

:class:`CapabilityError` is constructed from a capability report and must be
raisable *without* importing the missing distribution. An actionable install
message placed after a module-scope third-party import can never fire, because
the import raises first.

See Also
--------
scikitplot.cleanprompt._capabilities : Probes that populate :class:`CapabilityError`.
"""

from __future__ import annotations

__all__ = [
    "CapabilityError",
    "CleanPromptError",
    "DetectorError",
    "LeakError",
    "LimitExceededError",
    "OverlapError",
    "PatternError",
    "PolicyError",
    "RestorationError",
]


class CleanPromptError(Exception):
    """Base class for every error raised by :mod:`scikitplot.cleanprompt`."""


class PolicyError(CleanPromptError, ValueError):
    """
    A :class:`~scikitplot.cleanprompt._policy.RedactionPolicy` is invalid.

    Raised for contradictory or malformed configuration, such as an empty
    placeholder delimiter or a non-positive limit.
    """


class PatternError(CleanPromptError, ValueError):
    """
    A detection pattern is uncompilable or structurally unsafe.

    Parameters
    ----------
    message : str
        Human-readable description.
    name : str, optional
        Name of the offending pattern, when known.
    """

    def __init__(self, message: str, name: str | None = None) -> None:
        super().__init__(message)
        self.name = name


class DetectorError(CleanPromptError, RuntimeError):
    """
    A detector raised while scanning.

    Parameters
    ----------
    message : str
        Human-readable description.
    detector : str
        Name of the detector that failed. Carried separately so a caller can
        disable exactly that detector and retry.
    """

    def __init__(self, message: str, detector: str) -> None:
        super().__init__(message)
        self.detector = detector


class OverlapError(CleanPromptError, ValueError):
    """
    Two detections overlap under
    :attr:`~scikitplot.cleanprompt._policy.OverlapStrategy.STRICT`.

    Parameters
    ----------
    message : str
        Human-readable description.
    spans : tuple
        The two conflicting spans, as ``(start, end, kind)`` triples.
    """  # ruff: ignore[missing-blank-line-after-summary]

    def __init__(
        self,
        message: str,
        spans: tuple[tuple[int, int, str], tuple[int, int, str]],
    ) -> None:
        super().__init__(message)
        self.spans = spans


class LimitExceededError(CleanPromptError, ValueError):
    """
    A policy limit was exceeded.

    Parameters
    ----------
    message : str
        Human-readable description.
    limit_name : str
        Which limit was hit, for example ``"max_input_chars"``.
    limit : int
        The configured bound.
    actual : int
        The observed value.
    """

    def __init__(self, message: str, limit_name: str, limit: int, actual: int) -> None:
        super().__init__(message)
        self.limit_name = limit_name
        self.limit = limit
        self.actual = actual


class RestorationError(CleanPromptError, ValueError):
    """
    A placeholder in the text cannot be restored from the supplied vault.

    Parameters
    ----------
    message : str
        Human-readable description.
    labels : tuple of str
        The placeholder labels that could not be resolved.
    """

    def __init__(self, message: str, labels: tuple[str, ...]) -> None:
        super().__init__(message)
        self.labels = labels


class LeakError(CleanPromptError):
    """
    Outgoing text still holds a value the vault removed; it was not sent.

    Parameters
    ----------
    message : str
        Human-readable description. Names kinds and counts, never a value.
    kinds : tuple of str
        The kinds of the values found, one entry per distinct value.

    Notes
    -----
    **User notes.** Raised by :meth:`~scikitplot.cleanprompt.Guard.outgoing`
    *before* anything reaches a model. It means the final check found a value
    the encoding step should have hidden — usually because ``remember`` was
    turned off and the value recurred where no rule looks. Nothing was sent.
    """

    def __init__(self, message: str, kinds: tuple[str, ...]) -> None:
        super().__init__(message)
        self.kinds = kinds


class CapabilityError(CleanPromptError, ImportError):
    """
    An optional tier is not usable in this environment.

    Parameters
    ----------
    message : str
        Human-readable description, including the actionable install command.
    tier : str
        The tier name, for example ``"ner"``.
    status : str
        A :class:`~scikitplot.cleanprompt._capabilities.CapabilityStatus` value.
    install_hint : str, optional
        The exact command that would make the tier available.

    Notes
    -----
    **Developer notes.** This subclasses :class:`ImportError` so that existing
    ``except ImportError`` fallbacks around optional features keep working, while
    ``except CapabilityError`` gives the richer report. ``status`` preserves the
    ``BROKEN`` versus ``ABSENT`` distinction; collapsing them makes an installed
    but unusable dependency indistinguishable from a missing one.
    """

    def __init__(
        self,
        message: str,
        tier: str,
        status: str,
        install_hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.tier = tier
        self.status = status
        self.install_hint = install_hint
