"""
The secret side of a redaction: the label-to-original mapping.

Notes
-----
**User notes.** A :class:`Vault` is the only object in this submodule that holds
personal data after redaction. Treat it as you would a password: keep it on the
machine that created it, hand it to
:func:`~scikitplot.cleanprompt._engine.restore` when the reply comes back, and
call :meth:`Vault.clear` when you are done. Printing a vault, logging it, or
putting it in a traceback will not disclose its contents — its ``repr`` reports
counts, not values.

**Developer notes.** The vault exists as a separate type, rather than a bare
``dict`` returned alongside the text, for three reasons.

1. It makes "the safe half" and "the unsafe half" of a result structurally
   distinguishable, so a serializer cannot emit both by accident.
2. It carries the *grammar* fingerprint, so a vault cannot be applied under a
   different placeholder grammar than the one that produced it. Restoring
   ``[EMAIL-1]`` with a vault built for ``<<EMAIL_1>>`` would silently restore
   nothing; the fingerprint turns that into an error. It is the grammar digest
   and not the whole policy's, because narrowing ``kinds`` or raising a limit
   cannot change how a label is spelled and must not invalidate a vault.
3. It gives one place to implement a redacting ``repr``, an explicit
   :meth:`export`, and a :meth:`clear`.

``clear`` overwrites the mapping and marks the vault closed. It cannot promise
that Python has erased every copy of a ``str`` from memory — the interpreter
interns and copies strings and does not expose their buffers — and this module
does not claim otherwise. What it does guarantee is that the vault no longer
references the secrets and that any later use raises instead of silently
returning nothing.

See Also
--------
scikitplot.cleanprompt._engine.restore : Consumes a vault.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ItemsView, Iterator, Mapping

if TYPE_CHECKING:
    # Annotations are strings under ``from __future__ import annotations``, so
    # ``Self`` is needed only by a type checker. Importing it at runtime made
    # the base tier depend on typing_extensions, which Python 3.11+ does not
    # ship and nothing here declares (CP-054).
    from typing_extensions import Self


from ._exceptions import PolicyError

__all__ = [
    "Vault",
]


class Vault:
    """
    A closed mapping from placeholder label to original secret.

    Parameters
    ----------
    mapping : mapping of str to str, optional
        Initial label-to-secret pairs. Copied; the vault does not alias the
        caller's dictionary.
    grammar_fingerprint : str, optional
        Digest of the *placeholder grammar* that issued these labels — the
        :attr:`~scikitplot.cleanprompt._policy.TagStyle.fingerprint`, not the
        whole policy's. When set, restoration verifies it.

    Raises
    ------
    PolicyError
        If a label is empty or duplicated with a conflicting value.

    Notes
    -----
    **Developer notes.** The vault is append-only through :meth:`add` and is not
    thread-safe for concurrent writes. It does not need to be: one vault belongs
    to one redaction pass, which is single-threaded by construction. Sharing one
    vault across documents is the very defect this rewrite retires, so no
    locking is provided that would make it look supported.

    Examples
    --------
    >>> vault = Vault({"[EMAIL-1]": "a@b.c"})
    >>> vault["[EMAIL-1]"]
    'a@b.c'
    >>> repr(vault)
    'Vault(entries=1, closed=False)'
    >>> vault.clear()
    >>> len(vault)
    0
    """

    __slots__ = ("_closed", "_grammar_fingerprint", "_mapping")

    def __init__(
        self,
        mapping: Mapping[str, str] | None = None,
        grammar_fingerprint: str | None = None,
    ) -> None:
        self._mapping: dict[str, str] = {}
        self._closed = False
        self._grammar_fingerprint = grammar_fingerprint
        if mapping:
            for label, secret in mapping.items():
                self.add(label, secret)

    # -- construction -----------------------------------------------------

    def add(self, label: str, secret: str) -> None:
        """
        Record one label-to-secret pair.

        Parameters
        ----------
        label : str
            The rendered placeholder, including delimiters.
        secret : str
            The original text the label stands for.

        Raises
        ------
        PolicyError
            If ``label`` is empty, if the vault is closed, or if ``label`` is
            already present with a different secret.

        Notes
        -----
        **Developer notes.** Re-adding an identical pair is a no-op rather than
        an error, because the engine assigns one label per distinct value and
        may legitimately confirm an existing pair. A *conflicting* re-add means
        two different secrets were given the same label, which would make
        restoration ambiguous; that is always a bug and always raises.
        """
        if self._closed:
            raise PolicyError("cannot add to a cleared Vault")
        if not label:
            raise PolicyError("vault label must be a non-empty string")
        existing = self._mapping.get(label)
        if existing is not None and existing != secret:
            raise PolicyError(
                f"vault label {label!r} is already bound to a different value"
            )
        self._mapping[label] = secret

    # -- read -------------------------------------------------------------

    @property
    def grammar_fingerprint(self) -> str | None:
        """Str or None: Digest of the grammar that issued these labels."""
        return self._grammar_fingerprint

    @property
    def closed(self) -> bool:
        """bool: Whether :meth:`clear` has been called."""
        return self._closed

    def get(
        self,
        label: str,
        default: str | None = None,
    ) -> str | None:
        """
        Return the secret for ``label``, or ``default``.

        Parameters
        ----------
        label : str
            The placeholder to look up.
        default : str, optional
            Returned when ``label`` is absent.

        Returns
        -------
        str or None
            The stored secret, or ``default``.
        """
        return self._mapping.get(label, default)

    def __getitem__(self, label: str) -> str:
        return self._mapping[label]

    def __contains__(self, label: object) -> bool:
        return label in self._mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def labels(self) -> tuple:
        """
        Return every label in insertion order.

        Returns
        -------
        tuple of str
            The labels. Safe to log: labels are not secrets.
        """
        return tuple(self._mapping)

    def items(self) -> ItemsView[str, str]:
        """
        Return a live view of label-to-secret pairs.

        Returns
        -------
        ItemsView
            The mapping's items.

        Notes
        -----
        **Developer notes.** This exposes secrets and is named plainly rather
        than hidden, because restoration needs it. Prefer :meth:`labels` in any
        code path that logs, renders or serializes.
        """
        return self._mapping.items()

    # -- lifecycle --------------------------------------------------------

    def export(self) -> dict[str, str]:
        """
        Return a plain dictionary copy of the mapping, secrets included.

        Returns
        -------
        dict of str to str
            A copy of the label-to-secret mapping.

        Raises
        ------
        PolicyError
            If the vault has been cleared.

        Notes
        -----
        **User notes.** This is the deliberate, named way to get the secrets out
        — for instance to persist a session on the machine that owns the data.
        Nothing in this submodule calls it implicitly, and no serializer reaches
        it by accident.
        """
        if self._closed:
            raise PolicyError("cannot export a cleared Vault")
        return dict(self._mapping)

    def clear(self) -> None:
        """
        Drop every secret and close the vault.

        Notes
        -----
        **Developer notes.** Values are overwritten before deletion so that any
        surviving reference to the dictionary observes no plaintext. This does
        not and cannot guarantee that the interpreter has erased every copy of
        each string from process memory; see the module-level note.
        """
        for label in list(self._mapping):
            self._mapping[label] = ""
        self._mapping.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.clear()

    # -- presentation -----------------------------------------------------

    def __repr__(self) -> str:
        """
        Return a secret-free representation.

        Notes
        -----
        **Developer notes.** Never include values here. Reprs reach tracebacks,
        debuggers, notebook cells and log aggregators.
        """
        return f"Vault(entries={len(self._mapping)}, closed={self._closed})"

    __str__ = __repr__
