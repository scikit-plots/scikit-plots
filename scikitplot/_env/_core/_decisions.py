# scikitplot/_env/_core/_decisions.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The one place where the builder asks for permission.

Notes
-----
**User notes.** Three steps need an approval: installing a tool
(``bootstrap``), editing a shell profile (``shell_init``) and deleting an
environment (``remove``). Give approvals up front with ``approve=`` (Python)
or ``--yes`` / ``--approve KIND`` (command line); with ``interaction="auto"``
an interactive terminal is asked instead; an agent or notebook can pass a
callable ``approve(kind, message) -> bool``.

**Developer notes.** No backend, parser or helper reads ``input()``: they
return steps, and :py:meth:`Decider.require` is called by the executor before
the step runs. A configuration file cannot reach this module, so it cannot
approve itself.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Iterable, TextIO

from ._errors import ApprovalRequiredError, ConfigError
from ._types import Approval, Interaction, choice

__all__ = ["Decider"]

_HINTS = {
    Approval.BOOTSTRAP.value: (
        "pass --yes (or --approve bootstrap; approve='bootstrap' in Python)"
    ),
    Approval.SHELL_INIT.value: "pass --yes (or --approve shell_init)",
    Approval.REMOVE.value: "pass --yes (or --approve remove)",
}


class Decider:
    """
    Answers approval questions from pre-given approvals, a callable or a terminal.

    Parameters
    ----------
    interaction : {"never", "auto", "always"}, default="auto"
        Whether a person may be asked.
    approve : bool, str, iterable of str or callable, default=()
        ``True`` approves everything; strings name approval kinds; a callable
        ``approve(kind, message) -> bool`` decides each question.
    stdin, stderr : text stream or None, default=None
        Terminal used for questions; :py:data:`sys.stdin` / :py:data:`sys.stderr`.
    environ : mapping or None, default=None
        Used to detect CI (no questions in CI under ``auto``).

    Examples
    --------
    >>> from scikitplot._env._core._decisions import Decider
    >>> Decider("never", approve=["bootstrap"]).allows("bootstrap", "install uv")
    True
    >>> Decider("never").allows("remove", "delete env")
    False
    """

    def __init__(
        self,
        interaction: str = Interaction.AUTO.value,
        *,
        approve: Any = (),
        stdin: TextIO | None = None,
        stderr: TextIO | None = None,
        environ: Any = None,
    ) -> None:  # noqa: D107
        self.interaction = choice(interaction, Interaction, field="interaction").value
        self._callback: Callable[[str, str], bool] | None = None
        self._all = False
        self._kinds: set[str] = set()
        if approve is True:
            self._all = True
        elif callable(approve):
            self._callback = approve
        elif approve in (None, False):
            pass
        else:
            kinds = [approve] if isinstance(approve, str) else list(_iterable(approve))
            for kind in kinds:
                self._kinds.add(choice(kind, Approval, field="approve").value)
        self._stdin = stdin
        self._stderr = stderr
        self._environ = os.environ if environ is None else environ
        self.transcript: list[dict[str, Any]] = []

    def _can_ask(self) -> bool:
        if self.interaction == Interaction.NEVER.value:
            return False
        stdin = self._stdin if self._stdin is not None else sys.stdin
        tty = bool(getattr(stdin, "isatty", lambda: False)())
        if self.interaction == Interaction.ALWAYS.value:
            # A person answers only at a terminal: piped "y" is not a person.
            return tty
        in_ci = self._environ.get("CI", "").lower() not in ("", "0", "false", "no")
        return tty and not in_ci

    def _ask(self, kind: str, message: str) -> bool:
        stdin = self._stdin if self._stdin is not None else sys.stdin
        stderr = self._stderr if self._stderr is not None else sys.stderr
        stderr.write(f"{message}\nApprove {kind}? [y/N] ")
        stderr.flush()
        line = stdin.readline()
        if not line:  # end of input is a "no"
            return False
        return line.strip().lower() in ("y", "yes")

    def allows(self, kind: str, message: str) -> bool:
        """
        Decide one question without raising.

        Parameters
        ----------
        kind : str
            Approval kind.
        message : str
            What would happen.

        Returns
        -------
        bool
            The answer, recorded in :py:attr:`transcript`.
        """
        kind = choice(kind, Approval, field="approval").value
        if self._all or kind in self._kinds:
            answer, how = True, "pre-approved"
        elif self._callback is not None:
            answer, how = bool(self._callback(kind, message)), "callback"
        elif self._can_ask():
            answer, how = self._ask(kind, message), "asked"
        else:
            answer, how = False, "not-approved"
        self.transcript.append({"kind": kind, "answer": answer, "how": how})
        return answer

    def require(self, kind: str, message: str) -> None:
        """
        Decide one question and raise when it is not approved.

        Parameters
        ----------
        kind : str
            Approval kind.
        message : str
            What would happen.

        Raises
        ------
        ApprovalRequiredError
            When the answer is no.
        """
        if not self.allows(kind, message):
            raise ApprovalRequiredError(
                f"{message}: not approved", kind=kind, hint=_HINTS.get(kind)
            )


def _iterable(value: Any) -> Iterable[Any]:
    try:
        return iter(value)
    except TypeError:
        raise ConfigError(
            "approve must be True, an approval kind, a list of kinds or a callable"
        ) from None
