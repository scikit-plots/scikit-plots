"""
Authenticated vault encryption (tier: ``crypto``).

Notes
-----
**User notes.** Encrypting a vault protects the removed values at rest. You
supply the key; this module never generates one behind your back and never
writes one to disk::

    export CLEANPROMPT_VAULT_KEY="$(python -m scikitplot.cleanprompt doctor --new-key)"
    python -m scikitplot.cleanprompt redact --in p.txt --vault v.json --encrypt

Lose the key and the vault is unrecoverable. That is the point of it.

**Developer notes — why this exists at all.**

The ``crypto`` tier was declared in the capability table with nothing behind it,
and the maintenance record carried it as an ``UNAVAILABLE`` lane with the note
"build it or drop it; a declared capability with no implementation is the
decorative contract this project rejects". This is the build.

Fernet is used rather than a hand-assembled construction. It is authenticated
(AES-128-CBC with HMAC-SHA256), versioned, and carries its own IV and timestamp,
so there is no nonce discipline to get wrong here and no way to silently produce
an unauthenticated ciphertext. Every value is encrypted individually rather than
the mapping as a whole, so a corrupted entry is detected as *that* entry rather
than failing the entire vault.

Labels are **not** encrypted. A label is ``[EMAIL-1]`` — it is already public,
it appears in the text you are about to send, and encrypting it would only stop
a reader seeing which placeholders a vault covers while costing the ability to
inspect a vault's shape without the key.

What this does not do: it does not protect a vault in memory, it does not defend
against an attacker who has the key, and it does not make the key management
problem go away. A key stored next to the file it protects is decoration, which
is why the key comes from the environment and never from the vault document.

See Also
--------
scikitplot.cleanprompt._cli : Reads and writes encrypted vault documents.
"""

from __future__ import annotations

from typing import Any, Mapping

from ._capabilities import require
from ._exceptions import CleanPromptError

__all__ = ["decrypt_mapping", "encrypt_mapping", "new_key"]


def _fernet(key: bytes):
    """
    Return a Fernet instance for ``key``.

    Raises
    ------
    CapabilityError
        If the ``crypto`` tier is unavailable.
    CleanPromptError
        If the key is malformed.

    Notes
    -----
    **Developer notes.** The capability check precedes the import, so the
    actionable install message is reachable. Placing it after would let
    ``cryptography``'s own :class:`ModuleNotFoundError` fire first.
    """
    require("crypto")  # raises CapabilityError before cryptography is imported

    from cryptography.fernet import Fernet  # noqa: PLC0415 - deferred by design

    try:
        return Fernet(key)
    except Exception as exc:  # noqa: BLE001 - reported with guidance
        raise CleanPromptError(
            f"CLEANPROMPT_VAULT_KEY is not a valid Fernet key: {exc}. Generate a "
            "new one with: python -m scikitplot.cleanprompt doctor --new-key"
        ) from exc


def new_key() -> str:
    """
    Generate a fresh vault key.

    Returns
    -------
    str
        A URL-safe base64 Fernet key, suitable for ``CLEANPROMPT_VAULT_KEY``.

    Raises
    ------
    CapabilityError
        If the ``crypto`` tier is unavailable.

    Notes
    -----
    **User notes.** Store this in your secret manager, not in the repository
    and not beside the vault. Anyone holding it can read any vault it encrypted.
    """
    require("crypto")

    from cryptography.fernet import Fernet  # noqa: PLC0415

    return Fernet.generate_key().decode("ascii")


def encrypt_mapping(mapping: Mapping[str, str], key: bytes) -> dict[str, str]:
    """
    Encrypt every value of ``mapping``, leaving the labels in the clear.

    Parameters
    ----------
    mapping : mapping of str to str
        Label-to-secret pairs.
    key : bytes
        A Fernet key.

    Returns
    -------
    dict of str to str
        The same labels, mapped to base64 ciphertext.

    Raises
    ------
    CapabilityError
        If the ``crypto`` tier is unavailable.
    CleanPromptError
        If the key is malformed.

    Examples
    --------
    >>> key = new_key().encode()  # doctest: +SKIP
    >>> box = encrypt_mapping({"[EMAIL-1]": "a@b.co"}, key)  # doctest: +SKIP
    >>> decrypt_mapping(box, key)  # doctest: +SKIP
    {'[EMAIL-1]': 'a@b.co'}
    """
    box = _fernet(key)
    return {
        label: box.encrypt(secret.encode("utf-8")).decode("ascii")
        for label, secret in mapping.items()
    }


def decrypt_mapping(mapping: Any, key: bytes) -> dict[str, str]:
    """
    Decrypt a mapping produced by :func:`encrypt_mapping`.

    Parameters
    ----------
    mapping : object
        The stored mapping. Validated rather than trusted.
    key : bytes
        The Fernet key used to encrypt it.

    Returns
    -------
    dict of str to str
        Label-to-secret pairs.

    Raises
    ------
    CapabilityError
        If the ``crypto`` tier is unavailable.
    CleanPromptError
        If the document is malformed, or a value fails authentication.

    Notes
    -----
    **Developer notes.** A failed decryption names the label, not the value, and
    says the ciphertext failed authentication rather than guessing why. The two
    causes — wrong key, tampered file — are indistinguishable from here by
    design, and claiming to know which would be a guess presented as a fact.
    """
    if not isinstance(mapping, dict) or not all(
        isinstance(label, str) and isinstance(value, str)
        for label, value in mapping.items()
    ):
        raise CleanPromptError("the encrypted vault's 'entries' object is malformed")
    box = _fernet(key)

    from cryptography.fernet import InvalidToken  # noqa: PLC0415

    out: dict[str, str] = {}
    for label, token in mapping.items():
        try:
            out[label] = box.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:  # ruff: ignore[try-except-in-loop]
            raise CleanPromptError(
                f"vault entry {label!r} failed authentication: the key is wrong or "
                "the file was modified"
            ) from exc
    return out
