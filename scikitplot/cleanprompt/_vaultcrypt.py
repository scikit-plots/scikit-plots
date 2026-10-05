"""
Vault encryption that needs nothing installed (base tier).

Notes
-----
**User notes.** ``--encrypt`` works on any machine with a Python interpreter,
with no package to install and no platform to match::

    export CLEANPROMPT_VAULT_KEY='a long passphrase you chose'
    python -m scikitplot.cleanprompt encode --encrypt "Mail ada@example.com"
    python -m scikitplot.cleanprompt decode "I mailed [EMAIL-1]."

A vault written this way opens on any other machine with the same passphrase —
Linux, macOS, Windows, a container, an old interpreter. There is no key file to
copy and no library version to match.

Lose the passphrase and the vault is unrecoverable. That is what it is for.

**Developer notes — why a second cipher exists.**

The ``crypto`` tier encrypts with Fernet, which is the better primitive: AES in
a construction that has been reviewed far more than anything written here. It
is also an optional dependency with compiled extensions, and a vault encrypted
with it cannot be opened on a machine that does not have it.

That is the wrong trade for this submodule's central promise. The base tier is
pure standard library so that the tool works anywhere; an ``--encrypt`` that
only works where a compiled package is installed makes the one security feature
the least portable part of the program. Worse, it makes the advice in
``forget`` — "use ``--encrypt`` for values that must not be recoverable" —
conditional on something the reader may not have.

So this module provides the same guarantee from the standard library alone, and
:mod:`scikitplot.cleanprompt._crypto` remains available for callers who prefer
the audited primitive.

**The construction, stated so it can be reviewed.**

Nothing here is novel. It is the textbook composition, named at each step:

1. *Key derivation.* :func:`hashlib.scrypt` over the passphrase with a random
   16-byte salt, ``n=2**14, r=8, p=1``, producing 64 bytes. Where ``scrypt`` is
   unavailable — it needs OpenSSL 1.1 — :func:`hashlib.pbkdf2_hmac` with
   SHA-256 and 240,000 iterations is used instead. The document records which
   was used and every parameter, so a vault stays readable when the defaults
   here change.
2. *Key separation.* The 64 bytes are split: the first 32 encrypt, the last 32
   authenticate. One secret is never used for two purposes.
3. *Encryption.* A keystream from HMAC-SHA256 in counter mode —
   ``HMAC(k_enc, nonce || counter)`` — exclusive-ored with the plaintext. This
   is a stream cipher whose security reduces to HMAC-SHA256 being a
   pseudorandom function, which is the same assumption every HMAC use makes.
   The nonce is 16 random bytes from :mod:`secrets`, fresh for every value, so
   no keystream is ever reused.
4. *Authentication.* Encrypt-then-MAC: HMAC-SHA256 over the label, the cipher
   name, the nonce and the ciphertext. Verified with
   :func:`hmac.compare_digest` before a single byte is decrypted, so a modified
   vault fails rather than yielding altered values.

Binding the label into the tag matters: without it an attacker with write
access to the vault could swap two tokens and make ``[EMAIL-1]`` restore to a
different person's address, with every authentication check still passing.

**What this does not do.** It does not protect a vault while the process holds
it in memory, it does not help against someone who has the passphrase, and it
does not make key management disappear. It is also not a general-purpose
encryption API and is not offered as one: it encrypts short strings under a
freshly derived key, which is the only thing it is built for.

**On preferring this over Fernet by default.** A vault that cannot be opened is
a worse outcome than a difference between two authenticated constructions that
both rest on standard assumptions. ``--cipher fernet`` selects the audited
backend for anyone who weighs it the other way, and ``--cipher auto`` picks
Fernet when the tier is available.

See Also
--------
scikitplot.cleanprompt._crypto : The Fernet backend, when the tier is present.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
from typing import Any, Mapping

from ._exceptions import CleanPromptError

__all__ = [
    "CIPHERS",
    "DEFAULT_CIPHER",
    "PORTABLE_CIPHER",
    "decrypt_mapping",
    "encrypt_mapping",
    "new_passphrase",
    "resolve_cipher",
]

#: Name recorded in a vault encrypted by this module.
#:
#: Versioned, so that a change to the construction can be introduced without
#: making existing vaults unreadable: a new name is added and the old one keeps
#: decrypting.
PORTABLE_CIPHER = "cleanprompt-hmac-v1"

#: Cipher choices offered on the command line.
CIPHERS = ("auto", "portable", "fernet")

#: What ``--cipher`` does when not given.
#:
#: ``portable`` rather than ``auto``: the reason to encrypt a vault is usually
#: that it will outlive the moment, and a vault that can only be opened where a
#: compiled dependency happens to be installed has traded one risk for another.
DEFAULT_CIPHER = "portable"

#: scrypt parameters. ``n=2**14`` costs roughly 16 MiB and a few tens of
#: milliseconds, which is a sensible floor for an interactive tool and is the
#: value RFC 7914 gives as its interactive example.
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}

#: PBKDF2 iterations, used only where scrypt is unavailable. Chosen to be
#: roughly comparable in wall time on a current machine.
_PBKDF2_ROUNDS = 240_000

#: Derived key material: 32 bytes to encrypt, 32 to authenticate.
_KEY_BYTES = 64

#: Nonce width. 16 random bytes makes a repeat negligible long after any
#: plausible number of values has been encrypted under one key.
_NONCE_BYTES = 16

#: HMAC-SHA256 output width, which is both the tag size and the keystream block.
_BLOCK = 32


def resolve_cipher(choice: str = DEFAULT_CIPHER) -> str:
    """
    Resolve a ``--cipher`` choice into the cipher that will be used.

    Parameters
    ----------
    choice : str, default='portable'
        One of :data:`CIPHERS`.

    Returns
    -------
    str
        ``"portable"`` or ``"fernet"``.

    Raises
    ------
    PolicyError
        If ``choice`` is not a known cipher.

    Notes
    -----
    **Developer notes.** ``auto`` resolves by *usability*, the same rule the
    entity engines follow: an installed-but-broken ``cryptography`` is skipped
    exactly as a missing one, because it encrypts nothing either way. An
    explicitly named cipher is never silently swapped, so ``--cipher fernet``
    on a machine without the tier fails with an install hint rather than
    quietly writing a different format than the one that was asked for.

    Examples
    --------
    >>> resolve_cipher("portable")
    'portable'
    """
    from ._exceptions import PolicyError  # ruff: ignore[import-outside-top-level]

    if choice not in CIPHERS:
        msg = "unknown cipher {!r}; choose from {}".format(
            choice,
            ", ".join(CIPHERS),
        )
        raise PolicyError(msg)
    if choice != "auto":
        return choice

    from ._capabilities import probe  # ruff: ignore[import-outside-top-level]

    return "fernet" if probe("crypto").available else "portable"


#: Alphabet for generated passphrases: digits and upper case, minus the
#: characters people mistake for one another when reading a key off a screen
#: or dictating it — ``I``, ``L``, ``O``, ``U``, ``0`` and ``1``. This is the
#: Crockford base-32 set, and 26 symbols leave just over 4.7 bits each.
_PASSPHRASE_ALPHABET = (  # lint
    "23456789ABCDEFGHJKMNPQRSTVWXYZ"  # ruff: ignore[hardcoded-password-string]
)


def new_passphrase(groups: int = 4, width: int = 5) -> str:
    """
    Return a strong passphrase, using only the standard library.

    Parameters
    ----------
    groups : int, default=4
        Number of dash-separated groups.
    width : int, default=5
        Characters per group.

    Returns
    -------
    str
        For example ``'K7M2Q-4XTBH-9RJDN-PW3FS'``.

    Raises
    ------
    ValueError
        If either count is below one.

    Notes
    -----
    **Developer notes.** Grouped rather than one long run so it can be read
    aloud, typed from a screen and checked by eye. The default is twenty
    characters from a thirty-symbol alphabet, a little under 98 bits — beyond
    brute force, and short enough that people copy it instead of inventing
    something memorable, which is the failure this function exists to prevent.

    The alphabet excludes look-alike characters. An earlier version used
    :func:`secrets.token_urlsafe`, whose alphabet contains the ``-`` used as
    the separator, so the groups were not the count that was asked for; the
    doctest below is what caught it.

    Examples
    --------
    >>> phrase = new_passphrase()
    >>> len(phrase.split("-")), len(phrase)
    (4, 23)
    >>> set(phrase) <= set(_PASSPHRASE_ALPHABET + "-")
    True
    """
    if groups < 1 or width < 1:
        raise ValueError("groups and width must each be at least 1")
    return "-".join(
        "".join(secrets.choice(_PASSPHRASE_ALPHABET) for _ in range(width))
        for _ in range(groups)
    )


def _scrypt_is_usable() -> bool:
    """
    Return whether this interpreter can actually run :func:`hashlib.scrypt`.

    Notes
    -----
    **Developer notes.** The attribute exists on every supported Python, but
    the function raises where the interpreter was linked against an OpenSSL
    older than 1.1. So presence is not availability, and the only honest test
    is to run it once, cheaply, with throwaway parameters.

    The outcome is a *decision*, not a swallowed failure: either answer leads
    somewhere deliberate, and the document records which function was chosen.
    """
    if not hasattr(hashlib, "scrypt"):
        return False
    try:
        hashlib.scrypt(b"probe", salt=b"probe", n=2, r=1, p=1, dklen=16)
    except Exception:  # noqa: BLE001 - any failure means this build cannot
        return False
    return True


def _kdf_params(salt: bytes | None = None) -> dict[str, Any]:
    """
    Return the key-derivation parameters for a new document.

    Notes
    -----
    **Developer notes.** Every parameter is written into the vault, including
    the function's name, so a document stays readable when the defaults here
    change. A key-derivation function whose parameters live only in the code
    that wrote it makes the next change to that code a silent data loss.
    """
    salt = salt if salt is not None else secrets.token_bytes(16)
    encoded = base64.b64encode(salt).decode("ascii")
    if _scrypt_is_usable():
        return {
            "name": "scrypt",
            "salt": encoded,
            "length": _KEY_BYTES,
            **_SCRYPT,
        }
    return {
        "name": "pbkdf2_hmac_sha256",
        "salt": encoded,
        "length": _KEY_BYTES,
        "rounds": _PBKDF2_ROUNDS,
    }


def _derive(passphrase: bytes, params: Mapping[str, Any]) -> tuple[bytes, bytes]:
    """
    Derive ``(encryption_key, mac_key)`` from a passphrase.

    Raises
    ------
    CleanPromptError
        If the parameters name an unknown function or are malformed. A vault
        that cannot state how its key was derived cannot be decrypted, and
        guessing would produce a wrong key and an unhelpful failure much later.
    """
    try:
        name = str(params["name"])
        salt = base64.b64decode(params["salt"])
        length = int(params.get("length", _KEY_BYTES))
    except (KeyError, TypeError, ValueError) as exc:
        raise CleanPromptError(
            f"vault key-derivation parameters are malformed: {exc}"
        ) from exc

    if length < _KEY_BYTES:
        raise CleanPromptError(
            f"vault asks for {length} bytes of key material; {_KEY_BYTES} are required"
        )

    if name == "scrypt":
        try:
            material = hashlib.scrypt(
                passphrase,
                salt=salt,
                n=int(params["n"]),
                r=int(params["r"]),
                p=int(params["p"]),
                dklen=length,
                maxmem=0,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CleanPromptError(
                f"vault scrypt parameters are malformed: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - attributed, never suppressed
            raise CleanPromptError(
                f"this interpreter cannot run scrypt ({type(exc).__name__}: {exc}), and the vault "
                "was written with it. A build of Python linked against OpenSSL "
                "1.1 or newer can read it."
            ) from exc
    elif name == "pbkdf2_hmac_sha256":
        try:
            material = hashlib.pbkdf2_hmac(
                "sha256", passphrase, salt, int(params["rounds"]), dklen=length
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CleanPromptError(
                f"vault pbkdf2 parameters are malformed: {exc}"
            ) from exc
    else:
        raise CleanPromptError(
            f"vault uses an unknown key-derivation function {name!r}; this build "
            "knows scrypt and pbkdf2_hmac_sha256"
        )

    return material[:_BLOCK], material[_BLOCK:_KEY_BYTES]


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """Return ``length`` bytes of HMAC-SHA256 counter-mode keystream."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(
            key, nonce + struct.pack(">I", counter), hashlib.sha256
        ).digest()
        counter += 1
    return bytes(out[:length])


def _tag(key: bytes, label: str, nonce: bytes, ciphertext: bytes) -> bytes:
    """
    Return the authentication tag over a token and the label it belongs to.

    Notes
    -----
    **Developer notes.** The label is inside the tag. Without it, someone with
    write access to the vault could exchange two tokens and make ``[EMAIL-1]``
    restore to a different person's address with every check still passing —
    a swap that needs no key and would be invisible.
    """
    message = b"|".join(
        (
            PORTABLE_CIPHER.encode("utf-8"),
            label.encode("utf-8"),
            nonce,
            ciphertext,
        )
    )
    return hmac.new(key, message, hashlib.sha256).digest()


def encrypt_mapping(
    mapping: Mapping[str, str], passphrase: bytes
) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Encrypt a label-to-value mapping.

    Parameters
    ----------
    mapping : mapping of str to str
        Labels to values. Labels are not encrypted; see the module notes.
    passphrase : bytes
        The secret, as supplied by the user.

    Returns
    -------
    tokens : dict of str to str
        One base64 token per label.
    params : dict
        Key-derivation parameters to record in the document. Without them the
        vault cannot be decrypted, and they contain no secret.

    Raises
    ------
    CleanPromptError
        If the passphrase is empty.

    Notes
    -----
    **Developer notes.** Every value gets its own nonce, so a corrupted entry
    is detected as *that* entry rather than failing the whole vault, and two
    equal values do not produce equal tokens.

    Examples
    --------
    >>> tokens, params = encrypt_mapping({"[EMAIL-1]": "a@b.co"}, b"secret")
    >>> decrypt_mapping(tokens, b"secret", params)
    {'[EMAIL-1]': 'a@b.co'}
    """
    if not passphrase:
        raise CleanPromptError("a passphrase is required to encrypt a vault")

    params = _kdf_params()
    enc_key, mac_key = _derive(passphrase, params)

    tokens: dict[str, str] = {}
    for label, value in mapping.items():
        plaintext = value.encode("utf-8")
        nonce = secrets.token_bytes(_NONCE_BYTES)
        stream = _keystream(enc_key, nonce, len(plaintext))
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
        token = nonce + ciphertext + _tag(mac_key, label, nonce, ciphertext)
        tokens[label] = base64.b64encode(token).decode("ascii")
    return tokens, params


def decrypt_mapping(
    mapping: Any, passphrase: bytes, params: Mapping[str, Any]
) -> dict[str, str]:
    """
    Decrypt a mapping produced by :func:`encrypt_mapping`.

    Parameters
    ----------
    mapping : mapping of str to str
        Labels to tokens.
    passphrase : bytes
        The secret the vault was written with.
    params : mapping
        The key-derivation parameters recorded in the document.

    Returns
    -------
    dict of str to str
        Labels to values.

    Raises
    ------
    CleanPromptError
        If the document is malformed, or the passphrase is wrong, or any token
        fails authentication. These are deliberately one exception type with
        different messages: which of them happened is not something the caller
        can act on differently, and a wrong passphrase is indistinguishable
        from a tampered vault by design.

    Notes
    -----
    **Developer notes.** The tag is verified before the plaintext is produced,
    so a modified vault cannot yield an altered value even momentarily.
    """
    if not isinstance(mapping, dict):
        raise CleanPromptError("encrypted vault entries are not an object")
    if not passphrase:
        raise CleanPromptError("a passphrase is required to decrypt a vault")

    enc_key, mac_key = _derive(passphrase, params)

    out: dict[str, str] = {}
    for label, token in mapping.items():
        if not isinstance(label, str) or not isinstance(token, str):
            raise CleanPromptError("encrypted vault entry is not a string pair")
        try:
            raw = base64.b64decode(token, validate=True)
        except Exception as exc:  # noqa: BLE001 - attributed, never suppressed
            raise CleanPromptError(f"vault entry {label} is not valid base64") from exc

        if len(raw) < _NONCE_BYTES + _BLOCK:
            raise CleanPromptError(f"vault entry {label} is truncated")

        nonce = raw[:_NONCE_BYTES]
        ciphertext = raw[_NONCE_BYTES : len(raw) - _BLOCK]
        tag = raw[len(raw) - _BLOCK :]

        if not hmac.compare_digest(tag, _tag(mac_key, label, nonce, ciphertext)):
            raise CleanPromptError(
                f"vault entry {label} failed authentication: the passphrase is "
                "wrong, or the vault has been modified since it was "
                "written"
            )

        stream = _keystream(enc_key, nonce, len(ciphertext))
        try:
            out[label] = bytes(a ^ b for a, b in zip(ciphertext, stream)).decode(
                "utf-8",
            )
        # tag makes this unreachable
        except UnicodeDecodeError as exc:  # pragma: no cover
            raise CleanPromptError(
                f"vault entry {label} did not decrypt to text"
            ) from exc
    return out
