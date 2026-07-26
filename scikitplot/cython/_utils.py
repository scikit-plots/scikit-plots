"""Small utilities."""

from __future__ import annotations

from hashlib import sha256

__all__ = [
    "sanitize",
]


def _sha8(s: str) -> str:
    """Return an 8-hex-char content hash of ``s`` (stable, ASCII)."""
    return sha256(s.encode("utf-8")).hexdigest()[:8]


def sanitize(name: str) -> str:
    """
    Convert an arbitrary string into a valid, collision-resistant module name.

    Parameters
    ----------
    name : str
        Input string (path-like strings allowed).

    Returns
    -------
    str
        Sanitized module-like identifier.  The returned string is guaranteed to
        be a non-empty, valid Python identifier consisting only of ASCII
        alphanumerics and underscores.

    Raises
    ------
    TypeError
        If ``name`` is not a ``str``.

    Notes
    -----
    - Non-ASCII and non-alphanumeric characters (including ``/``, ``-``, ``.``,
      spaces, and any Unicode letter such as ``é``) are replaced with
      underscores, so the result is always pure ASCII (CYTHON-API-003).
    - If the first character of the result would be a digit, a leading
      underscore is prepended so that the output is always a valid identifier.
    - **Collision resistance**: when sanitisation actually alters the input
      (characters were replaced or a prefix added), a short ``_<hash>`` suffix
      derived from the *original* string is appended, so distinct inputs that
      would otherwise map to the same identifier (e.g. ``"a-b"`` and ``"a.b"``)
      get distinct names.  Inputs that are already valid ASCII identifiers are
      returned unchanged (no suffix), preserving backward-compatible names.
    - An empty input string returns ``"_"`` (the minimal valid identifier).

    Examples
    --------
    >>> sanitize("hello_world")  # already valid -> unchanged
    'hello_world'
    >>> sanitize("hello-world")  # altered -> disambiguated
    'hello_world_afa27b44'
    >>> sanitize("")
    '_'
    >>> sanitize("a-b") == sanitize("a.b")  # distinct inputs -> distinct names
    False
    """
    if not isinstance(name, str):
        raise TypeError(f"sanitize() requires a str, got {type(name).__name__!r}")
    # Empty input: return the minimal valid identifier.
    if not name:
        return "_"

    out: list[str] = []
    altered = False
    # Prepend underscore when the first character is a digit so that the
    # result is always a valid Python identifier start character.
    if name[0].isdigit():
        out.append("_")
        altered = True
    for ch in name:
        # ASCII-only alnum + underscore are kept as-is; underscore is a valid
        # identifier character and must NOT count as an alteration (otherwise
        # already-valid names like "hello_world" would be needlessly suffixed).
        # Non-ASCII letters (é, α, ①) are replaced, unlike str.isalnum().  # ruff:ignore[ambiguous-unicode-character-comment]
        if ch == "_" or (ch.isascii() and ch.isalnum()):
            out.append(ch)
        else:
            out.append("_")
            altered = True
    base = "".join(out)

    # Collision resistance: if sanitisation changed the input, append a short
    # hash of the ORIGINAL so distinct inputs don't collapse to one name.
    if altered:
        return f"{base}_{_sha8(name)}"
    return base
