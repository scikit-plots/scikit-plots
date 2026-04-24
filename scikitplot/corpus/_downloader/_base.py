# scikitplot/corpus/_downloader/_base.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._downloader._base
=====================================
Abstract base class and shared contracts for all ``[xxx]Downloader`` classes.

Design Invariants:

* **Dataclass-based.** Every concrete downloader is a ``@dataclass`` so all
  parameters are explicit, introspectable, and repr-friendly.
* **Security-first defaults.**  SSRF prevention, scheme allowlist, SSL
  verification, size cap, and redirect cap are all enabled by default.
  Users must explicitly opt-out via named parameters — there are no silent
  downgrades.
* **Context-manager lifecycle.**  ``with WebDownloader(...) as dl:`` cleans
  up any owned temporary directory on exit.  When used without a context
  manager the caller is responsible for calling :meth:`BaseDownloader.cleanup`.
* **DownloadResult carries all metadata** a caller needs to dispatch the file
  to the correct reader (extension, MIME type, suggested filename).  This
  decouples download from dispatch.
* **No code duplication** with ``_url_handler``.  Security helpers
  (:func:`~._url_handler._validate_url_security`,
  :func:`~._url_handler._is_private_ip`) are imported from there.

Python compatibility: 3.8 - 3.15+.
"""  # noqa: D205, D400

from __future__ import annotations

import abc
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional  # noqa: F401

from typing_extensions import Self

logger = logging.getLogger(__name__)

__all__ = [
    "BaseDownloader",
    "DownloadResult",
]

# ---------------------------------------------------------------------------
# DownloadResult — immutable transfer object from downloader → caller
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadResult:
    """
    Immutable result object returned by every :class:`BaseDownloader`.

    Parameters
    ----------
    input_url : str
        The original URL that was downloaded.
    output_path : pathlib.Path
        Absolute path to the downloaded local file.  The file exists and is
        readable when this object is returned.
    suffix : str
        File extension including the leading dot, e.g. ``".pdf"``.  Always
        lower-cased.  Used by :meth:`DocumentReader.create` to dispatch the
        file to the correct reader.  May differ from ``path.suffix`` when
        the extension was inferred from ``Content-Type`` or
        ``Content-Disposition`` rather than the URL path.
    content_type : str, optional
        MIME type from the HTTP ``Content-Type`` header, lower-cased and
        stripped of parameters (e.g. ``"application/pdf"``).
        Empty string when not available (local files, custom handlers).
    suggested_filename : str, optional
        Filename suggested by the server via ``Content-Disposition``, or the
        last URL path segment.  Empty string when not available.

    Notes
    -----
    **Why carry** ``suffix`` **separately from** ``output_path.suffix``?  The URL
    path may have no extension (API endpoints, GDrive share links).  In those
    cases the downloader infers the correct suffix from the ``Content-Type``
    header and stores a file with a synthetic name.  ``suffix`` is the
    *authoritative* extension; ``output_path.suffix`` is implementation detail.

    Examples
    --------
    >>> from pathlib import Path
    >>> r = DownloadResult(
    ...     input_url="https://example.com/paper",
    ...     output_path=Path("/tmp/skplt_abc123.pdf"),
    ...     suffix=".pdf",
    ...     content_type="application/pdf",
    ...     suggested_filename="paper.pdf",
    ... )
    >>> r.suffix
    '.pdf'
    """

    input_url: str
    output_path: Path
    suffix: str
    content_type: str = ""
    suggested_filename: str = ""


# ---------------------------------------------------------------------------
# BaseDownloader — abstract contract for all concrete downloaders
# ---------------------------------------------------------------------------

_DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (compatible; scikitplot-corpus/1.0; "
    "+https://github.com/scikit-plots/scikit-plots)"
)

_DEFAULT_MAX_BYTES: int = 100 * 1024 * 1024  # 100 MB — conservative default
_DEFAULT_TIMEOUT: float = 30.0
_DEFAULT_MAX_REDIRECTS: int = 5
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# ---------------------------------------------------------------------------
# Parameter normalization utility
# ---------------------------------------------------------------------------


def _coerce_param(
    val,
    n,
    *,
    name,
    default,
    allow_none_items=False,
):
    """
    Normalize a scalar / list / None parameter to a list of length n.

    T | list[T] | None  →  list[T] of length n.

    Parameters
    ----------
    val : T or list[T] or None
        Raw value: None uses default, scalar is broadcast, list is used as-is.
    n : int
        Expected list length (number of URLs).
    name : str
        Parameter name for error messages.
    default : T
        Substituted when val is None.
    allow_none_items : bool, optional
        Allow None entries inside a list (e.g. per-URL optional tokens).
        Default: False.

    Returns
    -------
    list
        Normalised list of exactly length n.

    Raises
    ------
    ValueError
        If val is a list of wrong length, or contains None and allow_none_items=False.

    Examples
    --------
    >>> _coerce_param(None, 3, name="timeout", default=30.0)
    [30.0, 30.0, 30.0]
    >>> _coerce_param(60.0, 3, name="timeout", default=30.0)
    [60.0, 60.0, 60.0]
    >>> _coerce_param([10.0, 20.0, 30.0], 3, name="timeout", default=30.0)
    [10.0, 20.0, 30.0]
    >>> _coerce_param(
    ...     [None, "tok"], 2, name="token", default=None, allow_none_items=True
    ... )
    [None, 'tok']
    """
    if val is None:
        return [default] * n
    if isinstance(val, list):
        if len(val) != n:
            raise ValueError(
                f"_coerce_param: '{name}' list length {len(val)} "
                f"does not match the number of URLs ({n}). "
                f"Supply a scalar to broadcast to all URLs, or a list "
                f"of exactly {n} values."
            )
        if not allow_none_items:
            for i, item in enumerate(val):
                if item is None:
                    raise ValueError(
                        f"_coerce_param: '{name}[{i}]' is None but "
                        f"allow_none_items=False. Supply a concrete value."
                    )
        return list(val)
    return [val] * n


@dataclass
class BaseDownloader(abc.ABC):
    """
    Abstract base class for all format-specific URL downloaders.

    Mirrors the :class:`~scikitplot.corpus._base.DocumentReader` design —
    a ``@dataclass`` ABC so all parameters are explicit and subclasses add
    only what they specialise.

    Parameters
    ----------
    input_url : str
        Fully-qualified HTTP/HTTPS URL to download.
        Validated in :meth:`__post_init__`.
    output_path : pathlib.Path or None, optional
        Directory to write the downloaded file into.  If ``None``, a fresh
        temporary directory is created on the first :meth:`download` call and
        owned by this instance (cleaned up on :meth:`cleanup` / context-manager
        exit).  Default: ``None``.
    timeout : float, optional
        HTTP connection + read timeout in seconds.  Default: ``30.0``.
    max_bytes : int, optional
        Maximum acceptable download size in bytes.  Downloads that exceed
        this limit are aborted and the partial file is deleted.
        Default: ``100 * 1024 * 1024`` (100 MB).
    verify_ssl : bool, optional
        Verify TLS/SSL certificates.  **Never set to** ``False`` **in
        production** — doing so silently disables MITM protection.
        Default: ``True``.
    block_private_ips : bool, optional
        Resolve the hostname before connecting and refuse to connect if any
        resolved address is RFC-1918 private, loopback, link-local, or
        reserved.  This is the primary SSRF defence.  Default: ``True``.
    max_redirects : int, optional
        Maximum number of HTTP 3xx redirects to follow.  Default: ``5``.
    user_agent : str, optional
        Value for the ``User-Agent`` HTTP request header.
        Default: scikitplot corpus bot string.

    Attributes
    ----------
    _tmp_dir : pathlib.Path or None
        Temporary directory created by this instance, if any.  ``None`` when
        *output_path* was supplied by the caller.

    Notes
    -----
    **Subclassing contract:**

    1. Decorate the subclass with ``@dataclass``.
    2. Call ``super().__post_init__()`` explicitly (or rely on the MRO if
       using cooperative multiple inheritance).
    3. Override :meth:`download` and call ``self._resolve_dest_dir()`` to
       obtain the write destination before streaming bytes to disk.
    4. Never log credentials (tokens, passwords) at any log level.

    **Security checklist enforced in** :meth:`__post_init__`:

    * Scheme must be ``http`` or ``https`` — no ``file://``, ``ftp://``, etc.
    * Hostname must not be empty.
    * (At download time) hostname is resolved and checked against private
      ranges when ``block_private_ips=True``.

    See Also
    --------
    scikitplot.corpus._downloader._web.WebDownloader :
        Generic HTTP/HTTPS downloader.
    scikitplot.corpus._downloader._github.GitHubDownloader :
        GitHub blob / raw URL downloader with automatic normalisation.
    scikitplot.corpus._downloader._gdrive.GoogleDriveDownloader :
        Google Drive share-link downloader.
    scikitplot.corpus._downloader._youtube.YouTubeDownloader :
        YouTube transcript downloader.
    scikitplot.corpus._downloader._downloader.AnyDownloader :
        Auto-dispatching downloader — routes to the correct specialist.
    scikitplot.corpus._downloader._downloader.CustomDownloader :
        User-supplied callable as a downloader.

    Examples
    --------
    Subclassing (minimal):

    >>> @dataclass
    ... class EchoDownloader(BaseDownloader):
    ...     def download(self) -> DownloadResult:
    ...         dest = self._resolve_dest_dir() / "echo.txt"
    ...         dest.write_text(self.input_url)
    ...         return DownloadResult(
    ...             input_url=self.input_url, output_path=dest, suffix=".txt"
    ...         )

    Context-manager usage (automatic cleanup):

    >>> with WebDownloader("https://example.com/doc.pdf") as dl:
    ...     result = dl.download()
    ...     reader = DocumentReader.create(result.path)
    """

    input_url: str
    output_path: Path | None = field(default=None, repr=False)
    timeout: float = _DEFAULT_TIMEOUT
    max_bytes: int = _DEFAULT_MAX_BYTES
    verify_ssl: bool = True
    block_private_ips: bool = True
    max_redirects: int = _DEFAULT_MAX_REDIRECTS
    user_agent: str = field(default=_DEFAULT_USER_AGENT, repr=False)

    # Internal: temporary directory owned by this instance.
    # Set in _resolve_dest_dir(); None until first download.
    _tmp_dir: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate ``input_url`` eagerly so invalid inputs fail at construction time.

        Raises
        ------
        TypeError
            If ``input_url`` is not a ``str``.
        ValueError
            If ``input_url`` does not start with ``http://`` or ``https://``.
        ValueError
            If ``input_url`` has no hostname component.
        ValueError
            If ``timeout`` or ``max_bytes`` or ``max_redirects`` are not
            positive numbers.
        """
        import urllib.parse  # noqa: PLC0415

        if not isinstance(self.input_url, str):
            raise TypeError(
                f"{type(self).__name__}: input_url must be a str; "
                f"got {type(self.input_url).__name__!r}."
            )
        import re  # noqa: PLC0415

        if not re.match(r"https?://", self.input_url, re.IGNORECASE):
            raise ValueError(
                f"{type(self).__name__}: input_url must start with 'http://' or "
                f"'https://'; got {self.input_url!r}."
            )
        parsed = urllib.parse.urlparse(self.input_url)
        scheme = parsed.scheme.lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise ValueError(
                f"{type(self).__name__}: unsupported scheme {scheme!r}. "
                f"Allowed: {sorted(_ALLOWED_SCHEMES)}."
            )
        if not parsed.hostname:
            raise ValueError(
                f"{type(self).__name__}: input_url has no hostname: {self.input_url!r}."
            )
        if self.timeout <= 0:
            raise ValueError(
                f"{type(self).__name__}: timeout must be > 0; got {self.timeout!r}."
            )
        if self.max_bytes <= 0:
            raise ValueError(
                f"{type(self).__name__}: max_bytes must be > 0; got {self.max_bytes!r}."
            )
        if self.max_redirects < 0:
            raise ValueError(
                f"{type(self).__name__}: max_redirects must be >= 0; "
                f"got {self.max_redirects!r}."
            )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def download(self) -> DownloadResult:
        """
        Download the resource and return a :class:`DownloadResult`.

        Returns
        -------
        DownloadResult
            Populated result object.  ``result.path`` is a readable local
            file; the caller must not delete it while using it.

        Raises
        ------
        ValueError
            On SSRF violation, size exceeded, unsupported scheme.
        OSError
            On filesystem errors (no space, permission denied).
        urllib.error.URLError
            On network errors.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------

    def _resolve_dest_dir(self) -> Path:
        """
        Return the write-destination directory, creating a temp dir if needed.

        Returns
        -------
        pathlib.Path
            Existing, writable directory.

        Notes
        -----
        When ``output_path`` was not supplied at construction time, a fresh
        ``tempfile.mkdtemp`` directory is created on first call and stored
        in ``self._tmp_dir``.  Subsequent calls return the same directory.
        On :meth:`cleanup`, the owned temp dir is removed recursively.
        """
        if self.output_path is not None:
            dest = Path(self.output_path)
            dest.mkdir(parents=True, exist_ok=True)
            return dest

        if self._tmp_dir is None:
            cls_tag = type(self).__name__.lower()[:8]
            object.__setattr__(
                self,
                "_tmp_dir",
                Path(tempfile.mkdtemp(prefix=f"skplt_{cls_tag}_")),
            )
        return self._tmp_dir  # type: ignore[return-value]

    def _check_ssrf(self) -> None:
        """
        Validate ``self.input_url`` against private/reserved IP ranges.

        Raises
        ------
        ValueError
            If the URL hostname resolves to any private, loopback,
            link-local, or reserved IP address.

        Notes
        -----
        Delegates to :func:`~scikitplot.corpus._url_handler._validate_url_security`
        to keep the security logic in a single authoritative location.
        Only called when ``block_private_ips=True``.
        """
        if not self.block_private_ips:
            return
        from .._url_handler import _validate_url_security  # noqa: PLC0415

        _validate_url_security(self.input_url)

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        """Support ``with downloader as dl:`` usage."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Clean up owned temporary directory on context exit."""
        self.cleanup()

    def cleanup(self) -> None:
        """
        Remove the temporary directory owned by this instance, if any.

        Safe to call multiple times.  If ``output_path`` was supplied at
        construction time (caller-owned), this method is a no-op.
        """
        tmp = self._tmp_dir
        if tmp is not None and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            logger.debug("%s.cleanup: removed %s", type(self).__name__, tmp)
            object.__setattr__(self, "_tmp_dir", None)
