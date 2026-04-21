# scikitplot/corpus/_downloader/_downloader.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._downloader._downloader
============================================
High-level dispatcher downloaders.

:class:`AnyDownloader`
    Auto-detects the URL type and routes to the correct specialist.
    Accepts a single URL or a list of URLs.  All parameters support
    ``T | list[T] | None`` — scalar broadcasts to every URL; per-URL list
    applies element-wise; ``None`` uses the parameter's built-in default.

:class:`CustomDownloader`
    Wraps a user-supplied callable as a :class:`BaseDownloader`.

Python compatibility: 3.8 - 3.15+.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union  # noqa: F401

from ._base import (
    _DEFAULT_MAX_BYTES,
    _DEFAULT_MAX_REDIRECTS,
    _DEFAULT_TIMEOUT,
    _DEFAULT_USER_AGENT,
    BaseDownloader,
    DownloadResult,
    _coerce_param,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AnyDownloader",
    "CustomDownloader",
]

# Sentinel for required-but-unfilled fields.
# Because BaseDownloader fields all have defaults, any non-default field in a
# subclass would violate dataclass MRO ordering. We give the field a sentinel
# default and validate eagerly in __post_init__.
_MISSING = object()


@dataclass
class AnyDownloader(BaseDownloader):
    """
    Auto-dispatching downloader with multi-URL and per-parameter list support.

    Accepts one URL **or** a list of URLs.  All parameters support
    ``T | list[T] | None``:

    * ``None``   → use the parameter's built-in default for every URL.
    * Scalar     → broadcast to every URL.
    * ``list``   → applied element-wise; must be the same length as *input_url*.

    Parameters
    ----------
    input_url : str or list[str]
        One URL or a list of URLs.  When a list is supplied,
        :meth:`download` returns ``list[DownloadResult]``.
    output_path : pathlib.Path or None, optional
        Directory shared across all URLs.  Default: ``None`` (temp dir).
    timeout : float or list[float] or None, optional
        HTTP timeout in seconds.  Default: ``30.0``.
    max_bytes : int or list[int] or None, optional
        Download size cap in bytes.  Default: ``100 MB``.
    verify_ssl : bool or list[bool] or None, optional
        Verify TLS certificates.  Default: ``True``.
    block_private_ips : bool or list[bool] or None, optional
        SSRF prevention.  Default: ``True``.
    max_redirects : int or list[int] or None, optional
        Maximum HTTP redirects.  Default: ``5``.
    user_agent : str or list[str] or None, optional
        ``User-Agent`` header value.  Default: scikitplot UA string.
    youtube_mode : str or list[str] or None, optional
        Mode for YouTubeDownloader: ``"transcript"``, ``"audio"``, or
        ``"video"``.  Default: ``"transcript"``.
    youtube_language : str or list[str] or None, optional
        BCP-47 language code for transcript fetching.  Default: ``"en"``.
    youtube_include_auto : bool or list[bool] or None, optional
        Include auto-generated captions as fallback.  Default: ``True``.
    github_token : str or list[str or None] or None, optional
        PAT for GitHubDownloader (private repos).  Per-URL ``None``
        allowed.  **Never logged.**  Default: ``None``.
    headers : dict or list[dict or None] or None, optional
        Extra HTTP headers for WebDownloader.  Per-URL ``None`` allowed.
        Default: ``None``.
    max_retries : int or list[int] or None, optional
        Retry attempts for WebDownloader.  Default: ``3``.
    retry_backoff : float or list[float] or None, optional
        Exponential back-off base for WebDownloader.  Default: ``1.0``.

    Notes
    -----
    **Single vs batch:**

    .. code-block:: python

        # Single URL — returns DownloadResult
        result = AnyDownloader("https://example.com/paper.pdf").download()

        # Batch — returns list[DownloadResult]
        results = AnyDownloader(
            [
                "https://example.com/paper.pdf",
                "https://github.com/org/repo/blob/main/data.csv",
                "https://www.youtube.com/watch?v=abc123",
            ]
        ).download()

    **Per-URL parameters:**

    .. code-block:: python

        dl = AnyDownloader(
            input_url=[
                "https://github.com/org/priv/blob/main/secret.csv",
                "https://example.com/public.pdf",
            ],
            github_token=["ghp_token", None],  # None = public, no token needed
            timeout=[120.0, 30.0],  # per-URL timeouts
            max_bytes=200 * 1024 * 1024,  # broadcast to all
        )
        results = dl.download()

    Examples
    --------
    Single URL:

    >>> dl = AnyDownloader("https://example.com/report.pdf")
    >>> isinstance(dl.download(), DownloadResult)  # doctest: +SKIP
    True

    Batch:

    >>> dl = AnyDownloader(
    ...     ["https://example.com/a.pdf", "https://example.com/b.pdf"],
    ...     timeout=60.0,
    ... )
    >>> len(dl.download())  # doctest: +SKIP
    2
    """

    # All list-capable params declared with defaults so no ordering issues.
    youtube_mode: object = "transcript"
    youtube_language: object = "en"
    youtube_include_auto: object = True
    github_token: object = field(default=None, repr=False)
    headers: object = field(default=None, repr=False)
    max_retries: object = 3
    retry_backoff: object = 1.0

    def __post_init__(self) -> None:
        """
        Validate input_url (str | list[str]) and all list-capable parameters.

        Raises
        ------
        TypeError
            If input_url is not a str or list[str].
        ValueError
            If any URL does not start with http(s)://, or any list param
            has the wrong length, or any numeric param is out of range.
        """
        import re  # noqa: PLC0415

        raw_url = self.input_url
        if isinstance(raw_url, str):
            urls: list = [raw_url]
        elif isinstance(raw_url, list):
            if not raw_url:
                raise ValueError("AnyDownloader: input_url list must not be empty.")
            urls = list(raw_url)
        else:
            raise TypeError(
                f"AnyDownloader: input_url must be str or list[str]; "
                f"got {type(raw_url).__name__!r}."
            )

        for i, u in enumerate(urls):
            if not isinstance(u, str):
                raise TypeError(
                    f"AnyDownloader: input_url[{i}] must be str; "
                    f"got {type(u).__name__!r}."
                )
            if not re.match(r"https?://", u, re.IGNORECASE):
                raise ValueError(
                    f"AnyDownloader: input_url[{i}]={u!r} must start with "
                    f"'http://' or 'https://'."
                )

        n = len(urls)

        # Validate numeric params element-wise after coercion
        def _check_positive(attr, default, label):
            lst = _coerce_param(getattr(self, attr), n, name=attr, default=default)
            for i, v in enumerate(lst):
                if v <= 0:
                    raise ValueError(f"AnyDownloader: {label}[{i}]={v!r} must be > 0.")

        def _check_nonneg(attr, default, label):
            lst = _coerce_param(getattr(self, attr), n, name=attr, default=default)
            for i, v in enumerate(lst):
                if v < 0:
                    raise ValueError(f"AnyDownloader: {label}[{i}]={v!r} must be >= 0.")

        _check_positive("timeout", _DEFAULT_TIMEOUT, "timeout")
        _check_positive("max_bytes", _DEFAULT_MAX_BYTES, "max_bytes")
        _check_nonneg("max_redirects", _DEFAULT_MAX_REDIRECTS, "max_redirects")
        _check_nonneg("max_retries", 3, "max_retries")
        _check_positive("retry_backoff", 1.0, "retry_backoff")

        # Validate youtube_mode values
        _valid_modes = ("transcript", "audio", "video")
        mode_list = _coerce_param(
            self.youtube_mode, n, name="youtube_mode", default="transcript"
        )
        for i, m in enumerate(mode_list):
            if m not in _valid_modes:
                raise ValueError(
                    f"AnyDownloader: youtube_mode[{i}]={m!r} must be one of "
                    f"{_valid_modes}."
                )

        # Coerce remaining params — validates lengths; None items allowed for
        # github_token and headers (per-URL optional values)
        _coerce_param(self.youtube_language, n, name="youtube_language", default="en")
        _coerce_param(
            self.youtube_include_auto, n, name="youtube_include_auto", default=True
        )
        _coerce_param(
            self.github_token,
            n,
            name="github_token",
            default=None,
            allow_none_items=True,
        )
        _coerce_param(
            self.headers, n, name="headers", default=None, allow_none_items=True
        )
        _coerce_param(self.verify_ssl, n, name="verify_ssl", default=True)
        _coerce_param(self.block_private_ips, n, name="block_private_ips", default=True)
        _coerce_param(
            self.user_agent, n, name="user_agent", default=_DEFAULT_USER_AGENT
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _urls(self):
        if isinstance(self.input_url, list):
            return self.input_url
        return [self.input_url]

    def _param_list(self, attr, default, allow_none_items=False):
        """Coerce attribute to list of len(_urls) with per-URL resolution."""
        return _coerce_param(
            getattr(self, attr),
            len(self._urls),
            name=attr,
            default=default,
            allow_none_items=allow_none_items,
        )

    def _build_specialist(self, idx: int) -> BaseDownloader:
        """
        Classify URL at index *idx* and build the appropriate specialist.

        Parameters
        ----------
        idx : int
            Index into ``self._urls``.

        Returns
        -------
        BaseDownloader
            Concrete specialist with all per-URL parameters applied.
        """
        from .._url_handler import URLKind, classify_url  # noqa: PLC0415
        from ._gdrive import GoogleDriveDownloader  # noqa: PLC0415
        from ._github import GitHubDownloader  # noqa: PLC0415
        from ._web import WebDownloader  # noqa: PLC0415
        from ._youtube import YouTubeDownloader  # noqa: PLC0415

        urls = self._urls
        url = urls[idx]
        n = len(urls)

        kind = classify_url(url)
        logger.debug(
            "AnyDownloader[%d/%d]: %s classified as %s",
            idx + 1,
            n,
            url,
            kind.value,
        )

        common = {
            "input_url": url,
            "output_path": self.output_path,
            "timeout": self._param_list("timeout", _DEFAULT_TIMEOUT)[idx],
            "max_bytes": self._param_list("max_bytes", _DEFAULT_MAX_BYTES)[idx],
            "verify_ssl": self._param_list("verify_ssl", True)[idx],
            "block_private_ips": self._param_list("block_private_ips", True)[idx],
            "max_redirects": self._param_list("max_redirects", _DEFAULT_MAX_REDIRECTS)[
                idx
            ],
            "user_agent": self._param_list("user_agent", _DEFAULT_USER_AGENT)[idx],
        }

        if kind in (URLKind.YOUTUBE, URLKind.YOUTUBE_CHANNEL, URLKind.YOUTUBE_PLAYLIST):
            return YouTubeDownloader(
                **common,
                mode=self._param_list("youtube_mode", "transcript")[idx],
                language=self._param_list("youtube_language", "en")[idx],
                include_auto_generated=self._param_list("youtube_include_auto", True)[
                    idx
                ],
            )

        if kind == URLKind.GOOGLE_DRIVE:
            return GoogleDriveDownloader(**common)

        if kind in (URLKind.GITHUB_BLOB, URLKind.GITHUB_RAW):
            return GitHubDownloader(
                **common,
                token=self._param_list("github_token", None, allow_none_items=True)[
                    idx
                ],
            )

        # Default: WebDownloader (DOWNLOADABLE, WEB_PAGE, extensionless, etc.)
        return WebDownloader(
            **common,
            max_retries=self._param_list("max_retries", 3)[idx],
            retry_backoff=self._param_list("retry_backoff", 1.0)[idx],
            headers=self._param_list("headers", None, allow_none_items=True)[idx],
        )

    # ------------------------------------------------------------------
    # BaseDownloader contract
    # ------------------------------------------------------------------

    def download(self):
        """
        Download one URL or all URLs and return the result(s).

        Returns
        -------
        DownloadResult
            When ``input_url`` was a single ``str``.
        list[DownloadResult]
            When ``input_url`` was a ``list[str]``.  Preserves input order.

        Notes
        -----
        Batch downloads are sequential.  For parallel execution, call
        :meth:`download_single` per URL in your own thread/process pool.
        """
        if isinstance(self.input_url, str):
            return self._download_single(0)
        return [self._download_single(i) for i in range(len(self._urls))]

    def download_all(self):
        """
        Download all URLs and always return ``list[DownloadResult]``.

        Normalises the return type so callers never need to branch on
        ``isinstance(result, list)``.

        Returns
        -------
        list[DownloadResult]
            One :class:`DownloadResult` per URL, in input order.

        Examples
        --------
        >>> dl = AnyDownloader("https://example.com/doc.pdf")
        >>> results = dl.download_all()  # doctest: +SKIP
        >>> len(results)
        1
        """
        raw = self.download()
        if isinstance(raw, list):
            return raw
        return [raw]

    def _download_single(self, idx: int) -> DownloadResult:
        """Build specialist for URL at *idx* and call its download()."""
        specialist = self._build_specialist(idx)
        logger.debug(
            "AnyDownloader: input_url[%d] → %s", idx, type(specialist).__name__
        )
        return specialist.download()


@dataclass
class CustomDownloader(BaseDownloader):
    """
    Wraps a user-supplied callable as a :class:`BaseDownloader`.

    Parameters
    ----------
    input_url : str
        HTTP/HTTPS URL passed through to ``handler``.
    handler : callable
        ``handler(input_url: str, output_path: Path, **kwargs) -> Path``.
        Must write content to ``output_path`` and return the path.
        Required — raises ``TypeError`` at construction if not supplied.
    handler_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``handler``.  Default: ``None``.
    output_path : pathlib.Path or None, optional
        Directory for the downloaded file.  Default: ``None`` (temp dir).
    timeout : float, optional
        Forwarded via ``handler_kwargs`` if not already present.
        Default: ``30.0``.
    max_bytes : int, optional
        Forwarded via ``handler_kwargs`` if not already present.
        Default: ``100 MB``.
    verify_ssl : bool, optional
        Verify TLS certificates.  Default: ``True``.
    block_private_ips : bool, optional
        SSRF prevention.  Default: ``True``.

    Raises
    ------
    TypeError
        If ``handler`` is not supplied or not callable.

    Examples
    --------
    >>> from pathlib import Path
    >>> def my_handler(input_url: str, output_path: Path, **kwargs) -> Path:
    ...     out = output_path / "file.txt"
    ...     out.write_text("content")
    ...     return out
    >>> dl = CustomDownloader("https://example.com/f", handler=my_handler)
    >>> result = dl.download()  # doctest: +SKIP
    """

    handler: object = field(default=_MISSING, repr=False)
    handler_kwargs: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate that handler was supplied and is callable."""
        super().__post_init__()
        if self.handler is _MISSING:
            raise TypeError(
                "CustomDownloader: 'handler' is required. "
                "Pass a callable: CustomDownloader(input_url=..., handler=my_fn)."
            )
        if not callable(self.handler):
            raise TypeError(
                f"CustomDownloader: handler must be callable; "
                f"got {type(self.handler).__name__!r}."
            )

    def download(self) -> DownloadResult:
        """
        Invoke the user-supplied ``handler`` and return a :class:`DownloadResult`.

        Returns
        -------
        DownloadResult
            Populated from the path returned by ``handler``.

        Raises
        ------
        ValueError
            If SSRF check fails (``block_private_ips=True``).
        TypeError
            If ``handler`` returns something that cannot be coerced to Path.
        FileNotFoundError
            If the returned path does not exist.
        """
        self._check_ssrf()

        dest = self._resolve_dest_dir()
        kwargs = dict(self.handler_kwargs or {})
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("max_bytes", self.max_bytes)

        raw_path = self.handler(self.input_url, dest, **kwargs)  # type: ignore[operator]

        if not isinstance(raw_path, Path):
            try:
                raw_path = Path(raw_path)
            except TypeError as exc:
                raise TypeError(
                    f"CustomDownloader: handler must return a Path or str; "
                    f"got {type(raw_path).__name__!r}."
                ) from exc

        if not raw_path.exists():
            raise FileNotFoundError(
                f"CustomDownloader: handler returned {raw_path!r} "
                f"but the file does not exist."
            )

        suffix = raw_path.suffix.lower() or ".bin"
        logger.info(
            "CustomDownloader: %s → %s (via %s)",
            self.input_url,
            raw_path.name,
            getattr(self.handler, "__name__", repr(self.handler)),
        )
        return DownloadResult(
            input_url=self.input_url,
            output_path=raw_path,
            suffix=suffix,
        )
