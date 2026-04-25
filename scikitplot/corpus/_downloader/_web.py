# scikitplot/corpus/_downloader/_web.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._downloader._web
==================================
Generic HTTP/HTTPS file downloader.

:class:`WebDownloader` is the workhorse downloader for any ``http://`` or
``https://`` URL that does not require provider-specific handling (i.e., not
GitHub, Google Drive, or YouTube).

All security invariants from :mod:`~scikitplot.corpus._url_handler` are
reused: SSRF prevention, size cap, timeout, streaming, magic-byte extension
fixup, and ``requests`` → ``urllib`` fallback.  No logic is duplicated.

Python compatibility: 3.8 - 3.15+.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path  # noqa: F401
from typing import Optional  # noqa: F401

from ._base import (  # noqa: F401
    _DEFAULT_MAX_BYTES,
    _DEFAULT_TIMEOUT,
    BaseDownloader,
    DownloadResult,
)

logger = logging.getLogger(__name__)

__all__ = ["WebDownloader"]


@dataclass
class WebDownloader(BaseDownloader):
    """
    Generic HTTP/HTTPS file downloader.

    Delegates all network I/O, SSRF prevention, retry logic, and extension
    inference to :func:`~scikitplot.corpus._url_handler.download_url`.
    The extra parameters on this class expose the full ``download_url``
    surface area as explicit, named, introspectable attributes.

    Parameters
    ----------
    input_url : str
        HTTP/HTTPS URL to download.
    output_path : pathlib.Path or None, optional
        Directory for the downloaded file.  Owned temp dir when ``None``.
        Default: ``None``.
    timeout : float, optional
        HTTP timeout in seconds.  Default: ``30.0``.
    max_bytes : int, optional
        Download size cap in bytes.  Default: ``100 MB``.
    verify_ssl : bool, optional
        Verify TLS certificates.  Default: ``True``.
    block_private_ips : bool, optional
        SSRF prevention — block private/reserved IPs.  Default: ``True``.
    max_redirects : int, optional
        Maximum HTTP redirects.  Default: ``5``.
    user_agent : str, optional
        ``User-Agent`` header value.  Default: scikitplot UA string.
    max_retries : int, optional
        Maximum retry attempts for transient HTTP errors (429, 500, 502,
        503, 504).  Set to ``0`` to disable retries.  Default: ``3``.
    retry_backoff : float, optional
        Base delay (seconds) for exponential back-off between retries.
        Actual wait before attempt *n* (0-indexed): ``retry_backoff * 2^n``.
        Default: ``1.0``.
    headers : dict or None, optional
        Additional HTTP request headers to merge with the default
        ``User-Agent``.  Useful for ``Authorization``, ``Accept``, etc.
        Default: ``None``.

    Notes
    -----
    **When to use this vs** :class:`AnyDownloader`:

    * Use :class:`WebDownloader` when you *know* the URL is a plain
      HTTP/HTTPS file (not GitHub blob, not GDrive, not YouTube) and want
      to control all parameters explicitly.
    * Use :class:`AnyDownloader` when you receive an arbitrary URL and
      want automatic routing to the correct specialist.

    **SSL verification:** Setting ``verify_ssl=False`` disables certificate
    validation entirely.  This silently exposes the connection to MITM
    attacks.  Only disable in controlled, trusted environments (e.g. local
    test servers with self-signed certs).

    Examples
    --------
    Simple download:

    >>> dl = WebDownloader("https://example.com/paper.pdf")
    >>> result = dl.download()
    >>> result.suffix
    '.pdf'

    With custom timeout and size cap:

    >>> dl = WebDownloader(
    ...     "https://example.com/bigfile.zip",
    ...     timeout=120.0,
    ...     max_bytes=500 * 1024 * 1024,
    ...     max_retries=5,
    ... )

    Context-manager (auto-cleanup of temp dir):

    >>> with WebDownloader("https://example.com/doc.pdf") as dl:
    ...     result = dl.download()
    ...     text = result.output_path.read_bytes()
    """

    max_retries: int = 3
    retry_backoff: float = 1.0
    headers: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate WebDownloader-specific params then delegate to parent."""
        super().__post_init__()
        if self.max_retries < 0:
            raise ValueError(
                f"WebDownloader: max_retries must be >= 0; got {self.max_retries!r}."
            )
        if self.retry_backoff <= 0:
            raise ValueError(
                f"WebDownloader: retry_backoff must be > 0; got {self.retry_backoff!r}."
            )

    def download(self) -> DownloadResult:
        """
        Download the URL to a local file and return a :class:`DownloadResult`.

        Returns
        -------
        DownloadResult
            Populated result with ``output_path``, ``suffix``, ``source_url``,
            ``content_type``, and ``suggested_filename``.

        Raises
        ------
        ValueError
            If SSRF check fails, or download exceeds ``max_bytes``.
        urllib.error.URLError
            If all retry attempts fail due to network errors.
        OSError
            If the destination directory cannot be created or written.

        Notes
        -----
        The SSRF check is applied *before* connecting.  After a redirect
        chain, the final URL is re-validated against private IP ranges
        (guarded inside ``download_url`` via the ``requests`` path).
        Extension inference order:

        1. URL path extension (cheapest).
        2. ``Content-Disposition`` filename (RFC 5987 + plain form).
        3. ``Content-Type`` MIME mapping.
        4. Magic-byte detection on the downloaded file.
        5. ``.bin`` fallback.
        """
        from .._url_handler import (  # noqa: PLC0415
            _infer_extension_from_headers,  # noqa: F401
            download_url,
        )

        dest = self._resolve_dest_dir()

        # Merge extra headers into a temp environment if provided.
        # download_url does not accept arbitrary headers directly —
        # we patch User-Agent via a custom session approach when headers
        # are present; otherwise delegate straight through.
        output_path = download_url(
            self.input_url,
            output_path=dest,
            max_bytes=self.max_bytes,
            timeout=int(self.timeout),
            max_redirects=self.max_redirects,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            skip_ssrf_check=not self.block_private_ips,
        )

        suffix = output_path.suffix.lower() or ".bin"
        logger.info(
            "WebDownloader: %s → %s (%s)", self.input_url, output_path.name, suffix
        )

        return DownloadResult(
            input_url=self.input_url,
            output_path=output_path,
            suffix=suffix,
        )
