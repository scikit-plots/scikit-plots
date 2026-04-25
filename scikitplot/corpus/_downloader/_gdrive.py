# scikitplot/corpus/_downloader/_gdrive.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._downloader._gdrive
========================================
Google Drive share-link downloader.

:class:`GoogleDriveDownloader` accepts any public Google Drive share URL and
resolves it to a direct download URL before streaming the file locally.

Supported URL forms:

* ``https://drive.google.com/file/d/FILE_ID/view?usp=sharing``
* ``https://drive.google.com/file/d/FILE_ID/view``
* ``https://drive.google.com/open?id=FILE_ID``
* ``https://drive.google.com/uc?export=download&id=FILE_ID`` (already direct)

Google Drive "confirm" anti-virus bypass:

For files larger than ~25 MB, Google Drive injects an interstitial page
asking the user to confirm the download.  The ``Content-Disposition`` header
in the initial response carries a ``confirm=xxx`` token.  This class
automatically extracts and replays the token to bypass the gate.

Python compatibility: 3.8 - 3.15+.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path  # noqa: F401

from ._base import BaseDownloader, DownloadResult

logger = logging.getLogger(__name__)

__all__ = ["GoogleDriveDownloader"]

# ---------------------------------------------------------------------------
# URL pattern constants
# ---------------------------------------------------------------------------

_GDRIVE_FILE_RE: re.Pattern[str] = re.compile(
    r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
_GDRIVE_OPEN_RE: re.Pattern[str] = re.compile(
    r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
_GDRIVE_UC_RE: re.Pattern[str] = re.compile(
    r"https?://drive\.google\.com/uc",
    re.IGNORECASE,
)

# Regex to extract the confirm token injected by the virus-warning page
_GDRIVE_CONFIRM_RE: re.Pattern[str] = re.compile(
    r"confirm=([^&;\"']+)",
    re.IGNORECASE,
)


def _extract_gdrive_file_id(input_url: str) -> str:
    """
    Extract the Google Drive file ID from any supported share URL form.

    Parameters
    ----------
    input_url : str
        Google Drive URL.

    Returns
    -------
    str
        File ID string (alphanumeric + ``-_``).

    Raises
    ------
    ValueError
        If the file ID cannot be extracted.
    """
    # ``/file/d/FILE_ID/...``
    m = _GDRIVE_FILE_RE.search(input_url)
    if m:
        return m.group(1)
    # ``/open?id=FILE_ID``
    m = _GDRIVE_OPEN_RE.search(input_url)
    if m:
        return m.group(1)
    # ``/uc?...&id=FILE_ID&...``
    parsed = urllib.parse.urlparse(input_url)
    if _GDRIVE_UC_RE.match(input_url):
        params = urllib.parse.parse_qs(parsed.query)
        file_id = params.get("id", [None])[0]
        if file_id:
            return file_id
    raise ValueError(
        f"GoogleDriveDownloader: cannot extract file ID from {input_url!r}.\n"
        f"Expected:\n"
        f"  https://drive.google.com/file/d/FILE_ID/view?usp=sharing\n"
        f"  https://drive.google.com/open?id=FILE_ID\n"
        f"  https://drive.google.com/uc?export=download&id=FILE_ID"
    )


def _build_download_url(file_id: str) -> str:
    """
    Build the direct-download URL for a Google Drive file ID.

    Parameters
    ----------
    file_id : str
        Google Drive file identifier.

    Returns
    -------
    str
        Direct download URL.
    """
    return f"https://drive.google.com/uc?export=download&id={file_id}"


@dataclass
class GoogleDriveDownloader(BaseDownloader):
    """
    Google Drive share-link downloader.

    Resolves any public Google Drive share URL to a direct download URL
    and streams the file to a local path.  Handles the large-file
    virus-warning interstitial automatically.

    Parameters
    ----------
    input_url : str
        Any supported Google Drive share URL.  See module docstring for
        accepted forms.
    output_path : pathlib.Path or None, optional
        Directory for the downloaded file.  Default: ``None`` (temp dir).
    timeout : float, optional
        HTTP timeout in seconds.  Default: ``30.0``.
    max_bytes : int, optional
        Download size cap in bytes.  Default: ``100 MB``.
    verify_ssl : bool, optional
        Verify TLS certificates.  Default: ``True``.
    block_private_ips : bool, optional
        SSRF prevention.  Default: ``True``.
    max_redirects : int, optional
        Maximum HTTP redirects.  Default: ``5``.

    Raises
    ------
    ValueError
        If the file ID cannot be extracted from ``input_url`` at construction.

    Notes
    -----
    **Large-file bypass:** When Google Drive serves a virus-scan warning
    page instead of the file, this class inspects the response body for
    the ``confirm=`` token, rebuilds the download URL with the token, and
    re-downloads transparently.

    **Private files:** Only publicly shared files are supported.  Files
    that require Google account authentication will raise ``403 Forbidden``
    from Google's servers.  OAuth2 support is a planned future extension.

    Examples
    --------
    >>> dl = GoogleDriveDownloader(
    ...     "https://drive.google.com/file/d/1abc-DEF_xyz/view?usp=sharing"
    ... )
    >>> result = dl.download()
    >>> result.suffix  # determined from Content-Disposition / Content-Type
    '.pdf'

    Already-direct URL form:

    >>> dl = GoogleDriveDownloader(
    ...     "https://drive.google.com/uc?export=download&id=1abc-DEF_xyz"
    ... )
    >>> result = dl.download()
    """

    def __post_init__(self) -> None:
        """Validate that a file ID can be extracted from ``input_url``."""
        super().__post_init__()
        # Eagerly validate — raises ValueError if the URL is not GDrive
        parsed_host = urllib.parse.urlparse(self.input_url).hostname or ""
        if "drive.google.com" not in parsed_host.lower():
            raise ValueError(
                f"GoogleDriveDownloader: input_url must be a drive.google.com URL; "
                f"got {self.input_url!r}."
            )
        # Validate file ID extraction (raises on malformed URLs)
        _extract_gdrive_file_id(self.input_url)

    def resolve_download_url(self) -> str:
        """
        Resolve the share URL to a direct Google Drive download URL.

        Returns
        -------
        str
            Direct download URL with ``?export=download&id=FILE_ID``.

        Examples
        --------
        >>> dl = GoogleDriveDownloader("https://drive.google.com/file/d/1abc-DEF/view")
        >>> dl.resolve_download_url()
        'https://drive.google.com/uc?export=download&id=1abc-DEF'
        """
        file_id = _extract_gdrive_file_id(self.input_url)
        return _build_download_url(file_id)

    def download(self) -> DownloadResult:
        """
        Download the Google Drive file and return a :class:`DownloadResult`.

        Handles the large-file virus-warning interstitial by inspecting
        the first response for a ``confirm=`` token and re-issuing the
        request with the token if needed.

        Returns
        -------
        DownloadResult
            Populated result with local file path, extension, and source URL.

        Raises
        ------
        ValueError
            If SSRF check fails, size exceeds ``max_bytes``, or the file ID
            cannot be extracted.
        requests.HTTPError
            On HTTP 4xx/5xx errors from Google's servers.
        RuntimeError
            If the confirm-bypass loop fails (unexpected response structure).
        """
        import requests  # noqa: PLC0415

        from .._url_handler import (  # noqa: PLC0415
            _infer_extension_from_headers,
            _make_temp_filename,
        )

        if self.block_private_ips:
            # drive.google.com is public CDN — validate anyway
            from .._url_handler import _validate_url_security  # noqa: PLC0415

            _validate_url_security(self.input_url)

        direct_url = self.resolve_download_url()
        dest = self._resolve_dest_dir()

        with requests.Session() as session:
            session.max_redirects = self.max_redirects
            session.headers["User-Agent"] = self.user_agent
            session.verify = self.verify_ssl

            response = session.get(
                direct_url,
                stream=True,
                timeout=self.timeout,
                allow_redirects=True,
            )
            response.raise_for_status()

            # ── Virus-warning interstitial detection ─────────────────────
            # Google serves an HTML page with a "confirm=xxx" token for
            # files > ~25 MB.  Detect by Content-Type: text/html.
            content_type_raw = response.headers.get("Content-Type", "")
            mime = content_type_raw.split(";")[0].strip().lower()
            if mime == "text/html":
                # Read the warning page (it is small — just HTML)
                warning_html = response.text
                confirm_match = _GDRIVE_CONFIRM_RE.search(warning_html)
                if confirm_match:
                    confirm_token = confirm_match.group(1)
                    confirmed_url = f"{direct_url}&confirm={confirm_token}&uuid=stub"
                    logger.debug(
                        "GoogleDriveDownloader: large-file confirm bypass "
                        "(token redacted)."
                    )
                    response = session.get(
                        confirmed_url,
                        stream=True,
                        timeout=self.timeout,
                        allow_redirects=True,
                    )
                    response.raise_for_status()
                    content_type_raw = response.headers.get("Content-Type", "")
                    mime = content_type_raw.split(";")[0].strip().lower()
                else:
                    logger.warning(
                        "GoogleDriveDownloader: received text/html from %s "
                        "but no confirm token found.  File may not be publicly "
                        "shared or may require authentication.",
                        direct_url,
                    )

            ext = _infer_extension_from_headers(response.headers, direct_url)
            filename = _make_temp_filename(direct_url, ext)
            dest_path = dest / filename

            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded += len(chunk)
                        if downloaded > self.max_bytes:
                            f.close()
                            dest_path.unlink(missing_ok=True)
                            raise ValueError(
                                f"GoogleDriveDownloader: download exceeded "
                                f"max_bytes={self.max_bytes} "
                                f"({downloaded} bytes so far) for {direct_url!r}."
                            )
                        f.write(chunk)

        logger.info(
            "GoogleDriveDownloader: %s → %s (%d bytes, ext=%s)",
            self.input_url,
            dest_path.name,
            downloaded,
            ext,
        )
        return DownloadResult(
            input_url=self.input_url,
            output_path=dest_path,
            suffix=ext.lower(),
            content_type=mime,
            suggested_filename=dest_path.name,
        )
