# scikitplot/corpus/_downloader/_github.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._downloader._github
========================================
GitHub URL downloader with automatic blob → raw normalisation.

:class:`GitHubDownloader` handles two GitHub URL forms:

* **Raw URL** — ``https://raw.githubusercontent.com/org/repo/refs/heads/main/README.md``
  or ``https://raw.githubusercontent.com/org/repo/main/file.txt``
  These are passed through unchanged to the HTTP layer.

* **Blob URL** — ``https://github.com/org/repo/blob/main/file.md``
  Rewritten to the equivalent raw URL before downloading.
  GitHub blob pages are HTML renderings; the raw content URL is the
  actual file bytes.

Both forms support an optional ``token`` parameter for private-repository
access via a ``Authorization: Bearer <token>`` header.

Python compatibility: 3.8 - 3.15+.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path  # noqa: F401
from typing import Optional  # noqa: F401

from ._base import BaseDownloader, DownloadResult

logger = logging.getLogger(__name__)

__all__ = ["GitHubDownloader"]

# ---------------------------------------------------------------------------
# URL pattern constants
# ---------------------------------------------------------------------------

# github.com/OWNER/REPO/blob/REF/path/to/file
_GITHUB_BLOB_RE: re.Pattern[str] = re.compile(
    r"^https?://github\.com/([^/]+/[^/]+)/blob/(.+)$",
    re.IGNORECASE,
)

# raw.githubusercontent.com/... (already raw)
_GITHUB_RAW_RE: re.Pattern[str] = re.compile(
    r"^https?://raw\.githubusercontent\.com/",
    re.IGNORECASE,
)

# github.com/OWNER/REPO (root — not a file, but a repo page)
_GITHUB_REPO_ROOT_RE: re.Pattern[str] = re.compile(
    r"^https?://github\.com/[^/]+/[^/]+/?$",
    re.IGNORECASE,
)

# github.com/OWNER/REPO/tree/... (directory listing page — not a file)
_GITHUB_TREE_RE: re.Pattern[str] = re.compile(
    r"^https?://github\.com/[^/]+/[^/]+/tree/",
    re.IGNORECASE,
)


@dataclass
class GitHubDownloader(BaseDownloader):
    """
    GitHub URL downloader with automatic blob → raw normalisation.

    Accepts both ``github.com/.../blob/...`` and
    ``raw.githubusercontent.com/...`` URLs.  Blob URLs are silently
    rewritten to their raw equivalent before downloading.

    Parameters
    ----------
    input_url : str
        GitHub blob or raw URL.
        Accepted forms:

        * ``https://github.com/OWNER/REPO/blob/REF/path/to/file``
        * ``https://raw.githubusercontent.com/OWNER/REPO/REF/path/to/file``
        * ``https://raw.githubusercontent.com/OWNER/REPO/refs/heads/BRANCH/path``
    token : str or None, optional
        GitHub personal access token (PAT) or fine-grained token.
        When provided, sent as ``Authorization: Bearer <token>`` so that
        private repositories can be accessed.
        **Never logged or included in repr.**
        Default: ``None`` (anonymous access, public repos only).
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
        If the URL is not a recognised GitHub blob or raw URL at
        construction time.
    ValueError
        If the URL points to a directory tree (``/tree/``), which is
        not a downloadable file.

    Notes
    -----
    **Blob → raw rewrite rule:**

    .. code-block:: text

        https://github.com/OWNER/REPO/blob/REF/path/to/file.md
              ↓
        https://raw.githubusercontent.com/OWNER/REPO/REF/path/to/file.md

    The ``refs/heads/`` prefix used by the GitHub UI for branch refs is
    preserved when already present in raw URLs and not added for blob URLs
    (blob URLs do not carry it).

    **Private repo access:** Tokens are passed as HTTP headers, never as
    URL query parameters.  Tokens are redacted from all log output.

    Examples
    --------
    Public repo — blob URL:

    >>> dl = GitHubDownloader(
    ...     "https://github.com/scikit-plots/scikit-plots/blob/main/README.md"
    ... )
    >>> result = dl.download()
    >>> result.suffix
    '.md'

    Public repo — raw URL:

    >>> dl = GitHubDownloader(
    ...     "https://raw.githubusercontent.com/scikit-plots/scikit-plots"
    ...     "/refs/heads/main/README.md"
    ... )
    >>> result = dl.download()

    Private repo with PAT:

    >>> dl = GitHubDownloader(
    ...     "https://github.com/myorg/private-repo/blob/main/data.csv",
    ...     token="ghp_xxxxxxxxxxxxxxxxxxxx",
    ... )
    >>> result = dl.download()
    """

    token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """
        Validate that ``input_url`` is a recognised GitHub URL form.

        Raises
        ------
        ValueError
            If the URL is a GitHub tree (directory) URL.
        ValueError
            If the URL is a GitHub domain URL but not a blob or raw URL.
        """
        super().__post_init__()

        parsed = urllib.parse.urlparse(self.input_url)
        host = (parsed.hostname or "").lower()
        is_github_host = host in ("github.com", "raw.githubusercontent.com")

        if not is_github_host:
            raise ValueError(
                f"GitHubDownloader: input_url must be a github.com or "
                f"raw.githubusercontent.com URL; got {self.input_url!r}.\n"
                f"For generic HTTP/HTTPS downloads use WebDownloader."
            )
        if _GITHUB_TREE_RE.match(self.input_url):
            raise ValueError(
                f"GitHubDownloader: input_url {self.input_url!r} points to a directory "
                f"tree page, not a downloadable file.  To download an entire "
                f"directory, use the GitHub API or a ZIP archive URL:\n"
                f"  https://github.com/OWNER/REPO/archive/refs/heads/BRANCH.zip"
            )
        if _GITHUB_REPO_ROOT_RE.match(self.input_url):
            raise ValueError(
                f"GitHubDownloader: input_url {self.input_url!r} is a repository root "
                f"page.  Provide a blob or raw file URL."
            )
        if not (
            _GITHUB_BLOB_RE.match(self.input_url)
            or _GITHUB_RAW_RE.match(self.input_url)
        ):
            raise ValueError(
                f"GitHubDownloader: url {self.input_url!r} is not a recognised "
                f"GitHub blob or raw URL.\n"
                f"Expected:\n"
                f"  https://github.com/OWNER/REPO/blob/REF/path/to/file\n"
                f"  https://raw.githubusercontent.com/OWNER/REPO/REF/path/to/file"
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def resolve_raw_url(self) -> str:
        """
        Normalise a GitHub blob URL to its raw.githubusercontent.com equivalent.

        Returns
        -------
        str
            Raw content URL.  If the input is already a raw URL, it is
            returned unchanged.

        Examples
        --------
        >>> dl = GitHubDownloader("https://github.com/user/repo/blob/main/data.csv")
        >>> dl.resolve_raw_url()
        'https://raw.githubusercontent.com/user/repo/main/data.csv'

        >>> dl2 = GitHubDownloader(
        ...     "https://raw.githubusercontent.com/user/repo/main/data.csv"
        ... )
        >>> dl2.resolve_raw_url() == dl2.input_url
        True
        """
        if _GITHUB_RAW_RE.match(self.input_url):
            return self.input_url  # already raw

        match = _GITHUB_BLOB_RE.match(self.input_url)
        if match is None:  # pragma: no cover — guarded in __post_init__
            raise ValueError(
                f"GitHubDownloader.resolve_raw_url: cannot parse {self.input_url!r}."
            )
        repo = match.group(1)  # "owner/repo"
        rest = match.group(2)  # "main/path/to/file.txt"
        return f"https://raw.githubusercontent.com/{repo}/{rest}"

    # ------------------------------------------------------------------
    # BaseDownloader contract
    # ------------------------------------------------------------------

    def download(self) -> DownloadResult:
        """
        Download the GitHub file and return a :class:`DownloadResult`.

        The blob URL (if given) is normalised to a raw URL first, then
        downloaded via :func:`~scikitplot.corpus._url_handler.download_url`
        with an optional ``Authorization`` header for private repos.

        Returns
        -------
        DownloadResult
            Populated result with local file path, extension, and source URL.

        Raises
        ------
        ValueError
            If SSRF check fails or size exceeds ``max_bytes``.
        urllib.error.URLError
            On network errors.

        Notes
        -----
        The ``source_url`` in the returned :class:`DownloadResult` is always
        the *original* URL passed at construction time, not the resolved raw
        URL.  This preserves the provenance label shown to end users.
        """
        import requests  # noqa: PLC0415

        raw_url = self.resolve_raw_url()

        if self.block_private_ips:
            # raw.githubusercontent.com is a CDN — validate hostname
            from .._url_handler import _validate_url_security  # noqa: PLC0415

            _validate_url_security(raw_url)

        dest = self._resolve_dest_dir()

        # Build headers — never log the token value
        req_headers: dict[str, str] = {"User-Agent": self.user_agent}
        if self.token:
            req_headers["Authorization"] = f"Bearer {self.token}"
            logger.debug(
                "GitHubDownloader: using token authentication (token redacted)."
            )

        with requests.Session() as session:
            session.max_redirects = self.max_redirects
            session.headers.update(req_headers)
            session.verify = self.verify_ssl

            response = session.get(
                raw_url,
                stream=True,
                timeout=self.timeout,
                allow_redirects=True,
            )
            response.raise_for_status()

            from .._url_handler import (  # noqa: PLC0415
                _infer_extension_from_headers,
                _make_temp_filename,
            )

            ext = _infer_extension_from_headers(response.headers, raw_url)
            filename = _make_temp_filename(raw_url, ext)
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
                                f"GitHubDownloader: download exceeded "
                                f"max_bytes={self.max_bytes} "
                                f"({downloaded} bytes so far) for {raw_url!r}."
                            )
                        f.write(chunk)

        logger.info(
            "GitHubDownloader: %s → %s (%d bytes, ext=%s)",
            raw_url,
            dest_path.name,
            downloaded,
            ext,
        )
        return DownloadResult(
            input_url=self.input_url,  # original URL, not the resolved raw URL
            output_path=dest_path,
            suffix=ext.lower(),
            content_type=response.headers.get("Content-Type", "")
            .split(";")[0]
            .strip()
            .lower(),
            suggested_filename=dest_path.name,
        )
