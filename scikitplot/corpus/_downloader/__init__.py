# scikitplot/corpus/_downloader/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._downloader
==============================
Composable, security-hardened URL downloaders for the corpus pipeline.

All downloaders share the :class:`BaseDownloader` contract:

* ``@dataclass`` construction — parameters are explicit and introspectable.
* SSRF prevention enabled by default (``block_private_ips=True``).
* SSL verification enabled by default (``verify_ssl=True``).
* Size cap enforced during streaming (``max_bytes``).
* Context-manager lifecycle — ``with XxxDownloader(...) as dl:`` auto-cleans.

Public API:

``DownloadResult``
    Immutable transfer object returned by every downloader's ``.download()``.
``BaseDownloader``
    Abstract base — subclass this to implement a custom protocol.
``WebDownloader``
    Generic HTTP/HTTPS file downloader.
``GitHubDownloader``
    GitHub blob/raw URL downloader with automatic normalisation.
``GoogleDriveDownloader``
    Google Drive share-link downloader with large-file bypass.
``YouTubeDownloader``
    YouTube transcript / audio / video downloader.
``AnyDownloader``
    Auto-dispatching downloader — routes to the right specialist.
``CustomDownloader``
    Wrap any callable as a downloader.
"""  # noqa: D205, D400

from . import (
    _base,
    _downloader,
    _gdrive,
    _github,
    _web,
    _youtube,
)
from ._base import *  # noqa: F403
from ._downloader import *  # noqa: F403
from ._gdrive import *  # noqa: F403
from ._github import *  # noqa: F403
from ._web import *  # noqa: F403
from ._youtube import *  # noqa: F403

__all__ = []
__all__ += _base.__all__
__all__ += _downloader.__all__
__all__ += _gdrive.__all__
__all__ += _github.__all__
__all__ += _web.__all__
__all__ += _youtube.__all__
