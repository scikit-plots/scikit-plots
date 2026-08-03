# scikitplot/corpus/_resources.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Offline-safe access to optional data resources (NLTK).

Local processing must not reach the network unless explicitly authorized. This
module centralises access to optional NLTK data corpora so that a missing
resource produces an actionable capability error by default instead of an
implicit ``nltk.download`` call (the CORPUS-RES-001 defect: hangs, network-policy
violations, nondeterministic CI, and browser/WASM incompatibility).

Policy
------
Managed downloads are **disabled by default**. They are enabled only when the
caller passes ``allow_download=True`` or sets the environment variable
``SCIKITPLOT_CORPUS_ALLOW_DOWNLOADS`` to a truthy value (``1``/``true``/``yes``/
``on``). When downloads are disabled and a resource is missing,
:func:`ensure_nltk_resource` raises :class:`ResourceUnavailableError` with
one-time install instructions and performs **no** network access.

Preflight
---------
:func:`nltk_resource_available` reports whether a resource is present without
ever downloading, so callers and CI can surface missing resources up front.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

__all__ = [
    "ResourceUnavailableError",
    "downloads_allowed",
    "ensure_nltk_resource",
    "nltk_resource_available",
]

#: Environment variable that authorizes managed data downloads when truthy.
ENV_ALLOW_DOWNLOADS = "SCIKITPLOT_CORPUS_ALLOW_DOWNLOADS"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class ResourceUnavailableError(RuntimeError):
    """A required optional data resource is missing and downloads are disabled."""


def downloads_allowed(explicit: bool | None = None) -> bool:
    """Return whether managed data downloads are permitted.

    Resolution order: the ``explicit`` argument (when not ``None``), then the
    :data:`ENV_ALLOW_DOWNLOADS` environment variable, then the offline-safe
    default ``False``.

    Parameters
    ----------
    explicit : bool or None, optional
        Caller override. ``None`` (default) defers to the environment/default.

    Returns
    -------
    bool
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(ENV_ALLOW_DOWNLOADS, "").strip().lower() in _TRUTHY


def nltk_resource_available(find_path: str) -> bool:
    """Preflight check: is an NLTK resource present? Never downloads.

    Parameters
    ----------
    find_path : str
        NLTK data path, e.g. ``"corpora/stopwords"`` or
        ``"tokenizers/punkt_tab"``.

    Returns
    -------
    bool
        ``True`` if NLTK is importable and the resource is found; ``False`` if
        NLTK is not installed or the resource is missing. No network access.
    """
    try:
        import nltk  # type: ignore[import]  # noqa: PLC0415
    except ImportError:
        return False
    try:
        nltk.data.find(find_path)
        return True
    except LookupError:
        return False


def ensure_nltk_resource(
    find_path: str,
    download_name: str,
    *,
    allow_download: bool | None = None,
    extra_hint: str = "",
    **kwargs: any,
) -> None:
    """Ensure an NLTK data resource is available WITHOUT implicit downloads.

    If the resource is present, returns immediately. If it is missing and
    downloads are authorized (via ``allow_download`` or the environment), a
    managed download is performed. Otherwise a :class:`ResourceUnavailableError`
    is raised with actionable instructions and no network access occurs.

    Parameters
    ----------
    find_path : str
        NLTK data path used by ``nltk.data.find`` (e.g. ``"corpora/wordnet"``).
    download_name : str
        NLTK downloader package name (e.g. ``"wordnet"``).
    allow_download : bool or None, optional
        Per-call override of the download policy. ``None`` (default) defers to
        :func:`downloads_allowed`.
    extra_hint : str, optional
        Additional text appended to the error message.
    **kwargs : any
        Allows arbitrary additional keyword args to be passed to `nltk`.

    Raises
    ------
    ResourceUnavailableError
        If NLTK is not installed, or the resource is missing and downloads are
        not authorized.
    """
    allow_download = allow_download or kwargs.pop("allow_download", allow_download)
    try:
        import nltk  # type: ignore[import]  # noqa: PLC0415
    except ImportError as exc:
        raise ResourceUnavailableError(
            f"NLTK is required for resource {download_name!r}. "
            "Install it with: pip install nltk"
        ) from exc

    try:
        nltk.data.find(find_path)
        return
    except LookupError:
        pass

    if downloads_allowed(allow_download):
        logger.info(
            "corpus: managed NLTK download of %r (%s) — authorized.",
            download_name,
            find_path,
        )
        nltk.download(download_name, quiet=True)
        return

    raise ResourceUnavailableError(
        f"NLTK resource {download_name!r} (data path {find_path!r}) is not "
        f"installed, and automatic downloads are disabled (offline-safe "
        f"default). Install it once with:\n"
        f"    python -m nltk.downloader {download_name}\n"
        f"or authorize managed downloads by setting {ENV_ALLOW_DOWNLOADS}=1."
        + (f" {extra_hint}" if extra_hint else "")
    )
