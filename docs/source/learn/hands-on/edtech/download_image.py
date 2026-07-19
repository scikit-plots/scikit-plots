#!/usr/bin/env python3
"""Download placeholder images."""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

from urllib.request import urlopen, urlretrieve

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PLACEHOLDER_IMAGES = {
    "workshops/rst/images/default-sphinx-page.png":
        "https://labs.bilimedtech.com/_images/default-sphinx-page.png",
    "workshops/rst/images/rtd-sphinx-page.png":
        "https://labs.bilimedtech.com/_images/rtd-sphinx-page.png",
    "workshops/rst/images/the_great_sphinx_david_roberts.jpg":
        "https://labs.bilimedtech.com/_images/the_great_sphinx_david_roberts.jpg",
    "cloud-computing/2/images/reverse-proxy.png":
        "https://labs.bilimedtech.com/_images/reverse-proxy.png",
    "cloud-computing/2/images/freenom5.png":
        "https://labs.bilimedtech.com/_images/freenom5.png",
    "cloud-computing/2/images/freenom6.png":
        "https://labs.bilimedtech.com/_images/freenom6.png",
    "cloud-computing/2/images/freenom7.png":
        "https://labs.bilimedtech.com/_images/freenom7.png",
    "cloud-computing/2/images/chat-domain-name.png":
        "https://labs.bilimedtech.com/_images/chat-domain-name.png",
}


def download_urlopen(url: str, destination: Path, *, timeout: float = 30.0) -> None:
    """Download a file using urllib.request.urlopen (recommended)."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    with (
        urlopen(url, timeout=timeout) as response,
        destination.open("wb") as file,
    ):
        shutil.copyfileobj(response, file)


def download_urlretrieve(url: str, destination: Path) -> None:
    """Download a file using urllib.request.urlretrieve."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination)


def download_images(
    images: dict[str, str],
    *,
    root: Path = Path("."),
    overwrite: bool = False,
    method: str = "urlopen",
) -> None:
    """
    Download all images.

    Parameters
    ----------
    images
        Mapping of destination paths to URLs.
    root
        Root directory for downloads.
    overwrite
        Whether to overwrite existing files.
    method
        Download backend:

        - ``"urlopen"`` (recommended; default)
        - ``"urlretrieve"``
    """
    downloaders = {
        "urlopen": download_urlopen,
        "urlretrieve": download_urlretrieve,
    }

    try:
        downloader = downloaders[method]
    except KeyError:
        raise ValueError(
            f"Unknown download method {method!r}. "
            f"Expected one of {tuple(downloaders)}."
        ) from None

    for rel_path, url in images.items():
        destination = root / rel_path
        logger.debug("Downloading %s -> %s", url, destination)

        if destination.exists() and not overwrite:
            # print(f"[SKIP] {destination}")
            logger.info("[SKIP] %s", destination)
            continue

        # print(f"[GET ] {url}")
        logger.info("[GET ] %s", url)

        try:
            downloader(url, destination)
        except Exception as exc:
            # print(f"[FAIL] {destination}: {exc}")
            # print(traceback.format_exc())
            # logger.error(..., exc_info=True)
            # logger.exception("Failed to download %s from %s", destination, url)
            logger.exception("[FAIL] %s from %s", destination, url)
        else:
            # print(f"[ OK ] {destination}")
            # logger.info("Downloaded %s", destination)
            logger.info("[ OK ] %s", destination)


if __name__ == "__main__":
    # Recommended
    # download_images(PLACEHOLDER_IMAGES)
    # Alternative
    # download_images(PLACEHOLDER_IMAGES, method="urlretrieve")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        choices=("urlopen", "urlretrieve"),
        default="urlopen",
        help="Download backend.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show errors.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (repeat for more detail).",
    )

    args = parser.parse_args()

    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    logger = logging.getLogger(__name__)

    download_images(
        PLACEHOLDER_IMAGES,
        overwrite=args.overwrite,
        method=args.method,
    )
