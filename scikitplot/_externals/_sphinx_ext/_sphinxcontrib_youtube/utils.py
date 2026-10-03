# scikitplot/_externals/_sphinx_ext/_sphinxcontrib_youtube/utils.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# Authors: Dr David Ham, Chris Pickel and others
# SPDX-License-Identifier: BSD-3-Clause

"""Skeleton of the video directive ready to be extended for specific providers."""

import re
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import requests
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.errors import ConfigError
from sphinx.util import logging
from sphinx.util.display import status_iterator

logger = logging.getLogger(__name__)

CONTROL_HEIGHT = 30

THUMBNAIL_DIR = "_video_thumbnail"

#: scikit-plots local patch: connect+read timeout, in seconds, for the latex
#: thumbnail fetch. Bounded so a slow host cannot stall a docs build.
DOWNLOAD_TIMEOUT = (5, 30)

#: scikit-plots local patch: ceiling on thumbnails fetched in one build.
#:
#: Legacy opt-in LaTeX thumbnail compatibility ceiling. The current LaTeX
#: visitor does not consume thumbnails, so ordinary builds perform no fetches.
#: If a downstream project explicitly enables the compatibility path, both
#: request count and byte budgets prevent a slow/large catalog from turning a
#: documentation build into an unbounded network job.
DEFAULT_DOWNLOAD_LIMIT = 200

#: Maximum bytes accepted for one LaTeX thumbnail.  A response is streamed
#: into a temporary file and promoted only after it completes, so a broken
#: connection cannot leave a corrupt image that looks cached on the next run.
DEFAULT_DOWNLOAD_MAX_BYTES = 8 * 1024 * 1024

#: Aggregate byte ceiling for the optional legacy thumbnail path. This bounds
#: total build-time network/disk exposure even when many responses are small.
DEFAULT_DOWNLOAD_MAX_TOTAL_BYTES = 64 * 1024 * 1024

_MAX_DOWNLOAD_LIMIT = 1000
_MAX_DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024
_MAX_DOWNLOAD_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_THUMBNAIL_HOSTS = frozenset({"i3.ytimg.com", "vumbnail.com"})


def _bounded_config_int(value, *, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ConfigError(
            f"{name} must be between {minimum} and {maximum} inclusive"
        )
    return value


def validate_download_config(app, config):
    """Fail closed on unsafe optional thumbnail-download budgets."""
    enabled = config.video_download_thumbnails
    if not isinstance(enabled, bool):
        raise ConfigError("video_download_thumbnails must be true or false")
    limit = _bounded_config_int(
        config.video_download_limit,
        name="video_download_limit",
        minimum=0,
        maximum=_MAX_DOWNLOAD_LIMIT,
    )
    max_bytes = _bounded_config_int(
        config.video_download_max_bytes,
        name="video_download_max_bytes",
        minimum=1,
        maximum=_MAX_DOWNLOAD_MAX_BYTES,
    )
    max_total = _bounded_config_int(
        config.video_download_max_total_bytes,
        name="video_download_max_total_bytes",
        minimum=1,
        maximum=_MAX_DOWNLOAD_MAX_TOTAL_BYTES,
    )
    if enabled and limit < 1:
        raise ConfigError(
            "video_download_limit must be at least 1 when thumbnail downloads are enabled"
        )
    if enabled and max_total < max_bytes:
        raise ConfigError(
            "video_download_max_total_bytes must be >= video_download_max_bytes "
            "when thumbnail downloads are enabled"
        )


def _safe_thumbnail_url(value):
    """Return one absolute HTTPS thumbnail URL, or None for unsupported templates."""
    from urllib.parse import urlsplit

    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _THUMBNAIL_HOSTS
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port not in (None, 443):
        return None
    return value


def _thumbnail_path(url):
    """Map one remote URL to a traversal-proof deterministic cache path."""
    digest = sha256(url.encode("utf-8")).hexdigest()
    return Path(THUMBNAIL_DIR, f"{digest}.jpg")

# -- helper methods ------------------------------------------------------------


# -- scikit-plots local patch: video reference normalisation -------------------
# Upstream takes ``self.arguments[0]`` as an opaque video id and interpolates it
# straight into the embed URL, so a pasted watch URL produced
# ``.../embed/https://www.youtube.com/watch?v=ID`` -- a dead iframe, emitted with
# no warning and a successful build.
#
# The URL grammar lives in one place (``_sphinx_youtube_core.reference``)
# rather than being duplicated here, so the standalone player, typed gallery,
# and sync tool cannot disagree about what a given URL means.

from .._sphinx_youtube_core.reference import (
    ReferenceError as _ReferenceError,
    parse_video_reference as _parse_video_reference,
)
from .._sphinx_youtube_core.video_options import LEAF_VIDEO_SPEC


def parse_youtube_id(value):
    """
    Normalise any single-video reference to its canonical id.

    Parameters
    ----------
    value : str
        A bare id or any YouTube URL naming one video.

    Returns
    -------
    str
        The canonical 11-character video id.

    Raises
    ------
    ValueError
        If the value names no single video.
    """
    return _parse_video_reference(value).video_id



def get_size(d, key):
    """Return a valid positive CSS size and unit."""
    if key not in d:
        return None
    m = re.fullmatch(r"([1-9]\d*)(|%|px)", d[key])
    if not m:
        raise ValueError("invalid size %r" % d[key])
    return int(m.group(1)), m.group(2) or "px"


def css(d):
    """Return a valid css style string."""
    return "; ".join(sorted("%s: %s" % kv for kv in d.items()))


# -- node and directive definition ---------------------------------------------


def _merge_url_parameters(url_parameters, start_at):
    """
    Fold a start offset into an embed's query string.

    Parameters
    ----------
    url_parameters : str
        The author's explicit ``:url_parameters:`` value, e.g. ``"?rel=0"``.
    start_at : int or None
        Start offset in seconds, recovered from the pasted URL.

    Returns
    -------
    str
        The query string to append to the embed URL. An explicit
        ``start``/``t`` written by the author always wins: the directive
        option is a deliberate instruction, while the offset in a pasted URL
        is incidental.
    """
    if start_at is None:
        return url_parameters
    if re.search(r"[?&](?:start|t)=", url_parameters):
        return url_parameters
    separator = "&" if url_parameters.startswith("?") else "?"
    return f"{url_parameters}{separator}start={start_at}"


class video(nodes.General, nodes.Element):
    """Video node."""

    pass


class Video(Directive):
    """Abstract Video directive."""

    _node = None
    "Subclasses should replace with node class."

    _thumbnail_url = "{}"
    "url to retrieve thumbnail images"

    _platform = ""
    "name of the platform"

    _platform_url = ""
    "url of the platform video provider"

    _platform_url_privacy = ""
    "the aleternative url to provide the video privately"

    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    # Standalone players and youtube-gallery-generated players share the exact
    # same validators.  ``privacy_mode`` therefore understands explicit false
    # values, sizes are positive, and query strings are bounded/normalized.
    option_spec: ClassVar = dict(LEAF_VIDEO_SPEC)

    def run(self):
        """Run the directive."""
        env = self.state.document.settings.env
        # scikit-plots local patch: accept watch/short/embed URLs, not just
        # bare ids, and fail with a located error instead of a dead embed.
        start_at = None
        if self._platform == "youtube":
            try:
                reference = _parse_video_reference(self.arguments[0])
            except (_ReferenceError, ValueError) as exc:
                return [
                    self.state_machine.reporter.error(
                        f"youtube: {exc}", line=self.lineno
                    )
                ]
            video_id = reference.video_id
            # A `t=`/`start=` offset in the pasted URL is intent, not noise:
            # carry it into the player instead of discarding it.
            start_at = getattr(reference, "start", None)
        else:
            video_id = self.arguments[0]
        # The current LaTeX visitor renders a URL box and does not consume a
        # thumbnail. Keep remote thumbnail I/O opt-in so normal documentation
        # builds remain deterministic/offline. Only fixed HTTPS provider URLs
        # enter the environment if compatibility downloading is explicitly on.
        if getattr(env.config, "video_download_thumbnails", False):
            url = _safe_thumbnail_url(self._thumbnail_url.format(video_id))
            if url:
                remote_images = getattr(env, "video_remote_images", None)
                if not isinstance(remote_images, dict):
                    remote_images = {}
                    env.video_remote_images = remote_images
                consumers = getattr(env, "video_remote_images_by_doc", None)
                if not isinstance(consumers, dict):
                    consumers = {}
                    env.video_remote_images_by_doc = consumers
                remote_images[url] = _thumbnail_path(url)
                consumers.setdefault(env.docname, set()).add(url)
                # Register the real owning document so Sphinx can purge its
                # standard image inventory correctly during incremental builds.
                env.images.add_file(env.docname, remote_images[url])

        if "aspect" in self.options:
            aspect = self.options.get("aspect")
            m = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", aspect)
            if m is None:
                # scikit-plots local patch: located error, not a traceback.
                return [
                    self.state_machine.reporter.error(
                        f"invalid aspect ratio {aspect!r}, expected e.g. '16:9'",
                        line=self.lineno,
                    )
                ]
            aspect = tuple(int(x) for x in m.groups())
        else:
            aspect = None

        alignment = ["left", "center", "right"]
        if "align" in self.options:
            align = self.options.get("align")
            if align not in alignment:
                # scikit-plots local patch: located error, not a traceback.
                return [
                    self.state_machine.reporter.error(
                        f"invalid alignment {align!r}, choices are: {alignment}",
                        line=self.lineno,
                    )
                ]
        else:
            align = None

        # custom platform url for peertube
        instance = self._platform_url
        if "instance" in self.options:
            instance = self.options.get("instance")

        try:
            width = get_size(self.options, "width")
            height = get_size(self.options, "height")
        except ValueError as exc:
            return [self.state_machine.reporter.error(str(exc), line=self.lineno)]
        return [
            self._node(
                id=video_id,
                title=self.options.get("title"),
                aspect=aspect,
                width=width,
                height=height,
                align=align,
                url_parameters=_merge_url_parameters(
                    self.options.get("url_parameters", ""), start_at
                ),
                privacy_mode=self.options.get("privacy_mode"),
                platform=self._platform,
                platform_url=self._platform_url,
                platform_url_privacy=self._platform_url_privacy,
                instance=instance,
            )
        ]


# -- builder specific methods --------------------------------------------------


def _privacy_enabled(value):
    """Interpret a validated privacy-mode option consistently."""
    return value is not None and value not in (False, "false", "off", "no", "0")


def visit_video_node_html(self, node, platform_url_privacy=None, additional_attr={}):
    """Visit html video node."""
    aspect = node["aspect"]
    width = node["width"]
    height = node["height"]
    url_parameters = node["url_parameters"]
    platform_url = node["platform_url"]
    platform_url_privacy = node["platform_url_privacy"]
    if _privacy_enabled(node.get("privacy_mode")) and platform_url_privacy:
        platform_url = platform_url_privacy

    if aspect is None:
        aspect = 16, 9

    div_style = {}
    # A player with an aspect ratio and no explicit height is responsive at
    # every width, including the historical no-option default.  Using the
    # native CSS aspect-ratio property avoids the fixed 560x345 frame that
    # became tall and distorted when max-width shrank it inside a mobile card.
    if height is None:
        if width is None:
            width = 560, "px"
        div_style = {
            "width": "%d%s" % width,
            "max-width": "100%",
            "aspect-ratio": "%d / %d" % aspect,
            "position": "relative",
        }
        style = {
            "position": "absolute",
            "top": "0",
            "left": "0",
            "width": "100%",
            "height": "100%",
            "border": "0",
        }
        attrs = {
            "src": "{}{}{}".format(platform_url, node["id"], url_parameters),
            "style": css(style),
            **additional_attr,
        }
    else:
        if width is None:
            if height[1] == "%":
                width = 100, "%"
            else:
                width = height[0] * aspect[0] / aspect[1], "px"
        style = {
            "width": "%d%s" % width,
            "height": "%d%s" % height,
            "border": "0",
        }
        attrs = {
            "src": "{}{}{}".format(platform_url, node["id"], url_parameters),
            "style": css(style),
            **additional_attr,
        }
    if node["align"] is not None:
        div_style["text-align"] = node["align"]
    attrs["allowfullscreen"] = "true"
    # -- scikit-plots local patch: accessibility + gallery performance -------
    # `title` gives the frame an accessible name (WCAG 2.1 SC 4.1.2); without
    # it a page of embeds is an unnavigable list of unnamed frames.
    # `loading="lazy"` matters at gallery scale: a 100- or 1000-video page
    # otherwise opens that many YouTube connections on first paint.
    # The responsive wrapper above keeps default and fixed-width players at
    # their requested aspect ratio inside narrow cards.
    attrs["title"] = node.get("title") or "{} video player".format(
        node["platform"] or "embedded"
    )
    attrs["loading"] = "lazy"
    if "max-width" not in attrs["style"]:
        attrs["style"] = attrs["style"] + "; max-width: 100%"
    # -- end scikit-plots local patch ----------------------------------------
    div_attrs = {
        "CLASS": "video_wrapper",
        "style": css(div_style),
    }
    if node["align"] is not None:
        div_attrs["CLASS"] += " align-%s" % node["align"]
    self.body.append(self.starttag(node, "div", **div_attrs))
    self.body.append(self.starttag(node, "iframe", **attrs))
    self.body.append("</iframe></div>")


def visit_video_node_epub(self, node):
    """Visit epub video node."""
    url_parameters = node["url_parameters"]
    link_url = "{}{}{}".format(node["platform_url"], node["id"], url_parameters)

    self.body.append(self.starttag(node, "a", CLASS="video_link_url", href=link_url))
    self.body.append(link_url)
    self.body.append("</a>")


def visit_video_node_latex(self, node):
    """Visit latex video node."""
    folder = r"\graphicspath{ {./%s/}{./} }" % THUMBNAIL_DIR
    if folder not in self.elements["preamble"]:
        self.elements["preamble"] += folder + "\n"

    macro = f"\\sphinxcontrib{node['platform']}"
    if macro not in self.elements["preamble"]:
        cmd = (
            r"\newcommand{%s}[3]{\begin{quote}\begin{center}\fbox{\url{#1#2#3}}\end{center}\end{quote}}"
            % macro
        )
        self.elements["preamble"] += cmd + "\n"

    self.body.append(
        "{}{{{}}}{{{}}}{{{}}}\n".format(
            macro, node["platform_url"], node["id"], node["url_parameters"]
        )
    )


def visit_video_node_unsupported(self, node):
    """Visit unsupported video node."""
    logger.warning(f"{node['platform']}: unsupported output format (node skipped)")
    raise nodes.SkipNode


def depart_video_node(self, node):
    """Depart any video node."""
    pass


_NODE_VISITORS = {
    "html": (visit_video_node_html, depart_video_node),
    "epub": (visit_video_node_epub, depart_video_node),
    "latex": (visit_video_node_latex, depart_video_node),
    "man": (visit_video_node_unsupported, depart_video_node),
    "texinfo": (visit_video_node_unsupported, depart_video_node),
    "text": (visit_video_node_unsupported, depart_video_node),
}

# -- manage downloaded images ---------------------------------------------------


def purge_download_images(app, env, docname):
    """Drop thumbnail registrations owned only by one purged document."""
    consumers = getattr(env, "video_remote_images_by_doc", None)
    remote_images = getattr(env, "video_remote_images", None)
    if not isinstance(consumers, dict) or not isinstance(remote_images, dict):
        return
    removed = set(consumers.pop(docname, set()) or ())
    if not removed:
        return
    still_used = set().union(*(set(urls or ()) for urls in consumers.values())) if consumers else set()
    for url in removed - still_used:
        remote_images.pop(url, None)


def merge_download_images(app, env, docnames, other):
    """Merge only worker-owned thumbnail registrations during parallel reads."""
    if not isinstance(getattr(env, "video_remote_images", None), dict):
        env.video_remote_images = {}
    if not isinstance(getattr(env, "video_remote_images_by_doc", None), dict):
        env.video_remote_images_by_doc = {}
    incoming = getattr(other, "video_remote_images", None) or {}
    incoming_by_doc = getattr(other, "video_remote_images_by_doc", None) or {}
    for docname in set(docnames or ()):
        purge_download_images(app, env, docname)
        urls = set(incoming_by_doc.get(docname, ()) or ())
        if not urls:
            continue
        env.video_remote_images_by_doc[docname] = urls
        for url in urls:
            if url in incoming:
                env.video_remote_images[url] = incoming[url]


def download_images(app, env):
    """
    Download thumbnails for the latex build.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application.
    env : sphinx.environment.BuildEnvironment
        The build environment carrying ``video_remote_images``.

    Notes
    -----
    This legacy compatibility path is disabled by default. The current LaTeX
    visitor does not reference thumbnails; projects that explicitly opt in get
    bounded request-count, per-file, and aggregate-byte budgets.
    """
    if not getattr(app.config, "video_download_thumbnails", False):
        return
    # images should only be downloaded if the builder is Latex related
    if "latex" not in app.builder.name:
        return

    iterator = (
        app.builder.status_iterator
        if hasattr(app.builder, "status_iterator")
        else status_iterator
    )
    msg = "Downloading remote images..."
    nb_images = len(env.video_remote_images)
    # scikit-plots local patch: aggregate download budget. Mutable
    # single-element lists rather than plain ints so the counters survive
    # the `continue` branches below without a nonlocal declaration.
    _limit = getattr(
        app.config, "video_download_limit", DEFAULT_DOWNLOAD_LIMIT
    )
    _attempted = [0]
    _downloaded = [0]
    _skipped = [0]
    _max_bytes = getattr(
        app.config, "video_download_max_bytes", DEFAULT_DOWNLOAD_MAX_BYTES
    )
    _max_total_bytes = getattr(
        app.config,
        "video_download_max_total_bytes",
        DEFAULT_DOWNLOAD_MAX_TOTAL_BYTES,
    )
    _received_total = [0]
    for src in iterator(env.video_remote_images, msg, "brown", nb_images):

        # Cached Sphinx environments predate this policy, so never trust a
        # persisted URL merely because current directive registration is safe.
        if _safe_thumbnail_url(src) is None:
            logger.warning(f'Cannot download unsafe thumbnail URL "{src}"')
            _skipped[0] += 1
            continue

        # scikit-plots local patch: bound the aggregate, not just each call.
        if _attempted[0] >= _limit or _received_total[0] >= _max_total_bytes:
            _skipped[0] += 1
            continue
        dst = Path(app.outdir) / _thumbnail_path(src)
        if not dst.is_file():
            _attempted[0] += 1
            logger.info(f"{src} -> {dst} (downloading)")
            dst.parent.mkdir(parents=True, exist_ok=True)
            # -- scikit-plots local patch: bounded, fully-handled fetch ----
            # Upstream calls `requests.get` with no timeout and catches only
            # `ConnectionError`, so a hung or slow thumbnail host stalls the
            # build indefinitely, and a read timeout / HTTP error / broken
            # pipe propagates as an unhandled exception. It also wrote the
            # response body unconditionally, so a 404 page was saved as a
            # `.jpg`. Fetch first, validate, then write only on success.
            response = None
            try:
                response = requests.get(
                    src,
                    timeout=DOWNLOAD_TIMEOUT,
                    stream=True,
                    allow_redirects=False,
                )
                if response.is_redirect or response.is_permanent_redirect:
                    raise ValueError("thumbnail redirects are not allowed")
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if not content_type.lower().startswith("image/"):
                    raise ValueError(f"unexpected content type {content_type!r}")
                length = response.headers.get("Content-Length")
                if length:
                    declared = int(length)
                    if declared < 0:
                        raise ValueError("negative Content-Length")
                    if declared > _max_bytes:
                        raise ValueError(
                            f"response is {declared:,} bytes; limit is {_max_bytes:,}"
                        )
                    if _received_total[0] + declared > _max_total_bytes:
                        raise ValueError(
                            "response would exceed aggregate thumbnail byte budget"
                        )
                temporary = dst.with_suffix(dst.suffix + ".part")
                received = 0
                try:
                    with temporary.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            if not chunk:
                                continue
                            chunk_size = len(chunk)
                            received += chunk_size
                            if received > _max_bytes:
                                raise ValueError(
                                    f"response exceeds {_max_bytes:,} bytes"
                                )
                            if _received_total[0] + chunk_size > _max_total_bytes:
                                raise ValueError(
                                    "aggregate thumbnail byte budget exceeded"
                                )
                            _received_total[0] += chunk_size
                            handle.write(chunk)
                    temporary.replace(dst)
                    _downloaded[0] += 1
                finally:
                    temporary.unlink(missing_ok=True)
            except (requests.RequestException, OSError) as exc:
                logger.warning(f'Cannot download thumbnail "{src}": {exc}')
                continue
            except (ValueError, TypeError) as exc:
                logger.warning(f'Cannot download thumbnail "{src}": {exc}')
                continue
            finally:
                if response is not None:
                    response.close()
            # -- end scikit-plots local patch ------------------------------
        else:
            logger.info(f"{src} -> {dst} (already in cache)")

    # scikit-plots local patch: report the budget rather than truncating in
    # silence -- a PDF quietly missing 1000 stills would look like a bug in
    # the document, not a deliberate limit.
    if _skipped[0]:
        logger.warning(
            f"video: attempted {_attempted[0]} thumbnail downloads, completed "
            f"{_downloaded[0]}, and skipped "
            f"{_skipped[0]} after reaching the download limit of {_limit}. "
            f"Raise 'video_download_limit' or the aggregate byte budget in "
            f"conf.py only if this explicitly-enabled legacy path needs them."
        )


def configure_image_download(app):
    """Prepare opt-in LaTeX thumbnail state without mutating HTML static paths."""
    if not getattr(app.config, "video_download_thumbnails", False):
        return
    if "latex" not in app.builder.name:
        return
    if not isinstance(getattr(app.env, "video_remote_images", None), dict):
        app.env.video_remote_images = {}
    if not isinstance(getattr(app.env, "video_remote_images_by_doc", None), dict):
        app.env.video_remote_images_by_doc = {}

    output_dir = Path(app.outdir) / THUMBNAIL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
