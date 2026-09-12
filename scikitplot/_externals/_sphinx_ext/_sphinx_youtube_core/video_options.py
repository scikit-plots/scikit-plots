"""
Validate gallery-wide options for generated YouTube players.

``video-*`` names map to the leaf directive's option names. Source values must
remain single-line; sizes/ratios are positive, alignment is enumerated, and
query parameters are normalized. A false privacy value omits privacy_mode,
because presence of that leaf option selects the privacy-enhanced host.
This module performs no network access or Sphinx extension registration.
"""

import re
from urllib.parse import parse_qsl, urlencode


def _single(argument):
    """Return stripped option text, rejecting source-line injection."""
    value = "" if argument is None else argument.strip()
    if any(char in value for char in "\r\n\x00"):
        raise ValueError("video options must be a single line")
    return value


def _size(argument):
    """Validate a positive integer size with optional px or percent units."""
    value = _single(argument)
    if not re.fullmatch(r"[1-9][0-9]*(?:px|%)?", value):
        raise ValueError("use a positive size such as 640, 640px or 100%")
    return value


def _aspect(argument):
    """Validate an exact, positive width:height ratio such as 16:9."""
    value = _single(argument)
    if not re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", value):
        raise ValueError("use a positive ratio such as 16:9")
    return value


def _align(argument):
    """Validate the leaf player's left, center or right alignment."""
    value = _single(argument)
    if value not in ("left", "center", "right"):
        raise ValueError("use left, center or right")
    return value


def _privacy(argument):
    """Accept an empty true flag or an explicit common boolean spelling."""
    value = _single(argument).lower()
    if value not in ("", "true", "false", "on", "off", "yes", "no", "1", "0"):
        raise ValueError("use an empty flag, true, or false")
    return value


def _query(argument):
    """Normalize at most 128 query pairs, adding the leading question mark."""
    value = _single(argument).lstrip("?&")
    return (
        ("?" + urlencode(parse_qsl(value, keep_blank_values=True, max_num_fields=128)))
        if value
        else ""
    )


LEAF_VIDEO_SPEC = {
    "width": _size,
    "height": _size,
    "aspect": _aspect,
    "align": _align,
    "title": _single,
    "privacy_mode": _privacy,
    "url_parameters": _query,
}

# The gallery-facing form is only a namespaced view of the same leaf option
# contract. Keeping one converter table prevents standalone ``youtube`` and
# generated ``youtube-gallery`` players from drifting on validation semantics.
VIDEO_SPEC = {
    "video-" + key.replace("_", "-"): converter
    for key, converter in LEAF_VIDEO_SPEC.items()
}


def player_options(options, title):
    """
    Build leaf options from validated gallery options and a record title.

    Parameters
    ----------
    options : mapping
        Parsed gallery directive options. VIDEO_SPEC has already validated
        any video-* values present in this mapping.
    title : str
        Record title, used as a whitespace-normalized accessible name unless
        video-title explicitly overrides it.

    Returns
    -------
    dict
        Leaf option names (including underscores where required) and source
        values. A false privacy setting is represented by an absent option.
    """
    result = {"title": " ".join(title.split())}
    for key in VIDEO_SPEC:
        if key not in options:
            continue
        value = options[key]
        if key == "video-privacy-mode":
            if value in ("false", "off", "no", "0"):
                continue
            value = ""
        result[key.removeprefix("video-").replace("-", "_")] = value
    return result
