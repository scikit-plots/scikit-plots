# scikitplot/_brand/_banner.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot/_brand/_banner.py — Generate deterministic ASCII-art terminal banners.

This script renders a project name into several ``figlet`` typefaces,
optionally wraps each rendering in a text-mode border, and writes the
results to a banner directory (default: ``.banners/``) as plain-text
files. The generated files are later selected at runtime via a
deterministic day-of-year rotation (``banner_index = day_of_year %
total_banners``); see ``terminal-banner-ideas.md`` for the full rotation
table.

Notes
-----
Install
::

  Requires: apt install figlet  (Debian/Ubuntu)
            brew install figlet (macOS)

Examples
--------
Generate the default eight banners into ``.banners/``::

    $ python3 docker/scripts/.bashrc.d/_banner.py   # writes .banners/001-standard.txt … 008-big.txt
    $ python3 -m scikitplot._brand._banner          # writes .banners/001-standard.txt … 008-big.txt

Preview banners without writing files::

    $ python3 scikitplot._brand._banner --dry-run --verbose

Render a different string into a custom directory::

    $ python3 scikitplot._brand._banner --text "my-project" --output-dir out/

Override every banner's case transform (ignoring each banner's own
curated default from `BANNER_SPECS`)::

    $ python3 scikitplot._brand._banner --case upper
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Sequence

__all__ = [
    "BANNER_SPECS",
    "CASE_TRANSFORMS",
    "DEFAULT_BANNER_DIR",
    "DEFAULT_TEXT",
    "DEFAULT_TEXT_CASE",
    "BannerGenerationError",
    "BannerSpec",
    "FigletNotFoundError",
    "apply_text_case",
    "box_block",
    "box_double",
    "box_none",
    "box_rounded",
    "box_solid",
    "figlet",
    "generate_all",
    "generate_banner",
    "main",
    "parse_args",
]

logger = logging.getLogger(__name__)

DEFAULT_FIGLET_WIDTH_FALLBACK = (200, 24)

FONT_SOURCES = {
    "ANSI Shadow": (
        "https://raw.githubusercontent.com/xero/figlet-fonts/master/ANSI%20Shadow.flf"
    ),
}

FONT_CACHE_DIR = pathlib.Path.home() / ".cache" / "scikit-plots" / "figlet"

#: Default directory that banner files are written into.
DEFAULT_BANNER_DIR = pathlib.Path(".banners")

#: Default text rendered into each banner.
DEFAULT_TEXT = "scikit-plots"

#: Default case transform applied when a `BannerSpec` does not request one.
#: Must be a key of `CASE_TRANSFORMS`.
DEFAULT_TEXT_CASE = "none"


class FigletNotFoundError(RuntimeError):
    """Raised when the ``figlet`` executable cannot be located on ``PATH``."""


class BannerGenerationError(RuntimeError):
    """Raised when a banner fails to render (non-zero exit, empty output, etc.)."""


# ---------------------------------------------------------------------------
# Text-case transforms
# ---------------------------------------------------------------------------

#: Registry of named case transforms available to `BannerSpec.text_case`
#: and the ``--case`` CLI flag. Each callable is a pure ``str -> str``
#: function with no side effects; values are drawn directly from
#: :class:`str` methods so behaviour matches ordinary Python string
#: semantics with no surprises.
CASE_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "none": lambda s: s,
    "lower": str.lower,
    "upper": str.upper,
    "title": str.title,
    "capitalize": str.capitalize,
    "swapcase": str.swapcase,
}


def apply_text_case(text: str, case: str = DEFAULT_TEXT_CASE) -> str:
    """
    Apply a named case transform to `text` before it is sent to figlet.

    Parameters
    ----------
    text : str
        The text to transform.
    case : str, optional
        A key of `CASE_TRANSFORMS` (default: `DEFAULT_TEXT_CASE`, i.e.
        ``"none"``).

    Returns
    -------
    str
        `text` with the named transform applied. ``"none"`` returns
        `text` unchanged.

    Raises
    ------
    ValueError
        If `case` is not a key of `CASE_TRANSFORMS`.

    See Also
    --------
    CASE_TRANSFORMS : The underlying name -> callable registry.
    BannerSpec.text_case : Per-banner default consumed by `generate_banner`.

    Notes
    -----
    The transform is applied to the *input* string, before figlet renders
    it into ASCII art — figlet draws one fixed glyph per input character,
    so e.g. ``"capitalize"`` only changes which glyph is drawn for the
    first character; it does not alter the rendered art afterwards.

    Examples
    --------
    >>> apply_text_case("scikit-plots", "capitalize")
    'Scikit-plots'
    >>> apply_text_case("scikit-plots", "title")
    'Scikit-Plots'
    >>> apply_text_case("scikit-plots", "upper")
    'SCIKIT-PLOTS'
    >>> apply_text_case("scikit-plots", "none")
    'scikit-plots'
    """
    try:
        transform = CASE_TRANSFORMS[case]
    except KeyError:
        valid = ", ".join(sorted(CASE_TRANSFORMS))
        raise ValueError(f"case must be one of {{{valid}}}, got {case!r}") from None
    return transform(text)


def _figlet_font_dir() -> pathlib.Path:
    try:
        result = subprocess.run(
            ["figlet", "-I2"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return pathlib.Path(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return pathlib.Path("/usr/share/figlet")


def ensure_font_installed(font: str) -> pathlib.Path | None:
    if not font or font in {
        "standard",
        "slant",
        "shadow",
        "small",
        "smslant",
        "big",
        "block",
    }:
        return None

    FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    font_file = FONT_CACHE_DIR / f"{font}.flf"

    if font_file.exists():
        return font_file

    url = FONT_SOURCES.get(font)
    if not url:
        return None

    try:
        urllib.request.urlretrieve(url, font_file)  # noqa: S310
        if font_file.stat().st_size < 100:  # noqa: PLR2004
            raise OSError("downloaded font file is unexpectedly small")
        logger.info("Downloaded figlet font %s -> %s", font, font_file)
        return font_file
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to download font %s: %s", font, exc)
        return None


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------


def figlet(text: str, font: str, kerning: bool = False) -> str:
    r"""
    Render `text` using the external ``figlet`` program.

    Parameters
    ----------
    text : str
        The text to render. Must be non-empty after stripping whitespace.
    font : str
        Name of an installed figlet font (e.g. ``"standard"``, ``"slant"``).
    kerning : bool
        False

    Returns
    -------
    str
        The rendered ASCII-art block, with the trailing newline removed.

    Raises
    ------
    ValueError
        If `text` or `font` is empty or whitespace-only.
    FigletNotFoundError
        If the ``figlet`` executable is not available on ``PATH``.
    BannerGenerationError
        If ``figlet`` exits with a non-zero status or produces empty output.

    See Also
    --------
    box_double, box_rounded, box_solid, box_none : Border wrappers applied
        to this function's output.

    Notes
    -----
    This shells out to the system ``figlet`` binary (https://www.figlet.org/)
    rather than reimplementing font rendering. The call passes an explicit
    argument list (no ``shell=True``), so it is not vulnerable to shell
    injection regardless of the contents of `text`.

    Examples
    --------
    >>> figlet("hi", "standard")  # doctest: +SKIP
    ' _     _ \n| |__ (_)\n| \'_ \\\| |\n| | | || |\n|_| |_|/ |\n      |__/'
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not font or not font.strip():
        raise ValueError("font must be a non-empty string")
    cached_font = ensure_font_installed(font)

    if shutil.which("figlet") is None:
        raise FigletNotFoundError(
            "The 'figlet' executable was not found on PATH. Install it via "
            "'apt install figlet' (Debian/Ubuntu) or 'brew install figlet' "
            "(macOS), then retry."
        )

    argv = ["figlet"]

    if cached_font is not None:
        argv.extend(["-d", str(FONT_CACHE_DIR)])

    argv.extend(
        [
            "-w",
            str(
                shutil.get_terminal_size(fallback=DEFAULT_FIGLET_WIDTH_FALLBACK).columns
            ),
        ]
    )
    if kerning:
        argv.append("-k")
    argv.extend(["-f", font, text])
    # argv.append("") if False else None

    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell=True
            argv,  # noqa: S607 - resolved via PATH check above
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise BannerGenerationError(
            f"figlet failed for font={font!r} (exit code {exc.returncode}): {stderr}"
        ) from exc

    rendered = result.stdout.rstrip("\n")
    if not rendered:
        raise BannerGenerationError(f"figlet produced empty output for font={font!r}")
    return rendered


def _max_width(lines: Sequence[str]) -> int:
    """
    Return the maximum character width across `lines`.

    Parameters
    ----------
    lines : Sequence[str]
        Lines of text. May be empty.

    Returns
    -------
    int
        The length of the longest line, or ``0`` if `lines` is empty.
    """
    return max((len(line) for line in lines), default=0)


def _require_lines(s: str) -> list[str]:
    """
    Split `s` into lines, raising if there are none.

    Parameters
    ----------
    s : str
        Multi-line text block.

    Returns
    -------
    list[str]
        The individual lines of `s`.

    Raises
    ------
    ValueError
        If `s` is empty or contains no lines.
    """
    lines = s.splitlines()
    if not lines:
        raise ValueError("s must contain at least one line")
    return lines


def box_none(s: str) -> str:
    """
    Return `s` unchanged (no border applied).

    Parameters
    ----------
    s : str
        Multi-line text block.

    Returns
    -------
    str
        The same text, unchanged.

    See Also
    --------
    box_block, box_double, box_rounded, box_solid : Other border styles.

    Examples
    --------
    >>> box_none("hi")
    'hi'
    """
    return s


def box_double(s: str) -> str:
    """
    Wrap `s` in a double-line box (``╔══╗`` style).

    Parameters
    ----------
    s : str
        Multi-line text block to enclose.

    Returns
    -------
    str
        The boxed text block, padded so every row shares one width.

    Raises
    ------
    ValueError
        If `s` is empty or contains no lines.

    See Also
    --------
    box_block, box_rounded, box_solid, box_none : Other border styles.

    Examples
    --------
    >>> print(box_double("hi"))
    ╔════╗
    ║ hi ║
    ╚════╝
    """
    lines = _require_lines(s)
    w = _max_width(lines)
    top = f"╔{'═' * (w + 2)}╗"
    bottom = f"╚{'═' * (w + 2)}╝"
    body = [f"║ {line:<{w}} ║" for line in lines]
    return "\n".join([top, *body, bottom])


def box_rounded(s: str) -> str:
    """
    Wrap `s` in a rounded box (``╭──╮`` style).

    Parameters
    ----------
    s : str
        Multi-line text block to enclose.

    Returns
    -------
    str
        The boxed text block, padded so every row shares one width.

    Raises
    ------
    ValueError
        If `s` is empty or contains no lines.

    See Also
    --------
    box_block, box_double, box_solid, box_none : Other border styles.

    Examples
    --------
    >>> print(box_rounded("hi"))
    ╭────╮
    │ hi │
    ╰────╯
    """
    lines = _require_lines(s)
    w = _max_width(lines)
    top = f"╭{'─' * (w + 2)}╮"
    bottom = f"╰{'─' * (w + 2)}╯"
    body = [f"│ {line:<{w}} │" for line in lines]
    return "\n".join([top, *body, bottom])


def box_solid(s: str) -> str:
    """
    Wrap `s` in a solid ``▓`` background frame.

    Parameters
    ----------
    s : str
        Multi-line text block to enclose.

    Returns
    -------
    str
        The boxed text block, padded so every row shares one width.

    Raises
    ------
    ValueError
        If `s` is empty or contains no lines.

    See Also
    --------
    box_block, box_double, box_rounded, box_none : Other border styles.

    Examples
    --------
    >>> print(box_solid("hi"))
    ▓▓▓▓▓▓▓▓
    ▓  hi  ▓
    ▓▓▓▓▓▓▓▓
    """
    lines = _require_lines(s)
    w = _max_width(lines)
    bar = "▓" * (w + 6)
    body = [f"▓  {line:<{w}}  ▓" for line in lines]
    return "\n".join([bar, *body, bar])


def box_block(s: str) -> str:
    """
    Wrap `s` in a solid Unicode full-block (``█``) background frame.

    Uses ``█`` (U+2588 FULL BLOCK) for the top, bottom, and side markers,
    giving the heaviest possible monochrome terminal border. The visual
    weight complements the figlet ``block`` font, whose glyphs themselves
    employ ``█`` strokes, so the frame blends organically with the art.

    Parameters
    ----------
    s : str
        Multi-line text block to enclose.

    Returns
    -------
    str
        The boxed text block, padded so every row shares one width.
        Layout::

            ██████████████
            █  content   █
            ██████████████

    Raises
    ------
    ValueError
        If `s` is empty or contains no lines.

    See Also
    --------
    box_double, box_rounded, box_solid, box_none : Other border styles.

    Notes
    -----
    ``█`` (FULL BLOCK) is heavier than ``▓`` (DARK SHADE) used by
    `box_solid`.  Both are in the Unicode *Block Elements* range
    (U+2580-U+259F) and render reliably in any UTF-8-capable terminal
    at a fixed-width font.  The three-space interior padding (two for
    the side markers, one for breathing room each side) is identical to
    `box_solid` so the two styles are visually interchangeable in terms
    of text alignment.

    Examples
    --------
    >>> print(box_block("hi"))
    ████████
    █  hi  █
    ████████
    """
    lines = _require_lines(s)
    w = _max_width(lines)
    bar = "█" * (w + 6)
    body = [f"█  {line:<{w}}  █" for line in lines]
    return "\n".join([bar, *body, bar])


# ---------------------------------------------------------------------------
# Banner specification table
# ---------------------------------------------------------------------------

#: A border function takes the raw figlet output and returns it framed.
BoxFn = Callable[[str], str]


@dataclasses.dataclass(frozen=True)
class BannerSpec:
    """
    Declarative description of one banner variant.

    Parameters
    ----------
    filename : str
        Output filename, relative to the banner directory
        (e.g. ``"001-standard.txt"``).
    font : str
        figlet font name to render with.
    border : BoxFn
        Border-wrapping function applied to the raw figlet output.
    border_name : str
        Human-readable border label, used only for logging.
    text_case : str, optional
        Key of `CASE_TRANSFORMS` applied to the input text before figlet
        renders it (default: `DEFAULT_TEXT_CASE`, i.e. ``"none"``).
        `generate_banner` may override this per call via its
        `case_override` parameter; the CLI's ``--case`` flag does so for
        every spec at once.

    Raises
    ------
    ValueError
        If `text_case` is not a key of `CASE_TRANSFORMS`. Raised eagerly,
        at construction time, so a misconfigured spec fails at module
        import rather than later during rendering.

    See Also
    --------
    BANNER_SPECS : The canonical ordered tuple of specs used by `generate_all`.
    apply_text_case : Function that consumes `text_case` at render time.
    """

    filename: str
    font: str
    border: BoxFn
    border_name: str
    text_case: str = DEFAULT_TEXT_CASE
    kerning: bool = False

    def __post_init__(self) -> None:  # noqa: D105
        if self.text_case not in CASE_TRANSFORMS:
            valid = ", ".join(sorted(CASE_TRANSFORMS))
            raise ValueError(
                f"{self.filename}: text_case must be one of {{{valid}}}, "
                f"got {self.text_case!r}"
            )


#: Canonical banner table, mirroring the rotation order documented in
#: ``terminal-banner-ideas.md``. Index 0 is selected when
#: ``day_of_year % len(BANNER_SPECS) == 0``.
#:
#: Each entry's `text_case` is a deliberate, curated choice (not just
#: ``"none"`` everywhere) so the rotation reads as visually varied rather
#: than six copies of the same casing:
#:
#: - ``001-standard`` — ``"capitalize"``    -> "Scikit-plots": a single
#:   capital first letter, undecorated and timeless.
#: - ``002-slant``    — ``"title"``         -> "Scikit-Plots": every word
#:   capitalized, fitting its formal double-line box.
#: - ``003-shadow``   — ``"none"``          -> "scikit-plots": unchanged;
#:   the shadow font's character is in its silhouette, not its casing.
#: - ``004-small``    — ``"upper"``         -> "SCIKIT-PLOTS": full caps to
#:   stays compact and unobtrusive in its rounded box.
#: - ``005-smslant``  — ``"capitalize"``    -> "Scikit-plots": a single
#:   keep a small footprint from reading as an afterthought.
#: - ``006-big``      — ``"title"``         -> "Scikit-Plots": every word
#:   match the font's already-maximal visual weight.
#: - ``007-block``       — ``"capitalize"`` -> "Scikit-plots": a single
#:   capital first letter; the figlet ``block`` font's own ``█``-stroke
#:   glyphs are framed in a Unicode Full Block Background (``box_block``).
#: - ``008-ansi-shadow`` — ``"upper"``      -> "SCIKIT-PLOTS": full caps to
#:   unlock ANSI Shadow's canonical ``███╗``/``╔══╝`` double-layer strokes;
#:   the font itself carries its block background so no additional border
#:   is applied (``box_none``), making it the most visually striking entry
#:   in the rotation.
BANNER_SPECS: tuple[BannerSpec, ...] = (
    BannerSpec(
        "001-standard.txt", "standard", box_none, "none", text_case="capitalize"
    ),
    BannerSpec("002-slant.txt", "slant", box_double, "double-line", text_case="title"),
    BannerSpec("003-shadow.txt", "shadow", box_none, "none", text_case="none"),
    BannerSpec("004-small.txt", "small", box_rounded, "rounded", text_case="none"),
    BannerSpec("005-smslant.txt", "smslant", box_none, "none", text_case="capitalize"),
    BannerSpec("006-big.txt", "big", box_solid, "solid", text_case="title"),
    BannerSpec("007-block.txt", "block", box_block, "block", text_case="upper"),
    # https://github.com/xero/figlet-fonts
    # https://raw.githubusercontent.com/xero/figlet-fonts/refs/heads/main/ANSI%20Shadow.flf
    BannerSpec(
        "008-ansi-shadow.txt",
        "ANSI Shadow",
        box_none,
        "none",
        text_case="upper",
        kerning=True,
    ),
)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_banner(
    spec: BannerSpec,
    text: str = DEFAULT_TEXT,
    case_override: str | None = None,
) -> str:
    """
    Render a single banner described by `spec`.

    Parameters
    ----------
    spec : BannerSpec
        Banner definition (font + border + default case transform).
    text : str, optional
        Text to render (default: ``"scikit-plots"``).
    case_override : str or None, optional
        A key of `CASE_TRANSFORMS` to use instead of `spec.text_case`
        for this call (default: None, i.e. use `spec.text_case`).

    Returns
    -------
    str
        The fully rendered, bordered banner text.

    Raises
    ------
    ValueError
        If `text` is empty, or if the effective case (`case_override`
        or `spec.text_case`) is not a key of `CASE_TRANSFORMS`.
    FigletNotFoundError
        If ``figlet`` is not installed.
    BannerGenerationError
        If rendering fails.

    See Also
    --------
    apply_text_case : Applies the effective case transform to `text`
        before it reaches `figlet`.

    Examples
    --------
    >>> generate_banner(BANNER_SPECS[0], text="hi")  # doctest: +SKIP
    '...'
    """
    case = spec.text_case if case_override is None else case_override
    cased_text = apply_text_case(text, case)
    raw = figlet(cased_text, spec.font, kerning=spec.kerning)
    return spec.border(raw)


def generate_all(
    specs: Sequence[BannerSpec] = BANNER_SPECS,
    text: str = DEFAULT_TEXT,
    output_dir: pathlib.Path = DEFAULT_BANNER_DIR,
    dry_run: bool = False,
    case_override: str | None = None,
) -> dict[str, str]:
    """
    Generate every banner in `specs` and write each to `output_dir`.

    Parameters
    ----------
    specs : Sequence[BannerSpec], optional
        Banner definitions to generate (default: `BANNER_SPECS`).
    text : str, optional
        Text to render in every banner (default: ``"scikit-plots"``).
    output_dir : pathlib.Path, optional
        Directory to write banner files into; created if missing
        (default: ``.banners/``).
    dry_run : bool, optional
        If True, render banners but write no files (default: False).
    case_override : str or None, optional
        A key of `CASE_TRANSFORMS` applied to every spec, overriding each
        spec's own `text_case` (default: None, i.e. each banner keeps its
        curated default from `BANNER_SPECS`).

    Returns
    -------
    dict[str, str]
        Mapping of filename to rendered banner content, in `specs` order.

    Raises
    ------
    ValueError
        If `specs` is empty, `text` is empty, or the effective case for
        any spec is not a key of `CASE_TRANSFORMS`.
    FigletNotFoundError
        If ``figlet`` is not installed.
    BannerGenerationError
        If any individual banner fails to render.
    OSError
        If `output_dir` cannot be created, or a file cannot be written.

    See Also
    --------
    generate_banner : Generates a single banner from one `BannerSpec`.

    Notes
    -----
    Banners are generated and written one at a time, in `specs` order. If
    rendering fails partway through, banners already written in earlier
    iterations of this call remain on disk; callers needing all-or-nothing
    semantics across the whole batch should write to a temporary directory
    and rename it into place once every banner has succeeded.

    Examples
    --------
    >>> generate_all(dry_run=True)  # doctest: +SKIP
    {'001-standard.txt': '...', ...}
    """
    if not specs:
        raise ValueError("specs must contain at least one BannerSpec")

    results: dict[str, str] = {}
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        effective_case = spec.text_case if case_override is None else case_override
        logger.info(
            "Rendering %s (font=%s, border=%s, case=%s)",
            spec.filename,
            spec.font,
            spec.border_name,
            effective_case,
        )
        try:
            content = generate_banner(spec, text=text, case_override=case_override)
        except (BannerGenerationError, ValueError) as exc:
            logger.warning("Skipping %s: %s", spec.filename, exc)
            continue

        results[spec.filename] = content

        if dry_run:
            logger.debug("Dry run: skipping write for %s", spec.filename)
            continue

        target = output_dir / spec.filename
        target.write_text(content + "\n", encoding="utf-8")
        logger.info("Wrote %s (%d lines)", target, content.count("\n") + 1)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for banner generation.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Argument vector to parse. If None, ``sys.argv[1:]`` is used
        (default: None).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes ``output_dir``, ``text``,
        ``dry_run``, and ``verbose``.
    """
    parser = argparse.ArgumentParser(
        prog="scikitplot._brand._banner",
        description=(
            "Generate deterministic ASCII-art terminal banners with figlet "
            "and write them to a banner directory for day-of-year rotation."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_BANNER_DIR,
        help=f"Directory to write banner files into (default: {DEFAULT_BANNER_DIR}).",
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default=DEFAULT_TEXT,
        help=f"Text to render in each banner (default: {DEFAULT_TEXT!r}).",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Render banners and print them, but do not write any files.",
    )

    parser.add_argument(
        "--case",
        choices=sorted(CASE_TRANSFORMS),
        default=None,
        help="Override every banner's case transform.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point for CLI use.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Argument vector to parse. If None, ``sys.argv[1:]`` is used
        (default: None).

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` on failure.

    Notes
    -----
    Errors are logged with an actionable message and converted to a
    non-zero exit code rather than propagating a traceback, since this
    function is the script's outermost boundary.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.text or not args.text.strip():
        logger.error("--text must be a non-empty string")
        return 1

    try:
        results = generate_all(
            text=args.text,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            case_override=args.case,
        )
    except FigletNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except (BannerGenerationError, ValueError, OSError) as exc:
        logger.error("Banner generation failed: %s", exc)
        return 1

    if args.dry_run:
        for filename, content in results.items():
            print(f"--- {filename} ---")  # noqa: T201
            print(content)  # noqa: T201
            print()  # noqa: T201
    else:
        logger.info("Wrote %d banner(s) to %s", len(results), args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
