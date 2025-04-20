"""utils.py"""

import os
import warnings
import platform
import logging

from typing import TYPE_CHECKING
from typing import (
    # Dict,
    Optional,
    Union,
)

# Attempt to import `cache` from `functools` (Python >= 3.9)
try:
    from functools import cache
except ImportError:
    # Fallback to `lru_cache` for Python < 3.9
    from functools import lru_cache as cache

import aggdraw  # Anti-Grain Geometry (AGG) graphics library
from PIL import (
    Image,
    ImageColor,
    ImageDraw,
    ImageFont,
)

# Set up logging
logging.basicConfig(level=logging.INFO)


if TYPE_CHECKING:  # Only imported during type checking
    from typing import Any


## Define __all__ to specify the public interface of the module
__all__ = [
    "Box",
    "Circle",
    "ColorWheel",
    "Ellipses",
    "RectShape",
    "fade_color",
    "get_keys_by_value",
    "get_rgba_tuple",
    "linear_layout",
    "self_multiply",
    "vertical_image_concat",
    "save_image_safely",
    "get_font",
]

######################################################################
## save_image_safely
######################################################################


def save_image_safely(
    img: Image.Image,
    to_file: str,
    use_matplotlib: bool = False,
    show_os_viewer: bool = False,
    dpi: int = None,
) -> None:
    """
    Save and optionally display an image using either Matplotlib or PIL.

    Parameters
    ----------
    img : PIL.Image.Image
        The image to save.

    to_file : str
        Full path to save the image (e.g., "output/image.png").
        If the directory does not exist, it will be created.

    use_matplotlib : bool, optional
        If True, use Matplotlib to save (and show) the image.
        This method provides better control over DPI, transparency, and layout.
        If False, uses PIL for a lightweight save. Default is False.

    show_os_viewer : bool, optional
        If True, displays the saved image (by PIL) in the system's default image viewer
        using PIL's `.show()` method. Default is False.

    dpi : int, optional
        The resolution in dots per inch when saving the image using Matplotlib or PIL.
        Only has an effect when supported by the format and method. Default is None.

    Notes
    -----
    - Falls back to PIL if Matplotlib fails, even if `use_matplotlib=True`.
    - `.show()` opens the image in the OS image viewer, which may block execution
      depending on the environment.
    - This function is designed for flexible debugging, not for high-performance pipelines.
    - Matplotlib output includes anti-aliasing, transparency control, and tighter layout.
    """
    # Ensure the directory exists before saving the image
    os.makedirs(os.path.dirname(to_file), exist_ok=True)
    if use_matplotlib:
        try:
            import matplotlib.pyplot as plt

            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
            # plt.draw()
            # plt.pause(0.1)  # Pause to allow for interactive drawing
            try:
                # Save the image using Matplotlib after showing it
                plt.savefig(to_file, dpi=dpi, bbox_inches="tight", pad_inches=0)
                print(f"Image saved using Matplotlib: {to_file}")
            except Exception as e:
                print(f"[ERROR] Failed to save plot: {e}")
            plt.show()
            # plt.gcf().clear()  # Clear the figure after saving
            # plt.close()
        except Exception as e:
            warnings.warn(
                f"[Matplotlib] Failed to save or show image: {e}. Falling back to PIL."
            )
            try:
                # Using PIL to save the image (default method)
                # img.save(to_file, dpi=(300, 300))  # dpi is ignored in PIL unless saving to PDF
                img.save(to_file)
                print(f"Image saved using PIL: {to_file}")
                if show_os_viewer:
                    # Will open in the system's default viewer
                    img.show()
            except Exception as pil_error:
                warnings.warn(f"[PIL] Final fallback failed: {pil_error}")
    else:
        try:
            # Using PIL to save the image (default method)
            # img.save(to_file, dpi=(300, 300))  # dpi is ignored in PIL unless saving to PDF
            img.save(to_file)
            print(f"Image saved using PIL: {to_file}")
            if show_os_viewer:
                # Will open in the system's default viewer
                img.show()
        except Exception as e:
            warnings.warn(f"[PIL] Could not save image to '{to_file}': {e}")


######################################################################
## get_font
######################################################################


@cache
def _cached_truetype(path: str, size: int) -> ImageFont.ImageFont:
    """
    Load a TrueType font with the specified path and size.

    Parameters
    ----------
    path : str
        Path to the TrueType (.ttf or .otf) font file.

    size : int
        Font size to be used.

    Returns
    -------
    ImageFont.ImageFont
        A loaded TrueType font object.

    Raises
    ------
    OSError
        If the font file cannot be opened or read.
    """
    return ImageFont.truetype(path, size)


def validate_font_size(font_size: Optional[int]) -> int:
    """
    Validate that the font size is a positive integer.

    Parameters
    ----------
    font_size : int or None
        The font size to validate.

    Returns
    -------
    int
        The validated font size.

    Raises
    ------
    ValueError
        If `font_size` is not a positive integer.
    """
    if not isinstance(font_size, int) or font_size <= 0:
        raise ValueError("font_size must be a positive integer.")
    return font_size


def load_default_font(font_size: int = 10) -> ImageFont.ImageFont:
    """
    Load the default PIL font with version compatibility for different Pillow versions.

    Parameters
    ----------
    font_size : int, optional
        Font size to apply. Will be used if supported by the current Pillow version.
        Default is 10.

    Returns
    -------
    ImageFont.ImageFont
        A default font object suitable for drawing text.
    """
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        # For Pillow versions < 9.2.0 that do not support the `size` argument
        return ImageFont.load_default()


def load_font(
    font_path: Optional[str] = None,
    font_size: int = 10,
    use_default_font: bool = True,
    verbose: bool = False,
) -> ImageFont.ImageFont:
    """
    Get a font object for text rendering, with an optional custom font path and size.

    Parameters
    ----------
    font_path : str, optional
        Path to the font file. If None, the default system font will be used.
        Default is None.

    font_size : int, optional
        Size of the font to load. Must be a positive integer. Default is 10.

    use_default_font : bool, optional
        If True, directly uses `ImageFont.load_default(size=font_size)`
        regardless of the platform or custom font path. Default is True.

    verbose : bool, optional
        If True, prints font path used. Default is False.

    Returns
    -------
    ImageFont.ImageFont
        The font object that can be used for text rendering.
    """
    font_size = validate_font_size(font_size)

    # Use PIL's default font directly
    if use_default_font:
        if verbose:
            print("Using PIL default font.")
        load_default_font(font_size=font_size)

    # Try loading custom font
    if font_path:
        supported_exts = os.getenv("SUPPORTED_EXTS", ".ttf .otf .ttc").split()
        if os.path.exists(font_path) and font_path.lower().endswith(supported_exts):
            try:
                if verbose:
                    print(f"Using custom font: {font_path}")
                return _cached_truetype(font_path, font_size)
            except OSError as e:
                logging.error(f"Failed to load font from '{font_path}': {e}")
        else:
            logging.warning(f"Invalid font path or unsupported font file: {font_path}")

    # Platform-specific fallback
    try:
        system_platform = platform.system().lower()
        default_font_paths = {
            # "windows": os.getenv("DEFAULT_WINDOWS_FONT", "C:/Windows/Fonts/segoeui.ttf"),
            "windows": os.getenv("DEFAULT_WINDOWS_FONT", "C:/Windows/Fonts/arial.ttf"),
            # "darwin": os.getenv("DEFAULT_MAC_FONT", "/System/Library/Fonts/Helvetica.ttc"),
            "darwin": os.getenv("DEFAULT_MAC_FONT", "/Library/Fonts/Arial.ttf"),
            # "DEFAULT_LINUX_FONT", "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
            # "DEFAULT_LINUX_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            "linux": os.getenv(
                "DEFAULT_LINUX_FONT",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            ),
        }
        system_font_path = default_font_paths.get(system_platform)

        if system_font_path:
            if verbose:
                print(f"Using system font: {system_font_path}")
            return _cached_truetype(system_font_path, font_size)
    except (OSError, ValueError) as e:
        logging.warning(f"Error loading system font: {e}")

    logging.warning("Falling back to PIL default font.")
    return load_default_font(font_size=font_size)


def get_font(
    font: Union[ImageFont.ImageFont, dict, None] = None,
) -> ImageFont.ImageFont:
    """
    Obtain a font object, either directly or by resolving parameters via `load_font`.

    Parameters
    ----------
    font : ImageFont.ImageFont or dict, optional
        If an ImageFont object, it is returned as is.
        If a dictionary, it is passed to `load_font()` to generate the font.

    Returns
    -------
    ImageFont.ImageFont
        Font object usable with PIL's drawing functions.

    Notes
    -----
    Pass `{"use_default_font": True, "font_size": 14}` to override size using default font.
    """
    if isinstance(font, ImageFont.ImageFont):
        return font

    if isinstance(font, dict):
        return load_font(**font)

    # for None
    return load_font()


######################################################################
## base
######################################################################


class RectShape:
    x1: int
    x2: int
    y1: int
    y2: int
    _fill: "Any"
    _outline: "Any"

    @property
    def fill(self):
        return self._fill

    @property
    def outline(self):
        return self._outline

    @fill.setter
    def fill(self, v):
        self._fill = get_rgba_tuple(v)

    @outline.setter
    def outline(self, v):
        self._outline = get_rgba_tuple(v)

    def _get_pen_brush(self):
        pen = aggdraw.Pen(self._outline)
        brush = aggdraw.Brush(self._fill)
        return pen, brush


class Box(RectShape):
    de: int
    shade: int

    def draw(self, draw: ImageDraw, draw_reversed: bool = False):
        pen, brush = self._get_pen_brush()

        if hasattr(self, "de") and self.de > 0:
            brush_s1 = aggdraw.Brush(fade_color(self.fill, self.shade))
            brush_s2 = aggdraw.Brush(fade_color(self.fill, 2 * self.shade))
            if draw_reversed:
                draw.line(
                    [
                        self.x2 - self.de,
                        self.y1 - self.de,
                        self.x2 - self.de,
                        self.y2 - self.de,
                    ],
                    pen,
                )
                draw.line(
                    [
                        self.x2 - self.de,
                        self.y2 - self.de,
                        self.x2,
                        self.y2,
                    ],
                    pen,
                )
                draw.line(
                    [
                        self.x1 - self.de,
                        self.y2 - self.de,
                        self.x2 - self.de,
                        self.y2 - self.de,
                    ],
                    pen,
                )
                draw.polygon(
                    [
                        self.x1,
                        self.y1,
                        self.x1 - self.de,
                        self.y1 - self.de,
                        self.x2 - self.de,
                        self.y1 - self.de,
                        self.x2,
                        self.y1,
                    ],
                    pen,
                    brush_s1,
                )
                draw.polygon(
                    [
                        self.x1 - self.de,
                        self.y1 - self.de,
                        self.x1,
                        self.y1,
                        self.x1,
                        self.y2,
                        self.x1 - self.de,
                        self.y2 - self.de,
                    ],
                    pen,
                    brush_s2,
                )
            else:
                draw.line(
                    [
                        self.x1 + self.de,
                        self.y1 - self.de,
                        self.x1 + self.de,
                        self.y2 - self.de,
                    ],
                    pen,
                )
                draw.line(
                    [
                        self.x1 + self.de,
                        self.y2 - self.de,
                        self.x1,
                        self.y2,
                    ],
                    pen,
                )
                draw.line(
                    [
                        self.x1 + self.de,
                        self.y2 - self.de,
                        self.x2 + self.de,
                        self.y2 - self.de,
                    ],
                    pen,
                )
                draw.polygon(
                    [
                        self.x1,
                        self.y1,
                        self.x1 + self.de,
                        self.y1 - self.de,
                        self.x2 + self.de,
                        self.y1 - self.de,
                        self.x2,
                        self.y1,
                    ],
                    pen,
                    brush_s1,
                )
                draw.polygon(
                    [
                        self.x2 + self.de,
                        self.y1 - self.de,
                        self.x2,
                        self.y1,
                        self.x2,
                        self.y2,
                        self.x2 + self.de,
                        self.y2 - self.de,
                    ],
                    pen,
                    brush_s2,
                )

        draw.rectangle([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Circle(RectShape):
    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()
        draw.ellipse([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Ellipses(RectShape):
    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()
        w = self.x2 - self.x1
        d = int(w / 7)
        draw.ellipse(
            [
                self.x1 + (w - d) / 2,
                self.y1 + 1 * d,
                self.x1 + (w + d) / 2,
                self.y1 + 2 * d,
            ],
            pen,
            brush,
        )
        draw.ellipse(
            [
                self.x1 + (w - d) / 2,
                self.y1 + 3 * d,
                self.x1 + (w + d) / 2,
                self.y1 + 4 * d,
            ],
            pen,
            brush,
        )
        draw.ellipse(
            [
                self.x1 + (w - d) / 2,
                self.y1 + 5 * d,
                self.x1 + (w + d) / 2,
                self.y1 + 6 * d,
            ],
            pen,
            brush,
        )


class ColorWheel:
    def __init__(self, colors: list = None):
        self._cache = dict()
        # The default color cycle is defined by the following colors:
        # self.colors = colors if colors is not None else [
        #   "#ffd166", "#ef476f", "#118ab2", "#073b4c", "#842da1", "#ffbad4", "#fe9775", "#83d483", "#06d6a0", "#0cb0a9"
        # ]
        self.colors = (
            colors
            if colors is not None
            else [
                "gray",
                "orange",
                "red",
                "pink",
                "salmon",
                "olive",
                "limegreen",
                "green",
                "dodgerblue",
                "cyan",
                "blue",
                "purple",
                "brown",
            ]
        )

    def get_color(self, class_type: type):
        if class_type not in self._cache.keys():
            index = len(self._cache.keys()) % len(self.colors)
            self._cache[class_type] = self.colors[index]
        return self._cache.get(class_type)


def fade_color(color: tuple, fade_amount: int) -> tuple:
    r = max(0, color[0] - fade_amount)
    g = max(0, color[1] - fade_amount)
    b = max(0, color[2] - fade_amount)
    return r, g, b, color[3]


def get_rgba_tuple(color: "Any") -> tuple:
    """
    :param color:
    :return: (R, G, B, A) tuple
    """
    if isinstance(color, tuple):
        rgba = color
    elif isinstance(color, int):
        rgba = (color >> 16 & 0xFF, color >> 8 & 0xFF, color & 0xFF, color >> 24 & 0xFF)
    else:
        rgba = ImageColor.getrgb(color)

    if len(rgba) == 3:
        rgba = (rgba[0], rgba[1], rgba[2], 255)
    return rgba


def get_keys_by_value(d, v):
    for key in d.keys():  # reverse search the dict for the value
        if d[key] == v:
            yield key


def self_multiply(tensor_tuple: tuple):
    """
    :param tensor_tuple:
    :return:
    """
    tensor_list = list(tensor_tuple)
    if None in tensor_list:
        tensor_list.remove(None)
    if len(tensor_list) == 0:
        return 0
    s = tensor_list[0]
    for i in range(1, len(tensor_list)):
        s *= tensor_list[i]
    return s


def vertical_image_concat(im1: Image, im2: Image, background_fill: "Any" = "white"):
    """
    Vertical concatenation of two PIL images.

    :param im1: top image
    :param im2: bottom image
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :return: concatenated image
    """
    dst = Image.new(
        "RGBA", (max(im1.width, im2.width), im1.height + im2.height), background_fill
    )
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def linear_layout(
    images: list,
    max_width: int = -1,
    max_height: int = -1,
    horizontal: bool = True,
    padding: int = 0,
    spacing: int = 0,
    background_fill: "Any" = "white",
):
    """
    Creates a linear layout of a passed list of images in horizontal or vertical orientation. The layout will wrap in x
    or y dimension if a maximum value is exceeded.

    :param images: List of PIL images
    :param max_width: Maximum width of the image. Only enforced in horizontal orientation.
    :param max_height: Maximum height of the image. Only enforced in vertical orientation.
    :param horizontal: If True, will draw images horizontally, else vertically.
    :param padding: Top, bottom, left, right border distance in pixels.
    :param spacing: Spacing in pixels between elements.
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :return:
    """
    coords = list()
    width = 0
    height = 0

    x, y = padding, padding

    for img in images:
        if horizontal:
            if max_width != -1 and x + img.width > max_width:
                # make a new row
                x = padding
                y = height - padding + spacing
            coords.append((x, y))

            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)

            x += img.width + spacing
        else:
            if max_height != -1 and y + img.height > max_height:
                # make a new column
                x = width - padding + spacing
                y = padding
            coords.append((x, y))

            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)

            y += img.height + spacing

    layout = Image.new("RGBA", (width, height), background_fill)
    for img, coord in zip(images, coords):
        layout.paste(img, coord)

    return layout
