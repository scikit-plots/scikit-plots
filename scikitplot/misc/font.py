"""
misc
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from PIL import ImageFont

__all__ = ["get_font"]


def get_font():
    import platform

    system_platform = platform.system().lower()
    # Detect platform and select font accordingly
    try:
        if system_platform == "windows":
            return ImageFont.truetype("arial.ttf", 32)
        if system_platform == "darwin":  # macOS
            return ImageFont.truetype(
                "/Library/Fonts/Arial.ttf", 32
            )  # or "/System/Library/Fonts/Helvetica.ttc"
        if system_platform == "linux":
            # Try a more common font path
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32
            )
        raise ValueError("Unsupported platform")
    except OSError:
        # Fallback font if the specified font is not found
        print("Font not found, using default font.")  # noqa: T201
        return ImageFont.load_default()


# Example usage
# _default_font = get_font()
