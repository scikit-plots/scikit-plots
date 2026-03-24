# galleries/examples/misc/plot_misc_script.py
#
# Authors: The scikit-plot developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Misc Showcase
=============

This gallery example demonstrates the
:py:class:`~scikitplot.misc.` module.
"""

# %%
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from scikitplot.misc import (
    closest_color_name,
    display_colors,
    plot_colortable,
    plot_overlapping_colors,
)

# %%

closest_color_name()

# %%

display_colors(['red', 'blue'], show_indices=True)

# %%

fig = plot_colortable()

# %%


# https://xkcd.com/color/rgb/
# https://xkcd.com/color/rgb.txt
# https://www.w3schools.com/cssref/css_colors.php
fig = plot_overlapping_colors()


# %%
#
# .. tags::
#
#    plot-type: barh
#    plot-type: text
#    level: beginner
#    purpose: showcase
