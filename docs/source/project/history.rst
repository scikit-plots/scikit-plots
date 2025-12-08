.. _project_history:

History
=======

This project was started in 2024.

scikit-plots logo (2024 - ).

.. seealso::

    :ref:`logo-index`

..
   This logo was added in ...

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, FancyBboxPatch, Wedge


    def _lerp_color(c1, c2, t):
        c1 = np.array(mcolors.to_rgb(c1))
        c2 = np.array(mcolors.to_rgb(c2))
        return (1 - t) * c1 + t * c2


    def draw_scikit_plots_logo(ax, seed=2):
        """
        Scikit-plots-style brand logo drawn with Matplotlib primitives only.

        No image reading, no external assets.
        """

        # --- Palette (tweak if you have official hex codes) ---
        NAVY = "#002030"  # noqa: N806
        BLUE = "#2f7fa3"  # noqa: N806
        BLUE_LIGHT = "#6fb7d2"  # noqa: N806
        ORANGE = "#c85028"  # noqa: N806
        ORANGE_LIGHT = "#e07b3a"  # noqa: N806
        BG = "white"  # noqa: N806

        # --- Canvas ---
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.axis("off")

        inner_r = 0.93
        inner_bg = Circle((0, 0), inner_r, facecolor=BG, edgecolor="none", zorder=0)
        ax.add_patch(inner_bg)

        # Clip all interior elements to a clean inner circle
        clip = Circle((0, 0), inner_r, transform=ax.transData)

        # --- Bottom bar chart (rounded, increasing) ---
        n_bars = 12
        xs = np.linspace(-0.8, 0.8, n_bars)
        width = 0.095
        baseline = -0.82
        heights = np.linspace(0.18, 0.75, n_bars)

        for i, (x, h) in enumerate(zip(xs, heights)):
            t = i / (n_bars - 1)
            col = _lerp_color(BLUE, BLUE_LIGHT, 0.6 * t)

            bar = FancyBboxPatch(
                (x - width / 2, baseline),
                width,
                h,
                boxstyle="round,pad=0.0,rounding_size=0.045",
                linewidth=0,
                facecolor=col,
                zorder=2,
            )
            bar.set_clip_path(clip)
            ax.add_patch(bar)

        # --- Rising orange "signal" dots ---
        orange_x = np.linspace(-0.15, 0.75, 7)
        orange_y = np.linspace(-0.15, 0.35, 7)
        for i, (x, y) in enumerate(zip(orange_x, orange_y)):
            c = ORANGE if i < 5 else ORANGE_LIGHT
            r = 0.055 if i < 5 else 0.045
            dot = Circle((x, y), r, facecolor=c, edgecolor="none", zorder=3)
            dot.set_clip_path(clip)
            ax.add_patch(dot)

        # --- Pie chart (top-left) ---
        pie_center = (-0.55, 0.25)
        pie_r = 0.22

        w_orange = Wedge(pie_center, pie_r, 220, 40, facecolor=ORANGE, edgecolor="none", zorder=4)
        w_blue = Wedge(pie_center, pie_r, 40, 140, facecolor=BLUE, edgecolor="none", zorder=4)
        w_gap = Wedge(pie_center, pie_r, 140, 220, facecolor=BG, edgecolor="none", zorder=4)

        for w in (w_orange, w_blue, w_gap):
            w.set_clip_path(clip)
            ax.add_patch(w)

        # --- Structured dot field (left-lower) ---
        grid_x = np.linspace(-0.75, -0.15, 5)
        grid_y = np.linspace(-0.15, -0.55, 4)

        for gx in grid_x:
            for gy in grid_y:
                col = NAVY if (int((gx + 1) * 10 + (gy + 1) * 10) % 3) else BLUE_LIGHT
                d = Circle((gx, gy), 0.045, facecolor=col, edgecolor="none", zorder=2.5)
                d.set_clip_path(clip)
                ax.add_patch(d)

        # --- Light decorative dots across upper area (deterministic) ---
        rng = np.random.default_rng(seed)
        pts = rng.uniform(-0.8, 0.8, size=(18, 2))
        for x, y in pts:
            if x * x + y * y > inner_r * inner_r:
                continue
            if y < -0.2 and x > -0.2:  # noqa: PLR2004
                continue

            col = NAVY if rng.random() < 0.6 else BLUE_LIGHT  # noqa: PLR2004
            r = rng.uniform(0.025, 0.045)
            d = Circle((x, y), r, facecolor=col, edgecolor="none", alpha=0.95, zorder=2)
            d.set_clip_path(clip)
            ax.add_patch(d)

        # --- Trend line with hollow nodes ---
        line_pts = np.array([[-0.05, -0.05], [0.25, 0.10], [0.55, 0.22]])
        line = Line2D(
            line_pts[:, 0],
            line_pts[:, 1],
            color=NAVY,
            linewidth=10,
            solid_capstyle="round",
            zorder=4,
        )
        line.set_clip_path(clip)
        ax.add_line(line)

        for (x, y) in line_pts[1:]:
            outer_node = Circle((x, y), 0.07, facecolor=NAVY, edgecolor="none", zorder=5)
            inner_node = Circle((x, y), 0.035, facecolor=BG, edgecolor="none", zorder=6)
            for n in (outer_node, inner_node):
                n.set_clip_path(clip)
                ax.add_patch(n)

        # --- "Spark" icon near top (stylized) ---
        spark_center = (0.02, 0.65)
        arms = 6
        for k in range(arms):
            ang = k * np.pi / arms
            dx = 0.12 * np.cos(ang)
            dy = 0.12 * np.sin(ang)

            l = Line2D(
                [spark_center[0] - dx, spark_center[0] + dx],
                [spark_center[1] - dy, spark_center[1] + dy],
                color=NAVY,
                linewidth=6,
                solid_capstyle="round",
                zorder=4,
            )
            l.set_clip_path(clip)
            ax.add_line(l)

        hole = Circle(spark_center, 0.04, facecolor=BG, edgecolor="none", zorder=5)
        hole.set_clip_path(clip)
        ax.add_patch(hole)

        # --- Fixed accent dots (right-upper) ---
        accents = [
            (0.75, 0.25, ORANGE),
            (0.68, 0.45, ORANGE_LIGHT),
            (0.50, 0.50, BLUE_LIGHT),
            (0.35, 0.60, NAVY),
        ]
        for x, y, c in accents:
            d = Circle((x, y), 0.04, facecolor=c, edgecolor="none", zorder=4)
            d.set_clip_path(clip)
            ax.add_patch(d)

        # --- Outer ring last ---
        outer = Circle((0, 0), 1.0, facecolor="none", edgecolor=NAVY, linewidth=18, zorder=10)
        ax.add_patch(outer)


    def scikit_plots_logo(figsize=(4, 4), dpi=200, seed=2):  # noqa: D103
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        draw_scikit_plots_logo(ax, seed=seed)
        fig.tight_layout(pad=0)
        return fig, ax


    if __name__ == "__main__":
        fig, _ = scikit_plots_logo(figsize=(4, 4), dpi=200, seed=2)

        # Vector-first export
        fig.savefig("scikit-plots-logo-mpl.svg", transparent=True)
        fig.savefig("scikit-plots-logo-mpl.png", transparent=True)

        plt.show()

..
   import numpy as np
   import matplotlib as mpl
   import matplotlib.pyplot as plt
   import matplotlib.cm as cm

   mpl.rcParams['xtick.labelsize'] = 10
   mpl.rcParams['ytick.labelsize'] = 12
   mpl.rcParams['axes.edgecolor'] = 'gray'


   axalpha = 0.05
   figcolor = 'white'
   dpi = 80
   fig = plt.figure(figsize=(6, 1.1), dpi=dpi)
   fig.patch.set_edgecolor(figcolor)
   fig.patch.set_facecolor(figcolor)


   def add_math_background():
       ax = fig.add_axes([0., 0., 1., 1.])

       text = []
       text.append(
           (r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = "
            r"U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2}"
            r"\int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 "
            r"\left[\frac{ U^{2\beta}_{\delta_1 \rho_1} - "
            r"\alpha^\prime_2U^{1\beta}_{\rho_1 \sigma_2} "
            r"}{U^{0\beta}_{\rho_1 \sigma_2}}\right]$", (0.7, 0.2), 20))
       text.append((r"$\frac{d\rho}{d t} + \rho \vec{v}\cdot\nabla\vec{v} "
                    r"= -\nabla p + \mu\nabla^2 \vec{v} + \rho \vec{g}$",
                    (0.35, 0.9), 20))
       text.append((r"$\int_{-\infty}^\infty e^{-x^2}dx=\sqrt{\pi}$",
                    (0.15, 0.3), 25))
       text.append((r"$F_G = G\frac{m_1m_2}{r^2}$",
                    (0.85, 0.7), 30))
       for eq, (x, y), size in text:
            ax.text(x, y, eq, ha='center', va='center', color="#11557c",
                   alpha=0.25, transform=ax.transAxes, fontsize=size)
       ax.set_axis_off()
       return ax


   def add_matplotlib_text(ax):
       ax.text(0.95, 0.5, 'matplotlib', color='#11557c', fontsize=65,
               ha='right', va='center', alpha=1.0, transform=ax.transAxes)


   def add_polar_bar():
       ax = fig.add_axes([0.025, 0.075, 0.2, 0.85], projection='polar')

       ax.patch.set_alpha(axalpha)
       ax.set_axisbelow(True)
       N = 7
       arc = 2. * np.pi
       theta = np.arange(0.0, arc, arc/N)
       radii = 10 * np.array([0.2, 0.6, 0.8, 0.7, 0.4, 0.5, 0.8])
       width = np.pi / 4 * np.array([0.4, 0.4, 0.6, 0.8, 0.2, 0.5, 0.3])
       bars = ax.bar(theta, radii, width=width, bottom=0.0)
       for r, bar in zip(radii, bars):
           bar.set_facecolor(cm.jet(r/10.))
           bar.set_alpha(0.6)

       ax.tick_params(labelbottom=False, labeltop=False,
                      labelleft=False, labelright=False)

       ax.grid(lw=0.8, alpha=0.9, ls='-', color='0.5')

       ax.set_yticks(np.arange(1, 9, 2))
       ax.set_rmax(9)


   main_axes = add_math_background()
   add_polar_bar()
   add_matplotlib_text(main_axes)
