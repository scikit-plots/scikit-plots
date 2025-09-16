"""
plot_rocplot_script with examples
==========================================

An example showing the :py:func:`~scikitplot.snsx.rocplot` function
used by a scikit-learn regressor.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Import scikit-plot
import scikitplot.snsx as sp


# %%
ax = sp.rocplot(
    x=[0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    y=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8  , 0.9, 1.0],
)


# %%
import numpy as np; np.random.seed(0)  # reproducibility
import pandas as pd
df = pd.DataFrame({
    "true": np.random.normal(0.5, 0.1, 100).round(),
    "prob": np.random.normal(0.5, 0.1, 100),
    "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]*10
})


# %%
# PR Curve
# ax = sp.prplot(x=df.true, y=df.prob, hue=df.group)
ax = sp.rocplot(df, x="true", y="prob", label=f"classA")


# %%
for i in range(10):
    np.random.seed(i)  # reproducibility
    # df = pd.DataFrame({
    #     "true": np.random.normal(0.5, 0.1, 100).round(),
    #     "prob": np.random.normal(0.5, 0.1, 100),
    #     "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]*10
    # })
    # ax = sp.prplot(df, x="true", y="prob", label=f"{i}")

    ax = sp.rocplot(
        x=np.random.normal(0.5, 0.1, 100).round(),
        y=np.random.normal(0.5, 0.1, 100),
        label=f"{i}",
    )

    # --- Collect unique handles and labels ---
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # deduplicate

    # Override legend
    ax.legend(by_label.values(), by_label.keys(), title="Legend")
