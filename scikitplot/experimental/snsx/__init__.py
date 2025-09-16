from numbers import Number
from functools import partial
import math
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection

from sklearn.metrics import precision_recall_curve, roc_curve, auc

from ...externals._seaborn.distributions import _DistributionPlotter


class ROCPRPlotter(_DistributionPlotter):
    """
    Precision-Recall and ROC Curve Plotter extending _DistributionPlotter.

    Parameters
    ----------
    data : DataFrame
        The dataset containing true labels, scores, and optional hue.
    variables : dict
        Dictionary like {"y_true": "true_label", "y_score": "pred_score", "hue": "model"}.
    """

    def __init__(self, data=None, variables={}, ax=None):
        super().__init__(data=data, variables=variables)
        self.ax = ax  # provide default axis

    def _plot_curve(self, x, y, label=None, alpha=0.7, **plot_kws):
        """
        Internal helper to plot a line on the axes with optional label.
        """
        ax = self.ax if self.ax is not None else plt.gca()
        ax.plot(x, y, label=label, alpha=alpha, **plot_kws)
        return ax

    def plot_precision_recall(self, alpha=0.7, legend=True, **plot_kws):
        """
        Plot Precision-Recall curves for each hue level.
        """
        ax = self.ax if self.ax is not None else plt.gca()

        for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):
            y_true = sub_data[self.variables["y_true"]]
            y_score = sub_data[self.variables["y_score"]]
            precision, recall, _ = precision_recall_curve(y_true, y_score)

            color = self._hue_map(sub_vars["hue"]) if "hue" in self.variables else plot_kws.pop("color", "C0")
            self._plot_curve(recall, precision, label=sub_vars.get("hue"), alpha=alpha, color=color, **plot_kws)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")

        if "hue" in self.variables and legend:
            ax.legend(title=self.variables["hue"])

    def plot_roc_auc(self, alpha=0.7, legend=True, **plot_kws):
        """
        Plot ROC curves with AUC annotation for each hue level.
        """
        ax = self.ax if self.ax is not None else plt.gca()

        for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):
            y_true = sub_data[self.variables["y_true"]]
            y_score = sub_data[self.variables["y_score"]]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            color = self._hue_map(sub_vars["hue"]) if "hue" in self.variables else plot_kws.pop("color", "C0")
            label = f"{sub_vars.get('hue')} (AUC={roc_auc:.2f})" if "hue" in self.variables else f"AUC={roc_auc:.2f}"
            self._plot_curve(fpr, tpr, label=label, alpha=alpha, color=color, **plot_kws)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")

        if "hue" in self.variables and legend:
            ax.legend(title=self.variables["hue"])
