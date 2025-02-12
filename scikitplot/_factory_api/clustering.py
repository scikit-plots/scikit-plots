"""
The Scikit-plots Factory API

This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""

# code that needs to be compatible with both Python 2 and Python 3

import types
import warnings

import six
from sklearn.utils import deprecated

import scikitplot as skplt

## Define __all__ to specify the public interface of the module,
## not required default all belove func
__all__ = ["clustering_factory"]


@deprecated(
    "This will be removed in v0.4.0. The Factory "
    "API has been deprecated. Please migrate "
    "existing code into the various new modules "
    "of the Functions API. Please note that the "
    "interface of those functions will likely be "
    "different from that of the Factory API."
)
def clustering_factory(clf):
    """
    Embeds scikit-plot plotting methods in an sklearn clusterer instance.

    Args:
        clf: Scikit-learn clusterer instance

    Returns:
        The same scikit-learn clusterer instance passed in **clf** with
        embedded scikit-plot instance methods.

    Raises:
        ValueError: If **clf** does not contain the instance methods necessary
            for scikit-plot instance methods.

    """
    required_methods = ["fit", "fit_predict"]

    for method in required_methods:
        if not hasattr(clf, method):
            raise TypeError(
                f'"{method}" is not in clf. Did you pass a clusterer instance?'
            )

    additional_methods = {
        "plot_silhouette": skplt.api.plotters.plot_silhouette,
        "plot_elbow_curve": skplt.api.plotters.plot_elbow_curve,
    }

    for key, fn in six.iteritems(additional_methods):
        if hasattr(clf, key):
            warnings.warn(
                f'"{key}" method already in clf. '
                "Overriding anyway. This may "
                "result in unintended behavior."
            )
        setattr(clf, key, types.MethodType(fn, clf))
    return clf
