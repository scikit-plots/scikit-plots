"""
Reset utility functions to restore global states for popular ML/plotting libraries.

This is useful when building galleries (e.g., with Sphinx-Gallery), testing pipelines, or ensuring
state consistency across example scripts, notebooks, or web apps (e.g., Streamlit, Dash).

Includes support for:
- scikit-learn
- matplotlib
- seaborn
- numpy

Usage (Sphinx-Gallery):
------------------------
sphinx_gallery_conf = {
    # ...
    'reset_modules': 'scikitplot._reset.reset',
}

Optional: Use atexit or test fixtures:
--------------------------------------
import atexit
atexit.register(reset)

# In test/conftest.py
import pytest
from scikitplot._reset import reset

@pytest.fixture(autouse=True)
def clean_modules():
    yield
    reset()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "reset",
    "reset_matplotlib",
    "reset_numpy",
    "reset_seaborn",
    "reset_sklearn",
]

# ----------------------
# Numpy: capture global config
# ----------------------

_numpy_err_config = np.geterr()
_numpy_print_config = np.get_printoptions()


def reset_numpy():
    """Reset NumPy error handling and print options to their defaults."""
    np.seterr(**_numpy_err_config)
    np.set_printoptions(**_numpy_print_config)


# ----------------------
# Scikit-learn
# ----------------------

try:
    import sklearn

    _default_sklearn_config = sklearn.get_config().copy()

    def reset_sklearn():
        """Reset scikit-learn global config to initial state."""
        sklearn.set_config(**_default_sklearn_config)

except ImportError:

    def reset_sklearn():
        """Reset scikit-learn global config to initial state."""
        # If sklearn isn't installed


# ----------------------
# Matplotlib
# ----------------------


def reset_matplotlib():
    """Close all plots and reset matplotlib style and rcParams."""
    plt.close("all")
    mpl.rcdefaults()
    try:  # noqa: SIM105
        mpl.style.use("default")
    except Exception:
        pass  # In case default style isn't available


# ----------------------
# Seaborn
# ----------------------

try:
    import seaborn as sns

    def reset_seaborn():
        """Reset seaborn styles to default."""
        sns.reset_defaults()
        try:
            sns.set_theme()  # Preferred reset from seaborn >=0.11
        except Exception:
            sns.set()

except ImportError:

    def reset_seaborn():
        """Reset seaborn styles to default."""
        # seaborn not available


# ----------------------
# Aggregate Reset
# ----------------------


def reset(_gallery_conf=None, _fname=None):
    """
    Reset global state for NumPy, scikit-learn, matplotlib, and seaborn.

    This function is designed to restore key scientific computing and visualization
    libraries to their **default configurations**. It helps prevent state leakage
    between different scripts, examples, or plotting sessions.

    This is particularly useful in:
        - Documentation builds using Sphinx-Gallery
        - Interactive notebook development (Jupyter, Colab)
        - Streamlit or Dash applications (reproducibility between reruns)
        - Automated testing pipelines
        - Educational or demo environments with sequential examples

    What this resets:
    -----------------
    - **NumPy**: Restores default error handling (`np.seterr`)
      and print formatting (`np.set_printoptions`).
    - **scikit-learn**: Resets global configuration via `sklearn.set_config`
      (e.g., `assume_finite`, working memory size).
    - **matplotlib**: Closes all open figures, resets `rcParams`, and applies the default style.
    - **seaborn** (if installed): Resets to default theme and styling to avoid residual settings.

    Parameters
    ----------
    _gallery_conf : dict, optional
        Required by Sphinx-Gallery's `reset_modules` hook, but ignored in this function.

    _fname : str, optional
        Filename of the example being executed. Included for compatibility with Sphinx-Gallery,
        but not used internally.

    Notes
    -----
    - This function is safe to call repeatedly; it is **idempotent**.
    - In Sphinx-Gallery, you can register this with:

        >>> sphinx_gallery_conf = {
        >>>     'reset_modules': 'scikitplot._reset.reset'
        >>> }

    - For testing or production environments, you may register it via `atexit` or use
      as a pytest fixture:

        >>> import atexit
        >>> atexit.register(reset)

        >>> @pytest.fixture(autouse=True)
        >>> def clean_state():
        >>>     yield
        >>>     reset()

    Examples
    --------
    >>> from scikitplot._reset import reset
    >>> reset()  # safely clean state between plotting functions

    See Also
    --------
    reset_numpy : Resets NumPy global state
    reset_sklearn : Resets scikit-learn configuration
    reset_matplotlib : Resets matplotlib figures and config
    reset_seaborn : Resets seaborn theme (if available)
    """
    reset_numpy()
    reset_sklearn()
    reset_matplotlib()
    reset_seaborn()
