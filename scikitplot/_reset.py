"""
Utility functions to reset global states for common ML and plotting libraries.

This module is useful when building galleries (e.g., with Sphinx-Gallery),
running tests, or maintaining consistent environments across scripts,
notebooks, and web apps (e.g., Streamlit, Dash).

Currently supported libraries:
- scikit-learn
- matplotlib
- seaborn
- numpy

Examples
--------
**Usage in Sphinx-Gallery configuration**:

.. code-block:: python

   sphinx_gallery_conf = {
       # ...
       "reset_modules": "scikitplot.reset",
   }

**Optional: Register via atexit**:

.. code-block:: python

   import atexit
   from scikitplot._reset import reset

   atexit.register(reset)

**With pytest (e.g., in `tests/conftest.py`)**:

.. code-block:: python

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

    This function restores common scientific computing and visualization libraries
    to their **default configurations**, preventing state leakage between scripts,
    examples, or plotting sessions.

    It is especially useful in:
    - Documentation builds (e.g., with Sphinx-Gallery)
    - Interactive notebooks (Jupyter, Colab)
    - Streamlit or Dash apps (ensuring consistent reruns)
    - Automated testing pipelines
    - Educational or demo environments

    What gets reset:
    - **NumPy**: Restores default error handling (`np.seterr`) and print formatting
      (`np.set_printoptions`).
    - **scikit-learn**: Resets global configuration via `sklearn.set_config`, including
      options like `assume_finite` and `working_memory`.
    - **matplotlib**: Closes all open figures, resets `rcParams`, and reapplies the default style.
    - **seaborn** (if installed): Restores default theme and style to avoid residual settings.

    Parameters
    ----------
    _gallery_conf : dict, optional
        Required for compatibility with Sphinx-Gallery's `reset_modules` hook.
        Not used internally.

    _fname : str, optional
        Filename of the current example (provided by Sphinx-Gallery).
        Not used internally.

    Notes
    -----
    - This function is **idempotent** and safe to call multiple times.
    - To use with Sphinx-Gallery, register it as:

      >>> sphinx_gallery_conf = {
      ...     "reset_modules": "scikitplot.reset",
      ... }

    - You can also register it with `atexit` or use it in a `pytest` fixture:

      >>> import atexit
      >>> atexit.register(reset)

      >>> import pytest
      >>> @pytest.fixture(autouse=True)
      ... def clean_state():
      ...     yield
      ...     reset()

    Examples
    --------
    >>> from scikitplot._reset import reset
    >>> reset()  # safely restore global state between examples

    See Also
    --------
    reset_numpy : Reset NumPy global state
    reset_sklearn : Reset scikit-learn configuration
    reset_matplotlib : Reset matplotlib figures and configuration
    reset_seaborn : Reset seaborn theme (if available)
    """
    reset_numpy()
    reset_sklearn()
    reset_matplotlib()
    reset_seaborn()
