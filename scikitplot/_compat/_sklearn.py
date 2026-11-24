from __future__ import annotations

from types import ModuleType

import sklearn

# The vendored ``packaging`` copy, not ``externals._seaborn``: the seaborn
# copy imports the real seaborn, which a base install does not have.
from ..externals._packaging.version import Version


def _version_predates(lib: ModuleType, version: str) -> bool:
    """Return whether ``lib.__version__`` is older than *version* (PEP 440)."""
    return Version(lib.__version__) < Version(version)


def learning_curve_params(val: dict | None):
    # dep use params 1.6
    if _version_predates(sklearn, "1.6.0"):
        return {"fit_params": val}
    return {"params": val}
