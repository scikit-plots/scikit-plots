# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""llm_provider."""

from .._testing._pytesttester import PytestTester  # Pytest testing
from ..utils.utils_mlflow import load_mlflow_gateway_config  # noqa: F401
from .model_registry import (  # noqa: F401
    LLM_MODEL_PROVIDER2API_KEY,
    LLM_MODEL_PROVIDER2CONFIG,
)

test = PytestTester(__name__)
del PytestTester

# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
__all__ = [
    # 'model_registry',
    "LLM_MODEL_PROVIDER2API_KEY",
    "LLM_MODEL_PROVIDER2CONFIG",
    "load_mlflow_gateway_config",
    "test",
]
