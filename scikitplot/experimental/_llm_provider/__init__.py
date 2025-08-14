# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
:py:mod:`~.llm_provider` for Large Language Models.

.. seealso::

   * https://huggingface.co/scikit-plots
   * https://huggingface.co/spaces/scikit-plots/model-sight

.. tip::

    * https://docs.litellm.ai/docs
    * https://docs.litellm.ai/docs/providers/huggingface
"""

from ..._testing._pytesttester import PytestTester  # Pytest testing
from ...utils.utils_mlflow import load_mlflow_gateway_config
from . import (
    chat_provider,
    clint_provider,
    model_registry,
)
from .chat_provider import get_response
from .model_registry import (
    LLM_PROVIDER_CONFIG_MAP,
    LLM_PROVIDER_ENV_CONNECTOR_MAP,
)

test = PytestTester(__name__)
del PytestTester

# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
__all__ = [
    # 'model_registry',
    "LLM_PROVIDER_CONFIG_MAP",
    "LLM_PROVIDER_ENV_CONNECTOR_MAP",
    "chat_provider",
    "clint_provider",
    "get_response",
    "load_mlflow_gateway_config",
    "model_registry",
    "test",
]
