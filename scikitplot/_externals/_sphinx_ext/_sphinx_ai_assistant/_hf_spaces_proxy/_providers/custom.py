# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit custom-backend resource adapter marker."""

from .registry import StaticProviderAdapter


class CustomAdapter(StaticProviderAdapter):
    def __init__(self, **kwargs):
        super().__init__("custom", **kwargs)
