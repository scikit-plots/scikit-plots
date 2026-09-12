# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Bundled/self-hosted model resource adapter marker."""

from .registry import StaticProviderAdapter


class SelfHostedAdapter(StaticProviderAdapter):
    def __init__(self, **kwargs):
        super().__init__("self_hosted", **kwargs)
