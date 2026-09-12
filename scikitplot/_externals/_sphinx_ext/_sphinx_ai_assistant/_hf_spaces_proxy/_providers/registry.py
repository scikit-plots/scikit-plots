# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Provider adapter registry and route planning for verified resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .base import ProviderResourceCapabilities, ResourceRoute
from .policy import ModelResourceOverride, capabilities_for, provider_names

try:
    from .._utils._resource_transport import ResourceUpload
except ImportError:  # standalone HF Space: _providers is top-level
    from _utils._resource_transport import ResourceUpload

_AUTO_ORDER = ("native", "tool", "extract", "context")


@dataclass
class StaticProviderAdapter:
    """
    Plan resource routes without performing provider I/O.

    Run 127 deliberately stops at a verifiable routing plan.  Provider-specific
    upload/execution modules added later consume these plans.  Keeping planning
    separate makes health discovery truthful and testable without credentials.
    """

    name: str
    overrides: Mapping[str, ModelResourceOverride] | None = None
    max_files: int | None = None
    max_file_bytes: int | None = None
    max_total_bytes: int | None = None

    def capabilities(self, model: str) -> ProviderResourceCapabilities:
        return capabilities_for(
            adapter=self.name,
            model=model,
            overrides=self.overrides,
            max_files=self.max_files,
            max_file_bytes=self.max_file_bytes,
            max_total_bytes=self.max_total_bytes,
        )

    async def route_resources(
        self, model: str, resources: Sequence[ResourceUpload]
    ) -> Sequence[ResourceRoute]:
        caps = self.capabilities(model)
        planned: list[ResourceRoute] = []
        for upload in resources:
            verified = upload.verified
            candidates = tuple(caps.routes_for(verified.detected_modality))
            intent = verified.intent
            route = "unsupported"
            if intent == "raw":
                for candidate in ("native", "tool"):
                    if candidate in candidates:
                        route = candidate
                        break
            elif intent == "extract":
                if "extract" in candidates:
                    route = "extract"
            elif intent == "context":
                if "context" in candidates:
                    route = "context"
            else:
                for candidate in _AUTO_ORDER:
                    if candidate in candidates:
                        route = candidate
                        break
            planned.append(
                ResourceRoute(
                    resource_id=verified.id,
                    route=route,
                    reason=(
                        "planned provider route"
                        if route != "unsupported"
                        else f"{caps.provider} adapter has no route for {verified.detected_modality}/{intent}"
                    ),
                    metadata={
                        "provider": caps.provider,
                        "model": model,
                        "modality": verified.detected_modality,
                        "intent": intent,
                        "candidates": list(candidates),
                        "execution": "plan-only",
                    },
                )
            )
        return tuple(planned)

    async def close(self) -> None:
        return None


class ProviderRegistry:
    def __init__(
        self,
        *,
        overrides: Mapping[str, ModelResourceOverride] | None = None,
        max_files: int | None = None,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
    ) -> None:
        self._adapters = {
            name: StaticProviderAdapter(
                name,
                overrides=overrides,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
            )
            for name in provider_names()
        }

    def get(self, name: str) -> StaticProviderAdapter:
        return self._adapters.get(
            str(name or "").strip().lower(), self._adapters["custom"]
        )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))
