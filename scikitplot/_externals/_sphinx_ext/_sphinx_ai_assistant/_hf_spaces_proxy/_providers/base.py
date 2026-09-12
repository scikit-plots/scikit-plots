# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Provider-neutral resource routing interfaces.

Run 127 defines the provider-routing authority boundary before enabling any
credential-bearing executor.  Adapters added in later runs must explicitly map each
verified modality; an unknown modality fails closed instead of being silently
flattened or dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

try:
    from .._utils._resource_transport import ResourceUpload
except ImportError:  # standalone HF Space: _providers is top-level
    from _utils._resource_transport import ResourceUpload

RESOURCE_ROUTES = frozenset({"native", "tool", "extract", "context", "unsupported"})


@dataclass(frozen=True)
class ResourceRoute:
    resource_id: str
    route: str
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.route not in RESOURCE_ROUTES:
            raise ValueError("invalid resource route")


@dataclass(frozen=True)
class ProviderResourceCapabilities:
    provider: str
    model: str
    modality_routes: dict[str, tuple[str, ...]]
    max_files: int | None = None
    max_file_bytes: int | None = None
    max_total_bytes: int | None = None

    def routes_for(self, modality: str) -> tuple[str, ...]:
        return tuple(self.modality_routes.get(modality, ("unsupported",)))


class ProviderAdapter(Protocol):
    """Minimal contract every provider adapter must implement."""

    name: str

    def capabilities(self, model: str) -> ProviderResourceCapabilities: ...

    async def route_resources(
        self, model: str, resources: Sequence[ResourceUpload]
    ) -> Sequence[ResourceRoute]: ...

    async def close(self) -> None: ...
