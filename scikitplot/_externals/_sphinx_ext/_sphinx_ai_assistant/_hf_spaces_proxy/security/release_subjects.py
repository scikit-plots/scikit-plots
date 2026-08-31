#!/usr/bin/env python3
"""Print canonical non-secret subjects that production release evidence must bind."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import verify_release_evidence  # noqa: E402
from _utils._shared_logic import PROXY_VERSION  # noqa: E402

logger = logging.getLogger(__name__)


def subjects() -> dict[str, object]:  # ruff: ignore[undocumented-public-function]
    supply = verify_release_evidence.SUPPLY
    return {
        "schema_version": 1,
        "proxy_version": PROXY_VERSION,
        "target_platform": verify_release_evidence.POLICY["target_platform"],
        "requirements_lock_sha256": (
            hashlib.sha256((ROOT / supply["lock_file"]).read_bytes()).hexdigest()
        ),
        "python_sbom_sha256": (
            hashlib.sha256((ROOT / supply["sbom_file"]).read_bytes()).hexdigest()
        ),
        "runtime_source_sha256": verify_release_evidence._runtime_source_sha256(),
        "base_image_index_digest": supply["base_image"]["index_digest"],
    }


if __name__ == "__main__":
    sys.stdout.write(json.dumps(subjects(), sort_keys=True) + "\n")
