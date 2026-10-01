"""
Synchronize the standalone HF proxy page-feedback runtime mirror.

The authoritative generic feedback backend lives in ``_sphinx_feedback``.
The HF Space proxy is deployed as a standalone Docker context, so it carries a
byte-identical mirror of the server-only contracts/service package.  This helper
is deterministic and idempotent; CI compares the mirror against the source.
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve()
PROXY = HERE.parents[1]
EXT_ROOT = PROXY.parents[1]
SOURCE = EXT_ROOT / "_sphinx_feedback"
DEST = PROXY / "_page_feedback"
SERVICE_FILES = (
    "__init__.py",
    "_config.py",
    "_core.py",
    "_github.py",
    "_sqlite.py",
    "app.py",
)


def sync() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "_service").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE / "__init__.py", DEST / "__init__.py")
    shutil.copyfile(SOURCE / "_contracts.py", DEST / "_contracts.py")
    for name in SERVICE_FILES:
        shutil.copyfile(SOURCE / "_service" / name, DEST / "_service" / name)


if __name__ == "__main__":
    sync()
