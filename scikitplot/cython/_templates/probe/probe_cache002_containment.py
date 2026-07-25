"""CYTHON-CACHE-002 probe: cache artifact selection must be contained to the
cache entry and integrity-checked.

Checks (no compiler needed):
1. an absolute artifact path in meta.json is refused (not selected);
2. a traversal artifact path is refused;
3. a contained basename is selected;
4. a recorded artifact_sha256 that mismatches raises ValueError.

Exit 0 = all properties hold.
"""
from __future__ import annotations

import sys
import tempfile
from hashlib import sha256
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

# Dev/AI probe bootstrap.
try:
    import scikitplot.cython  # noqa: F401
except ImportError:
    _here = Path(__file__).resolve()
    for _cand in _here.parents:
        if (_cand / "scikitplot" / "__init__.py").exists():
            sys.path.insert(0, str(_cand))
            break

from scikitplot.cython._cache import _artifact_from_meta_or_guess  # noqa: E402

SUF = EXTENSION_SUFFIXES[0]


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        entry = root / "entry"
        entry.mkdir()
        external = root / f"evil{SUF}"
        external.write_bytes(b"\x7fELF-EVIL")

        # 1. absolute path refused
        r1 = _artifact_from_meta_or_guess(entry, {"artifact": str(external)})
        print(f"absolute artifact refused: {'OK' if r1 is None else 'FAIL'}")
        ok = ok and r1 is None

        # 2. traversal refused
        r2 = _artifact_from_meta_or_guess(entry, {"artifact": f"../evil{SUF}"})
        print(f"traversal artifact refused: {'OK' if r2 is None else 'FAIL'}")
        ok = ok and r2 is None

        # 3. contained basename selected
        art = entry / f"mod{SUF}"
        art.write_bytes(b"ELF")
        r3 = _artifact_from_meta_or_guess(entry, {"artifact": f"mod{SUF}"})
        print(f"contained artifact selected: {'OK' if r3 == art else 'FAIL'}")
        ok = ok and r3 == art

        # 4. integrity mismatch raises
        raised = False
        try:
            _artifact_from_meta_or_guess(
                entry, {"artifact": f"mod{SUF}", "artifact_sha256": "0" * 64}
            )
        except ValueError:
            raised = True
        print(f"integrity mismatch raises: {'OK' if raised else 'FAIL'}")
        ok = ok and raised

        # (sanity) matching hash accepted
        good = sha256(b"ELF").hexdigest()
        r5 = _artifact_from_meta_or_guess(
            entry, {"artifact": f"mod{SUF}", "artifact_sha256": good}
        )
        print(f"matching hash accepted: {'OK' if r5 == art else 'FAIL'}")
        ok = ok and r5 == art

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
