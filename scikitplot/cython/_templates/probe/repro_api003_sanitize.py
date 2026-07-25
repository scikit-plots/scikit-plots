"""CYTHON-API-003 repro/probe: sanitize() collisions + non-ASCII.

BEFORE: 'a-b' and 'a.b' both -> 'a_b' (collision); 'café' -> 'café' (non-ASCII
leak, violating the ASCII-only contract).
AFTER: distinct inputs -> distinct names; output is always pure ASCII; already
-valid identifiers are unchanged.

Exit 0 = fixed behavior holds.
"""
from __future__ import annotations

import sys
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

from scikitplot.cython._utils import sanitize  # noqa: E402


def main() -> int:
    ok = True

    no_collision = sanitize("a-b") != sanitize("a.b")
    print(f"distinct inputs stay distinct: {'OK' if no_collision else 'FAIL'} "
          f"({sanitize('a-b')!r} vs {sanitize('a.b')!r})")
    ok = ok and no_collision

    ascii_only = all(sanitize(s).isascii() for s in ["café", "αβγ", "Ä", "①"])
    print(f"output is pure ASCII: {'OK' if ascii_only else 'FAIL'} "
          f"(café -> {sanitize('café')!r})")
    ok = ok and ascii_only

    unchanged = all(sanitize(s) == s for s in ["hello_world", "_private", "abc123"])
    print(f"valid identifiers unchanged: {'OK' if unchanged else 'FAIL'}")
    ok = ok and unchanged

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
