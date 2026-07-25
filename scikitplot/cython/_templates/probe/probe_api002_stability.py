"""CYTHON-API-002 probe: every public symbol has a stability tier.

Checks:
1. total coverage — every name in __all__ has a tier;
2. tiers partition the surface (disjoint + exhaustive);
3. accessors behave (str/enum, unknown raises).

Exit 0 = stability contract holds.
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

import scikitplot.cython as skc  # noqa: E402
from scikitplot.cython import Stability, api_stability, list_api  # noqa: E402
from scikitplot.cython._api import API_STABILITY  # noqa: E402


def main() -> int:
    ok = True

    missing = [n for n in skc.__all__ if n not in API_STABILITY]
    cover = not missing
    print(f"total coverage: {'OK' if cover else 'FAIL'} ({len(missing)} untiered)")
    ok = ok and cover

    stable = set(list_api(Stability.STABLE))
    adv = set(list_api(Stability.ADVANCED))
    exp = set(list_api(Stability.EXPERIMENTAL))
    partition = (
        stable.isdisjoint(adv)
        and stable.isdisjoint(exp)
        and adv.isdisjoint(exp)
        and (stable | adv | exp) == set(skc.__all__)
    )
    print(f"tiers partition surface: {'OK' if partition else 'FAIL'} "
          f"(S={len(stable)} A={len(adv)} E={len(exp)})")
    ok = ok and partition

    try:
        api_stability("nope_not_real")
        acc = False
    except KeyError:
        acc = list_api("stable") == list_api(Stability.STABLE)
    print(f"accessors behave: {'OK' if acc else 'FAIL'}")
    ok = ok and acc

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
