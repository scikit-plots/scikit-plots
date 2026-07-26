"""CYTHON-SEC-002 probe: SecurityPolicy.strict is operative.

Checks:
1. strict=True → unset allow_* flags are False (restrictive);
2. strict=False → unset allow_* flags are True (permissive) and a shell-meta
   arg is now permitted;
3. an explicit per-flag value overrides strict either way.

Exit 0 = strict is operative.
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

from scikitplot.cython._security import (  # noqa: E402
    SecurityError,
    SecurityPolicy,
    validate_build_inputs,
)

BAD = "-O2; rm -rf /"
FLAGS = [
    "allow_absolute_include_dirs",
    "allow_shell_metacharacters",
    "allow_reserved_macros",
    "allow_dangerous_compiler_args",
]


def _rejects(policy) -> bool:
    try:
        validate_build_inputs(policy=policy, extra_compile_args=[BAD])
        return False
    except SecurityError:
        return True


def main() -> int:
    ok = True

    strict = SecurityPolicy(strict=True)
    r1 = all(getattr(strict, f) is False for f in FLAGS) and _rejects(strict)
    print(f"strict=True restrictive + rejects: {'OK' if r1 else 'FAIL'}")
    ok = ok and r1

    loose = SecurityPolicy(strict=False)
    r2 = all(getattr(loose, f) is True for f in FLAGS) and not _rejects(loose)
    print(f"strict=False permissive + permits: {'OK' if r2 else 'FAIL'}")
    ok = ok and r2

    override = SecurityPolicy(strict=False, allow_shell_metacharacters=False)
    r3 = override.allow_shell_metacharacters is False and _rejects(override)
    print(f"explicit per-flag override wins: {'OK' if r3 else 'FAIL'}")
    ok = ok and r3

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
