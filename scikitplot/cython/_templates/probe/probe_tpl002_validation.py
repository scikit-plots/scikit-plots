"""CYTHON-TPL-002 probe: strict template-metadata validation.

Checks:
1. unknown schema_version rejected;
2. wrong-typed list entries (silently dropped by the lenient parser) rejected;
3. absolute/escaping support/extra-source references rejected;
4. clean metadata passes.

Exit 0 = strict validation behaves correctly.
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

from scikitplot.cython._templates_api import (  # noqa: E402
    TEMPLATE_SCHEMA_VERSION,
    TemplateInfo,
    TemplateValidationError,
    validate_template_info,
)


def _info(**kw):
    base = dict(template_id="c/n", path=Path("n.pyx"), meta_path=None)
    base.update(kw)
    return TemplateInfo(**base)


def _rejects(**kw) -> bool:
    raw = kw.pop("raw", None)
    try:
        validate_template_info(_info(**kw), raw=raw)
        return False
    except TemplateValidationError:
        return True


def main() -> int:
    ok = True

    r1 = _rejects(schema_version=TEMPLATE_SCHEMA_VERSION + 1)
    print(f"unknown schema rejected: {'OK' if r1 else 'FAIL'}")
    ok = ok and r1

    r2 = _rejects(raw={"tags": ["ok", 42]})
    print(f"wrong-typed entry rejected: {'OK' if r2 else 'FAIL'}")
    ok = ok and r2

    r3 = _rejects(support_paths=("../evil.pxi",)) and _rejects(
        extra_sources=("/etc/passwd",)
    )
    print(f"uncontained reference rejected: {'OK' if r3 else 'FAIL'}")
    ok = ok and r3

    try:
        validate_template_info(
            _info(support_paths=("helpers/util.pxi",)), raw={"tags": ["fast"]}
        )
        r4 = True
    except TemplateValidationError:
        r4 = False
    print(f"clean metadata passes: {'OK' if r4 else 'FAIL'}")
    ok = ok and r4

    print("VERDICT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
