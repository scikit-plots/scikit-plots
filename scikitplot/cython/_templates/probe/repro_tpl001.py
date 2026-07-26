"""CYTHON-TPL-001 repro: template/workflow/example resolvers must contain
paths to the _templates root. On a fixed tree, escapes raise ValueError.
"""
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
from scikitplot.cython._templates_api import get_template_path, get_workflow_path, get_package_example_path, _TEMPLATE_ROOT

root = _TEMPLATE_ROOT.resolve()
print("template root:", root)

# 1) Absolute path escape via get_template_path (extension form)
try:
    r = get_template_path("/etc/hostname.pyx")  # absolute → _TEMPLATE_ROOT / abs == abs
    inside = str(Path(r).resolve()).startswith(str(root))
    print(f"1) get_template_path('/etc/hostname.pyx') -> {r}  inside_root={inside}")
    if not inside: print("   BUG: escaped template root")
except FileNotFoundError as e:
    # file may not exist, but check what path it TRIED
    tried = str(e)
    print(f"1) tried path: {tried}  (escaped={'_templates' not in tried})")

# 2) Traversal via get_workflow_path
try:
    r = get_workflow_path("../../..")  # climbs out
    inside = str(Path(r).resolve()).startswith(str(root))
    print(f"2) get_workflow_path('../../..') -> {Path(r).resolve()}  inside_root={inside}")
    if not inside: print("   BUG: escaped workflow root")
except FileNotFoundError as e:
    print(f"2) FileNotFoundError (dir may not exist): {e}")
