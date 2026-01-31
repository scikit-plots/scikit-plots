"""Module Python template: package build notes.

This template is documentation-as-code for building a small extension package.
Use :func:`scikitplot.cython.build_package_from_code` to compile multiple modules
in one build directory.
"""

EXAMPLE = r"""
from scikitplot.cython import build_package_from_code

pkg = {
    "core": "def add(int a, int b):\n    return a+b\n",
    "metrics": "def f1(double p, double r):\n    return 2*p*r/(p+r)\n",
}
res = build_package_from_code(pkg, package_name="myfast")
"""
