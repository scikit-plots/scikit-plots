# cython: language_level=3
"""
t14_string_reverse — Python string manipulation.

What it demonstrates
--------------------
- Not everything needs to be C-typed; Cython can still help structure code.
- Deterministic behavior with Unicode strings.

How to run
----------
>>> from scikitplot.cython import compile_template
>>> m = compile_template("t14_string_reverse")
>>> m.reverse("héllo")
'olléh'
"""


def reverse(str s):
    # Python-level reversal is correct for Unicode code points.
    return s[::-1]
