# cython: language_level=3
"""
t09_enum_state — cdef enum.

What it demonstrates
--------------------
- Defining an enum and exposing stable integer codes to Python.

How to run
----------
>>> from scikitplot.cython import compile_template
>>> m = compile_template("t09_enum_state")
>>> m.state_code("ok")
0
"""


cdef enum State:
    OK = 0
    WARN = 1
    ERROR = 2


def state_code(str s):
    if s == "ok":
        return OK
    if s == "warn":
        return WARN
    if s == "error":
        return ERROR
    raise ValueError("unknown state")
