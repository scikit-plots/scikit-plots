#!/usr/bin/env bash

python - <<'PY'
from scikitplot.corpus._chunkers._custom_tokenizer import (
    ScriptType,
    detect_script,
)

samples = [
    "",
    " ",
    "!!!",
    "!@#$%^&*()",
    "[]",
    r"\^_`",
    "_",
    "`",
    "^",
    "[",
    "]",
    "123",
    "12345!!!",
    "abc",
    "ABC",
    "a_b",
    "hello!",
    "😀",
    "😀😀",
    "©",
    "™",
    "→",
    "★",
]

for text in samples:
    try:
        result = detect_script(text)
        print(f"{text!r:<18} -> {result!r}")
    except Exception as exc:
        print(f"{text!r:<18} -> ERROR {type(exc).__name__}: {exc}")
PY

python - <<'PY'
import inspect
from scikitplot.corpus._chunkers._custom_tokenizer import detect_script

print(inspect.getsource(detect_script))
PY
