#!/usr/bin/env bash

python - <<'PY'
import inspect

from scikitplot.corpus._chunkers._sentence import _split_regex

samples = [
    "Hello. Goodbye",
    "Hello! Goodbye",
    "Hello? Goodbye",
    "你好。再见",
    "你好！再见",
    "你好？再见",
    "こんにちは。さようなら",
    "One sentence.",
    "一句。",
]

for text in samples:
    try:
        result = _split_regex(text, multi_script=True)
        print(f"{text!r}")
        print(f"  -> {result!r}")
    except TypeError:
        # If current signature differs, inspect it below.
        print(f"{text!r} -> signature differs")

print("\n--- _split_regex source ---")
print(inspect.signature(_split_regex))
print(inspect.getsource(_split_regex))
PY
