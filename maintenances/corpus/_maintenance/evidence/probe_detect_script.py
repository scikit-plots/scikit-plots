"""
Digest and time ``detect_script`` over a fixed set of cases (CX-08).

Run from the repository root:
``python -B maintenances/corpus/_maintenance/evidence/probe_detect_script.py [ROOT] [--texts DIR]``.
``ROOT`` defaults to this checkout. ``--texts`` adds every paragraph of the
``.md``/``.txt``/``.rst`` files under ``DIR`` as a case. Writes the number of
cases, a SHA-256 digest of every result and the time spent on ASCII and other
samples: run it on two checkouts and compare the digests; equal digests mean
equal results for every case. Cases are derived from SHAKE-256, not from a
random generator, so every run and every interpreter sees the same ones.
"""

import argparse
import hashlib
import logging
import math
import pathlib
import sys
import time

#: Lone surrogates, which no Unicode text holds.
SURROGATES = range(0xD800, 0xE000)

#: Alphabets for the generated cases: printable ASCII, all of ASCII, ASCII
#: with Latin-1 and Greek, ASCII with CJK and Arabic, and the whole BMP
#: without surrogates.
ALPHABETS = [
    [chr(c) for c in range(0x20, 0x7F)],
    [chr(c) for c in range(0x80)],
    [chr(c) for c in (*range(0x20, 0x7F), *range(0xA0, 0x100), *range(0x370, 0x400))],
    [
        chr(c)
        for c in (*range(0x20, 0x7F), *range(0x4E00, 0x4F00), *range(0x600, 0x700))
    ],
    [chr(c) for c in range(0x20, 0x10000) if c not in SURROGATES],
]
THRESHOLDS = (0.0, 0.55, 1.0, 1.01, math.nan)
SAMPLE_SIZES = (500, 0, 1, 499, 501, -5)
GENERATED = 20_000


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")


def _generated():
    for i in range(GENERATED):
        stream = hashlib.shake_256(f"cx08:{i}".encode()).digest(2 + 3 * 700)
        alphabet = ALPHABETS[stream[0] % len(ALPHABETS)]
        size = (stream[1] * 7) % 700
        yield "".join(
            alphabet[
                int.from_bytes(stream[2 + 3 * j : 5 + 3 * j], "big") % len(alphabet)
            ]
            for j in range(size)
        )


def _edges():
    for c in map(chr, range(0x80)):
        yield from (c, c * 499, c * 500, c * 501)
    for filler in ("1", "!", " ", "_"):
        for n in (498, 499, 500, 501):
            yield filler * n + "a"
            yield filler * n + "a\u00e9"
            yield "\u00e9" + filler * n + "a"
    yield from ("", "a\u00a0", "\u00a0", "abc\u4e2d", "\u4e2dabc", "a" * 500 + "\u4e2d")


def _paragraphs(directory):
    for path in sorted(pathlib.Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".rst"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            yield from (p for p in text.split("\n\n") if p.strip())


def main() -> int:
    """Run every case, print the digest and the timings."""
    parser = argparse.ArgumentParser()
    default_root = pathlib.Path(__file__).resolve().parents[4]
    parser.add_argument("root", nargs="?", type=pathlib.Path, default=default_root)
    parser.add_argument("--texts", type=pathlib.Path, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    logging.disable(logging.WARNING)
    from scikitplot.corpus._chunkers._custom_tokenizer import (  # noqa: PLC0415
        detect_script,
    )

    digest = hashlib.sha256()
    cases = 0
    for text in _edges():
        for size in SAMPLE_SIZES:
            for threshold in THRESHOLDS:
                result = detect_script(
                    text, sample_size=size, majority_threshold=threshold
                )
                digest.update(result.value.encode() + b"\n")
                cases += 1
    real = list(_paragraphs(args.texts)) if args.texts else []
    timed = {"ascii": [0, 0.0], "other": [0, 0.0]}
    for text in list(_generated()) + real:
        kind = "ascii" if text[:500].isascii() else "other"
        start = time.perf_counter()
        results = [detect_script(text, majority_threshold=t) for t in THRESHOLDS]
        timed[kind][1] += time.perf_counter() - start
        timed[kind][0] += len(THRESHOLDS)
        for result in results:
            digest.update(result.value.encode() + b"\n")
        cases += len(THRESHOLDS)
    _say(f"root={args.root} python={sys.version.split()[0]}")
    _say(f"cases {cases} (generated {GENERATED}, real paragraphs {len(real)})")
    _say(f"digest {digest.hexdigest()}")
    for kind, (calls, seconds) in timed.items():
        per_call = seconds / calls * 1e6 if calls else 0.0
        _say(f"{kind:5} calls {calls:7}  {seconds:7.3f} s  {per_call:7.2f} us/call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
