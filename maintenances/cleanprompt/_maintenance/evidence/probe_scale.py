"""
Measure scale behaviour that must stay linear (round 15, CP-064, CP-065).

1. A CSV above the document limit is encoded in record-aligned pieces, and the
   output equals one pass with the limit raised (D-16.1).
2. Random cut sizes over CSV and JSON Lines: pieces equal one pass.
3. Log scrubbing with thousands of live holders: creation and per-record cost.

Run from anywhere; the repository root is found from this file's location.
Exits non-zero on any failure. Timings are reported, and bounded generously
so a slow machine does not fail while a quadratic regression does.
"""

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4]))
import warnings as _warnings

_warnings.simplefilter("ignore")
import io
import json
import random
import time

from scikitplot.cleanprompt import FluentCleanPrompt, configure_logging
from scikitplot.cleanprompt._logging import _SHARED, VaultScrubber, get_logger
from scikitplot.cleanprompt._policy import Limits
from scikitplot.cleanprompt._runtime import Cleaner

fails = []


def check(name, ok, detail=""):
    print("%-44s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        fails.append(name)


def rows(n):
    return "name,email,note\n" + "".join(
        'P%d Q,p%d@example.com,"row %d\nsecond line"\n' % (i % 997, i, i)
        for i in range(n)
    )


print("=" * 78)
print("1. Large CSV: chunked by the document limit vs one pass with it raised")
print("=" * 78)
times = {}
for n in (30_000, 60_000, 120_000):
    text = rows(n)
    chunked = Cleaner(FluentCleanPrompt().plan())
    t0 = time.monotonic()
    out = chunked.encode_text(text, "csv")
    times[n] = time.monotonic() - t0
    whole = Cleaner(FluentCleanPrompt().plan())
    whole._policy = whole._policy.evolve(
        limits=Limits(max_input_chars=len(text) + 1, max_entries=10**7, max_spans=10**7)
    )
    same = whole.encode_text(text, "csv").text == out.text
    check(
        "%6d rows, %8d chars, %3d pieces" % (n, len(text), out.report.get("chunks", 1)),
        same and "@example.com" not in out.text,
        "%.2fs" % times[n],
    )
ratio = times[120_000] / max(times[30_000], 1e-9)
check("120k/30k time ratio is linear (< 8, ideal 4)", ratio < 8, "%.2f" % ratio)

print()
print("=" * 78)
print("2. Random cut sizes: pieces equal one pass (CSV and JSON Lines)")
print("=" * 78)
rng = random.Random(15)
bad = 0
trials = 300
for _ in range(trials):
    if rng.random() < 0.5:
        fmt = "csv"
        text = "name,email\n" + "".join(
            "%s,u%d@example.org\n"
            % (rng.choice(["Ann Lee", '"Lee, Ann"', '"A\nB"', "x"]), rng.randint(0, 20))
            for _ in range(rng.randint(0, 40))
        )
    else:
        fmt = "jsonl"
        text = "".join(
            json.dumps(
                {
                    "email": "u%d@example.org" % rng.randint(0, 20),
                    "n": rng.randint(0, 9),
                }
            )
            + rng.choice(["\n", "\n\n"])
            for _ in range(rng.randint(0, 40))
        )
    one = Cleaner(FluentCleanPrompt().plan()).encode_text(text, fmt).text
    size = rng.randint(1, max(2, len(text)))
    pieces = (
        Cleaner(FluentCleanPrompt().plan(), chunk_chars=size)
        .encode_text(text, fmt)
        .text
    )
    bad += pieces != one
check("%d random documents and cut sizes" % trials, bad == 0, "%d differ" % bad)

print()
print("=" * 78)
print("3. Log scrubbing with many live holders (CP-065)")
print("=" * 78)
buffer = io.StringIO()
configure_logging("info", "json", stream=buffer)
t0 = time.monotonic()
holders = []
for i in range(5000):
    holder = VaultScrubber()
    holder.add(["holder%d@example.com" % i])
    holders.append(holder)
created = time.monotonic() - t0
held = _SHARED.held  # includes every value the cleaners above still hold
t0 = time.monotonic()
logger = get_logger("scikitplot.cleanprompt._scale")
for i in range(2000):
    logger.warning("record about holder%d@example.com", i % 5000)
logged = time.monotonic() - t0
for holder in holders:
    holder.close()
configure_logging("warning", stream=io.StringIO())
check("create 5000 holders", created < 10, "%.2fs" % created)
check("2000 records against %d held values" % held, logged < 10, "%.2fs" % logged)
check("no held value reached the log", "@example.com" not in buffer.getvalue())

print()
print("=" * 78)
print("4. Streamed decoding: a long surrogate reply in 5-character chunks")
print("=" * 78)
guard = FluentCleanPrompt().style("surrogate").guard()
guard.outgoing(
    "name,email\n" + "".join("Person%d Q,p%d@example.com\n" % (i, i) for i in range(50)), "csv"
)
safe = guard.outgoing(" ".join("Person%d Q wrote p%d@example.com." % (i, i) for i in range(50)))
reply = (safe + " ") * 12
t0 = time.monotonic()
streamed = "".join(guard.decode_stream(reply[i : i + 5] for i in range(0, len(reply), 5)))
dt = time.monotonic() - t0
check("%d chars in 5-char chunks equals the whole" % len(reply), streamed == guard.incoming(reply), "%.2fs" % dt)
check("streamed decode under 3 s (was 3.44 s in v18)", dt < 3.0, "%.2fs" % dt)

# CP-111: international groupings found whole, ranges and dates left, at scale.
from scikitplot.cleanprompt import encode as _encode_cp111

_numbers = ["+33 1 23 45 67 89", "+49 30 12345678", "+49 (0)30 1234 5678",
            "030 12345678", "+44 7700 900123", "+90 212 555 12 34"]
_kept = ["2019-2024", "15.01.2024", "15-01-2024", "ISBN 978-3-16-148410-0"]
_doc = " ".join("Row %d: tel %s, period %s." % (i, _numbers[i % 6], _kept[i % 4]) for i in range(2000))
t0 = time.monotonic()
_out = _encode_cp111(_doc).text
dt = time.monotonic() - t0
check("2000 phone rows: no number sent, not even its prefix",
      not any(n in _out or n.split()[0] + " " + n.split()[1] in _out for n in _numbers), "%.2fs" % dt)
check("2000 rows: every range, date and ISBN kept", all(_out.count(k) == _doc.count(k) for k in _kept))
check("2000 phone rows under 5 s", dt < 5.0, "%.2fs" % dt)

# CP-117: VINs at scale, checked and labelled, with 17-character look-alikes.
_vins = ["1M8GDM9AXKP042788", "1HGCM82633A004352", "jh4ka4650jc000403"]
_labelled = ["WVWZZZ3BZWE689725"]
_alike = ["ABCDEFGHJK1234567", "11111111111111111"]
_vdoc = " ".join(
    "Row %d: vin %s, loan VIN: %s, part %s." % (i, _vins[i % 3], _labelled[0], _alike[i % 2])
    for i in range(2000)
)
_vcleaner = FluentCleanPrompt().packs("vehicle").materialize()
t0 = time.monotonic()
_vout = _vcleaner.encode_text(_vdoc).text
dt = time.monotonic() - t0
check("2000 VIN rows: no VIN sent, checked or labelled",
      not any(v in _vout for v in _vins + _labelled), "%.2fs" % dt)
check("2000 VIN rows: every look-alike kept", all(_vout.count(a) == _vdoc.count(a) for a in _alike))
check("2000 VIN rows restore exactly", _vcleaner.decode(_vout) == _vdoc)

# CP-119: German tax identification numbers at scale, with look-alikes.
_ids = ["86504957138", "86 504 957 138", "12345678911"]
_not_ids = ["86504957139", "47198320652", "11123456786"]
_ddoc = " ".join(
    "Zeile %d: Steuer-ID %s, Bestellung %s." % (i, _ids[i % 3], _not_ids[i % 3])
    for i in range(2000)
)
_dcleaner = FluentCleanPrompt().packs("de").materialize()
t0 = time.monotonic()
_dout = _dcleaner.encode_text(_ddoc).text
dt = time.monotonic() - t0
check("2000 Steuer-ID rows: no Steuer-ID sent", not any(n in _dout for n in _ids), "%.2fs" % dt)
check("2000 Steuer-ID rows: every look-alike kept", all(_dout.count(a) == _ddoc.count(a) for a in _not_ids))
check("2000 Steuer-ID rows restore exactly", _dcleaner.decode(_dout) == _ddoc)

print()
print("TOTAL FAILURES:", len(fails), fails if fails else "")
_sys.exit(1 if fails else 0)
