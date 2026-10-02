"""Import-isolation and tier probes, under an adversarial __import__ blocker.

The blocker is stronger than an uninstalled dependency: it refuses the import
even though the distribution is present on this machine. A probe that passes
here would pass on a base installation, and it also catches the case an
uninstalled-dependency test cannot — a module that imports its tier lazily but
*unconditionally* on some other code path.
"""
import subprocess
import sys

import pathlib as _pathlib
# Computed from this file's location, never hard-coded (CP-053).
ROOT = str(_pathlib.Path(__file__).resolve().parents[4])

#: Every optional distribution, plus the scientific stack this submodule must
#: never reach for. ``nltk`` joined in round 3 with the second NER engine: it is
#: a tier like any other, and the probe is the reason we know the facade treats
#: it like one.
BLOCK = (
    "spacy",
    "nltk",
    "flask",
    "cryptography",
    "numpy",
    "pandas",
    "pydantic",
    "scipy",
    "sklearn",
    "click",
    "yaml",
    # A typing backport is third-party too; the base tier imported it at
    # module scope until round 12 (CP-054).
    "typing_extensions",
)


#: An empty stand-in for the parent package, registered before anything
#: imports it (CP-085). ``scikitplot/__init__.py`` imports NumPy, so under the
#: real parent every probe below would report a leak, or be refused by the
#: blocker, for a reason that is not this submodule's. The stand-in's
#: ``__path__`` is the real package directory: submodules load from the real
#: files and no parent code runs.
ISOLATED_PARENT = (
    "import sys, types\n"
    "_parent = types.ModuleType('scikitplot')\n"
    "_parent.__path__ = [%r]\n"
    "sys.modules['scikitplot'] = _parent\n"
) % (str(_pathlib.Path(ROOT, "scikitplot")),)


def run(body):
    script = ISOLATED_PARENT + body
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


BLOCKER = """
import builtins
_real = builtins.__import__
_blocked = %r
def _guard(name, *a, **k):
    if name.split('.')[0] in _blocked:
        raise ImportError('blocked by probe: ' + name)
    return _real(name, *a, **k)
builtins.__import__ = _guard
""" % (BLOCK,)

LEAKED = "[n for n in %r if n in sys.modules]" % (BLOCK,)

probes = [
    ("plain import", "import scikitplot.cleanprompt\nprint('ok', %s)" % LEAKED),
    ("star import", "exec('from scikitplot.cleanprompt import *')\nprint('ok', %s)" % LEAKED),
    ("dir()", "import scikitplot.cleanprompt as c\nd=dir(c)\nprint('ok', 'spacy_detector' in d and 'nltk_detector' in d, %s)" % LEAKED),
    ("capabilities()", "import scikitplot.cleanprompt as c\nr=c.capabilities()\nprint('ok', sorted(r), %s)" % LEAKED),
    ("redact+restore", "import scikitplot.cleanprompt as c\nr=c.Redactor().redact('mail a@b.co and 4242 4242 4242 4242')\nprint('ok', c.restore(r.text,r.vault).text=='mail a@b.co and 4242 4242 4242 4242', %s)" % LEAKED),
    ("hasattr miss", "import scikitplot.cleanprompt as c\nhasattr(c,'nope')\nprint('ok', %s)" % LEAKED),
    # Attribute access must never import the tier. Where the tier is ABSENT it
    # raises with an install hint; where the distribution is present but its
    # import is blocked it resolves the *description* and defers the failure to
    # first use. Both are correct; leaking the dependency is not, and that is
    # the invariant asserted here rather than one environment's branch.
    ("getattr spacy leaks nothing", "import scikitplot.cleanprompt as c\ntry:\n c.spacy_detector\nexcept c.CapabilityError as e:\n assert 'pip install' in str(e)\nprint('ok', %s)" % LEAKED),
    ("getattr nltk leaks nothing", "import scikitplot.cleanprompt as c\ntry:\n c.nltk_detector\nexcept c.CapabilityError as e:\n assert 'pip install' in str(e)\nprint('ok', %s)" % LEAKED),
    ("pickle a result", "import pickle, scikitplot.cleanprompt as c\nr=c.Redactor().redact('a@b.co')\nprint('ok', pickle.loads(pickle.dumps(r.entries)) == r.entries)"),
    ("deepcopy a policy", "import copy, scikitplot.cleanprompt as c\nprint('ok', copy.deepcopy(c.DEFAULT_POLICY) == c.DEFAULT_POLICY)"),
    # --- round 3 surface -------------------------------------------------
    ("encode/decode round trip", "import scikitplot.cleanprompt as c\ns,h=c.encode('mail a@b.co')\nprint('ok', 'a@b.co' not in s and c.decode(s,h)=='mail a@b.co', %s)" % LEAKED),
    ("handle export/load", "import json, scikitplot.cleanprompt as c\ns,h=c.encode('mail a@b.co')\nw=c.Handle.load(json.loads(json.dumps(h.export())))\nprint('ok', c.decode(s,w)=='mail a@b.co', %s)" % LEAKED),
    # Session.encode returns a plain str, not an EncodedPrompt: the session
    # owns the handle, and handing back a per-turn copy would invite a caller
    # to decode turn 3 with turn 1's handle. The asymmetry is the guard rail.
    ("session context manager", "import scikitplot.cleanprompt as c\nwith c.session() as s:\n  a=s.encode('mail a@b.co')\n  b=s.encode('again a@b.co')\nprint('ok', a.split()[-1]==b.split()[-1], %s)" % LEAKED),
    ("describe_engines", "import scikitplot.cleanprompt as c\nr=c.describe_engines()\nprint('ok', sorted(r['engines'])==['nltk','spacy'], %s)" % LEAKED),
    ("resolve_model", "import scikitplot.cleanprompt as c\nm,_=c.resolve_model('de','sm')\nprint('ok', m=='de_core_news_sm', %s)" % LEAKED),
    ("logging configured", "import io, scikitplot.cleanprompt as c\nb=io.StringIO()\nc.configure_logging('debug','json',b)\nc.Redactor().redact('mail a@b.co')\nprint('ok', 'a@b.co' not in b.getvalue(), %s)" % LEAKED),
    # CP-040: vault encryption must work with nothing installed, which is the
    # whole reason the portable cipher exists. Encrypt, decrypt, and confirm no
    # value survives in the document.
    ("encrypt round trip", "import scikitplot.cleanprompt as c\nfrom scikitplot.cleanprompt._vaultcrypt import encrypt_mapping, decrypt_mapping\nt,p=encrypt_mapping({'[EMAIL-1]':'ada@example.com'}, b'pass')\nimport json\nprint('ok', decrypt_mapping(t,b'pass',p)=={'[EMAIL-1]':'ada@example.com'} and 'ada@example.com' not in json.dumps([t,p]), %s)" % LEAKED),
    ("wrong passphrase refused", "from scikitplot.cleanprompt._vaultcrypt import encrypt_mapping, decrypt_mapping\nfrom scikitplot.cleanprompt import CleanPromptError\nimport sys\nt,p=encrypt_mapping({'[E-1]':'v'}, b'a')\ntry:\n decrypt_mapping(t,b'b',p)\n print('FAIL decrypted')\nexcept CleanPromptError:\n print('ok', %s)" % LEAKED),
    ("new passphrase", "from scikitplot.cleanprompt._vaultcrypt import new_passphrase\nimport sys\nprint('ok', len(new_passphrase().split('-'))==4, %s)" % LEAKED),
    # CP-023 and the BROKEN-tier fix, as one invariant: entity detection that
    # was asked for and cannot run must raise CapabilityError, never return a
    # clean result and never surface as a generic detector crash. Which of the
    # two messages appears depends on whether the distribution is absent or
    # merely unimportable; that it is a CapabilityError does not.
    # Round 12: packs and formats are read from the compiled JSON with the
    # standard library; PyYAML is blocked here, so this is the proof.
    ("fluent plan without PyYAML", "import scikitplot.cleanprompt as c\nk=c.FluentCleanPrompt().packs('patient').materialize()\nt=k.encode_text('{\"mrn\": 12}', 'json').text\nprint('ok', '12' not in t and k.decode(t)=='{\"mrn\": 12}', %s)" % LEAKED),
    ("custom JSON pack without PyYAML", "import json, tempfile, pathlib, scikitplot.cleanprompt as c\nd=pathlib.Path(tempfile.mkdtemp())/'hr.json'\nd.write_text(json.dumps({'name':'hr','version':1,'summary':'x','fields':[{'names':['badge_id'],'kind':'EMPLOYEE'}]}))\nprint('ok', 'hr' in c.load_custom(d).packs, %s)" % LEAKED),
    ("custom YAML asks for PyYAML", "import tempfile, pathlib, scikitplot.cleanprompt as c\nd=pathlib.Path(tempfile.mkdtemp())/'hr.yaml'\nd.write_text('name: hr')\ntry:\n c.load_custom(d)\n print('FAIL loaded')\nexcept c.CapabilityError as e:\n print('ok', 'pyyaml' in e.install_hint, %s)" % LEAKED),
    # Round 13: the gate and its logging, with every optional package blocked.
    ("guard ask/stream/objects", "import scikitplot.cleanprompt as c\ng=c.FluentCleanPrompt().guard()\na=g.ask('mail ann@example.com', lambda s: s)\nd=g.stream(); o=d.feed(g.outgoing('x ann@example.com'))+d.flush()\nprint('ok', a=='mail ann@example.com' and o=='x ann@example.com' and g.decode_object(g.encode_object({'e':'ann@example.com'}))=={'e':'ann@example.com'}, %s)" % LEAKED),
    ("audit and scrubbing", "import io, scikitplot.cleanprompt as c\nb=io.StringIO(); c.configure_logging('info','json',b)\ng=c.FluentCleanPrompt().guard(); g.outgoing('ann@example.com')\nc.get_logger('scikitplot.cleanprompt._x').warning('ann@example.com')\nprint('ok', 'ann@example.com' not in b.getvalue() and '\"event\": \"encoded\"' in b.getvalue(), %s)" % LEAKED),
    # Round 14: the MCP server and plan files, every optional package blocked.
    ("mcp server read/write", "import json, tempfile, pathlib, scikitplot.cleanprompt as c\nfrom scikitplot.cleanprompt._mcp import McpServer\nr=pathlib.Path(tempfile.mkdtemp()); (r/'a.txt').write_text('mail ann@example.com')\ns=McpServer(lambda: c.Guard(), [r])\nx=s.handle({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'cleanprompt_read_file','arguments':{'path':'a.txt'}}})\nprint('ok', x['result']['content'][0]['text']=='mail [EMAIL-1]', %s)" % LEAKED),
    ("plan file save/load", "import tempfile, pathlib, scikitplot.cleanprompt as c\nfrom scikitplot.cleanprompt._plan import save_plan, load_plan\nf=pathlib.Path(tempfile.mkdtemp())/'p.json'\np=c.FluentCleanPrompt().packs('patient').plan(); save_plan(p, f)\nprint('ok', load_plan(f)==p, %s)" % LEAKED),
    # Round 15: chunked record files, folder surveys, the shared log filter.
    ("chunked CSV equals one pass", "import scikitplot.cleanprompt as c\nfrom scikitplot.cleanprompt._runtime import Cleaner\nt='email\\n'+''.join('u'+str(i)+'@x.org\\n' for i in range(50))\none=Cleaner(c.FluentCleanPrompt().plan()).encode_text(t,'csv').text\nk=Cleaner(c.FluentCleanPrompt().plan(), chunk_chars=40); e=k.encode_text(t,'csv')\nprint('ok', e.text==one and e.report['chunks']>1 and k.decode(e.text)==t, %s)" % LEAKED),
    ("folder survey writes nothing", "import tempfile, pathlib, scikitplot.cleanprompt as c\nr=pathlib.Path(tempfile.mkdtemp()); (r/'a.csv').write_text('name,email\\nAnn Lee,ann@example.com\\n')\nk=c.FluentCleanPrompt().materialize(); i=k.survey_tree(r)\nprint('ok', sorted(p.name for p in r.iterdir())==['a.csv'] and i[0].kinds=={'EMAIL':1,'PERSON':1} and k.decode('[EMAIL-1]')=='[EMAIL-1]', %s)" % LEAKED),
    ("shared log filter, 1000 holders", "import io, scikitplot.cleanprompt as c\nfrom scikitplot.cleanprompt._logging import VaultScrubber\nb=io.StringIO(); c.configure_logging('info','json',b)\nh=[VaultScrubber() for _ in range(1000)]\n[x.add(['u'+str(i)+'@example.com']) for i,x in enumerate(h)]\nc.get_logger('scikitplot.cleanprompt._x').warning('u999@example.com and u0@example.com')\n[x.close() for x in h]\nprint('ok', '@example.com' not in b.getvalue(), %s)" % LEAKED),
    # Round 16: async clients and threads, every optional package blocked.
    ("async aask and stream", "import asyncio, scikitplot.cleanprompt as c\ng=c.FluentCleanPrompt().guard()\nasync def m(p):\n return 'to '+p.split()[-1]\nasync def s():\n for x in ['to [EMA','IL-1]']:\n  yield x\nasync def run():\n a=await g.aask('mail ann@example.com', m)\n b=''.join([y async for y in g.adecode_stream(s())])\n return a, b\na,b=asyncio.run(run())\nprint('ok', a=='to ann@example.com' and b=='to ann@example.com', %s)" % LEAKED),
    # Round 17: the vault's file primitives, every optional package blocked.
    ("vault lock and atomic write", "import os, tempfile\nfrom scikitplot.cleanprompt._files import atomic_write, locked\nt=os.path.join(tempfile.mkdtemp(), 'v.json')\nwith locked(t, timeout=5):\n atomic_write(t, '{}')\nprint('ok', open(t).read()=='{}' and oct(os.stat(t).st_mode & 0o777)=='0o600', %s)" % LEAKED),
    # Round 20: least-privilege tool arguments, every optional package blocked.
    ("tool arguments by kind", "import scikitplot.cleanprompt as c\ng=c.FluentCleanPrompt().guard(); g.outgoing('mail ann@example.com, card 4242 4242 4242 4242')\na=g.decode_tool_arguments({'to':'[EMAIL-1]'}, allow={'EMAIL'})\ntry:\n g.decode_tool_arguments({'u':'[CREDIT_CARD-1]'}, allow=()); r=False\nexcept c.LeakError:\n r=True\nprint('ok', a=={'to':'ann@example.com'} and r, %s)" % LEAKED),
    # Round 21: tool results as JSON documents, every optional package blocked.
    ("tool result keys and numbers", "import json, scikitplot.cleanprompt as c\ng=c.FluentCleanPrompt().guard()\no=json.dumps(g.encode_object({'ann@example.com': {'card': 4242424242424242, 'note': 'password: hunter2hunter2'}}))\nprint('ok', 'ann@' not in o and '4242' not in o and 'hunter2' not in o, %s)" % LEAKED),
    ("ner requested, none usable", "import scikitplot.cleanprompt as c\ntry:\n c.encode('Ada Lovelace', ner=True)\n print('FAIL silent')\nexcept c.CapabilityError as e:\n print('ok', 'install' in str(e), %s)\nexcept Exception as e:\n print('FAIL wrong type', type(e).__name__)" % LEAKED),
]

print("=" * 72)
print("IMPORT ISOLATION UNDER AN __import__ BLOCKER")
print("blocked:", ", ".join(BLOCK))
print("=" * 72)
failures = 0
for name, body in probes:
    code, out, err = run(BLOCKER + body)
    # "ok False" must not count. The first version of this harness tested only
    # that a probe printed "ok", so a probe whose assertion evaluated False
    # was reported as PASS. A probe that cannot fail is not evidence.
    status = (
        "PASS"
        if code == 0
        and out.startswith("ok")
        and "FAIL" not in out
        and "False" not in out
        else "FAIL"
    )
    if status == "FAIL":
        failures += 1
    print("%-30s %s  %s" % (name, status, out or err.splitlines()[-1:]))

print()
print("=" * 72)
print("CLI ENTRY POINTS UNDER THE SAME BLOCKER")
print("=" * 72)
for args in (
    ["--help"],
    ["doctor"],
    ["doctor", "--format", "json"],
    ["kinds"],
    ["inspect", "--in", "/dev/null"],
    ["redact", "--in", "/dev/null", "--vault", "/dev/null"],
    ["packs"],
    ["packs", "--check"],
    ["skill"],
    ["plan"],
    ["ask", "--via", "cat", "mail ann@example.com"],
):
    p = subprocess.run(
        [
            sys.executable,
            "-c",
            ISOLATED_PARENT
            + BLOCKER
            + "import runpy,sys;sys.argv=['m']+%r;runpy.run_module('scikitplot.cleanprompt',run_name='__main__')"
            % args,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    ok = p.returncode == 0
    if not ok:
        failures += 1
    print("%-30s %s  (exit %d)" % ("python -m ... " + " ".join(args), "PASS" if ok else "FAIL", p.returncode))

print()
print("TOTAL FAILURES:", failures)
sys.exit(1 if failures else 0)
