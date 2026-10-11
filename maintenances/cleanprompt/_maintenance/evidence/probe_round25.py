"""
Round 25 reproductions: what each finding looked like, printed as observed.

Run from the wide checkout root::

    python -B maintenances/cleanprompt/_maintenance/evidence/probe_round25.py

Run against the uploaded tree (before round 25) it printed, in order:
``spacy download en_core_web_lg`` against a runtime default of
``en_core_web_sm``; ``-p 5000:5000``; ``CLEANPROMPT_NER`` present;
``resolve_bind`` without a ``debug`` parameter; ``create_app`` calling
``spacy_detector(model=None)``; and six obfuscated values sent unchanged
(one card as ``4111\u200b[PHONE-1]``). After round 25 each line shows the
corrected behaviour. ``CP-093`` needs spaCy or NLTK installed without their
data and is exercised by the ``doctor`` lines at the end when they are.

This script reports; it asserts nothing. The assertions are the named
regression tests in ``tests/test_regressions.py`` and ``probe_negative.py``.
"""
import inspect
import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from scikitplot.cleanprompt import _serve, encode  # noqa: E402
from scikitplot.cleanprompt._languages import resolve_model  # noqa: E402

files = _serve.container_files(port=5000, with_ner=True)
dockerfile = files["Dockerfile"]
print("CP-095 image installs :", [l.strip() for l in dockerfile.splitlines() if "spacy download" in l])
print("CP-095 runtime default:", resolve_model("en", "sm")[0])
print("CP-095 image requests :", [l.strip() for l in dockerfile.splitlines() if l.startswith("CMD")])
print("CP-096 docker run     :", [l.strip() for l in "\n".join(files.values()).splitlines() if "docker run" in l])
print("CP-096 unread env     :", "CLEANPROMPT_NER" in files["docker-compose.yml"])
print("CP-097 debug guarded  :", "debug" in inspect.signature(_serve.resolve_bind).parameters)

from scikitplot.cleanprompt import _engines  # noqa: E402

calls = []
real = _engines.build_detectors
_engines.build_detectors = lambda **kw: calls.append(kw) or []
try:
    from scikitplot.cleanprompt._app import create_app

    create_app(ephemeral_secret_key=True, enable_ner=True, ner_engine="nltk", language="tr", model_size="lg")
    print("CP-094 create_app     :", calls)
except Exception as exc:  # noqa: BLE001 - a report, not a test
    print("CP-094 create_app     :", type(exc).__name__, exc)
finally:
    _engines.build_detectors = real

for text in (
    "mail ada\u200b@example.com now",
    "mail \uff41\uff44\uff41\uff20\uff45\uff58\uff41\uff4d\uff50\uff4c\uff45\uff0e\uff43\uff4f\uff4d now",
    "call +1 555\u00a00100",
    "call +1\u2011555\u20110100",
    "ip 192.0.2.\u200b10",
    "card 4111\u200b1111 1111 1111",
):
    print("CP-098", ascii(text), "->", ascii(encode(text).text))

from scikitplot.cleanprompt._cli import main  # noqa: E402

for args in (["doctor", "--ner"], ["doctor", "--ner", "--ner-engine", "nltk"]):
    out = io.StringIO()
    main([*args, "--format", "json"], stdin=io.StringIO(), stdout=out, stderr=io.StringIO())
    report = json.loads(out.getvalue())
    detection = report["detection"]
    print("CP-093", " ".join(args), "healthy=%s ner_ready=%s remedy=%s" % (
        report["healthy"], detection.get("ner_ready"), detection.get("ner_remedy")))
