"""
Cleanprompt Packs and Formats: Records, Configs and Whole Folders
=================================================================

.. currentmodule:: scikitplot.cleanprompt

Patterns find values by their *shape*: an address looks like an address. A
record says what its values are in its *keys*, and a pattern cannot read a key:

.. code-block:: text

    {"mrn": "00412345"}          an eight-digit number, to a pattern
    DB_PASSWORD=pw-1             four characters, to a pattern
    member_id,M-00412            nothing at all, to a pattern

Packs close that gap. A **pack** is a validated YAML definition of what to hide
in one domain — ``personal``, ``addressbook``, ``patient``, ``finance``,
``secrets``, ``records``, ``email``, ``cloud``, and ``pandas``/``numpy``/
``sklearn`` for code. A **format** says how a kind of file is read — CSV by its
header, JSON by its keys, ``.env`` by ``KEY=value``, a Word file by extracting
its text — and which packs suit it. :class:`FluentCleanPrompt` combines them
into one immutable plan, and :class:`Cleaner` applies the plan to text, files,
folders and zip archives under one shared vault.

Every value in this example is reserved and cannot reach anybody.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from scikitplot.cleanprompt import (
    CleanPromptError,
    FluentCleanPrompt,
    builtin_catalog,
    encode,
    load_custom,
)

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-cleanprompt-packs-")
_HOME = Path(_WORKSPACE.name)
os.environ["CLEANPROMPT_VAULT"] = str(_HOME / "vault.json")


def show(label: str, text: str) -> None:
    """Print a labelled block with the workspace path masked."""
    print("── {0} ──".format(label))
    print(text.replace(_WORKSPACE.name, "<workspace>").rstrip("\n"))
    print()


# %%
# 1. The leak packs exist to close
# --------------------------------
# A patient record through the pattern engine alone. Nothing in it has the
# shape of an address, a card or a telephone number, so nothing is hidden.

RECORD = (
    '{"patient": {"mrn": "00412345", "name": "Marion Holt", '
    '"diagnosis": "E11.9", "insurer_id": "ZX-4471-09", "age": 61}}'
)
patterns_only, _ = encode(RECORD)
show("patterns only", patterns_only)
assert "00412345" in patterns_only  # the measured defect, CP-048

cleaner = FluentCleanPrompt().materialize()  # packs('auto'): the format chooses
safe = cleaner.encode_text(RECORD, "json")
show("with packs", safe.text)
for secret in ("00412345", "Marion Holt", "E11.9", "ZX-4471-09"):
    assert secret not in safe.text
assert cleaner.decode(safe.text) == RECORD
print("report:", json.dumps(safe.report["kinds"]))

# %%
# The output is still JSON, and it stays JSON even for numbers: a hidden JSON
# number becomes a reserved negative number rather than ``[MRN-1]``, which no
# parser would open and which a model would "repair" into a string.

numbers = FluentCleanPrompt().packs("patient").materialize()
numeric = numbers.encode_text('{"mrn": 20417, "visits": 3}', "json")
show("a hidden number", numeric.text)
json.loads(numeric.text)
assert numbers.decode(numeric.text) == '{"mrn": 20417, "visits": 3}'

# %%
# 2. What is available
# --------------------
# Everything is data, compiled into a JSON file the standard library reads, so
# listing it needs nothing installed.

catalog = builtin_catalog()
for spec in catalog.resolve_packs("all"):
    print(
        "pack   {0:<12} fields={1:<2} patterns={2}  {3}".format(
            spec.name, len(spec.fields), len(spec.patterns), spec.summary[:60]
        )
    )
print()
for name in ("csv", "json", "env", "notebook", "docx", "archive"):
    spec = catalog.formats[name]
    print(
        "format {0:<9} {1:<16} round trip={2!s:<5} auto packs={3}".format(
            name, ",".join(spec.extensions), spec.round_trip, ",".join(spec.packs)
        )
    )

# %%
# 3. Three ways to choose
# -----------------------
# ``all`` for everything, one pack for one domain, or any combination. A pack's
# requirements come with it: ``addressbook`` needs ``personal``.

CONTACTS = "name,email,title,org\nMarion Holt,marion@example.com,Analyst,Acme Ltd\n"
for selection in (("all",), ("addressbook",), ("personal", "finance")):
    one = FluentCleanPrompt().packs(*selection).materialize()
    encoded = one.encode_text(CONTACTS, "csv")
    print("{0:<22} packs={1}".format(" + ".join(selection), encoded.report["packs"]))
    print("  ", encoded.text.splitlines()[1])
    assert one.decode(encoded.text) == CONTACTS

# %%
# 4. A plan is immutable, validated and fingerprinted
# ---------------------------------------------------
# Each setter returns a new builder. Setting one domain twice is refused unless
# you say what you meant, and a broken plan reports every problem at once.

base = FluentCleanPrompt().packs("patient")
print(base.packs("finance", conflict="extend").plan().packs)
try:
    base.packs("finance")
except CleanPromptError as error:
    print("refused:", error)

problems = FluentCleanPrompt().packs("patinet").formats("docz").style("fancy").validate()
for problem in problems:
    print("problem:", problem.splitlines()[0])
assert len(problems) == 3

first = FluentCleanPrompt().packs("patient", "finance").formats("csv", ".json").plan()
second = FluentCleanPrompt().formats(".json", "csv").packs("finance", "patient").plan()
assert first.fingerprint() == second.fingerprint()
print("same plan, any order:", first.fingerprint()[:16])

# %%
# 5. Your own pack, from a JSON file
# ----------------------------------
# A custom definition is validated exactly like a built-in: unknown keys are
# errors, and every pattern's examples are executed before it is used. JSON
# needs nothing installed; YAML needs PyYAML and says so when it is missing.

hr_pack = {
    "name": "hr",
    "version": 1,
    "summary": "Our HR identifiers.",
    "requires": ["personal"],
    "fields": [{"names": ["badge_id", "employee_number"], "kind": "EMPLOYEE", "role": "id"}],
    "patterns": [
        {
            "kind": "EMPLOYEE",
            "pattern": r"(?i)\bbadge\s*:?\s*(?P<value>EMP-\d{6})\b",
            "intent": "A badge number written with its label.",
            "examples_yes": ["badge: EMP-004121"],
            "examples_no": ["EMP-12"],
        }
    ],
}
hr_path = _HOME / "hr.json"
hr_path.write_text(json.dumps(hr_pack), encoding="utf-8")
print("loaded:", sorted(load_custom(hr_path).packs))

hr = FluentCleanPrompt().custom(str(hr_path)).packs("hr").materialize()
roster = hr.encode_text("badge_id,full_name\nEMP-004121,Marion Holt\n", "csv")
note = hr.encode_text("Visitor badge: EMP-004121 signed in.\n", "text")
show("roster", roster.text)
show("note", note.text)
assert "[EMPLOYEE-1]" in roster.text and "[EMPLOYEE-1]" in note.text  # one value, one label

try:
    import yaml  # noqa: F401
except ImportError:
    print("SKIP: PyYAML is not installed, so the YAML form is not shown")
else:
    yaml_path = _HOME / "hr.yaml"
    yaml_path.write_text(yaml.safe_dump(hr_pack), encoding="utf-8")
    print("YAML loads the same:", sorted(load_custom(yaml_path).packs) == ["hr"])

# %%
# 6. A whole folder
# -----------------
# Every file a selected format reads is encoded into a mirror folder under one
# vault. A file no format reads is skipped, one that cannot be read safely is
# refused; neither is written, so nothing unexamined leaves.


def _docx(paragraphs):
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join("<w:p><w:r><w:t>{0}</w:t></w:r></w:p>".format(p) for p in paragraphs)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="{0}"><w:body>{1}</w:body></w:document>'.format(namespace, body),
        )
    return buffer.getvalue()


project = _HOME / "project"
(project / "config").mkdir(parents=True)
(project / ".git").mkdir()
(project / "contacts.csv").write_text(
    "name,phone,member_id\nMarion Holt,+1 555 010 4477,M-00412\n", encoding="utf-8"
)
(project / "patient.json").write_text(RECORD, encoding="utf-8")
(project / "config" / ".env").write_text(
    "DB_HOST=db.internal.example\nDB_PASSWORD=pw-correct-horse\n", encoding="utf-8"
)
(project / "visit.docx").write_bytes(_docx(["Seen today. MRN: 00412345.", "Follow up in two weeks."]))
(project / "broken.json").write_text("{", encoding="utf-8")
(project / "logo.png").write_bytes(b"\x89PNG")
(project / ".git" / "config").write_text("[remote]\n", encoding="utf-8")

team = FluentCleanPrompt().materialize()
for item in team.encode_tree(project, _HOME / "safe"):
    print("{0:<8} {1:<18} {2}".format(item.status, item.relative, item.reason or item.output))

show("safe/visit.docx.txt", (_HOME / "safe" / "visit.docx.txt").read_text(encoding="utf-8"))
show("safe/config/.env", (_HOME / "safe" / "config" / ".env").read_text(encoding="utf-8"))

# %%
# The Word note and the JSON record name the same patient number, and both say
# ``[MRN-1]``: the vault is shared, and a labelled pattern hides the value, not
# its label. Decoding the folder restores every round-trip file byte for byte.

list(team.decode_tree(_HOME / "safe", _HOME / "back"))
for name in ("contacts.csv", "patient.json", "config/.env"):
    assert (_HOME / "back" / name).read_bytes() == (project / name).read_bytes()
print("restored byte for byte: contacts.csv, patient.json, config/.env")

# %%
# 7. A zip archive
# ----------------
# Members are handled like files. A member whose name climbs out of the archive
# is refused, a nested archive is not opened, and the output is deterministic.

bundle = _HOME / "bundle.zip"
with zipfile.ZipFile(bundle, "w") as archive:
    archive.write(project / "contacts.csv", "contacts.csv")
    archive.write(project / "patient.json", "records/patient.json")
    archive.writestr("../escape.txt", "x")
for item in team.encode_archive(bundle, _HOME / "bundle.safe.zip"):
    print("{0:<8} {1:<22} {2}".format(item.status, item.relative, item.reason))
with zipfile.ZipFile(_HOME / "bundle.safe.zip") as archive:
    print("written:", archive.namelist())

# %%
# 8. With the corpus submodule
# ----------------------------
# :mod:`scikitplot.corpus` reads documents; cleanprompt makes them safe before
# they are embedded, indexed or prompted. Either works without the other.

try:
    from scikitplot.corpus import CorpusDocument
except ImportError as error:  # the bridge is optional in both directions
    print("SKIP: scikitplot.corpus is not importable here:", error)
else:
    from scikitplot.cleanprompt import redact_documents

    documents = [
        CorpusDocument.create("visit.txt", 0, "Seen today. MRN: 00412345.", metadata={"patient_name": "Marion Holt"})
    ]
    (safe_doc,) = redact_documents(documents, team)
    print(safe_doc.text, safe_doc.metadata)
    assert "00412345" not in safe_doc.text

team.clear()

# %%
# 9. The same from the command line
# ---------------------------------
# ``packs`` lists and checks definitions; ``batch`` encodes a folder or a zip
# and writes the vault that ``batch --decode`` and ``decode`` read.


def _cli() -> list[str]:
    executable = shutil.which("scikitplot")
    if executable:
        return [executable, "cleanprompt"]
    return [sys.executable, "-m", "scikitplot.cleanprompt"]


def run(*arguments: str, limit: int = 12):
    completed = subprocess.run([*_cli(), *arguments], input="", capture_output=True, text=True)
    shown = " ".join(one.replace(_WORKSPACE.name, "<workspace>") for one in arguments)
    print("$ cleanprompt", shown)
    lines = completed.stdout.replace(_WORKSPACE.name, "<workspace>").rstrip("\n").split("\n")
    if len(lines) > limit:
        lines = lines[:limit] + ["... ({0} more)".format(len(lines) - limit)]
    print("  " + "\n  ".join(lines))
    print("  exit", completed.returncode)
    return completed


run("packs", "--check")
done = run("batch", str(project), "--out", str(_HOME / "cli-safe"), "--pack", "all", "-f", "json", limit=6)
assert done.returncode == 1  # broken.json was refused, and a pipeline must see that
run("batch", str(_HOME / "cli-safe"), "--decode", "--out", str(_HOME / "cli-back"), limit=4)
assert (_HOME / "cli-back" / "patient.json").read_text(encoding="utf-8") == RECORD

# %%
# 10. Cleanup
# -----------

os.environ.pop("CLEANPROMPT_VAULT", None)
_WORKSPACE.cleanup()
print("Temporary workspace cleaned:", not Path(_WORKSPACE.name).exists())

# %%
#
# .. tags::
#
#    model-workflow: cleanprompt
#    plot-type: text
#    level: intermediate
#    purpose: showcase
