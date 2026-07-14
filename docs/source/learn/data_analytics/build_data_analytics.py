#!/usr/bin/env python3
"""
Generator for the Data Analytics hub — a NESTED, two-level Sphinx tree.

Structure (unlike the four flat sealed courses)::

    data_analytics/
      index.rst                      <- HUB browser, grouped by the 8 sections
      <n>_<slug>/
        index.rst                    <- SECTION browser, grouped by that section's stages
        NNN-*.rst                    <- lesson pages, per-section 3-digit numbering

Every index.rst -- hub and section alike -- is produced by the SAME
``render_browser`` routine, so all of them share the v2 filterable-browser logic
(a dependency-free JS filter box + collapsed dropdowns + an A-Z master list). The
hub groups lessons by section; a section index groups its lessons by stage.

The generator is deterministic and idempotent: sorted iteration, no RNG, no
timestamps -> byte-identical rebuilds. It reads three inputs
(``da_inventory.tsv``, ``da_content.py``, and the registries below) and writes
the hub index, one folder per populated section, and one page per lesson.

Sections with no inventory rows yet (harvest pending) are listed in the hub with
their known post totals but are not built until their rows are added.
"""

from __future__ import annotations

import csv
import re
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import da_content as _c  # noqa: E402

CONTENT: dict[str, str] = _c.CONTENT
MINDMAP: dict[str, list[str]] = _c.MINDMAP
GLOSS: dict[str, str] = getattr(_c, "GLOSS", {})

ANCHOR_PREFIX = "da"
N_LESSONS_TOTAL = 216  # full course target across all 8 sections

# --- Section registry: key -> (order, emoji, title, blurb, folder) ------------
SECTIONS: dict[str, tuple[int, str, str, str, str]] = {
    "foundations": (1, "\U0001F331", "Foundations",
                    "The case for data, the analysis process and data life cycle, "
                    "analytical thinking, and the core tools of the trade.",
                    "1_foundations"),
    "ddd": (2, "\U0001F3AF", "Data-Driven Decisions",
            "Turning questions into decisions: stakeholders, metrics, and "
            "communicating results that drive action.",
            "2_data_driven_decisions"),
    "prep": (3, "\U0001F4E6", "Data Preparation",
             "Sourcing, structuring, and organising data before analysis: types, "
             "formats, databases, and sampling.",
             "3_data_preparation"),
    "cleaning": (4, "\U0001F9FD", "Data Cleaning & Preparation",
                 "Finding and fixing dirty data: missing values, duplicates, "
                 "outliers, and validation for trustworthy inputs.",
                 "4_data_cleaning_preparation"),
    "analyze": (5, "\U0001F4CA", "Analyze Data",
                "Organising, formatting, aggregating, and computing on data to "
                "surface patterns and answer the question.",
                "5_analyze_data"),
    "viz": (6, "\U0001F3A8", "Data Visualization",
            "Turning results into visuals that inform: chart choice, design "
            "principles, and honest, accessible graphics.",
            "6_data_visualization"),
    "python": (7, "\U0001F40D", "Data Analysis Using Python",
               "Doing the whole workflow in Python: NumPy, pandas, and plotting "
               "for real analytical tasks.",
               "7_data_analysis_python"),
    "jobsearch": (8, "\U0001F4BC", "Job Search",
                  "From portfolio to offer: resumes, the analyst interview, "
                  "case studies, and landing the role.",
                  "8_job_search"),
}
SECTION_ORDER = [k for k, _ in sorted(SECTIONS.items(), key=lambda kv: kv[1][0])]

# Known post totals per section (from the source category pages).
SECTION_TOTALS = {
    "foundations": 27, "ddd": 27, "prep": 25, "cleaning": 32,
    "analyze": 30, "viz": 27, "python": 33, "jobsearch": 15,
}

# --- Stage registry: (section, stage) -> (emoji, title, blurb) -----------------
STAGES: dict[tuple[str, str], tuple[str, str, str]] = {
    ("foundations", "why"): (
        "\U0001F31F", "Why Data Analytics",
        "Why data matters today and how it reshapes work and decision-making."),
    ("foundations", "process"): (
        "\U0001F504", "The Analysis Process & Data Life Cycle",
        "The six phases of analysis and the life cycle data moves through."),
    ("foundations", "thinking"): (
        "\U0001F9E0", "Analytical Skills & Thinking",
        "Analytical skills, structured questioning, and root-cause methods."),
    ("foundations", "tools"): (
        "\U0001F9F0", "Tools, Applications & Ethics",
        "Spreadsheets, SQL, visualization, industry uses, and fairness."),
    ("ddd", "framing"): (
        "\U0001F9ED", "Framing the Problem",
        "Problem types, asking the right questions, and how data drives decisions."),
    ("ddd", "metrics"): (
        "\U0001F4D0", "Metrics & Dashboards",
        "Data versus metrics, dashboards, and quantitative thinking."),
    ("ddd", "spreadsheets"): (
        "\U0001F4D7", "Spreadsheets for Analysis",
        "Organising, calculating, and troubleshooting analysis in spreadsheets."),
    ("ddd", "execution"): (
        "\U0001F5E3", "Stakeholders, Communication & Execution",
        "Scoping with stakeholders and communicating results that drive action."),
    ("prep", "types"): (
        "\U0001F9EC", "Data Types & Structure",
        "How data is generated, its types and formats, and structured vs tabular shapes."),
    ("prep", "bias_ethics"): (
        "\u2696\uFE0F", "Bias & Data Ethics",
        "Recognising bias, judging sources with ROCCC, and the ethics of data use."),
    ("prep", "sources"): (
        "\U0001F5C4\uFE0F", "Databases & Data Sources",
        "Relational databases, metadata and governance, and accessing data."),
    ("prep", "spreadsheets_sql"): (
        "\U0001F522", "Spreadsheets, SQL & Organization",
        "Importing, sorting, querying with SQL, organising, and securing data."),
    ("cleaning", "integrity"): (
        "\U0001F9F1", "Data Integrity & Sampling",
        "Why clean data matters, integrity risks, sampling, power, and margin of error."),
    ("cleaning", "dirty"): (
        "\U0001F9F9", "Dirty Data & Spreadsheet Cleaning",
        "Recognising dirty data and cleaning it with spreadsheet tools and functions."),
    ("cleaning", "sql"): (
        "\U0001F42C", "Cleaning with SQL",
        "SQL from first queries to CAST, COALESCE, and advanced cleaning functions."),
    ("cleaning", "verify"): (
        "\u2705", "Verification, Documentation & Next Steps",
        "Verifying and reporting cleaning work, documenting changes, and moving forward."),
    ("analyze", "organize"): (
        "\U0001F5C2\uFE0F", "Organizing & Formatting Data",
        "Sorting, filtering, formatting, validating, and string work in sheets and SQL."),
    ("analyze", "combine"): (
        "\U0001F517", "Problem-Solving & Combining Data",
        "Getting unstuck, choosing tools, and joining data with VLOOKUP, JOIN, and subqueries."),
    ("analyze", "calc"): (
        "\U0001F9EE", "Calculations & Aggregation",
        "Formulas, conditional aggregation, pivot tables, and SQL GROUP BY calculations."),
    ("analyze", "advanced"): (
        "\U0001F680", "Validation & Temporary Tables",
        "Ongoing validation and temporary tables with WITH for cleaner SQL workflows."),
    ("viz", "principles"): (
        "\U0001F3A8", "Visualization Principles & Design",
        "What makes a visual work: purpose, art elements, chart choice, and accessibility."),
    ("viz", "tableau"): (
        "\U0001F4CA", "Tableau",
        "From first steps in Tableau Public to creative, linked, effective visuals."),
    ("viz", "story"): (
        "\U0001F4D6", "Storytelling & Dashboards",
        "Turning findings into narrative, key messages, dashboards, and focused views."),
    ("viz", "present"): (
        "\U0001F3A4", "Presentations & Q&A",
        "Structuring persuasive presentations, slide design, delivery, and handling Q&A."),
    ("python", "basics"): (
        "\U0001F40D", "Python Fundamentals",
        "From first program and Jupyter to variables, types, functions, and clean code."),
    ("python", "control"): (
        "\U0001F500", "Control Flow",
        "Booleans, branching, and iteration with while, for, and range."),
    ("python", "structures"): (
        "\U0001F4DA", "Strings & Data Structures",
        "Strings, lists, tuples, comprehensions, dictionaries, and sets in practice."),
    ("python", "libraries"): (
        "\U0001F43C", "NumPy & pandas",
        "Vectorized arrays and DataFrames: masking, groupby, and combining data."),
    ("jobsearch", "identity"): (
        "\U0001F9ED", "Career Identity & Planning",
        "Transferable skills, a career identity statement, and an AI-assisted search plan."),
    ("jobsearch", "apply"): (
        "\U0001F4C4", "Resume, Brand & Applications",
        "Tailoring resumes, building an online presence, platforms, tracking, networking."),
    ("jobsearch", "interview"): (
        "\U0001F3AF", "Interviews & Follow-Up",
        "Interview prep, the STAR method, AI practice tools, and post-interview strategy."),
}
# Per-section stage display order.
STAGE_ORDER: dict[str, list[str]] = {
    "foundations": ["why", "process", "thinking", "tools"],
    "ddd": ["framing", "metrics", "spreadsheets", "execution"],
    "prep": ["types", "bias_ethics", "sources", "spreadsheets_sql"],
    "cleaning": ["integrity", "dirty", "sql", "verify"],
    "analyze": ["organize", "combine", "calc", "advanced"],
    "viz": ["principles", "tableau", "story", "present"],
    "python": ["basics", "control", "structures", "libraries"],
    "jobsearch": ["identity", "apply", "interview"],
}

# Legacy anchors from the pre-existing hand hub, emitted on every page -> hub top.
COMPAT_ANCHORS = [
    "data-analytics", "da-foundations", "da-decisions", "da-prep",
    "da-cleaning", "da-analyze", "da-viz", "da-python", "da-jobsearch",
]


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
def slugify(title: str) -> str:
    s = title.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"-+", "-", s)
    return s


def bar(width_for: str, ch: str) -> str:
    return ch * max(72, len(width_for) + 2)


def load_inventory() -> list[dict]:
    rows: list[dict] = []
    with open(HERE / "da_inventory.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append({k: v.strip() for k, v in r.items()})
    return rows


def validate(rows: list[dict]) -> None:
    errs: list[str] = []
    titles = [r["title"] for r in rows]
    inv = set(titles)
    if len(titles) != len(inv):
        seen: set[str] = set()
        for t in titles:
            if t in seen:
                errs.append(f"duplicate inventory title: {t!r}")
            seen.add(t)
    for r in rows:
        sec, st = r["section"], r["stage"]
        if sec not in SECTIONS:
            errs.append(f"unknown section {sec!r} for {r['title']!r}")
        elif (sec, st) not in STAGES:
            errs.append(f"unknown stage {sec}/{st} for {r['title']!r}")
        elif st not in STAGE_ORDER.get(sec, []):
            errs.append(f"stage {sec}/{st} missing from STAGE_ORDER")
    for key in CONTENT:
        if key not in inv:
            errs.append(f"CONTENT key not in inventory: {key!r}")
    for key in MINDMAP:
        if key not in inv:
            errs.append(f"MINDMAP key not in inventory: {key!r}")
        for nb in MINDMAP[key]:
            if nb not in inv:
                errs.append(f"MINDMAP neighbour not in inventory: {nb!r}")
    for key in GLOSS:
        if key not in inv:
            errs.append(f"GLOSS key not in inventory: {key!r}")
    if errs:
        for e in errs:
            print("ERROR:", e, file=sys.stderr)
        raise SystemExit(1)


def render(template, style="format", **kwargs):
    """
    FORMAT = "Hello {name}"
    PERCENT = "Hello %(name)s"
    TEMPLATE = "Hello $name"

    print(render(FORMAT, style="format", name="Alice"))
    print(render(PERCENT, style="percent", name="Alice"))
    print(render(TEMPLATE, style="template", name="Alice"))
    """
    if style == "format":
        return template.format(**kwargs)
    if style == "percent":
        return template % kwargs
    if style == "template":
        from string import Template
        return Template(template).substitute(**kwargs)
    raise ValueError(f"Unknown style: {style}")


# ------------------------------------------------------------------------------
# the shared v2 browser (used by BOTH the hub and every section index)
# ------------------------------------------------------------------------------
FILTER_JS_BASE = "doc"  # div class

FILTER_JS_DIV_CLASS = """.. raw:: html

   <div style="text-align:center;margin:0.4rem 0 0.4rem">
   <input type="text" id="term-filter" placeholder="\U0001F50D Type to filter %(scope)s &mdash; by title or keyword\u2026"
          style="width:100%%;padding:.6em .8em;margin:.4em 0 1em;font-size:1em;
                 border:1px solid var(--pst-color-border,#ccc);border-radius:6px;box-sizing:border-box;">
   <div id="term-filter-count" style="margin:-.6em 0 1em;font-size:.85em;opacity:.7;"></div>
   </div>
   <script>
   (function(){
     var box=document.getElementById('term-filter');
     if(!box)return;
     var count=document.getElementById('term-filter-count');
     function norm(s){return (s||'').toLowerCase();}
     function run(){
       var q=norm(box.value), shown=0, total=0;
       document.querySelectorAll('.da-row').forEach(function(row){
         total++;
         var hit=q===''||norm(row.getAttribute('data-k')).indexOf(q)>=0;
         row.style.display=hit?'':'none';
         if(hit)shown++;
       });
       document.querySelectorAll('details.sd-dropdown, details.term-az').forEach(function(d){
         var any=d.querySelectorAll('.da-row:not([style*="none"])').length>0;
         d.style.display=any?'':'none';
         if(q!==''&&any)d.setAttribute('open','');
       });
       count.textContent=q===''?'':(shown+' of '+total+' match');
     }
     box.addEventListener('input',run);
   })();
   </script>
"""

# ---- live filter: type-to-search across every term (progressive JS) ----
# Static, dependency-free, deterministic. Without JS the page degrades
# gracefully to plain collapsible dropdowns.
# Same pattern/classes as learn/terminology (details.sd-dropdown, .term-az)
FILTER_JS_DOC = """.. raw:: html

   <div style="text-align:center;margin:0.4rem 0 0.4rem">
   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"
           placeholder="&#128269;&nbsp; Type to filter %(scope)s lessons &mdash; by title or keyword&hellip;"
          style="width:100%%;max-width:100%%;padding:0.55rem 1rem;font-size:1rem;
                 border:1px solid var(--pst-color-border,#ccc);border-radius:0.55rem;box-sizing:border-box;
                 background:transparent;color:inherit"/>
   <div id="term-filter-count" style="opacity:0.65;font-size:0.85rem;
        min-height:1.2em;margin-top:0.35rem"></div>
   </div>
   <script>
   document.addEventListener('DOMContentLoaded',function(){
     var inp=document.getElementById('term-filter');if(!inp){return;}
     var dds=[].slice.call(document.querySelectorAll('details.sd-dropdown'));
     var az=document.querySelector('details.term-az');
     var items=[];
     dds.forEach(function(d){[].slice.call(d.querySelectorAll('li')).forEach(
       function(li){items.push({li:li,d:d,t:li.textContent.toLowerCase()});});});
     var cnt=document.getElementById('term-filter-count');
     inp.addEventListener('input',function(){
       var q=inp.value.trim().toLowerCase();var n=0;
       dds.forEach(function(d){d.tHits=0;});
       items.forEach(function(it){
         var hit=!q||it.t.indexOf(q)!==-1;
         it.li.style.display=hit?'':'none';
         if(hit){it.d.tHits+=1;if(az&&it.d===az){n+=1;}}});
       dds.forEach(function(d){
         if(q){d.style.display=d.tHits?'':'none';d.open=d.tHits>0;}
         else{d.style.display='';d.open=false;}});
        if(cnt){{cnt.textContent=(q&&az)?(n+' of {n_items} match'+(n===1?'':'s')):'';}}
     });
   });
   </script>
"""


def browser_row(label: str, key: str, doc_target: str) -> str:
    """One clickable line inside a dropdown, tagged for the filter."""
    safe = key.replace('"', "&quot;")
    return (f"   <div class=\"da-row\" data-k=\"{safe}\">"
            f"<a href=\"{doc_target}.html\">{label}</a></div>")


def render_browser(page_anchor: str, h1: str, intro_lines: list[str],
                   scope_word: str, groups: list[dict], az_entries: list[tuple]) -> str:
    """
    Emit a v2 filterable browser. Used identically by hub and section indexes.

    ``groups`` : ordered list of dicts with keys emoji,title,blurb,anchor,rows
                 where each row is (label, filter_key, doc_target).
    ``az_entries`` : (label, filter_key, doc_target) for the A-Z master dropdown.
    """
    I: list[str] = []
    w = I.append
    w(":html_theme.sidebar_secondary.remove:")
    w("")
    w(".. role:: raw-html(raw)")
    w("   :format: html")
    w("")
    w(".. |br| raw:: html")
    w("")
    w("   <br/>")
    w("")
    w(f".. _{page_anchor}:")
    w("")
    w(f":raw-html:`<div style=\"text-align:center\"><strong>` {h1}")
    w("|br| |full_version| - |today|")
    w(":raw-html:`</strong></div>`")
    w("")
    w(bar(h1, "="))
    w(h1)
    w(bar(h1, "="))
    w("")
    I.extend(intro_lines)
    w("")
    if FILTER_JS_BASE == "doc":
        w(FILTER_JS_DOC % {"scope": scope_word})
    else:
        w(FILTER_JS_DIV_CLASS % {"scope": scope_word})
    w("")

    for g in groups:
        head = f"{g['emoji']} {g['title']}"
        w(f".. dropdown:: {head}")
        w("   :animate: fade-in-slide-down")
        w("   :class-container: sd-dropdown")
        w("")
        if g.get("blurb"):
            w(f"   {g['blurb']}")
            w("")

        if FILTER_JS_BASE == "doc":
            for label, _key, tgt in g["rows"]:
                w(f"   * :doc:`{label} <{tgt}>`")
            w("")
        else:
            w("   .. raw:: html")
            w("")
            for label, key, tgt in g["rows"]:
                w("      " + browser_row(label, key, tgt))
            w("")

    # ---- dictionary view: one A-Z master list (auto-sorted) ----------
    az_head = "\U0001F524 Every lesson, A\u2013Z"
    w(az_head)
    w("-" * (len(az_head) + 2))
    w("")
    w(".. dropdown:: \U0001F520 A\u2013Z index")
    w("   :animate: fade-in-slide-down")
    w("   :class-container: term-az")
    w("")

    if FILTER_JS_BASE == "doc":
        w("   .. hlist::")
        w("      :columns: 2")
        w("")
        # for g in sorted(groups, key=lambda r: str.casefold(r["title"])):
        #     for label, _key, tgt in sorted(g["rows"], key=lambda r: str.casefold(r[0].split(' · ')[1])):
        for label, _key, tgt in sorted(az_entries, key=lambda e: e[1].lower()):
            w(f"      * :doc:`{label} <{tgt}>`")
        w("")
    else:
        w("   .. raw:: html")
        w("")
        for label, key, tgt in sorted(az_entries, key=lambda e: e[1].lower()):
            w("      " + browser_row(label, key, tgt))
        w("")

    # hidden ordered toctree so Sphinx builds the sequence + sidebar nav
    w(".. toctree::")
    w("   :hidden:")  # → Do not render this toctree in the page body.
    w("   :includehidden:")  # → Include descendant toctrees that are themselves marked :hidden: when building the overall hierarchy.
    w(f"   :maxdepth: {(2 if any(os.sep in i[2] for i in az_entries) else 1)}")  # → Control how many levels of the hierarchy are exposed to navigation.
    w("")
    _index = set()
    for _label, _key, tgt in sorted(az_entries, key=lambda e: e[2].lower()):
        i = tgt.split(os.sep)[0]
        if os.sep in tgt and i not in _index:
            _index.add(i)
            w(f"   {i}/index")
        if os.sep not in tgt:
            w(f"   {tgt}")
    w("")
    w(".. tags:: purpose: reference, topic: data analytics")
    w("")
    return "\n".join(I) + "\n"


# ------------------------------------------------------------------------------
# lesson page
# ------------------------------------------------------------------------------
def lesson_page(row: dict, num: int, stem: str, sec_rows: list[tuple],
                idx_in_sec: int) -> str:
    title, sec, st = row["title"], row["section"], row["stage"]
    _, s_emoji, s_title, _, _ = SECTIONS[sec]
    st_emoji, st_title, _ = STAGES[(sec, st)]
    anchor = f"{ANCHOR_PREFIX}-{sec}-{num:03d}"

    I: list[str] = []
    w = I.append
    w(":html_theme.sidebar_secondary.remove:")
    w("")
    w(".. role:: raw-html(raw)")
    w("   :format: html")
    w("")
    w(".. |br| raw:: html")
    w("")
    w("   <br/>")
    w("")
    w(f".. _{anchor}:")
    for legacy in COMPAT_ANCHORS:
        w(f".. _{legacy}-{sec}-{num:03d}:")
    w("")
    w(bar(title, "="))
    w(title)
    w(bar(title, "="))
    w("")
    w(f":bdg-primary:`{s_emoji} {s_title}` "
      f":bdg-secondary:`{st_emoji} {st_title}` "
      f":bdg-info:`Lesson {num:03d}`")
    w("")
    # prev / next within the section
    nav: list[str] = []
    if idx_in_sec > 0:
        pstem = sec_rows[idx_in_sec - 1][1]
        nav.append(f"\u25C0 :doc:`Previous <{pstem}>`")
    if idx_in_sec < len(sec_rows) - 1:
        nstem = sec_rows[idx_in_sec + 1][1]
        nav.append(f":doc:`Next <{nstem}>` \u25B6")
    nav.append(":doc:`\u2191 Section <index>`")
    nav.append(":doc:`\u2191 Hub <../index>`")
    w(" \u00b7 ".join(nav))
    w("")
    body = CONTENT.get(title)
    if body is not None:
        w(body.rstrip("\n"))
        w("")
    else:
        g = GLOSS.get(title, "")
        if g:
            w(g)
            w("")
        w(".. admonition:: Lesson in progress")
        w("   :class: note")
        w("")
        w("   Full content for this lesson has not been written yet.")
        w("")

    # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#admonitions-messages-and-warnings
    # Note      → "Be aware of this clarification or detail."          # 📝 Neutral observations, assumptions, clarifications, conventions, or exceptions.
    # See also  → "Explore these related topics and resources."        # 📚 Internal/external references, further reading, related topics, prerequisites.
    # Hint      → "This may help you understand the concept."          # 💡 Intuition, conceptual connections, mind maps, learning aids.
    # Tip       → "This may help you work more effectively."           # 💡 Best practices, shortcuts, recommendations, efficient/advice workflows.
    # Info      → "Here's additional background or context."           # ℹ️ Background, implementation notes, **sources used by this page**, supplementary factual information where the information came from.
    # Important → "Do not overlook this; it's essential."              # ⭐ Critical/Essential concepts, requirements, or limitations.

    # lateral cross-links
    nbs = MINDMAP.get(title, [])
    if nbs:
        w(".. hint::")
        w("")
        for nb in nbs:
            tgt = TITLE_TO_DOC.get(nb)
            if tgt:
                w(f"   - :doc:`{nb} <{tgt}>`")
        w("")

    # source (context/traceability)
    w(".. seealso::")
    w("")
    w(f"   **Source article** Adapted (context, re-expressed) in our own words from: `{row['url']} <{row['url']}>`__ "
               f"(insightful-data-lab.com).")
    w("")

    # tags
    w(f".. tags:: purpose: reference, topic: data analytics, topic: {sec}, topic: {st}")
    w("")
    return "\n".join(I) + "\n"


# ------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------
TITLE_TO_DOC: dict[str, str] = {}  # title -> doc target relative to the linking file


def build() -> None:
    rows = load_inventory()
    validate(rows)

    # group rows by section, preserve inventory order, assign per-section numbers
    by_section: dict[str, list[dict]] = {}
    for r in rows:
        by_section.setdefault(r["section"], []).append(r)

    # first pass: assign stems + fill TITLE_TO_DOC (section-relative + hub-relative)
    # section-relative doc target for a lesson in the SAME section is just its stem;
    # for cross-section links we store the hub-relative path and fix up per file.
    section_stems: dict[str, list[tuple]] = {}  # section -> [(row, stem, num)]
    for sec in SECTION_ORDER:
        srows = by_section.get(sec, [])
        stems: list[tuple] = []
        for i, r in enumerate(srows, 1):
            stem = f"{i:03d}-{slugify(r['title'])}"
            stems.append((r, stem, i))
        section_stems[sec] = stems

    # global title -> hub-relative doc (folder/stem); section pages will rewrite
    # same-section targets to bare stem, cross-section to ../<folder>/<stem>.
    global_doc: dict[str, tuple[str, str]] = {}  # title -> (section, stem)
    for sec in SECTION_ORDER:
        for r, stem, _ in section_stems[sec]:
            global_doc[r["title"]] = (sec, stem)

    # write each populated section
    built_sections: list[str] = []
    for sec in SECTION_ORDER:
        stems = section_stems[sec]
        if not stems:
            continue
        built_sections.append(sec)
        folder = HERE / SECTIONS[sec][4]
        folder.mkdir(exist_ok=True)
        # clear only numbered pages
        for p in folder.glob("[0-9]*.rst"):
            p.unlink()

        sec_rows = [(r, stem) for r, stem, _ in stems]  # (row, stem) in order

        # per-file TITLE_TO_DOC: from a page in THIS section, same-section -> stem,
        # cross-section -> ../<folder>/<stem>
        def doc_target_from_section(title: str, cur_sec: str) -> str | None:
            if title not in global_doc:
                return None
            tsec, tstem = global_doc[title]
            if tsec == cur_sec:
                return tstem
            return f"../{SECTIONS[tsec][4]}/{tstem}"

        # emit lesson pages
        for idx, (r, stem, num) in enumerate(stems):
            # rebuild TITLE_TO_DOC for this file's perspective
            TITLE_TO_DOC.clear()
            for t in global_doc:
                td = doc_target_from_section(t, sec)
                if td:
                    TITLE_TO_DOC[t] = td
            page = lesson_page(r, num, stem, sec_rows, idx)
            (folder / f"{stem}.rst").write_text(page, encoding="utf-8")

        # section index: v2 browser grouped by stage, links = bare stems
        groups: list[dict] = []
        az: list[tuple] = []
        for st in STAGE_ORDER[sec]:
            st_emoji, st_title, st_blurb = STAGES[(sec, st)]
            rws: list[tuple] = []
            for r, stem, num in stems:
                if r["stage"] != st:
                    continue
                g = GLOSS.get(r["title"], "")
                label = f"{num:03d} \u00b7 {r['title']}" + (f" \u2014 {g}" if g else "")
                key = f"{r['title']} {g}"
                rws.append((label, key, stem))
                az.append((f"{r['title']}", r["title"], stem))
            groups.append({"emoji": st_emoji, "title": st_title,
                           "blurb": st_blurb, "anchor": f"{ANCHOR_PREFIX}-{sec}-stage-{st}",
                           "rows": rws})
        order, s_emoji, s_title, s_blurb, _ = SECTIONS[sec]
        h1 = f"{s_emoji} {s_title}"
        intro = [
            f"*Section {order} of the Data Analytics hub — "
            f"{len(stems)} of {SECTION_TOTALS[sec]} lessons.*",
            "",
            s_blurb,
            "",
            ":doc:`\u2191 Back to the Data Analytics hub <../index>`",
        ]
        idx_rst = render_browser(f"{ANCHOR_PREFIX}-{sec}-index", h1, intro,
                                 "this section", groups, az)
        (folder / "index.rst").write_text(idx_rst, encoding="utf-8")

    # write the HUB index: v2 browser grouped by section, links = folder/stem
    hub_groups: list[dict] = []
    hub_az: list[tuple] = []
    for sec in SECTION_ORDER:
        order, s_emoji, s_title, s_blurb, folder_name = SECTIONS[sec]
        stems = section_stems[sec]
        total = SECTION_TOTALS[sec]
        rws: list[tuple] = []
        if stems:
            for r, stem, num in stems:
                g = GLOSS.get(r["title"], "")
                label = f"{num:03d} \u00b7 {r['title']}" + (f" \u2014 {g}" if g else "")
                key = f"{r['title']} {g} {s_title}"
                tgt = f"{folder_name}/{stem}"
                rws.append((label, key, tgt))
                hub_az.append((f"{r['title']}  ({s_title})", r["title"], tgt))
            blurb = (f"{s_blurb}  \u2014 **{len(stems)}/{total} lessons.** "
                     f":doc:`Open the section <{folder_name}/index>`")
        else:
            blurb = f"{s_blurb}  \u2014 *{total} lessons, harvest pending.*"
        hub_groups.append({"emoji": s_emoji,
                           "title": f"{order}. {s_title}",
                           "blurb": blurb, "anchor": f"{ANCHOR_PREFIX}-section-{sec}",
                           "rows": rws})

    n_built = sum(len(section_stems[s]) for s in SECTION_ORDER)
    hub_h1 = "\U0001F4C8 Data Analytics"
    hub_intro = [
        "A hands-on data-analytics curriculum in eight sections, from first "
        "principles to the job hunt \u2014 rewritten and cross-linked for scikit-plots.",
        "",
        f"*{n_built} of {N_LESSONS_TOTAL} lessons written across "
        f"{len(built_sections)} of {len(SECTION_ORDER)} sections.* Each section "
        "below is its own browsable mini-course; use the filter to search every "
        "lesson at once.",
    ]
    hub_rst = render_browser(f"{ANCHOR_PREFIX.replace('da', 'data-analytics')}-index"
                             if False else "data-analytics-index",
                             hub_h1, hub_intro, "all lessons", hub_groups, hub_az)
    (HERE / "index.rst").write_text(hub_rst, encoding="utf-8")

    print(f"Wrote hub index + {len(built_sections)} section(s): "
          f"{', '.join(built_sections)}")
    full = sum(1 for r in rows if r["title"] in CONTENT)
    print(f"lessons: {n_built} pages ({full} full, {n_built - full} stub) "
          f"of {N_LESSONS_TOTAL} total")


if __name__ == "__main__":
    build()
