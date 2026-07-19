#!/usr/bin/env python3
# ======================================================================
# build_deep_learning.py  —  deterministic generator for learn/deep_learning
# ----------------------------------------------------------------------
# Reads dl_inventory.tsv (title <TAB> url <TAB> stage), joins it with the
# STAGES + GLOSS tables below and the CONTENT / MINDMAP store in
# dl_content.py, and emits an idempotent, ORDERED LEARNING PATH:
#   * index.rst        -> staged hub (4 stages) + hidden ordered toctree
#   * NN-<slug>.rst     -> one lesson page per post (NN = curriculum order),
#                          each with prev/next navigation, a re-expressed
#                          body, lateral "See also" links, and a source.
#
# Re-runnable: same inputs -> byte-identical output (ordered, no RNG).
# Old NN-*.rst pages are cleared first so the set never drifts; only the
# NN-*.rst glob is removed (toolkit files + index.rst are never touched).
#
# Fail-fast: the build aborts (nothing written) if any inventory title
# lacks a GLOSS entry (or vice-versa), any inventory stage is unknown, or
# any CONTENT / MINDMAP key (or MINDMAP neighbour) is not an exact
# inventory title.  No silent gaps, no dangling cross-links.
#
# To edit/extend: change dl_content.py (bodies), append a row to
# dl_inventory.tsv (+ a GLOSS line here), then re-run.  See DEEP_LEARNING.md.
# ======================================================================
from __future__ import annotations
import csv
import re
import sys
import unicodedata
from pathlib import Path

# --- AI-generated-content notice (emitted at the top of every lesson body) ------
AI_NOTICE = (
    ".. important::\n"
    "\n"
    "   **AI-generated content.** This page was written with the assistance of an\n"
    "   AI language model and is provided as a learning aid. Despite careful\n"
    "   review, it may still contain mistakes, omissions, or out-of-date\n"
    "   information. Whether you are new to the topic, a team lead, or a senior\n"
    "   practitioner, treat it as a starting point rather than an authoritative\n"
    "   reference: read it critically and independently verify anything you act on\n"
    "   (code, commands, figures, and factual claims) against official\n"
    "   documentation and primary sources before relying on it."
)



HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # make dl_content importable regardless of CWD
INVENTORY = HERE / "dl_inventory.tsv"
OUT = HERE / "index.rst"               # generated in place, inside deep_learning/
PAGES_DIR = HERE                        # NN-*.rst lesson pages are siblings of this script
N_LESSONS = 17

# ---- stage registry: key -> (emoji, title, level, blurb) -------------
STAGES: dict[str, tuple[str, str, str, str]] = {
    "intro": ("\U0001F9E0", "Introduction to Deep Learning", "beginner",
              "What a neural network is, where supervised deep learning applies, and why it took off."),
    "neuron": ("\U0001F535", "Logistic Regression as a Neuron", "beginner",
               "A single sigmoid unit \u2014 the binary-classification model, its loss and its cost function."),
    "calculus": ("\U0001F4C9", "Derivatives & the Computation Graph", "intermediate",
                 "The calculus that trains a neuron: gradients, the computation graph and the chain rule."),
    "training": ("\u2699\uFE0F", "Backprop & Vectorization", "intermediate",
                 "Backpropagation across a whole training set, then vectorising it for speed with numpy."),
}
STAGE_ORDER = ["intro", "neuron", "calculus", "training"]

# ---- one-line summary per lesson (used on the index cards) ------------
GLOSS: dict[str, str] = {
    "What is a Neural Network?":
        "How stacked neurons map inputs to outputs, learning the mapping from examples by gradient descent.",
    "Supervised Learning and Neural Networks":
        "Learning input\u2192output mappings from labelled data, and which network suits structured, image or sequence inputs.",
    "Why Deep Learning is Taking Off":
        "The three drivers \u2014 data scale, compute, and algorithms \u2014 that made deep networks practical.",
    "Geoffrey Hinton Interview":
        "Perspectives from a pioneer of backpropagation on how the field's key ideas emerged.",
    "Binary Classification and Logistic Regression (Neural Network Basics)":
        "Framing a yes/no prediction, the notation for one training example, and stacking data into matrices.",
    "Logistic Regression (Binary Classification Model)":
        "A single neuron: a linear score passed through the sigmoid to output a probability.",
    "Logistic Regression \u2013 Loss Function and Cost Function":
        "The per-example cross-entropy loss and the averaged cost that training minimises.",
    "Gradient Descent in Logistic Regression":
        "Iteratively stepping the parameters downhill along the cost gradient toward a minimum.",
    "Derivatives":
        "The slope of a function \u2014 the building block of every gradient used to train a network.",
    "More Derivative Examples":
        "Worked derivatives that build intuition for how small input changes move the output.",
    "Computation Graph":
        "Breaking a computation into elementary steps \u2014 the structure backpropagation runs on.",
    "Derivatives with a Computation Graph":
        "Applying the chain rule backward through the graph to get every partial derivative efficiently.",
    "Logistic Regression Gradient Descent":
        "Backpropagation for one example: the derivatives of the loss w.r.t. each weight and the bias.",
    "Gradient Descent on m Training Examples":
        "Averaging the single-example gradients over the whole training set for one update step.",
    "Vectorization in Logistic Regression":
        "Replacing explicit loops with matrix operations so the whole batch computes at once.",
    "More Vectorization Examples":
        "Further numpy patterns \u2014 broadcasting and whole-array functions \u2014 that remove Python loops.",
    "Vectorizing Logistic Regression":
        "A fully vectorised forward and backward pass over all examples \u2014 the gateway to deep networks.",
}

BAR = "=" * 70
SUB = "-" * 70


def slugify(text: str) -> str:
    """Lowercase, hyphenated, ASCII slug (stable; used for filenames + anchors)."""
    t = text.replace("\u2013", "-").replace("\u2014", "-")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^A-Za-z0-9]+", "-", t).strip("-").lower()
    return t


def load_inventory() -> list[tuple[str, str, str]]:
    """Return [(title, url, stage), ...] in curriculum (file) order."""
    out: list[tuple[str, str, str]] = []
    with INVENTORY.open(encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)  # skip 'title  url  stage'
        for row in rd:
            if not row or not row[0].strip():
                continue
            title, url, stage = row[0].strip(), row[1].strip(), row[2].strip()
            out.append((title, url, stage))
    return out


def title_bar(title: str) -> str:
    """An overline/underline rule guaranteed >= the title length."""
    return "=" * max(72, len(title) + 2)


def main() -> int:
    inv = load_inventory()
    titles = [t for t, _u, _s in inv]
    title_set = set(titles)

    # ---- fail-fast: inventory <-> GLOSS <-> STAGES -------------------
    if len(inv) != N_LESSONS:
        print(f"ERROR: inventory has {len(inv)} lessons, expected {N_LESSONS}", file=sys.stderr)
        return 1
    miss_gloss = [t for t in titles if t not in GLOSS]
    extra_gloss = [t for t in GLOSS if t not in title_set]
    if miss_gloss or extra_gloss:
        if miss_gloss:
            print("ERROR: inventory titles with no GLOSS entry:", file=sys.stderr)
            for t in miss_gloss:
                print("   -", repr(t), file=sys.stderr)
        if extra_gloss:
            print("ERROR: GLOSS entries not in inventory:", file=sys.stderr)
            for t in extra_gloss:
                print("   -", repr(t), file=sys.stderr)
        return 1
    bad_stage = [(t, s) for t, _u, s in inv if s not in STAGES]
    if bad_stage:
        print("ERROR: inventory stages not in STAGES:", file=sys.stderr)
        for t, s in bad_stage:
            print(f"   - {s!r} ({t!r})", file=sys.stderr)
        return 1

    # ---- fail-fast: CONTENT / MINDMAP keys are exact inventory titles -
    try:
        from dl_content import CONTENT  # type: ignore
    except Exception:
        CONTENT = {}
    try:
        from dl_content import MINDMAP  # type: ignore
    except Exception:
        MINDMAP = {}
    bad_content = [k for k in CONTENT if k not in title_set]
    if bad_content:
        print("ERROR: CONTENT keys not in inventory:", file=sys.stderr)
        for k in bad_content:
            print("   -", repr(k), file=sys.stderr)
        return 1
    bad_mm = [k for k in MINDMAP if k not in title_set]
    bad_nb = sorted({n for k in MINDMAP for n in MINDMAP[k] if n not in title_set})
    if bad_mm or bad_nb:
        if bad_mm:
            print("ERROR: MINDMAP keys not in inventory:", file=sys.stderr)
            for k in bad_mm:
                print("   -", repr(k), file=sys.stderr)
        if bad_nb:
            print("ERROR: MINDMAP neighbours not in inventory:", file=sys.stderr)
            for n in bad_nb:
                print("   -", repr(n), file=sys.stderr)
        return 1

    # ---- derived maps -----------------------------------------------
    idx = {t: i + 1 for i, t in enumerate(titles)}          # 1-based lesson number
    slug = {t: slugify(t) for t in titles}
    docname = {t: f"{idx[t]:02d}-{slug[t]}" for t in titles}
    anchor = {t: f"dl-{slug[t]}" for t in titles}

    # ---- clear stale lesson pages (idempotent set) ------------------
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    for old in PAGES_DIR.glob("[0-9][0-9]-*.rst"):
        old.unlink()

    # ---- emit one page per lesson -----------------------------------
    n_rich = 0
    for pos, (title, url, stage) in enumerate(inv):
        i = pos + 1
        emoji, stage_title, level, _blurb = STAGES[stage]
        sn = STAGE_ORDER.index(stage) + 1
        L: list[str] = []
        a = L.append

        a(f".. _{anchor[title]}:")
        a("")
        bar = title_bar(title)
        a(bar)
        a(title)
        a(bar)
        a("")
        a(f"**Stage {sn} \u00b7 {emoji} {stage_title}**  \u00b7  "
          f"Lesson {i:02d} of {N_LESSONS}  \u00b7  *{level}*")
        a("")

        # prev / next navigation
        nav_parts: list[str] = []
        if i > 1:
            p = inv[pos - 1][0]
            nav_parts.append(f":doc:`\u25c0 Previous \u00b7 {p} <{docname[p]}>`")
        if i < N_LESSONS:
            nx = inv[pos + 1][0]
            nav_parts.append(f":doc:`Next \u00b7 {nx} \u25b6 <{docname[nx]}>`")
        nav_parts.append(":doc:`\u2191 Section <index>`")
        if nav_parts:
            a("   \u00b7   ".join(nav_parts))
            a("")

        a("")

        a(AI_NOTICE); a("")

        # body: full content if written, else a working stub
        if title in CONTENT:
            n_rich += 1
            body = CONTENT[title].strip("\n")
            a(body)
        else:
            a(GLOSS[title])
            a("")
            a(".. admonition:: Lesson in progress")
            a("   :class: note")
            a("")
            a("   The full write-up for this lesson is being prepared. Until then, use the")
            a("   one-line summary above and follow the navigation to adjacent lessons.")
        a("")

        # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#admonitions-messages-and-warnings
        # Note      → "Be aware of this clarification or detail."          # 📝 Neutral observations, assumptions, clarifications, conventions, or exceptions.
        # See also  → "Explore these related topics and resources."        # 📚 Internal/external references, further reading, related topics, prerequisites.
        # Hint      → "This may help you understand the concept."          # 💡 Intuition, conceptual connections, mind maps, learning aids.
        # Tip       → "This may help you work more effectively."           # 💡 Best practices, shortcuts, recommendations, efficient/advice workflows.
        # Info      → "Here's additional background or context."           # ℹ️ Background, implementation notes, **sources used by this page**, supplementary factual information where the information came from.
        # Important → "Do not overlook this; it's essential."              # ⭐ Critical/Essential concepts, requirements, or limitations.

        # lateral cross-links
        if MINDMAP.get(title):
            a(".. hint::")
            a("")
            links = "  \u00b7  ".join(f":doc:`{n} <{docname[n]}>`" for n in MINDMAP[title])
            a(f"   **Related lessons:** {links}")
            a("")

        # source (context/traceability)
        a(".. seealso::")
        a("")
        a(f"   **Source article** Adapted (context, re-expressed) in our own words from: `{url} <{url}>`__ "
          f"(insightful-data-lab.com).")
        a("")

        # tags
        a(f".. tags:: purpose: reference, topic: deep learning, level: {level}")
        a("")

        (PAGES_DIR / f"{docname[title]}.rst").write_text("\n".join(L), encoding="utf-8")

    # ---- emit the staged learning-path index ------------------------
    by_stage: dict[str, list[str]] = {k: [] for k in STAGE_ORDER}
    for t, _u, s in inv:
        by_stage[s].append(t)

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
    w(".. _deep-learning-index:")
    w("")
    w(":raw-html:`<div style=\"text-align:center\"><strong>` \U0001F9E0 Deep Learning")
    w("|br| From a single neuron to deep networks")
    w("|br| |full_version| - |today|")
    w(":raw-html:`</strong></div>`")
    w("")
    w(BAR)
    w("Deep Learning")
    w(BAR)
    w("")
    w("**Deep learning** stacks simple, differentiable units into networks trained end-to-end by")
    w("**gradient descent**. This course follows the ground-up path the source corpus takes \u2014 view")
    w("**logistic regression as a single neuron**, learn the **computation graph** and")
    w("**backpropagation** that train it, then **vectorise** it into the efficient building block that")
    w("scales to deep networks \u2014 as an ordered, self-contained course of 17 lessons.")
    w("")
    w("Read it at any depth:")
    w("")
    w("* **newcomers** \u2014 what a neural network is, and a single neuron;")
    w("* **practitioners** \u2014 the sigmoid model, the cross-entropy cost, and gradient descent;")
    w("* **researchers / engineers** \u2014 the computation graph, backpropagation, and full vectorisation with ``numpy``.")
    w("")
    w(".. warning::")
    w("")
    w("   These lessons build the mathematics **from scratch** in ``numpy`` to show what a framework")
    w("   does under the hood. In production you would use a framework (**PyTorch** / **Keras**) \u2014 its")
    w("   automatic differentiation computes exactly these gradients for you.")
    w("")
    w(".. note::")
    w("")
    w("   Follow the lessons in order with **Next \u25b6**, or jump in by stage below. Snippets use")
    w("   real ``numpy`` / ``PyTorch`` / ``Keras`` calls and connect to scikit-plots' ``visualkeras``")
    w("   architecture views. This course pairs with the")
    w("   :ref:`Terminology reference <terminology-index>` (classification metrics for evaluating networks).")
    w("")
    w(SUB.replace("-", "="))   # a visual transition rule (== line, blank-separated)
    w("")

    # ---- live filter: type-to-search across every term (progressive JS) ----
    # Static, dependency-free, deterministic. Without JS the page degrades
    # gracefully to plain collapsible dropdowns.
    # ---- v2 hub: live filter + collapsed stage dropdowns + A-Z --------
    # Same pattern/classes as learn/terminology (details.sd-dropdown, .term-az)
    n_items = len(titles)
    w(".. raw:: html")
    w("")
    w('   <div style="text-align:center;margin:0.4rem 0 0.4rem">')
    w('   <input id="term-filter" type="search" autocomplete="off" spellcheck="false"')
    w(f'          placeholder="&#128269;&nbsp; Type to filter {n_items} lessons &mdash; by title or keyword&hellip;"')
    w('          style="width:100%;max-width:100%;padding:0.55rem 1rem;font-size:1rem;')
    w('                 border:1px solid var(--pst-color-border,#ccc);border-radius:0.55rem;box-sizing:border-box;')  # 1px solid rgba(128,128,128,0.45)
    w('                 background:transparent;color:inherit"/>')
    w('   <div id="term-filter-count" style="opacity:0.65;font-size:0.85rem;')
    w('        min-height:1.2em;margin-top:0.35rem"></div>')
    w("   </div>")
    w("   <script>")
    w("   document.addEventListener('DOMContentLoaded',function(){")
    w("     var inp=document.getElementById('term-filter');if(!inp){return;}")
    w("     var dds=[].slice.call(document.querySelectorAll('details.sd-dropdown'));")
    w("     var az=document.querySelector('details.term-az');")
    w("     var items=[];")
    w("     dds.forEach(function(d){[].slice.call(d.querySelectorAll('li')).forEach(")
    w("       function(li){items.push({li:li,d:d,t:li.textContent.toLowerCase()});});});")
    w("     var cnt=document.getElementById('term-filter-count');")
    w("     inp.addEventListener('input',function(){")
    w("       var q=inp.value.trim().toLowerCase();var n=0;")
    w("       dds.forEach(function(d){d.tHits=0;});")
    w("       items.forEach(function(it){")
    w("         var hit=!q||it.t.indexOf(q)!==-1;")
    w("         it.li.style.display=hit?'':'none';")
    w("         if(hit){it.d.tHits+=1;if(az&&it.d===az){n+=1;}}});")
    w("       dds.forEach(function(d){")
    w("         if(q){d.style.display=d.tHits?'':'none';d.open=d.tHits>0;}")
    w("         else{d.style.display='';d.open=false;}});")
    w(f"       if(cnt){{cnt.textContent=(q&&az)?(n+' of {n_items} match'+(n===1?'':'s')):'';}}")
    w("     });")
    w("   });")
    w("   </script>")
    w("")

    for stage in STAGE_ORDER:
        emoji, stage_title, level, blurb = STAGES[stage]
        sn = STAGE_ORDER.index(stage) + 1
        lessons = by_stage[stage]
        # stable per-stage anchor (linkable from other hubs)
        w(f".. _dl-stage-{stage}:")
        w("")
        w(f".. dropdown:: {emoji} Stage {sn} \u2014 {stage_title}  \u00b7  {len(lessons)} lessons")
        w("   :animate: fade-in-slide-down")
        w("")
        w(f"   *{blurb}*  \u00b7  *{level}*")
        w("")
        for t in lessons:
            w(f"   * :doc:`{idx[t]:02d} \u00b7 {t} <{docname[t]}>` \u2014 {GLOSS[t]}")
        w("")

    # ---- dictionary view: one A-Z master list (auto-sorted) ----------
    az_head = "\U0001F524 Every lesson, A\u2013Z index"
    w(az_head)
    w("-" * (len(az_head) + 2))
    w("")
    w(".. dropdown:: \U0001F520 Open the full alphabetical index")
    w("   :animate: fade-in-slide-down")
    w("   :class-container: term-az")
    w("")
    w("   .. hlist::")
    w("      :columns: 2")
    w("")
    for t in sorted(titles, key=str.casefold):
        w(f"      * :doc:`{t} <{docname[t]}>`")
    w("")

    # hidden ordered toctree so Sphinx builds the sequence + sidebar nav
    w(".. toctree::")
    w("   :hidden:")
    w("   :maxdepth: 1")
    w("")
    for t in titles:
        w(f"   {docname[t]}")
    w("")
    w(".. tags:: purpose: reference, topic: deep learning")
    w("")

    OUT.write_text("\n".join(I), encoding="utf-8")

    print(f"Wrote index.rst + {N_LESSONS} lesson pages "
          f"({n_rich} with full content, {N_LESSONS - n_rich} stub) "
          f"across {len(STAGE_ORDER)} stages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
