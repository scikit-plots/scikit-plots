# BilimEdtech Labs mirror — course guide (Mode B, verbatim CC BY 4.0)

A **verbatim mirror** of the BilimEdtech Labs site
(`https://labs.bilimedtech.com`, © 2022 BilimEdtech Labs) reproduced under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license into
`learn/hands-on/edtech/`. This is a **Mode B** build in the sense of
`NEW_COURSE_PLAYBOOK.md` §7: the body of every page is reproduced *without
modification* from the source's reStructuredText; the only additions are a
per-page attribution notice and (for pages not yet fetched) a clearly-labelled
placeholder stub. No content is rewritten, reordered, or "improved".

> **The deal (CC BY 4.0).** You may copy, redistribute, and adapt the material,
> including commercially, provided you (a) give appropriate credit, (b) link to
> the license, and (c) indicate if changes were made. This mirror credits the
> author, links the license on every page, ships the full legal text
> (`LICENSE_edtech`), and states plainly that pages are reproduced *without
> modification*. See `LICENSE_edtech.rst` and
> https://creativecommons.org/licenses/by/4.0/deed.en.

## What is in this folder

```
learn/hands-on/edtech/
  build_edtech.py         # generator: verbatim body + attribution + auto-stubs
  edtech_manifest.py      # HAND INPUT: VERBATIM set, DROP_SEARCH, STUB_TITLES
  _sources_cache/         # HAND INPUT: raw upstream RST, byte-for-byte as fetched
    index.rst.txt
    LICENSE.txt           # CC BY 4.0 legalcode.txt, verbatim
    cloud-computing/index.rst.txt
    nasm/index.rst.txt
    operating-systems/index.rst.txt
    workshops/index.rst.txt
  _buildcheck.py          # drop-in validator (identical across every learn/ course)
  EDTECH.md               # this guide
  LICENSE_edtech.rst      # GENERATED: title + scope header + full legal text
  index.rst               # GENERATED: verbatim site index (Search section dropped)
  <section>/index.rst     # GENERATED: verbatim section indexes / stubs
  <section>/<lab>/...      # GENERATED: verbatim leaf pages / stubs
```

**Hand inputs** are `edtech_manifest.py` and everything under `_sources_cache/`.
Everything else (`*.rst`) is **generated** and owned by `build_edtech.py` — the
generator deletes and rewrites all `*.rst` on every run, so never hand-edit a
generated page; edit the manifest and/or the cache instead.

## Upstream structure and fetch-map

The source is a Sphinx / Read-the-Docs site. Its root `toctree` has five
sections; each section index has its own `toctree` down to leaf pages. The map
below is the frozen structure this mirror targets (harvested by walking the
toctrees — see "Fetch strategy"). Page URLs are `…/<doc path>.html`; sources are
`…/_sources/<doc path>.rst.txt`.

| Section | Leaf pages (upstream doc paths under the section) |
| --- | --- |
| `index` (root) | site index → the five section indexes |
| `cloud-computing` | `index`; `1/{index,overview,1.1..1.9}`; `2/{index,overview,2.1..2.4}`; `3/{index,overview,docker-compose,3.1..3.6}`; `4/{index,overview,docker-terms,4.1..4.3}`; `5/{index,overview,dockerfile,5.1..5.3}`; `6/{index,overview,6.1..6.4}`; `8/{index,overview,8.1..8.5}`; `references/{index,nginx,security,wordpress,sphinx-readthedocs,dockerfile,paas}` |
| `nasm` | `index`; `1..6/{index + steps}`; `resources/{index,templates,resources}`; `windows-install/{index,overview,1,2,3,4}` |
| `c` | `index` (+ include `c-urls.rst`); `1..6/{index + steps}`; `templates/index`; `references/{index,c-programming-resources}`; `windows-install/index` |
| `operating-systems` | `index`; `overview`; `1..8/{index + steps}`; `references/{index,system-calls-resources,misc}` |
| `workshops` | `index`; `rst/{index,dev-env,writing-rst-setup,vps-config,writing-rst-1..8}` |

Full site ≈ 190–240 pages. The lab-step lists for `nasm/c/operating-systems`
labs `1..N` are discovered by fetching each lab `index` (see below); the table
records what the top-level and section indexes reveal.

## How the generator works

`build_edtech.py` is a pure, deterministic, idempotent function of its inputs.
Each run:

1. **Clears** every generated `*.rst` under the folder (the generator owns them;
   `_sources_cache/*.rst.txt` are `.txt` and are never touched).
2. **Emits `LICENSE_edtech.rst`** — an `:orphan:` page with a 70-rule title, a
   scope header, and the full CC BY 4.0 legal text placed in a **literal block**
   (4-space indented) so its `====` rule lines are rendered as preformatted text,
   never parsed as RST section underlines.
3. **Emits each `VERBATIM` page**: reads `_sources_cache/<path>.rst.txt`,
   optionally strips a trailing `Search` section (only for paths in
   `DROP_SEARCH` — currently just the root `index`, since Sphinx provides search
   itself — the one edit the plan permits), writes the body **byte-for-byte**,
   then appends the **Source & license** admonition (below).
4. **Auto-stubs the rest**: collects every `toctree` target across the emitted
   pages; any target not in `VERBATIM` is written as a placeholder stub (title
   from `STUB_TITLES` or a humanized path segment, a "verbatim mirror pending"
   note, and the same license link). Stubs carry **no `toctree`**, so they add no
   further unresolved targets — the tree is always buildable, and the full
   upstream structure is visible while pages are filled in batches.
5. **Writes placeholder images** for every entry in `PLACEHOLDER_IMAGES` (a
   deterministic 640×360 labelled PNG), then **prunes empty directories** left
   behind when a stub is removed (e.g. after a section is fully mirrored).

Both the `toctree` scan (step 4) and the validator's role scan skip literal /
code-block bodies via the shared `_mask_code_blocks` rule, so RST *about* RST
mirrors correctly — example directives in code blocks never become real structure.

Determinism: inputs are sorted, there is no RNG and no timestamp, so two runs are
byte-identical. `_buildcheck.py` proves this every run (`idem OK <sha12>`).

### The Source & license admonition (every mirrored page)

Appended **after** the verbatim body (not before — see landmine *Title
promotion*). CC BY 4.0 requires author credit, a license link, and a change
notice; the block supplies all three plus the page's own source URL:

```rst
.. admonition:: Source & license
   :class: note

   Reproduced **verbatim, without modification** from
   `© 2022, BilimEdtech Labs <https://labs.bilimedtech.com/index.html>`__,
   licensed under
   `Creative Commons Attribution 4.0 International (CC BY 4.0) <https://creativecommons.org/licenses/by/4.0/deed.en>`__.

   Source page:
   https://labs.bilimedtech.com/<doc path>.html

   See :doc:`LICENSE <<relative>/LICENSE_edtech>` for the full license text.
```

The `:doc:` link to `LICENSE_edtech` is **depth-correct per page**
(`lic_rel()` emits `../` × the path depth), so it resolves whether the tree is
validated from `edtech/`, from `hands-on/`, or by the real Sphinx build — all use
relative `:doc:` resolution. Because the mirror is verbatim, the notice states
"without modification"; **if you ever adapt a page, change that wording to record
the change** (CC BY change-notice requirement).

### Extending / editing later (the whole point of the generator)

- **Add a page (Mode B fetch):** fetch its source RST (see below), drop it at
  `_sources_cache/<path>.rst.txt`, add `<path>` to `VERBATIM` in the manifest,
  rerun `_buildcheck.py`. The matching stub is automatically replaced by the
  verbatim page. If the new page's `toctree` references further pages, those
  become new stubs until fetched.
- **Add a whole lab atomically:** fetch the lab `index` **and all its step
  pages** in one batch, add them all to `VERBATIM`, rebuild. (Do not add a lab
  `index` with a step `toctree` without its steps, or the real Sphinx build will
  warn about missing toctree docs — the stub `index` has no toctree precisely to
  avoid this.)
- **Edit a page:** re-fetch it into `_sources_cache/` and rebuild; the idempotency
  check confirms nothing else moved.
- **Re-mirror:** re-fetch only changed pages. The frozen manifest + cache make it
  reproducible; update the change notice if the upstream page changed.

## Fetch strategy (proven)

`web_fetch` only accepts URLs already surfaced in the conversation; **constructed
`_sources/…` URLs are rejected**. So the crawl follows links down the toctree:

1. Fetch a page's rendered **`.html`** (root `index.html` is the entry point).
   The RTD theme exposes real `href` links to every child page **and** a
   **"View page source"** link to that page's `_sources/<path>.rst.txt`.
2. Fetch that **`_sources/<path>.rst.txt`** (now allowed, because the link
   appeared in the HTML result) to get the exact reStructuredText.
3. Parse its `.. toctree::` blocks for child doc paths, fetch each child's
   `.html`, recurse to the leaves.

### Bulk crawl — `fetch_edtech.py` (fast path, run locally)

The link-following crawl above is what the in-notebook fetcher is limited to. From
**any machine with ordinary network access to the site**, `fetch_edtech.py`
does the whole thing in one pass — the constructed-URL restriction is only the
notebook fetcher's, not the site's. The script (stdlib only, no `pip` installs):

- walks every `.. toctree::` from the root to discover all reachable pages
  (orphan pages outside the nav are skipped — matching the mirror's scope);
- writes each page's source to `_sources_cache/<path>.rst.txt`;
- resolves and fetches every `.. include::` fragment (`urls.rst`, `*-content.rst`)
  to `_sources_cache/<frag>.txt`;
- downloads every referenced image (`.. image::` / `.. figure::` /
  `.. |x| image::`) from `_images/<basename>` to `_sources_cache/_images/`;
- writes `_sources_cache/_discovered.json` (`{pages, fragments, images}`).

It reuses the generator's exact code-block/inline-literal **masking**, so example
`.. toctree::` / `.. include::` blocks inside tutorial code are not mistaken for
real structure. It is **idempotent** (skips cached files unless `--force`),
**fails safe** (a page/fragment/image that won't fetch is listed in a failure
report and skipped — those stay stubs to hand-mirror), and polite (a short delay
and a descriptive User-Agent).

```
python3 fetch_edtech.py            # crawl + fetch everything new
python3 fetch_edtech.py --dry-run  # discover + report only, fetch nothing
python3 fetch_edtech.py --force    # refetch even cached files
python3 _buildcheck.py             # then regenerate + validate the full mirror
```

`edtech_manifest.py` folds `_discovered.json` into `VERBATIM` / `INCLUDES` /
`PLACEHOLDER_IMAGES` at import (`_apply_discovered()`), and `write_placeholder`
copies a real downloaded image from `_sources_cache/_images/` when present,
falling back to a labelled placeholder otherwise. So the **same** generator
produces either the hand-built subset (no crawl) or the complete mirror (after a
crawl) with no other change — and real screenshots replace the placeholders
automatically. If the script can't reach the site or a run fails, nothing is lost:
the hand-built path in this guide still applies.

### Fetch landmines (seen in practice)

- **Windowed allow-list.** Surfaced URLs seem to age out after enough
  intervening fetches/searches. If a `_sources/...rst.txt` fetch is rejected even
  though you saw the page, **re-fetch that page's `.html` and immediately fetch
  the source** (adjacent calls), or `web_search` the page title to re-surface it.
- **CC legal text is off-domain.** `creativecommons.org` is not pre-approved for
  direct fetch. `web_search` for "creativecommons.org licenses by 4.0
  legalcode.txt", then fetch the surfaced `…/legalcode.txt` result. (Done — cached
  at `_sources_cache/LICENSE.txt`.)
- **`.. include:: <name>-urls.rst` fragments — a hard blocker for lab step
  pages.** Many lab pages begin with `.. include:: 1-urls.rst` (Lab 1),
  `c-urls.rst` (`c/index`), etc. — a per-lab fragment that defines the
  ``|Name|`` raw-HTML link substitutions the page body uses. These fragments have
  **no HTML page of their own**, so Sphinx never publishes them to `_sources`, and
  there is no "View page source" link to surface — they **cannot be fetched** by
  the link-following strategy, and no public source repo was found. Because the
  fragment content is not in any fetchable source, reproducing these pages verbatim
  is blocked, and the fragment must not be *invented* (that would violate the
  verbatim / no-inference rule). Options when you hit one: **(a)** obtain the
  fragment from the upstream source repo if you can get it, then mirror normally;
  **(b)** reconstruct the fragment from the rendered link targets on the pages that
  use it (an explicit, documented adaptation — every ``|Name|`` becomes
  ``.. |Name| raw:: html`` / ``<a href="…" target="_blank">Name</a>`` — flagged as
  a CC BY change notice), which also needs the generator to inline it or a
  `conf.py` `exclude_patterns` entry so the fragment file does not orphan-warn;
  **(c)** keep the include-dependent pages stubbed and mirror only the
  self-contained pages. **Self-contained pages have no such include** — section and
  lab **index** pages, lab **overview** pages (they carry their own inline
  ``.. |…| raw:: html`` defs), and **most `references/`** pages — so those mirror
  cleanly today. Until a decision is made on (a)/(b)/(c), the lab **step** pages
  (`1/1.1..1.9`, etc.) stay stubbed.
- **A second include variant: `.. include:: <name>-content.rst`.** A few pages put
  their *entire body* in an unfetchable content fragment — e.g.
  `references/sphinx-readthedocs` includes `sphinx-readthedocs-content.rst` and
  `references/dockerfile` includes `dockerfile-content.rst`. Same root problem as
  the `-urls.rst` fragments (no HTML page → not in `_sources` → unfetchable), but
  here the fragment *is* the whole page. These stay stubbed under the same options.
  Always fetch a page's `.rst.txt` and check for `.. include::` before assuming it
  is mirrorable — rendered HTML hides includes entirely.
- **A third variant: `.. include:: urls.rst` (shared, unprefixed).** Some lab
  **overview** pages pull a site-shared `urls.rst` fragment for their link
  substitutions (e.g. Lab 2 and Lab 5 overviews use ``|Rocket.Chat|`` / ``|Flask|``
  from it), while others define their substitutions **inline** and are clean
  (Lab 1 and Lab 4 overviews). So overviews are **mixed** — each must be checked
  individually; there is no way to tell from the rendered page. Same unfetchable-
  fragment problem; the `urls.rst`-dependent overviews stay stubbed under the
  options above. Overviews that use it may also carry a real screenshot (e.g. Lab
  2's `reverse-proxy.png`) needing a placeholder.
- **Cross-section `:ref:` labels.** Some pages reference labels defined on other
  pages. Fetch a lab's pages as a unit so intra-lab refs resolve; a
  cross-*section* ref (like the C one above) needs both endpoints present.

## Canonical URL / duplicate-content (SEO) — the one hook outside RST

Verbatim reproduction risks a search-engine duplicate-content penalty. The fix is
to point each mirrored page's **canonical URL at the upstream source page** so
engines credit the original (the upstream pages already declare
`canonical: https://labs.bilimedtech.com/…`). Sphinx sets canonical at the
`conf.py` / theme level, **not** from page RST, so this is wired once in the
docs `conf.py` with an `html-page-context` handler that maps each edtech page to
its source URL. Drop-in for the scikit-plots `conf.py`:

```python
# conf.py — point edtech mirror pages' canonical link at the upstream source.
def _edtech_canonical(app, pagename, templatename, context, doctree):
    prefix = "learn/hands-on/edtech/"
    if pagename.startswith(prefix):
        rel = pagename[len(prefix):]            # e.g. "cloud-computing/index"
        context["pageurl"] = f"https://labs.bilimedtech.com/{rel}.html"

def setup(app):
    app.connect("html-page-context", _edtech_canonical)
```

(The PyData / RTD themes emit `<link rel="canonical" href="{{ pageurl }}">` in
`<head>` when `pageurl` is set. Confirm the exact context key for the active
theme; `pageurl` is standard.) The `LICENSE_edtech` page is our own artifact —
leave its canonical at the local build.

## RST & validation landmines specific to this mirror

- **Title promotion → notice goes at the bottom.** docutils only promotes a
  page's first heading to the document title if it is the *first* structural
  element. Prepending the attribution admonition would demote the title and
  change every toctree caption, so the notice is **appended after** the verbatim
  body instead.
- **License legal text in a literal block.** The CC text contains 71-char `====`
  rules; if placed as raw RST they parse as section under/overlines. Indenting
  every line 4 spaces inside a `::` literal block renders them as preformatted
  text, and the `_buildcheck.py` underline check skips indented lines, so no false
  positives.
- **Grid card `^^^` separator.** The `sphinx-design` card header uses `^^^`,
  which is shorter than the bold title above it and would trip the underline
  check — *unless* the card body is indented (its head line then starts with
  spaces and is skipped). Keep card content indented (8 spaces), as in the hub
  `hands-on/index.rst` and the site `learn/index.rst`.
- **Dropped `Search` section.** The source root index ends with a `Search`
  heading and `* :ref:`search``. `search` is a Sphinx built-in with no explicit
  label, so leaving it would trip the `:ref:` check; dropping the section (the
  one permitted edit) removes it cleanly.
- **Code-block example roles / directives (RST-workshop pages) — handled.** Pages
  that *teach* RST embed example roles like `:ref:`genindex`` and example
  directives like `.. toctree::` / `.. include::` inside literal/code blocks. Real
  Sphinx treats code-block contents as literal and ignores them. Both tools here
  now do the same via a shared `_mask_code_blocks` helper that blanks
  literal/code-block bodies (a `.. code-block::`/`code`/`sourcecode`/
  `parsed-literal`/`literalinclude` directive, or a line ending in `::` that is not
  itself a directive, followed by its blank-or-more-indented body) **and blanks
  inline-literal spans** (double backticks), so an example role written inline like
  ``` ``:ref:`Label``` ``` is also ignored. `_buildcheck.py` masks before the
  `:doc:`/`:ref:` scan, and `build_edtech.py` masks before the
  `toctree` scan (so an example `.. toctree::` inside a code block does **not**
  spawn spurious stub pages — this is exactly what created the phantom
  `web-guide/index` and `. . .` stubs before the fix). The masker only *skips*
  checks/parsing, so it cannot regress existing courses (all remained green with
  identical shas). Real `.. automodule::` (Task 9) still must never appear as a
  live directive — reproduce it only inside a code block; and `.. image::`
  targets need a real file present (see `PLACEHOLDER_IMAGES`).

## Validation & rebuilding

Run from inside the folder:

```bash
cd learn/hands-on/edtech
python3 _buildcheck.py          # build + validate + idempotency
```

Green run:

```
[edtech] edtech mirror: 5 verbatim, 28 stub, 34 rst (incl LICENSE)
validate: doc 0 | ref 0 | underline 0  ||  idem OK <sha12>
```

`_buildcheck.py` runs `build_edtech.py`, then validates every `:doc:` target
(resolved per file directory), every `:ref:` target, and every section underline,
then rebuilds once and compares a sha256 of all `.rst` for byte-identical
idempotency. The `hands-on/` parent carries the same drop-in in **validate-only**
mode (no generator there): `cd learn/hands-on && python3 _buildcheck.py` checks
the hub page plus the whole edtech tree.

### Current mirror status

- **Workshops section: COMPLETE.** All 16 RST-workshop pages are verbatim —
  `workshops/index`, `workshops/rst/index`, the four support pages, the two guide
  pages (`writing-rst-overview`, `setting-up-sphinx`), and all nine Tasks
  (`writing-rst-1..9`).
- **COMPLETE — 134 verbatim pages, 13 include-fragments, 78 real images, 0
  stubs.** Populated in one pass by `fetch_edtech.py` (see the Bulk crawl section);
  `_sources_cache/_discovered.json` drives `VERBATIM` / `INCLUDES` /
  `PLACEHOLDER_IMAGES` via `edtech_manifest._apply_discovered()`. Every section is
  mirrored end to end: cloud-computing (all 8 labs + references), NASM, C, operating
  systems, and the RST workshop. All `.. include::` fragments (`urls.rst`,
  `N-urls.rst`, `*-content.rst`) are fetched and resolved in place; every downloaded
  screenshot replaces its placeholder (0 placeholders remain). Validation:
  `doc 0 | ref 0 | underline 0`, byte-identical idempotent rebuild.
- **The site-wide prolog (`/includes/prolog.inc`).** Upstream pages carry
  `.. include:: /includes/prolog.inc` — a leading-slash, Sphinx-source-root path
  that has no `_sources` entry and, in scikit-plots, would resolve into *its own*
  `/includes/` tree. The crawler skips it (`find_includes` drops leading-slash
  targets) and the overlay filters it; the generator's `rewrite_prolog` repoints
  the directive at a **mirror-local** `includes/prolog.inc` (emitted from cache),
  which reconstructs only the one substitution it provides — the standard
  ``|br|`` → ``<br />``. Documented CC BY change notice. (The `|Memory leak|` /
  `|linked list|` refs on `c/5/index` are upstream capitalization typos vs. the
  `|memory leak|` / `|Linked List|` defs in `c-urls.rst`; reproduced verbatim.)
- **Validator anchor collection** now matches Sphinx: explicit labels are collected
  when **indented** (e.g. `.. _simple-wp-in-using-docker-compose:` inside a list)
  and when they contain **spaces** (e.g. `.. _Step 5.3:`, referenced as
  ``:ref:`text <Step 5.3>```). The pattern `^\s*\.\. _([^:\n]+):\s*$` still
  excludes ``.. _name: url`` external-link targets. Propagated to all 11 drop-in
  copies; the six existing courses stay byte-identical (idempotency shas unchanged).
- **conf.py requirement (fragments + prolog).** Sphinx's "document isn't included
  in any toctree" warning fires only for files it treats as *documents* — i.e.
  those whose extension is in `source_suffix` (default `.rst`). Two consequences
  for this mirror:

  1. The emitted include-fragments are real `.rst` (`urls.rst`, `*-urls.rst`,
     `*-content.rst`) and so **are** documents — `.. include::` does not exempt
     them (inclusion is parse-time text substitution; the file is still discovered
     independently). They must be excluded or they orphan-warn.
  2. Everything under `_sources_cache/` is `.txt` / `.png` / `.json` (never `.rst`)
     and `includes/prolog.inc` is `.inc`, so Sphinx ignores them **by extension**
     already. But the cache is build *input*, not publishable content, and its
     safety hinges on `source_suffix` staying `.rst` — add `.txt` there and every
     `*.rst.txt` would flip into a (broken) document. So exclude the cache outright
     rather than rely on the extension.

  ::

      exclude_patterns += [
          "learn/hands-on/edtech/_sources_cache/**",   # raw source cache — build input, never a doc
          "learn/hands-on/edtech/**/urls.rst",
          "learn/hands-on/edtech/**/*-urls.rst",
          "learn/hands-on/edtech/**/*-content.rst",
          "learn/hands-on/edtech/**/includes/*.inc",
          # + "learn/hands-on/edtech/**/*.md"  only if you build with MyST
      ]

  This mirrors upstream's own build config (upstream excludes the same fragments).
  **If you build with MyST** (i.e. `.md` is in `source_suffix`), also exclude the
  guide files, which would otherwise orphan-warn as documents::

      "learn/hands-on/edtech/**/*.md",   # EDTECH.md and friends — docs about the mirror, not pages
- **Generator note (toctree comments).** A lab index toctree may contain a
  commented-out entry (e.g. Lab 4's ``..   4.4``). `toctree_targets` skips lines
  whose stripped form starts with `..`, so commented entries don't spawn spurious
  stub pages (same class of fix as the code-block masking).
- **cloud-computing progress & blocker:** the self-contained pages mirror cleanly
  (section/lab indexes, lab overviews, `references/` pages). The lab **step** pages
  (`1/1.1..1.9`, and every other lab's numbered steps) begin with
  `.. include:: <lab>-urls.rst` — an unfetchable fragment (see the landmine) — so
  they stay stubbed pending a decision on how to handle those includes. Step pages
  also carry screenshots that will each need a placeholder image.
- **cloud-computing scope note:** the main `cloud-computing/index` toctree covers
  Labs 1-6, 8 and `references/` only. The upstream also has `6-open-vpn/` and
  `7-scripting/` folders that are **not** in that toctree (orphan-ish upstream);
  they are outside the mirror's structure. This is why `writing-rst-8`'s
  ``:ref:`Step 1. Initialize OpenVPN using Docker``` (target lives in `6-open-vpn/`)
  stays a documented pending reference rather than resolving via the main nav.
- **Placeholder assets (3):** `workshops/rst/images/default-sphinx-page.png`,
  `rtd-sphinx-page.png`, and `the_great_sphinx_david_roberts.jpg` (a JPEG — the
  generator matches the placeholder format to the file extension). These are the
  upstream screenshots the fetch pipeline cannot retrieve; `PLACEHOLDER_IMAGES`
  records the upstream URLs. **CC BY change notice:** these three image files are
  substitutes; all page *text* is verbatim.
- **One pending cross-section reference:** `writing-rst-8` (Tables) links to the
  cloud-computing OpenVPN lab via ``:ref:`Step 1. Initialize OpenVPN using
  Docker```. That label lives on a not-yet-mirrored cloud-computing page, so the
  reference is validator-green (space-form, so the `:ref:` check skips it) but will
  emit one "undefined label" warning under a real Sphinx build until that page is
  mirrored. It resolves automatically when cloud-computing's OpenVPN lab lands.
- **Stubs (27):** `c/index` (blocked on its `c-urls.rst` include); every lab /
  reference / overview page under `cloud-computing`, `nasm`, `operating-systems`.
- **Remaining work:** each lab's `index` + steps (and `c/index` with its include)
  per the batch procedure above; add to `VERBATIM`; rebuild.

## Delivery / sync (nested folder note)

This course lives at `learn/hands-on/edtech/`, one level deeper than the flat
`learn/<course>/` courses. The drop-in `_buildcheck.py --sync` mirrors a folder to
`/mnt/user-data/outputs/learn/<folder-name>`, which is correct only for
top-level courses; for this nested folder, **sync the whole `learn/` tree at once**
(copy `learn/` into `/mnt/user-data/outputs/learn` and rezip
`scikit-plots-learn.zip`) rather than `edtech/ --sync`. The build/validate/
idempotency parts of `_buildcheck.py` are unaffected and correct here.

## Provenance

- Source: `https://labs.bilimedtech.com` — © 2022, BilimEdtech Labs; Sphinx +
  Read-the-Docs theme; site license CC BY 4.0 (declared in every page footer).
- License text: fetched from
  `https://creativecommons.org/licenses/by/4.0/legalcode.txt` (verbatim,
  `_sources_cache/LICENSE.txt`).
- Every page and structure detail was grounded by fetching the live site; nothing
  here is inferred. Lab-step page lists for `nasm/c/operating-systems` labs `1..N`
  are completed by fetching each lab index in later batches.
