"""
Hand-maintained inputs for the BilimEdtech Labs CC BY 4.0 mirror.

This is the single source of truth the generator (``build_edtech.py``) reads.
It never renders anything itself. To extend the mirror you only edit this file
and drop the matching raw source under ``_sources_cache/`` — see ``EDTECH.md``.

Convention
----------
A *mirrored doc path* is the Sphinx document name (POSIX, no ``.rst``) relative
to this folder, e.g. ``"cloud-computing/index"``. The raw upstream
reStructuredText for a verbatim page lives at
``_sources_cache/<doc path>.rst.txt`` and is reproduced **byte-for-byte** under
the page's attribution notice. Every upstream page URL is derived
deterministically as ``f"{SITE_BASE}/{doc path}.html"``.
"""

# --- Upstream identity / attribution -------------------------------------
SITE_BASE = "https://labs.bilimedtech.com"
ROOT_SOURCE = "https://labs.bilimedtech.com/index.html"
COPYRIGHT = "© 2022, BilimEdtech Labs"
LICENSE_DEED = "https://creativecommons.org/licenses/by/4.0/deed.en"
LICENSE_LEGAL = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"

# --- Verbatim pages ------------------------------------------------------
# Doc paths reproduced exactly from ``_sources_cache/<path>.rst.txt``.
# Grow this set (and add the cache file) as each page is fetched.
VERBATIM = {
    "index",
    "cloud-computing/index",
    "cloud-computing/1/index",
    "cloud-computing/1/overview",
    "cloud-computing/2/overview",
    "cloud-computing/2/2.1",
    "cloud-computing/2/2.2",
    "cloud-computing/2/2.3",
    "cloud-computing/2/2.4",
    "cloud-computing/4/overview",
    "cloud-computing/5/overview",
    "cloud-computing/5/dockerfile",
    "cloud-computing/5/5.1",
    "cloud-computing/5/5.2",
    "cloud-computing/5/5.3",
    "cloud-computing/2/index",
    "cloud-computing/3/index",
    "cloud-computing/4/index",
    "cloud-computing/5/index",
    "cloud-computing/6/index",
    "cloud-computing/8/index",
    "cloud-computing/4/overview",
    "cloud-computing/references/index",
    "cloud-computing/references/nginx",
    "cloud-computing/references/security",
    "cloud-computing/references/wordpress",
    "cloud-computing/references/paas",
    "cloud-computing/references/dockerfile",
    "nasm/index",
    "operating-systems/index",
    "workshops/index",
    "workshops/rst/index",
    "workshops/rst/presentation",
    "workshops/rst/dev-env",
    "workshops/rst/writing-rst-setup",
    "workshops/rst/vps-config",
    "workshops/rst/writing-rst-overview",
    "workshops/rst/setting-up-sphinx",
    "workshops/rst/writing-rst-1",
    "workshops/rst/writing-rst-2",
    "workshops/rst/writing-rst-3",
    "workshops/rst/writing-rst-4",
    "workshops/rst/writing-rst-5",
    "workshops/rst/writing-rst-6",
    "workshops/rst/writing-rst-7",
    "workshops/rst/writing-rst-8",
    "workshops/rst/writing-rst-9",
}

# The ONE allowed edit (per the plan): drop the trailing ``Search`` section
# from these pages, since Sphinx provides site search itself. Root index only.
DROP_SEARCH = {"index"}

# --- Stub display titles -------------------------------------------------
# Nice titles for pages referenced by a mirrored toctree but not yet fetched.
# Harvested verbatim from the upstream navigation so the in-progress tree reads
# correctly. Any stub path absent here falls back to a humanized path segment.
STUB_TITLES = {
    "c/index": "C Programming",

    "cloud-computing/1/index": "Lab 1: Set up your VPS",
    "cloud-computing/1/1.1": "Step 1: Interacting with your VPS",
    "cloud-computing/1/1.2": "Step 2: Update and reboot",
    "cloud-computing/1/1.3": "Step 3: Essential Configuration",
    "cloud-computing/1/1.4": "Step 4: Configure the firewall",
    "cloud-computing/1/1.5": "Step 5: Install Docker",
    "cloud-computing/1/1.6": "Step 6: Install Nginx",
    "cloud-computing/1/1.7": "Step 7: Install PHP",
    "cloud-computing/1/1.8": "Step 8: Create a Name-based Site",
    "cloud-computing/1/1.9": "Step 9: Install a Web Admin panel",

    "cloud-computing/2/overview": "Lab 2: Overview",
    "cloud-computing/2/2.1": "Step 1: Install a snap Application",
    "cloud-computing/2/2.2": "Step 2: Configure Nginx using the Command Line",
    "cloud-computing/2/2.3": "Step 3: Load a Simple Docker Project",
    "cloud-computing/2/2.4": "Step 4: Using Docker Compose",

    "cloud-computing/3/overview": "Lab 3: Overview",
    "cloud-computing/3/docker-compose": "Docker Compose",
    "cloud-computing/3/3.1": "Step 1: Clean Up",
    "cloud-computing/3/3.2": "Step 2: Configure Volumes",
    "cloud-computing/3/3.3": "Step 3: Persist Data Using Volumes",
    "cloud-computing/3/3.4": "Step 4: Add additional Services",
    "cloud-computing/3/3.5": "Step 5: Configure Wordpress to use Redis",
    "cloud-computing/3/3.6": "Step 6: Configure Nextcloud using Docker Compose",

    "cloud-computing/4/overview": "Lab 4: Overview",
    "cloud-computing/4/docker-terms": "Docker Terms Review",
    "cloud-computing/4/4.1": "Step 1: Building a Docker Image",
    "cloud-computing/4/4.2": "Step 2: Modifying a Docker Image",
    "cloud-computing/4/4.3": "Step 3: Extending a Docker Image",

    "cloud-computing/5/overview": "Lab 5: Overview",
    "cloud-computing/5/dockerfile": "Dockerfile",
    "cloud-computing/5/5.1": "Step 1: Create a Basic Dockerfile",
    "cloud-computing/5/5.2": "Step 2: Daemonize Docker",
    "cloud-computing/5/5.3": "Step 3: Building a Hello World Python Image",

    "cloud-computing/6/overview": "Lab 6: Overview",
    "cloud-computing/6/6.1": "Step 1: Create a Python Flask image for your development projects",
    "cloud-computing/6/6.2": "Step 2: Setup the docker-compose environment",
    "cloud-computing/6/6.3": "Step 3: The Development process",
    "cloud-computing/6/6.4": "Step 4: Developing Your Flask App",

    "cloud-computing/8/overview": "Lab 8: VPS Security",
    "cloud-computing/8/8.1": "Step 1: Enable Automatic Updates",
    "cloud-computing/8/8.2": "Step 2: Change the Default SSH Port",
    "cloud-computing/8/8.3": "Step 3: Change the root Password",
    "cloud-computing/8/8.4": "Step 4: Restrict root Login",
    "cloud-computing/8/8.5": "Step 5: Create another User",
    "cloud-computing/2/index": "Lab 2: Configure Nginx as a Reverse Proxy",
    "cloud-computing/3/index": "Lab 3: Docker Compose and Volumes",
    "cloud-computing/4/index": "Lab 4: Modifying a Dockerfile",
    "cloud-computing/5/index": "Lab 5: Building a Docker Image",
    "cloud-computing/6/index": "Lab 6: Flask Python App",
    "cloud-computing/8/index": "Lab 8: VPS Security",
    "cloud-computing/references/index": "Reference Guides",
    "cloud-computing/references/security": "Security Quick References",
    "cloud-computing/references/wordpress": "WordPress Hacks",
    "cloud-computing/references/sphinx-readthedocs": "Sphinx and Read the Docs",
    "cloud-computing/references/dockerfile": "Dockerfile",
    "cloud-computing/references/paas": "PaaS Tools",

    "nasm/1/index": "Lab 1: Building an ASM File",
    "nasm/2/index": "Lab 2: User Input and Arithmetic",
    "nasm/3/index": "Lab 3: Control Structure",
    "nasm/4/index": "Lab 4: Nested Loops and Sub Routines",
    "nasm/5/index": "Lab 5: Subprograms",
    "nasm/6/index": "Lab 6: Subprograms: Arguments",
    "nasm/resources/index": "Reference Guides",
    "nasm/windows-install/index": "Set up your NASM Environment on Windows",

    "operating-systems/overview": "C System Calls: Overview",
    "operating-systems/1/index": "1. Computer Boot Sequence",
    "operating-systems/2/index": "2. Simple IO in C",
    "operating-systems/3/index": "3. C Fundamentals",
    "operating-systems/4/index": "4. Reading a File using System Calls",
    "operating-systems/5/index": "5. Writing to a File using System Calls",
    "operating-systems/6/index": "6: Creating a Process in C",
    "operating-systems/7/index": "7: Terminating a Process in C",
    "operating-systems/8/index": "8: Deleting a File using System Calls",
    "operating-systems/references/index": "Reference Guides",
}

# --- Placeholder image assets -------------------------------------------
# Upstream binary screenshots the fetch pipeline cannot retrieve (web_fetch
# rejects image bytes and the labs domain is unreachable from the sandbox shell).
# The generator writes a labelled placeholder (format matched to the extension)
# at each mirror path so the real Sphinx build stays warning-free. Replace each
# with the upstream original (URL given) when it can be downloaded; the mirrored
# RST already references these paths.
PLACEHOLDER_IMAGES = {
    "workshops/rst/images/default-sphinx-page.png":
        "https://labs.bilimedtech.com/_images/default-sphinx-page.png",
    "workshops/rst/images/rtd-sphinx-page.png":
        "https://labs.bilimedtech.com/_images/rtd-sphinx-page.png",
    "workshops/rst/images/the_great_sphinx_david_roberts.jpg":
        "https://labs.bilimedtech.com/_images/the_great_sphinx_david_roberts.jpg",
    "cloud-computing/2/images/reverse-proxy.png":
        "https://labs.bilimedtech.com/_images/reverse-proxy.png",
    "cloud-computing/2/images/freenom5.png":
        "https://labs.bilimedtech.com/_images/freenom5.png",
    "cloud-computing/2/images/freenom6.png":
        "https://labs.bilimedtech.com/_images/freenom6.png",
    "cloud-computing/2/images/freenom7.png":
        "https://labs.bilimedtech.com/_images/freenom7.png",
    "cloud-computing/2/images/chat-domain-name.png":
        "https://labs.bilimedtech.com/_images/chat-domain-name.png",
}

# --- Include fragments ---------------------------------------------------
# Files pulled into pages via ``.. include:: <name>`` that have no HTML page of
# their own (so Sphinx never publishes them to ``_sources``). Their raw text was
# supplied out-of-band; cached under ``_sources_cache/<path>.txt`` and emitted
# byte-verbatim WITHOUT an attribution footer (they are inlined into pages that
# already carry it). Paths keep the ``.rst`` extension. Because these are not
# toctree documents, scikit-plots' conf.py MUST exclude them from the build to
# avoid "document isn't included in any toctree" warnings, e.g.::
#
#     exclude_patterns += [
#         "learn/hands-on/edtech/**/urls.rst",
#         "learn/hands-on/edtech/**/*-urls.rst",
#         "learn/hands-on/edtech/**/*-content.rst",
#     ]
#
# Each lab's ``urls.rst`` is shared by every page in that lab (overview + steps),
# so adding one fragment unblocks the whole lab once the page sources are mirrored.
INCLUDES = {
    "cloud-computing/2/urls.rst",
    "cloud-computing/5/urls.rst",
    "cloud-computing/references/dockerfile-content.rst",
    # Site-wide Sphinx prolog. Upstream pages carry ``.. include:: /includes/prolog.inc``
    # (a leading-slash, source-root-relative path). That file has no _sources entry
    # and, in scikit-plots, ``/includes/`` would be *its own* tree — so the generator
    # rewrites the include to this mirror-local copy (see build_edtech.py). Only the
    # standard ``|br|`` line-break substitution it provides is reconstructed here;
    # any other prolog-only substitution would render as an upstream-undefined ref
    # (none are used except case-mismatch typos already present upstream).
    "includes/prolog.inc",
}

# --- Bulk-crawl overlay ---------------------------------------------------
# ``fetch_edtech.py`` (run from a machine with network access to the site) writes
# ``_sources_cache/_discovered.json`` describing every page, include fragment, and
# image it pulled. If that file is present we fold it into the hand-maintained
# lists above, so the same generator produces either the hand-built subset (no
# crawl) or the complete mirror (after a crawl) with no other change. The
# hand-built entries remain authoritative; the overlay only adds.
def _apply_discovered() -> None:
    import json
    from pathlib import Path

    disc = Path(__file__).resolve().parent / "_sources_cache" / "_discovered.json"
    if not disc.exists():
        return
    try:
        data = json.loads(disc.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return

    global VERBATIM, INCLUDES, PLACEHOLDER_IMAGES
    # Drop leading-slash (source-root-relative) include targets such as
    # ``/includes/prolog.inc`` — they are not per-page fragments; the generator
    # rewrites them to the mirror-local copy declared in INCLUDES above.
    frags = {f for f in data.get("fragments", []) if not f.startswith("/")}
    INCLUDES = set(INCLUDES) | frags
    # a fetched page that is actually an include fragment must not be a page
    VERBATIM = (set(VERBATIM) | set(data.get("pages", []))) - INCLUDES
    merged = dict(PLACEHOLDER_IMAGES)
    merged.update(data.get("images", {}))
    PLACEHOLDER_IMAGES = merged


_apply_discovered()
