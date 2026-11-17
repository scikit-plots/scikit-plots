# History

2026-09-12: first dedicated `_brand` maintenance/skill campaign created from the supplied snapshot.

2026-11-17: PR 949, brand kit and the Eukrina Mind AI brand (patch 1 of 3).
R: BRD-EUK-001 (fixed) the Eukrina brand existed only as a bundle of 410 recoloured PNGs and 21 SVGs made by nearest-colour pixel mapping. Defects seen on that bundle: one-colour symbols merge the diagonal into the U and two are misnamed (symbol-forest is taupe, symbol-navy is charcoal); the diagonal's end cap reaches 132 units from the round icon's centre, past its 124-unit circle; the mark sits 7 units right of centre in its 256 box; gradient ids `blue`/`green` collide when two SVGs share a page; some gradients have two equal stops; on light backgrounds the accents reach 2.0-3.0:1 and the subtitle 3.4-4.1:1 (WCAG asks 3 and 4.5).
R: rewritten from scratch as `scikitplot/_brand/_kit/` (colour scales and WCAG contrast, safe SVG writer, font outlines, atomic idempotent writing with a manifest) and `scikitplot/_brand/_eukrina/` (the mark, two editions and the type as data; six layouts drawn by rule; builds). Nothing reads an image.
R: design decisions of the maintainer: contrast-safe colours by rule (nearest passing step of each colour's own scale); live text with an outline option; SVG first, raster export next.
R: found during visual checks: fitted gradient stops can swap lightness; a rule keeps the edition's light-to-dark order. A transform printed at 3 decimals could round the mark past its clear space; scales are rounded down before the translation is computed.
R: BRD-BAN-002 (open, high) `_banner.py` line 574 subscripts `collections.abc.Callable` at import time, so `scikitplot._brand` does not import on Python 3.8. Not changed here.
R: tests: 215 new (test__kit.py, test__eukrina.py), whole `_brand` suite 486 passed (CPython 3.13, Matplotlib 3.11.2); 215 passed on CPython 3.11 with Matplotlib 3.6.3 and on 3.8.20 with 3.7.5 (isolated stand-in, because of BRD-BAN-002). Six mutations each caught. Tracker: maintenance PASS, runtime FAIL with the same 7 findings as main once the new files were added to the inventory; maintenance tests 17 passed, 1 failed, as on main.
