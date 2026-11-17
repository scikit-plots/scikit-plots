# Maintaining `scikitplot._brand`

`scikitplot._brand` owns deterministic Matplotlib logo generation and terminal-banner generation. It does not own Matplotlib, NumPy, the system `figlet` executable, remote font hosting, or root package entrypoint configuration.

Start with `_maintenance/FRESH_CHAT_HANDOFF.md`, run the maintenance contract/review tools, then run the focused logo suite. Keep logo rendering and banner generation as separate evidence lanes.

## Kit and brands

`_kit/` is the generic, brand-free part: colours (`_color`: scales, WCAG contrast, `readable`), a safe SVG writer (`_svg`: no scripts, handlers, styles or external references; ids namespaced per document), text (`_text`: font lookup without fallback, measured advances, glyph outlines through Matplotlib) and output (`_io`: confined, atomic, idempotent writes and a checksummed manifest). A brand package such as `_eukrina/` holds data (`_spec`), drawing rules (`_render`, `RULES`) and builds (`_build`), and depends on the kit, never the reverse.

Rules for a change: positions come from the mark's bounds and the type metrics, never by hand; a new colour role is checked by the contrast tests; a new layout gets the well-formed/safe/deterministic tests by joining `LAYOUTS`; look at a contact sheet of every combination before delivering (render the SVGs with `rsvg-convert` on their backgrounds).
