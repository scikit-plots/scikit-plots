# R173T85 — PDF thumb geometry parity and cascade authority

Status: COMPLETE

Base: R173T84 workspace tab composition.

## Reported UI issue

The PDF mode-switch thumb appeared misplaced while the Panel visibility switch
was visually correct. The markup used the same base track/thumb primitives, so
the defect had to be resolved at the CSS cascade/geometry layer rather than by
moving DOM nodes or changing PDF state logic.

## Root cause

The PDF checked transform was declared early in the stylesheet, before the
shared `.ai-assistant-mic-popup-toggle[aria-checked="true"]` transform. Those
selectors have equal specificity. The later generic rule therefore won the
cascade and could apply the Mic-sized 12px travel to the larger 34px PDF track.

R173T83 then added a PDF-only 17px optical exception, but because it lived in
the earlier declaration it did not fix the authority problem and also diverged
from the known-good Panel reference (16px).

## Repair

Checked thumb travel now has one CSS authority:

- the shared checked rule consumes `--ai-assistant-toggle-thumb-travel`, with
  `12px` as the Mic fallback;
- PDF, Copy, and Panel mode switches each set the large-control token to `16px`;
- PDF/Copy/Panel-specific checked `transform` declarations are removed;
- PDF and Panel retain identical 34×18 tracks and 14×14 thumbs at `top/left:1px`;
- PDF JavaScript, `aria-checked`, persistence, and action semantics are unchanged.

This supersedes the R173T83 optical-offset conclusion. R173T83 remains
historical evidence of the visual report and the insufficient first repair.

## Verification

- `test_ai_assistant__pdf_thumb_geometry_parity.mjs`: 10/10;
- `test_ai_assistant__pdf_toggle_layout.mjs`: 16/16;
- `test_ai_assistant__panel_trigger.mjs`: 100/100;
- `test_ai_assistant__copy_mode.mjs`: 13/13;
- centralized maintenance/review gates;
- packaged-byte replay before delivery.
