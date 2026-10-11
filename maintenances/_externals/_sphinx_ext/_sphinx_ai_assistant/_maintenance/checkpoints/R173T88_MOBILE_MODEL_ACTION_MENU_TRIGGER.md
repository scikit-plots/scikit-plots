# R173T88 — Mobile model action menu trigger anchor

Status: COMPLETE

Base: R173T87 Share artifact responsive action layout.

## Reported UI issue

On narrow/mobile model lists, activating the vertical-ellipsis model action
button could open Edit/Delete/Reset noticeably far below the trigger. The gap
was larger on rows containing more metadata, description text, or parameter
visualization.

## Root cause

The trigger and popup had different layout owners. The `⋮` button was a
flex child at the top-right of `.ai-assistant-panel-model-row`, while the
absolute `.ai-assistant-panel-model-actions` popup used that **entire model
row** as its containing block and `top: calc(100% - 0.28rem)`. A tall model
card therefore moved the popup farther down even though the trigger itself had
not moved.

The responsive harness also had a maintenance gap: it inferred the CSS file
from the JavaScript path instead of honoring the mutation runner's explicit CSS
target, so CSS mutants for this surface could not be exercised correctly.

## Repair

- Introduce `.ai-assistant-panel-model-action-host` as the shared local owner
  of the disclosure trigger and action surface.
- Keep that host `position: relative` and append both `actionsWrap` and
  `menuBtn` to it.
- On mobile, place the popup at `top: calc(100% + 0.24rem); right: 0`, so its
  distance is measured from the 1.8rem trigger host rather than the variable
  model-card height.
- Measure the visible model-sheet/viewport intersection on open and flip the
  popup above the trigger when there is insufficient room below.
- Preserve one DOM/handler path for Edit/Delete/Edited/Reset across desktop and
  mobile; only presentation/placement changes.
- Teach the responsive harness to honor the explicit CSS path supplied by the
  mutation runner.

## Verification

- model responsive actions: **38/38**;
- model override/edit neighbor: **125/125**;
- model remove/revert neighbor: **23/23**;
- quick-model neighbor: **91/91**;
- all registered Node/UI harnesses: **161/161**;
- mutation catalogue: **473/473**;
- maintenance core: **35/35**;
- family maintenance: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- independent AI review: **PR_READY / release ELIGIBLE**;
- JavaScript syntax: GREEN.

## Prevention

A popup must be positioned by the element that visually owns it, not by a
larger content row whose height is unrelated to the trigger. Mutation controls
now reintroduce full-row anchoring, remove the positioned host, and disable the
edge flip; each must be caught.
