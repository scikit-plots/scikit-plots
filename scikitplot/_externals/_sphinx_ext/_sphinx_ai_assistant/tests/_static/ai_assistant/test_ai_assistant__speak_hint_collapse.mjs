// Run 173 T55 - the speak hint collapses; it is never destroyed.
//
// R173T52 made it removable and remembered the removal. That was the wrong
// shape: the row is onboarding for a shortcut, and once removed there was no
// way back short of a new session. A hint that can only ever be destroyed is
// one a reader will not risk putting away.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
function extract(name){const st=src.indexOf('function '+name+'(');if(st<0)throw new Error('missing '+name);
 let d=0,b=false;for(let i=st;i<src.length;i++){if(src[i]==='{'){d++;b=true;}else if(src[i]==='}'){if(--d===0&&b)return src.slice(st,i+1);}}
 throw new Error('unterminated '+name);}
let n = 0, f = 0;
const ok = (c, m) => { c ? n++ : (f++, console.error('FAIL ' + m)); };

// ── Reversible ───────────────────────────────────────────────────────────
ok(!/_SPEAK_HINT_DISMISSED_KEY/.test(src),'the destructive dismissal is gone');
ok(/_SPEAK_HINT_COLLAPSED_KEY = 'ai-assistant-speak-hint-collapsed'/.test(src),'state is collapsed-or-not, not present-or-gone');
ok(!/speakRow\.remove\(\)/.test(src),'the row is never removed from the panel');
ok(src.includes("if (speakBanner && hasSpeech) {"),'and it is always built, so it can always be restored');

// ── The toggle says which way the next press goes ────────────────────────
ok(src.includes("speakToggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');"),'the control reports the hint state it owns');
ok(src.includes("collapsed ? 'Show the speak hint' : 'Collapse the speak hint'"),'its label names the action, not just the state');
ok(src.includes('ev.stopPropagation();'),'collapsing does not also start speech recognition');
ok(src.includes("_ssSet(_SPEAK_HINT_COLLAPSED_KEY, collapsed ? '0' : '1');"),'the choice is remembered for the session');
ok(src.includes("_applySpeakCollapsed(_ssGet(_SPEAK_HINT_COLLAPSED_KEY) === '1');"),'and applied when the panel is built');

// A sibling button: a button inside a button is invalid and browsers drop one
// of the two click targets.
ok(src.includes('speakRow.appendChild(speakBannerEl);') && src.includes('speakRow.appendChild(speakToggle);'),'the toggle is a sibling of the banner');
ok(!/speakBannerEl\.appendChild\(speakToggle\)/.test(src),'never nested inside it');

// ── Collapsed keeps the glyph and the accessible name ────────────────────
ok(/\[data-collapsed="true"\] \.ai-assistant-panel-speak-banner \{[^}]*flex:\s*0 0 auto/.test(css),'collapsed, the banner shrinks to its icon');
const clipped = (css.match(/\[data-collapsed="true"\][^{]*span,[\s\S]*?\}/) || [''])[0];
ok(/clip-path:\s*inset\(50%\)/.test(clipped) && !/display:\s*none/.test(clipped),'the label is clipped, not removed, so the hit area and the accessible name survive');
ok(src.includes("speakBannerEl.setAttribute('aria-label', 'Speak with your assistant"),'the banner still announces what it does when collapsed');

// ── Collapsed sits where the thumb is ────────────────────────────────────
//
// Collapsed the row is a single small target whose only purpose is to be
// pressed, and on a phone the trailing edge is where the thumb rests. Expanded
// it is text to read, so it stays leading-aligned.
ok(/\[data-collapsed="true"\] \{ justify-content: flex-end; \}/.test(css),'the collapsed row sits at the trailing edge');
ok(!/\[data-collapsed="true"\] \{ justify-content: flex-start; \}/.test(css),'not at the leading edge, which is the far corner one-handed');
ok(!/\[data-collapsed="true"\][^{]*\{[^}]*justify-content:\s*right/.test(css),'expressed with flex-end, so a right-to-left interface mirrors without a second rule');
// DOM order is unchanged: only the alignment moves, so tab order and the
// screen-reader reading order are the same in both states.
ok(src.indexOf('speakRow.appendChild(speakBannerEl);') < src.indexOf('speakRow.appendChild(speakToggle);'),'the banner still precedes the toggle in the DOM in both states');

// ── The automatic path and the manual one reach the same state ───────────
//
// `_dismissSpeakBanner` used to hide the banner element with display:none.
// Since the banner sits in a row alongside a collapse toggle, that left the
// toggle behind on its own -- a control whose only purpose was to show and
// hide something no longer there. Expanding it produced an empty row.
const dismiss = extract('_dismissSpeakBanner');
ok(dismiss.includes("row.setAttribute('data-collapsed', 'true');"),'sending a message collapses the row, it does not hide half of it');
ok(dismiss.includes("querySelector('.ai-assistant-panel-speak-row')"),'it acts on the row, not on the banner element inside it');
ok(dismiss.indexOf('return;') < dismiss.indexOf("banner.style.display = 'none'"),'the element fallback runs only when there is no row');
ok(dismiss.includes("toggle.setAttribute('aria-expanded', 'false');"),'the toggle is told the state it now controls');
ok(dismiss.includes("toggle.setAttribute('aria-label', 'Show the speak hint');"),'and its label names the way back');

// Either path must be undoable by the other: the reader's toggle and the
// automatic collapse now write the same attribute.
ok(src.includes("speakRowEl.setAttribute('data-collapsed', 'false');"),'clearing the conversation expands the hint again');
ok(src.includes("speakToggleEl.setAttribute('aria-label', 'Collapse the speak hint');"),'with the toggle relabelled to match');
ok(!/_dismissSpeakBanner[\s\S]{0,400}?\.remove\(\)/.test(src),'nothing removes the row, so expanding always has something to expand');

// ── Collapsed, the row stops costing a row ───────────────────────────────
//
// One icon was still holding a full-width flex line with empty space beside
// it. The transcript now takes that width.
// Selected by content, not by position: the selector has more than one rule --
// T56's alignment and this checkpoint's placement -- and matching the first
// tested the wrong one, which is how two of these assertions first failed
// against correct CSS.
// Selected by content, not by position: the selector has more than one rule.
// The geometry moved onto the base rule: BOTH states are zero-height with
// their contents lifted. Expanded was still an in-flow row, so the space
// beside the banner was panel background -- opaque, nothing behind it.
const rowBase = (css.match(/^\.ai-assistant-panel-speak-row \{[^}]*\}/m) || [''])[0];
ok((css.match(/^\.ai-assistant-panel-speak-row \{/gm) || []).length === 1,'the row is defined once, not by two rules read together');
ok(/min-height:\s*2rem/.test(rowBase) && /height:\s*auto/.test(rowBase),'the row keeps a positive paint box in both states');
ok(/margin:\s*-2rem\s+0\.75rem\s+0/.test(rowBase),'an equal negative block margin reclaims its flow space');
ok(/overflow:\s*visible/.test(rowBase),'its floating controls are not clipped');
ok(/isolation:\s*isolate/.test(rowBase),'the floating paint is isolated from transcript compositing');
ok((css.match(/\.ai-assistant-panel-speak-row > \* \{ transform:\s*none; \}/g) || []).length === 1,'base children stay in ordinary paint coordinates');
ok(/\[data-collapsed="true"\] > \* \{[\s\S]*?transform:\s*none/.test(css),'collapsed children also stay out of transform-based lifting');
ok(!/\.ai-assistant-panel-speak-row[^\n{]*[\s\S]{0,180}?translateY\(-100%\)/.test(css),'the speak row no longer depends on a zero-height translate layer');
ok(!/position:\s*absolute/.test(rowBase),'it is not absolutely positioned against an ancestor it does not have');
ok(/\.ai-assistant-panel:has\(\.ai-assistant-panel-speak-row\) \.ai-assistant-panel-body \{[^}]*padding-bottom:\s*2\.75rem/.test(css),'the transcript keeps end padding so the pill never sits on the final line');
ok((css.match(/:has\(\.ai-assistant-panel-speak-row/g) || []).length === 1,'one reservation rule, covering both states');
// ── Both ends read as the same kind of thing ─────────────────────────────
//
// Collapsed, both ends already carried a surface. Expanded, the banner was a
// filled pill and the toggle bare, so one end looked like a control and the
// other like a glyph resting beside it.
//
// Bounded to the rule body: an unbounded lazy match runs past the closing brace
// into a later rule setting the same token, and would pass with the toggle's own
// declaration removed.
ok(/^\.ai-assistant-panel-speak-toggle \{[^}]*background-color:\s*var\(--ai-speak-toggle-surface\)/m.test(css),'the toggle carries its semantic surface at every width');
const speakToggleBase = (css.match(/^\.ai-assistant-panel-speak-toggle \{[^}]*\}/m) || [''])[0];
ok(!/background\s*:\s*transparent/.test(speakToggleBase),'no background shorthand erases the toggle ground later in the same rule');
ok(/\[data-collapsed="false"\] \.ai-assistant-panel-speak-toggle \{[^}]*align-self:\s*stretch/.test(css),'and matches its height, so the pair sits on one baseline');
// The row itself stays transparent: only the two controls are drawn.
ok(!/^\.ai-assistant-panel-speak-row \{[^}]*background/m.test(css),'the row draws nothing, so the space around the controls stays transparent');

console.log(`${n} passed, ${f} failed`); if (f) process.exit(1);
