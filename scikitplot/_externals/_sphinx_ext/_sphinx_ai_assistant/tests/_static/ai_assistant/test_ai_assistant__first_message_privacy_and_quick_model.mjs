// Run 171 — first-message privacy status + quick answer-menu model switcher.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function extract(name) {
  const start = src.indexOf('function ' + name + '(');
  if (start < 0) throw new Error('missing ' + name);
  let depth = 0, began = false, quote = '', esc = false, line = false, block = false;
  for (let i = start; i < src.length; i++) {
    const c = src[i], n = src[i + 1];
    if (line) { if (c === '\n') line = false; continue; }
    if (block) { if (c === '*' && n === '/') { block = false; i++; } continue; }
    if (quote) { if (esc) { esc = false; continue; } if (c === '\\') { esc = true; continue; } if (c === quote) quote = ''; continue; }
    if (c === '/' && n === '/') { line = true; i++; continue; }
    if (c === '/' && n === '*') { block = true; i++; continue; }
    if (c === '"' || c === "'" || c === '`') { quote = c; continue; }
    if (c === '{') { depth++; began = true; }
    else if (c === '}' && --depth === 0 && began) return src.slice(start, i + 1);
  }
  throw new Error('unterminated ' + name);
}

const append = extract('_appendPanelMessage');
const replay = extract('_replayTranscript');
const banner = extract('_buildFirstMessagePrivacyBanner');
const more = extract('_buildBubbleMore');
const close = extract('_closeBubbleMoreWrapper');
const position = extract('_positionBubbleMoreMenuWithinPanelBody');
const select = extract('_selectQuickModel');
const candidates = extract('_quickModelCandidates');

ok(append.includes("body.querySelector('.ai-assistant-pagehelp')") && append.includes('pageHelp.remove()'), 'first send removes stale Explain-this-page onboarding');
ok(append.includes('var firstRealMessage = _transcript.length === 0') && append.includes('_ensureFirstMessagePrivacyBanner(body)'), 'privacy row is installed at first real message commit');
ok(replay.includes('if (_transcript.length) _ensureFirstMessagePrivacyBanner(body)'), 'restored transcript restores its top privacy status row');
ok(banner.includes("banner.setAttribute('role', 'note')"), 'privacy row has semantic note role');
ok(banner.includes("copy.appendChild(document.createTextNode(_firstMessagePrivacyText() + ' '))"), 'privacy copy is inserted as text rather than trusted HTML');
ok(banner.includes("'ai-assistant-open-privacy'"), 'More information routes to existing privacy sheet');
ok(src.includes(".addEventListener('ai-assistant-open-privacy'"), 'panel owns internal privacy-sheet routing event');
ok(src.includes(".addEventListener('ai-assistant-open-model-configuration'"), 'panel owns model-configuration routing event');

ok(more.indexOf("modelToggleLabel.textContent = 'Change model'") < more.indexOf("retryMenuLbl.textContent = 'Retry'"), 'Change model is first answer-menu action before Retry');
ok(more.includes("modelHeading.textContent = 'Try a different model'"), 'expanded list uses requested try-a-different-model heading');
// The six-item bound is gone. It kept the first six and dropped the rest in
// silence: a twelve-model configuration showed six, indistinguishable from a
// six-model one, under a heading reading "Try a different model". The cut fell
// wherever the configured order happened to put it.
ok(more.includes('if (activeQuickModel) quickDisplayModels.push(activeQuickModel)'), 'the active model is listed first');
ok(!/quickDisplayModels\.length\s*>=\s*\d/.test(more), 'no model is dropped from the quick menu');
ok(more.includes("if (!activeQuickModel || candidate.id !== activeQuickModel.id)"), 'the active model is not listed twice');
// Length is handled by scrolling, so a long configuration costs a scroll
// rather than six missing entries.
const qm_css = fs.readFileSync(process.argv[3], 'utf8');
// Reversed deliberately. Its own `max-height: min(50vh, 18rem)` plus
// `overflow-y: auto` is right for a list standing alone and wrong for one
// nested in a menu that is itself bounded: on a short panel the outer menu was
// clamped to ~180px while this list claimed up to 288px, so two scroll regions
// sat one inside the other, the inner took the wheel, and no model was
// reachable at all.
const listRule = (qm_css.replace(/\/\*[\s\S]*?\*\//g, '')
  .match(/^\.ai-assistant-panel-bubble-model-list \{[^}]*\}/gm) || []);
ok(listRule.length === 1, 'the model list is defined once, not by two rules read together');
ok(/max-height:\s*none/.test(listRule[0]) && /overflow-y:\s*visible/.test(listRule[0]), 'the list has no scroller of its own inside the menu');
ok(/overscroll-behavior:\s*auto/.test(listRule[0]), 'so the wheel reaches the menu that actually scrolls');
// The outer menu is the single scroller, and must be told when the list grows.
ok(src.includes("modelList.closest('.ai-assistant-panel-bubble-action-more-menu')"), 'expanding the list finds the menu whose bound it changed');
ok(src.includes('if (ownerMenu) _positionBubbleMoreMenuWithinPanelBody(ownerMenu);'), 'and re-places it, so the new rows are inside the recomputed bound');
ok(/\.ai-assistant-panel-bubble-model-heading\s*\{[^}]*position:\s*sticky/.test(qm_css), 'the heading stays visible while the options scroll');

// Every candidate reaches the list: stubs and custom models are not filtered
// out on the way, so "all configured models" means all of them.
const cand = extract('_quickModelCandidates');
ok(!/\bstub\b/i.test(cand), 'stub models are not filtered out of the candidate list');
ok(cand.includes('_MODEL_STORE.listCustom()'), 'custom models are candidates too');
ok(more.includes("'View all ' + quickModels.length + ' models…'") && more.includes("'Model configuration…'"), 'full model sheet remains escape hatch for long/short lists');
ok(more.includes("modelBtn.setAttribute('role', 'menuitemradio')") && more.includes("modelBtn.setAttribute('aria-checked'"), 'model choices expose radio semantics and active state');
ok(more.includes("providerLabel + ' · ' + modelWire"), 'quick rows include provider and wire-model metadata');
ok(more.includes('var currentModels = _quickModelCandidates(_cfg())') && more.includes("modelList.addEventListener('keydown'") && more.includes("'ArrowDown'") && more.includes("'Home'") && more.includes("'End'"), 'quick model list refreshes active state and supports keyboard traversal');
ok(close.includes("'.ai-assistant-panel-bubble-action--model-toggle'") && close.includes("'.ai-assistant-panel-bubble-model-list'"), 'closing More also collapses model disclosure');
ok(position.includes('minWidth: hasModels ? 224 : 144') && position.includes('maxWidth: hasModels ? 320 : 220'), 'boundary coordinator grants model menu a wider bounded surface');
ok(select.includes('_setActiveModelId(m.id)') && select.includes("'ai-assistant-model-change'") && select.includes('_syncInlinePickers(m.id)') && select.includes('_syncModelSheet(m.id)'), 'quick selection uses canonical model state and sync event');
ok(candidates.includes('!m.disabled') && candidates.includes('!_MODEL_STORE.isHiddenBuiltin(m.id)') && candidates.includes('_MODEL_STORE.listCustom()'), 'quick list excludes disabled/removed models and includes custom models');
ok(css.includes('.ai-assistant-panel-chat-privacy') && css.includes('.ai-assistant-panel-chat-privacy-more'), 'privacy status has dedicated responsive styling');
ok(css.includes('.ai-assistant-panel-bubble-model-option') && css.includes('[aria-checked="true"]'), 'quick model rows and active checkmark are styled');

// Runtime: provider display labels are stable and human-readable.
const providerLabel = (0, eval)('(' + extract('_quickModelProviderLabel') + ')');
ok(providerLabel('openai') === 'OpenAI' && providerLabel('huggingface') === 'Hugging Face' && providerLabel('acme') === 'acme', 'provider label normalization preserves unknown custom providers');

// Runtime: candidate filtering mirrors canonical override/tombstone authority.
const candidateFactory = new Function(`
  const _MODEL_STORE = {
    registerBuiltin(){},
    applyOverrides(xs){ return xs.map(x => Object.assign({}, x)); },
    isHiddenBuiltin(id){ return id === 'hidden'; },
    listCustom(){ return [{id:'custom',label:'Custom',provider:'custom',model:'wire/custom'}]; }
  };
  ${candidates}
  return _quickModelCandidates;
`);
const list = candidateFactory()({panelApiModels:[
  {id:'a',label:'A',provider:'openai',model:'gpt-a'},
  {id:'disabled',disabled:true},
  {id:'hidden'}
]});
ok(list.map(x => x.id).join(',') === 'a,custom', 'runtime candidate list filters disabled/tombstoned rows and appends custom rows');

// Runtime: defaults never fabricate zero-retention/no-training guarantees.
const privacyTextSource = extract('_firstMessagePrivacyText');
function privacyText(cfg, localFallback) {
  return new Function('_cfg','_getActiveModel','_stubUsesLocalFallback', `return (${privacyTextSource})();`)(
    () => cfg, () => ({id:'m'}), () => localFallback
  );
}
ok(privacyText({panelChatPrivacyText:'Verified private route'}, false) === 'Verified private route', 'operator verified plain-text privacy copy can override default');
ok(privacyText({panelApiEnabled:false,panelChatPrivacyText:''}, false).startsWith('Local chat.'), 'local default clearly says local chat');
ok(privacyText({panelApiEnabled:true,panelChatPrivacyText:''}, false).includes('depend on that provider'), 'API default defers retention/training claims to provider policy');
ok(!privacyText({panelApiEnabled:true,panelChatPrivacyText:''}, false).includes('Zero data retention'), 'API default does not invent a zero-retention claim');

// ── Toggle switches survive forced-colours modes ─────────────────────────
//
// The PDF switch already had this treatment and the panel and mic switches did
// not. Author colours are discarded there, so a track drawn rgba(0,0,0,.22)
// and a white thumb both collapse toward the system background: the switch
// becomes a pill with no visible state, and on/off is the one thing a switch
// has to communicate.
const tog_css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
const forced = (tog_css.match(/@media \(forced-colors: active\) \{[\s\S]*?\n\}/g) || []).join('\n');
['mic-toggle-track','panel-toggle-track','mic-toggle-thumb','panel-toggle-thumb','pdf-toggle-track','pdf-toggle-thumb']
  .forEach(function (cls) {
    ok(forced.includes(cls), cls + ' has a forced-colours treatment');
  });
// Named per toggle: the PDF block has its own `background: Highlight`, so a
// bare search for it passed while the panel toggle's rule had been removed.
ok(/panel-toggle-track \{\s*background:\s*Highlight;/.test(forced), 'the checked panel track uses a system colour, so on and off differ');
ok(/mic-toggle-track,[\s\S]{0,200}?background:\s*Highlight;/.test(forced), 'and the mic track with it');
ok(!/forced-colors[\s\S]*?panel-toggle-track \{\s*background:\s*rgba/.test(forced), 'no author colour is used for the panel track there');
ok(/background:\s*HighlightText;/.test(forced), 'and the thumb contrasts against it');
ok(/box-shadow:\s*none;[\s\S]{0,140}?border:\s*1px solid Canvas/.test(forced), 'the thumb gains a border, since its drop shadow is discarded too');

// ── A quick-swap chevron beside the model picker ─────────────────────────
//
// The picker opens the full sheet, where a model is chosen deliberately from a
// list with descriptions. The chevron is for the other case: swapping to a
// model the reader already knows, without leaving the composer. Two intents,
// two targets, rather than one control guessing which was meant.
ok(src.includes("_buildOverflowMenu('Try a different model', function () {"),'the chevron opens a quick model list');
ok(src.includes("'ai-assistant-panel-inline-picker-more', ICONS.chevronDown)"),'built from the shared menu, so it inherits its keyboard behaviour');
// Resolved per open: the configured list and the active model both change
// while the panel is up.
ok(src.includes('var live = _quickModelCandidates(_cfg());'),'candidates are read when the menu opens, not when it was built');
ok(/m\.id === currentId \? '.{0,8} ' : ''/.test(src),'the current model is marked, so the list says where you are');
ok(src.includes('_setActiveModelId(m.id);'),'choosing one switches the model');
// The picker itself must stay the button: the sync code writes aria-expanded
// on it, and a wrapper would have swallowed that silently.
ok(src.includes('pickerWrap.appendChild(inlinePicker);') && src.includes('pickerWrap.appendChild(quickModelBtn);'),'the two are siblings inside a wrapper');
ok(src.includes('var inlinePicker = _buildInlineModelPicker();') && src.includes("inlinePicker.setAttribute('aria-expanded'"),'inlinePicker still refers to the button the sync code writes to');
const pickerCss = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
ok(/\.ai-assistant-panel-inline-picker-wrapper \{[^}]*display:\s*inline-flex/.test(pickerCss),'the pair sits on one line');
ok(/\.ai-assistant-panel-inline-picker-more::before \{[^}]*width:\s*1px/.test(pickerCss),'a hairline joins them, so they read as one control with two targets');
ok(/\.ai-assistant-panel-inline-picker-more\[aria-expanded="true"\] svg \{ transform: rotate\(180deg\); \}/.test(pickerCss),'the chevron rotates from its announced state');
ok(/forced-colors: active[\s\S]{0,200}?inline-picker-more[\s\S]{0,60}?ButtonText/.test(pickerCss),'and survives forced-colours mode');

// ── The hamburger menu scrolls instead of running past the panel ─────────
//
// It had no height bound at all: on a small panel its lower entries ran past
// the bottom edge with nothing to scroll and no indication they existed.
const ham_css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
const hamBlocks = ham_css.match(/^\.ai-assistant-panel-hamburger \{[^}]*\}/gm) || [];
ok(hamBlocks.length === 1,'the hamburger is defined once, not by two rules read together');
const ham = hamBlocks[0] || '';
ok(/max-height:\s*min\(calc\(100% - 3\.3rem - 0\.75rem\), 80vh\)/.test(ham),'it is bounded by the panel, with the screen as the outer limit');
ok(/overflow-y:\s*auto/.test(ham),'and scrolls rather than overflowing');
ok(/overscroll-behavior:\s*contain/.test(ham),'without scrolling the transcript behind it');
// `100%` is correct here precisely because this menu is inside the panel --
// unlike the shared popups, which are placed against the viewport.
ok(/position:\s*absolute/.test(ham),'the percentage resolves against the panel, so a resize needs no JavaScript');
ok(/@media \(pointer: coarse\)[\s\S]{0,220}?ai-assistant-panel-hamburger[\s\S]{0,80}?85vh/.test(ham_css),'a coarse pointer gets more of the screen for the same list');

// ── The chevron on a touch device ────────────────────────────────────────
//
// 1.6rem with a transparent background is a fine cursor target and a poor
// thumb one, and on touch there is no hover to reveal that it is a control at
// all -- hard to hit and hard to notice, each failure making the other worse.
const chev_css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
const chevBlocks = chev_css.match(/^\.ai-assistant-panel-inline-picker-more \{[^}]*\}/gm) || [];
ok(chevBlocks.length === 1,'the chevron is defined once, not by two rules read together');
const chev = chevBlocks[0] || '';
// These moved onto a rule naming both chevrons, so a third inherits them by
// being added there rather than by someone remembering to copy them.
const bothChev = (chev_css.match(/\.ai-assistant-panel-inline-picker-more,\s*\n\.ai-assistant-panel-attachment-preview-menu-btn \{[^}]*\}/) || [''])[0];
ok(bothChev.length > 0,'the two chevrons share one treatment rule');
ok(/appearance:\s*none/.test(bothChev) && /-webkit-appearance:\s*none/.test(bothChev),'Safari and Firefox button chrome is reset for both');
ok(/touch-action:\s*manipulation/.test(bothChev),'no 300ms tap delay on iOS Safari or Android Chrome');
ok(/-webkit-tap-highlight-color:\s*transparent/.test(bothChev),'and no grey flash that reads as a control that did not respond');
ok(/\.ai-assistant-panel-attachment-preview-menu-btn:active/.test(chev_css),'both get pressed feedback, the only signal touch has');
ok(/@media \(pointer: coarse\)[\s\S]*?attachment-preview-menu-btn \{[^}]*min-width:\s*2\.75rem/.test(chev_css),'and the same 44px target, dialog or composer');

const coarse = (chev_css.match(/@media \(pointer: coarse\) \{[\s\S]*?\n\}/g) || []).join('\n');
// Scoped to the chevron's own rule: the mic carries the same numbers, so an
// unscoped search matched that instead and passed with the chevron shrunk.
ok(/inline-picker-more \{[^}]*min-width:\s*2\.75rem[^}]*min-height:\s*2\.75rem/.test(coarse),'a coarse pointer gets a 44px target');
ok(/inline-picker-more svg \{ width: 1\.1rem/.test(coarse),'with a glyph big enough to see');
ok(/inline-picker-more[\s\S]{0,200}?background:\s*color-mix/.test(coarse),'and a resting ground, since hover cannot supply one');
ok(/inline-picker-more::before \{ display: none; \}/.test(coarse),'the divider is dropped rather than narrowing the target');
ok(/inline-model-picker \{\s*min-height:\s*2\.75rem/.test(coarse),'the picker beside it grows to match');
ok(/\[data-bs-theme="dark"\][\s\S]{0,200}?inline-picker-more[\s\S]{0,120}?rgba\(255, 255, 255, 0\.10\)/.test(chev_css),'the dark theme states its own resting ground');
ok(/\.ai-assistant-panel-inline-picker-more:active/.test(chev_css),'a pressed state, which on touch is the only feedback there is');
ok(/forced-colors: active[\s\S]{0,200}?inline-picker-more[\s\S]{0,60}?ButtonText/.test(chev_css),'and a border where colours are discarded');

// ── The enlarged chevron must not crowd the mic ──────────────────────────
//
// R173T74 gave the chevron a 44px target. In a row with `gap: 0.1rem` and no
// wrapping that pushed it into the microphone beside it: two adjacent
// thumb-sized controls with almost nothing between them, so the reader aims at
// one and hits the other.
const crowd_css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
const coarseAll = (crowd_css.match(/@media \(pointer: coarse\) \{[\s\S]*?\n\}/g) || []).join('\n');
ok(/footer-actions-right \{[^}]*gap:\s*0\.4rem/.test(coarseAll),'the gap grows with the targets rather than staying fixed');
ok(/footer-actions-right \{[^}]*flex-wrap:\s*wrap/.test(coarseAll),'and the row may wrap before either control is squeezed');
ok(/footer-btn--mic \{[^}]*min-width:\s*2\.75rem/.test(coarseAll),'the mic keeps a target of its own');
ok(/footer-btn--mic \{[^}]*flex:\s*0 0 auto/.test(coarseAll),'and is not shrunk to make room for the chevron');

// ── The chevron is drawn from the picker's tokens ────────────────────────
//
// The two are joined into one segmented control, so they should come from one
// set of values. The chevron was transparent with a border only on touch, so at
// rest on a desktop the pair read as a bordered button with a bare glyph stuck
// to it.
const pair_css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
const moreBlocks = pair_css.match(/^\.ai-assistant-panel-inline-picker-more \{[^}]*\}/gm) || [];
ok(moreBlocks.length === 1,'the chevron is defined once, not by two rules read together');
const chevRule = moreBlocks[0] || '';
ok(/border:\s*1px solid var\(\s*--pst-color-border/.test(chevRule),'it uses the picker border token, not one of its own');
ok(/color:\s*var\(--pst-color-text-muted, var\(--color-foreground-secondary/.test(chevRule),'and the picker text token');
ok(!/border:\s*0;/.test(chevRule),'no stale borderless declaration is left above it to read past');
// One outline for the pair, not two meeting in the middle.
ok(/border-inline-start:\s*0/.test(chevRule),'the shared edge is dropped on the chevron');
ok(/\.ai-assistant-panel-inline-picker-wrapper \.ai-assistant-panel-inline-model-picker \{[^}]*border-inline-end:\s*1px solid/.test(pair_css),'and supplied once by the picker');
ok(/\.ai-assistant-panel-inline-picker-more::before \{ display: none; \}/.test(pair_css),'the separate divider is gone, so the hairline is not doubled');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
