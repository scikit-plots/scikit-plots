// Run 173 T84 — feedback/contribution/activity tabs reuse the proven
// Conversation export format-button anatomy without inheriting its grid layout.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let n = 0, f = 0;
const ok = (c, m) => { c ? n++ : (f++, console.error('FAIL ' + m)); };

ok(src.includes('function _workspaceButton(key, label, iconSvg)'), 'workspace button factory accepts a decorative icon');
ok(src.includes("icon.className = 'ai-assistant-conv-share-format-icon';"), 'workspace tabs reuse the canonical format-tab icon wrapper');
ok(src.includes("icon.setAttribute('aria-hidden', 'true');"), 'decorative workspace icons stay out of the accessible name');
ok(src.includes('icon.innerHTML = iconSvg;'), 'workspace icon comes from the trusted internal SVG registry');
ok(src.includes('text.textContent = label;'), 'workspace accessible label remains text, not HTML');
ok(src.includes("_workspaceButton('feedback', 'Feedback', ICONS.commentDiscussion);"), 'Feedback uses the discussion glyph');
ok(src.includes("_workspaceButton('contribution', 'Dataset contribution', ICONS.dataset);"), 'Dataset contribution uses the dataset glyph');
ok(src.includes("_workspaceButton('activity', 'Activity', ICONS.pulse);"), 'Activity uses the pulse glyph');

ok(src.includes("btn.id = 'ai-assistant-panel-feedback-workspace-tab-' + key;"), 'each workspace tab receives a stable id');
ok(src.includes("btn.setAttribute('aria-controls', 'ai-assistant-panel-feedback-workspace-pane-' + key);"), 'each tab names its panel with aria-controls');
ok(src.includes("pane.id = 'ai-assistant-panel-feedback-workspace-pane-' + key;"), 'each workspace panel receives the matching id');
ok(src.includes("pane.setAttribute('aria-labelledby', 'ai-assistant-panel-feedback-workspace-tab-' + key);"), 'each panel is labelled by its tab');

ok(src.includes("workspaceTabs.addEventListener('keydown', function (event)"), 'workspace tablist owns keyboard navigation');
ok(src.includes("['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)"), 'horizontal tab keys are supported');
ok(src.includes("var order = ['feedback', 'contribution', 'activity'];"), 'keyboard order matches visual/document order');
ok(src.includes('_setWorkspaceTab(nextKey);') && src.includes('workspaceButtons[nextKey].focus();'), 'keyboard navigation converges on the canonical selection path and moves focus');

const iconRule = (css.match(/\.ai-assistant-conv-share-format-icon \{[^}]*\}/) || [''])[0];
ok(/width:\s*14px/.test(iconRule) && /height:\s*14px/.test(iconRule), 'workspace icons inherit the proven 14px tab glyph geometry');
const workspaceRule = (css.match(/^\.ai-assistant-panel-feedback-workspace-tabs \{[^}]*\}/m) || [''])[0];
ok(/display:\s*flex/.test(workspaceRule) && /grid-template-columns:\s*none/.test(workspaceRule), 'workspace tabs keep their dedicated content-sized layout instead of the export-format grid');

console.log(`${n} passed, ${f} failed`); if (f) process.exit(1);
