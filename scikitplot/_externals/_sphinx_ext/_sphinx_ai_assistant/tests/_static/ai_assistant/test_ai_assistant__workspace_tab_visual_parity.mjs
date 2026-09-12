// Run 173 T86 — workspace tabs keep their dedicated layout but share the
// exact typography and colour-state authority of Conversation format tabs.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let n = 0, f = 0;
const ok = (c, m) => { c ? n++ : (f++, console.error('FAIL ' + m)); };

ok(src.includes("workspaceTabs.className = 'ai-assistant-conv-share-format-switcher ai-assistant-panel-feedback-workspace-tabs';"),
   'workspace tablist still opts into the canonical format-tab visual family');
ok(src.includes("btn.className = 'ai-assistant-conv-share-format-btn';"),
   'workspace buttons use the same canonical button class as export formats');
ok(src.includes("icon.className = 'ai-assistant-conv-share-format-icon';"),
   'workspace glyphs use the same canonical icon wrapper');

const buttonRule = (css.match(/\.ai-assistant-conv-share-format-btn \{[^}]*\}/) || [''])[0];
ok(/font:\s*inherit/.test(buttonRule), 'canonical tabs retain host font-family inheritance');
ok(/font-size:\s*0\.76rem/.test(buttonRule), 'canonical tabs own one font size');
ok(/font-weight:\s*600/.test(buttonRule), 'canonical tabs own one font weight');
ok(/line-height:\s*1\.55/.test(buttonRule), 'canonical tabs own line-height instead of inheriting it from their container');
ok(/color:\s*var\(--color-foreground-secondary,\s*#586069\)/.test(buttonRule), 'canonical tabs own the same neutral text colour');

const selectedRule = (css.match(/\.ai-assistant-conv-share-format-btn\[aria-selected="true"\] \{[^}]*\}/) || [''])[0];
ok(/background:\s*rgba\(9,\s*105,\s*218,\s*0\.1\)/.test(selectedRule), 'selected background remains owned by the shared button rule');
ok(/border-color:\s*rgba\(9,\s*105,\s*218,\s*0\.22\)/.test(selectedRule), 'selected border remains owned by the shared button rule');
ok(/color:\s*var\(--color-accent,\s*#0969da\)/.test(selectedRule), 'selected text/icon colour remains owned by the shared button rule');

const workspaceButtonRule = (css.match(/\.ai-assistant-panel-feedback-workspace-tabs > \.ai-assistant-conv-share-format-btn \{[^}]*\}/) || [''])[0];
ok(/flex:\s*0 1 auto/.test(workspaceButtonRule), 'workspace keeps only its content-sized layout override');
ok(!/(?:font|color|background|border|box-shadow|opacity)\s*:/.test(workspaceButtonRule),
   'workspace layout rule cannot fork typography or colour');
ok(!/\.ai-assistant-panel-feedback-workspace-tabs > \.ai-assistant-conv-share-format-btn\[aria-selected="true"\]\s*\{/.test(css),
   'workspace has no parallel selected-state colour or indicator rule');

console.log(`${n} passed, ${f} failed`); if (f) process.exit(1);
