import fs from 'node:fs';

const js = fs.readFileSync(process.argv[2], 'utf8');
let pass = 0, fail = 0;
function t(name, ok) { if (ok) { console.log('ok - ' + name); pass++; } else { console.error('not ok - ' + name); fail++; } }

t('close button advertises Escape', /closeBtn\.setAttribute\('aria-keyshortcuts', 'Escape'\)/.test(js));
t('close button tooltip exposes Esc', /closeBtn\.title = 'Close ' \+ _escapeHtml\(title\) \+ ' \(Esc\)'/.test(js));
t('shortcut sheet names panel close', /shortcutRow\('Close AI Assistant', \['Escape'\]/.test(js));
t('mic popup closes before panel', /if \(micPopup && micPopup\.getAttribute\('data-pinned'\) === 'true'\)[\s\S]*?return;/.test(js));
t('feedback popup closes before panel', /pinnedFeedbackPopup[\s\S]*?ai-assistant-fbk-popup\[data-pinned="true"\][\s\S]*?return;/.test(js));
t('menu or sheet closes before panel', /if \(hasMenu \|\| hasSheet\)[\s\S]*?hamburgerMenuEl\._exit\(\);[\s\S]*?return;/.test(js));
t('Escape falls through to same close lifecycle', /Nothing lighter owns Escape[\s\S]*?closeAIPanel\(\);/.test(js));
t('panel-scoped listener remains used', /panel\.addEventListener\('keydown', function \(e\) \{[\s\S]*?if \(e\.key !== 'Escape'\) return;/.test(js));
t('active microphone remains first priority', /if \(_isListening \|\| _speechStartPending \|\| _micSpaceHeld\)[\s\S]*?_stopSpeechRecognition\(\)/.test(js));
t('active response remains before navigation close', /if \(_stopActivePanelResponse\(\)\)[\s\S]*?return;/.test(js));

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
