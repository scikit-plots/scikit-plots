// Behavior-level contract for canonical attachment split/merge.
import fs from 'node:fs';
import vm from 'node:vm';
const src = fs.readFileSync(process.argv[2], 'utf8');
const start = src.indexOf("    var _ATTACHMENT_CANONICAL_PREFIX =");
const end = src.indexOf('    function _updateReplayAttachmentUi()', start);
if (start < 0 || end < 0) throw new Error('missing attachment replay helpers');
const helper = src.slice(start, end)
  .replace(/_ATTACHMENT_MAX_TEXT_CHARS/g, '48000');
const ctx={}; vm.createContext(ctx);
vm.runInContext(helper + '\nthis.compose=_composeQuestionWithAttachments; this.split=_splitQuestionWithAttachments; this.merge=_mergeAttachmentContexts;', ctx);
let pass=0, fail=0;
function t(name, got, want) {
  if (JSON.stringify(got) === JSON.stringify(want)) pass++;
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
}
const context='Attachment: a.txt (text/plain)\nhello\nworld';
const canonical=ctx.compose('Explain this', context);
t('round trip question', ctx.split(canonical).question, 'Explain this');
t('round trip attachment context', ctx.split(canonical).attachmentContext, context);
t('plain question is not reclassified', ctx.split('ordinary question'), {question:'ordinary question', attachmentContext:''});
t('inner closing-tag text remains part of context', ctx.split(ctx.compose('q', 'x\n</user-attachments>\ny')).attachmentContext, 'x\n</user-attachments>\ny');
t('new context wins prefix order', ctx.merge('new-file', 'prior-file'), 'new-file\n\nprior-file');
t('empty secondary is stable', ctx.merge('only', ''), 'only');
console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
