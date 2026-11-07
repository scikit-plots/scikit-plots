// Run 89 regression: feedback popup owns feedback context only; answer utilities
// live in the canonical bubble-action More menu, with Home under inline disclosure.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let pass = 0, fail = 0;
function t(name, got, want=true) {
  if (got === want) pass++;
  else { fail++; console.log(`FAIL ${name}\n  got: ${JSON.stringify(got)}\n want: ${JSON.stringify(want)}`); }
}
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let depth = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { depth++; started = true; }
    else if (src[j] === '}') {
      depth--;
      if (started && depth === 0) return src.slice(i, j + 1);
    }
  }
  throw new Error('unbalanced: ' + name);
}

const fbk = extract('_buildFbkFloat');
const more = extract('_buildBubbleMore');
const closeMore = extract('_closeBubbleMoreWrapper');
const projection = extract('_publicAssistantEventDetail');
const readBookmarks = extract('_readAnswerBookmarkIds');
const setBookmark = extract('_setAnswerBookmarked');

const detailAt = fbk.indexOf("label: 'Detailed feedback'");
const centerAt = fbk.indexOf("label: 'Feedback center\\u2026'");
const contributeAt = fbk.indexOf("label: 'Contribute this Q&A\\u2026'");
t('feedback popup order is detail -> center -> contribution', detailAt >= 0 && centerAt > detailAt && contributeAt > centerAt);
t('feedback popup no longer owns Bookmark', !fbk.includes('Bookmark answer'));
t('feedback popup no longer owns Home', !fbk.includes("label: 'Home'"));
t('feedback popup no longer owns a nested More control', !fbk.includes('ai-assistant-fbk-more-trigger') && !fbk.includes('ai-assistant-fbk-more-body'));
t('feedback popup still routes to Feedback center', fbk.includes("new CustomEvent('ai-assistant-open-feedback-center'"));
t('feedback popup still routes to Contribution', fbk.includes("new CustomEvent('ai-assistant-open-contribution'"));

const shareAt = more.indexOf("shareMenuLbl.textContent = 'Share'");
const bookmarkAt = more.indexOf("bookmarkMenuLbl.textContent = bookmarked ? 'Remove bookmark' : 'Bookmark answer'");
const sepAt = more.indexOf("secondarySep.className = 'ai-assistant-panel-bubble-action-more-sep'");
const secondaryAt = more.indexOf("secondaryLabel.textContent = 'More'");
const bodyAt = more.indexOf("secondaryBody.className = 'ai-assistant-panel-bubble-action-more-secondary'");
const homeAt = more.indexOf("homeMenuLbl.textContent = 'Home'");
t('Bookmark belongs to canonical bubble More menu', bookmarkAt >= 0);
t('Bookmark follows existing Retry/Listen/Share region', bookmarkAt > shareAt);
t('divider follows Bookmark', sepAt > bookmarkAt);
t('secondary More follows divider', secondaryAt > sepAt);
t('secondary body follows its toggle', bodyAt > secondaryAt);
t('Home lives under secondary More', homeAt > bodyAt);
t('secondary More uses vertical dots', more.includes('secondaryToggle.innerHTML = ICONS.overflowV'));
t('secondary More uses chevron', more.includes('secondaryChevron.innerHTML = ICONS.chevronDown'));
t('secondary disclosure is inline/hidden by default', more.includes("secondaryBody.hidden = true") && more.includes("secondaryBody.setAttribute('data-open', 'false')"));
t('secondary expansion reuses boundary repositioning', more.includes('if (open) _schedulePinnedFeedbackPopupPosition();'));
t('closing root More uses centralized atomic close', /if \(isOpen\)[\s\S]*?_closeBubbleMoreWrapper\(wrapper, false\)/.test(more));
t('focus leaving root More uses centralized atomic close', /focusout[\s\S]*?_closeBubbleMoreWrapper\(wrapper, false\)/.test(more));
t('central close collapses nested disclosure', closeMore.includes("secondaryToggle.setAttribute('aria-expanded', 'false')") && closeMore.includes("secondaryBody.setAttribute('data-open', 'false')") && closeMore.includes('secondaryBody.hidden = true'));

t('Bookmark is reversible aria-pressed state', more.includes("bookmarkMenuBtn.setAttribute('aria-pressed', next ? 'true' : 'false')"));
t('Bookmark failure remains visible', more.includes('Bookmark could not be saved in this browser.'));
t('bookmark storage remains versioned', src.includes("_ANSWER_BOOKMARKS_KEY = 'ai-assistant-answer-bookmarks-v1'"));
t('bookmark storage remains bounded', src.includes('_ANSWER_BOOKMARKS_MAX = 500'));
t('bookmark reader accepts only marker ids', readBookmarks.includes('/^abm-[0-9a-f]{8}-[0-9a-z]+$/i'));
t('bookmark writer persists ids only', setBookmark.includes('ids: ids') && !setBookmark.includes('answerText') && !setBookmark.includes('questionText'));

t('Home uses stable internal event', more.includes("new CustomEvent('ai-assistant-open-home'"));
t('Home source reflects answer actions, not feedback', more.includes("source: 'answer-actions'"));
t('Home remains private from host-page projection', !projection.includes("case 'ai-assistant-open-home'"));
t('unconfigured Home remains visibly handled', more.includes('Home workspace is not configured yet.'));

t('bubble menu has separator style', /\.ai-assistant-panel-bubble-action-more-sep\s*\{/.test(css));
t('bubble secondary disclosure is inline', /\.ai-assistant-panel-bubble-action-more-secondary\s*\{[\s\S]*?border-left:/.test(css));
t('bubble secondary hidden state is explicit', /\.ai-assistant-panel-bubble-action-more-secondary\[hidden\]\s*\{/.test(css));
t('secondary chevron rotates when expanded', /secondary-toggle\[aria-expanded="true"\][\s\S]*?transform:\s*rotate\(180deg\)/.test(css));
t('touch menu items retain larger targets', /@media \(pointer: coarse\)[\s\S]*?bubble-action-more-menu[\s\S]*?min-height:\s*2\.5rem/.test(css));
t('obsolete feedback-specific More CSS is removed', !css.includes('.ai-assistant-fbk-more-trigger') && !css.includes('.ai-assistant-fbk-more-body'));

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
