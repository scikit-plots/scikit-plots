// Run 88-89 compatibility: Bookmark remains answer-scoped but now lives in the
// existing answer More menu before the secondary More -> Home disclosure.
import fs from 'node:fs'; const src=fs.readFileSync(process.argv[2],'utf8'); let pass=0,fail=0;
function t(n,g){if(g)pass++;else{fail++;console.log(`FAIL ${n}`)}}
function ex(n){const i=src.indexOf('function '+n+'(');let d=0,s=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;s=true}else if(src[j]==='}'){d--;if(s&&d===0)return src.slice(i,j+1)}}}
const f=ex('_buildFbkFloat'), m=ex('_buildBubbleMore');
const b=m.indexOf("bookmarkMenuLbl.textContent = bookmarked ? 'Remove bookmark' : 'Bookmark answer'"), sep=m.indexOf("secondarySep.className = 'ai-assistant-panel-bubble-action-more-sep'"), more=m.indexOf("secondaryLabel.textContent = 'More'"), home=m.indexOf("homeMenuLbl.textContent = 'Home'");
t('feedback popup keeps only feedback/contribution', !f.includes('Bookmark answer')&&!f.includes("label: 'Home'"));
t('Bookmark is in answer menu', b>=0);
t('divider follows Bookmark', sep>b);
t('secondary More follows divider', more>sep);
t('Home follows secondary More', home>more);
t('bookmark reversible', m.includes("bookmarkMenuBtn.setAttribute('aria-pressed', next ? 'true' : 'false')"));
t('bookmark privacy store retained', src.includes("_ANSWER_BOOKMARKS_KEY = 'ai-assistant-answer-bookmarks-v1'")&&src.includes('_ANSWER_BOOKMARKS_MAX = 500'));
console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
