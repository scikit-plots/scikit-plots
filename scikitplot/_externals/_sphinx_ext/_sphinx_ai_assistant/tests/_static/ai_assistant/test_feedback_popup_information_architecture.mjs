// Run 86-89 compatibility regression: feedback popup remains a focused feedback
// surface while bookmark persistence stays privacy-minimal in answer actions.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8');
let pass=0, fail=0;
function t(n,g,w=true){if(g===w)pass++;else{fail++;console.log(`FAIL ${n}`)}}
function extract(name){const i=src.indexOf('function '+name+'(');let d=0,st=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;st=true}else if(src[j]==='}'){d--;if(st&&d===0)return src.slice(i,j+1)}}throw Error(name)}
const fbk=extract('_buildFbkFloat'), more=extract('_buildBubbleMore'), read=extract('_readAnswerBookmarkIds'), set=extract('_setAnswerBookmarked');
const d=fbk.indexOf("label: 'Detailed feedback'"), c=fbk.indexOf("label: 'Feedback center\\u2026'"), q=fbk.indexOf("label: 'Contribute this Q&A\\u2026'");
t('focused feedback order', d>=0&&c>d&&q>c);
t('old mixed label removed', !fbk.includes('Manage feedback & sharing'));
t('old coming soon removed', !fbk.includes('Coming soon: Flag'));
t('feedback popup excludes bookmark', !fbk.includes('Bookmark answer'));
t('feedback popup excludes Home', !fbk.includes("label: 'Home'"));
t('feedback popup excludes duplicate More', !fbk.includes('ai-assistant-fbk-more-trigger'));
t('bookmark moved to bubble menu', more.includes("'Bookmark answer'"));
t('Home moved to bubble menu', more.includes("homeMenuLbl.textContent = 'Home'"));
t('bookmark key versioned', src.includes("_ANSWER_BOOKMARKS_KEY = 'ai-assistant-answer-bookmarks-v1'"));
t('bookmark storage bounded', src.includes('_ANSWER_BOOKMARKS_MAX = 500'));
t('bookmark reader validates markers', read.includes('/^abm-[0-9a-f]{8}-[0-9a-z]+$/i'));
t('bookmark writer stores ids only', set.includes('ids: ids')&&!set.includes('answerText'));
t('popup rows remain native full width', /\.ai-assistant-fbk-popup-row\s*\{[\s\S]*?width:\s*100%;[\s\S]*?border:\s*0;/.test(css));
t('obsolete feedback More css removed', !css.includes('.ai-assistant-fbk-more-trigger'));
console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
