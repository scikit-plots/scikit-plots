// Run 87-89 compatibility: Feedback center is primary in feedback; global Home
// belongs to the canonical answer More menu.
import fs from 'node:fs'; const src=fs.readFileSync(process.argv[2],'utf8'); let pass=0,fail=0;
function t(n,g){if(g)pass++;else{fail++;console.log(`FAIL ${n}`)}}
function ex(n){const i=src.indexOf('function '+n+'(');let d=0,s=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;s=true}else if(src[j]==='}'){d--;if(s&&d===0)return src.slice(i,j+1)}}}
const f=ex('_buildFbkFloat'), m=ex('_buildBubbleMore');
const detail=f.indexOf("label: 'Detailed feedback'"), center=f.indexOf("label: 'Feedback center\\u2026'"), contribute=f.indexOf("label: 'Contribute this Q&A\\u2026'");
t('feedback center primary order', detail>=0&&center>detail&&contribute>center);
t('feedback center unique', (f.match(/label: 'Feedback center\\u2026'/g)||[]).length===1);
t('Home absent from feedback popup', !f.includes("label: 'Home'"));
t('Home present once in answer menu', (m.match(/homeMenuLbl\.textContent = 'Home'/g)||[]).length===1);
t('Home internal hook preserved', m.includes("new CustomEvent('ai-assistant-open-home'"));
t('feedback hook preserved', f.includes("new CustomEvent('ai-assistant-open-feedback-center'"));
console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
