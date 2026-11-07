import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }

function node(tag) {
  return {
    tagName: String(tag).toUpperCase(), children: [], parentNode: null,
    textContent: '', className: '', id: '', attrs: {}, listeners: {},
    setAttribute(k,v){this.attrs[k]=String(v);},
    appendChild(c){c.parentNode=this; this.children.push(c); return c;},
    addEventListener(e,fn){(this.listeners[e]=this.listeners[e]||[]).push(fn);},
    focus(){ document.activeElement=this; },
    remove(){ if(this.parentNode){ const i=this.parentNode.children.indexOf(this); if(i>=0)this.parentNode.children.splice(i,1); } },
    async click(){ for(const fn of (this.listeners.click||[])) await fn({target:this,preventDefault(){},stopPropagation(){}}); }
  };
}
function allText(n) { return [n.textContent, ...(n.children||[]).map(allText)].join(' '); }
function findText(n, text) {
  if (n.textContent === text) return n;
  for (const c of n.children || []) { const f=findText(c,text); if(f) return f; }
  return null;
}
const body = node('body');
const docListeners = {};
globalThis.document = {
  body, activeElement: null,
  createElement: node,
  createTextNode(t){const n=node('#text'); n.textContent=String(t); return n;},
  addEventListener(e,fn){(docListeners[e]=docListeners[e]||[]).push(fn);},
  removeEventListener(e,fn){ if(!docListeners[e]) return; docListeners[e]=docListeners[e].filter(x=>x!==fn); }
};

const start = src.indexOf('var _SECRET_PATTERNS = [');
const end = src.indexOf('var _HIDDEN_SELECTORS = [', start);
if (start < 0 || end <= start) throw new Error('helper block not found');
const ctx = { Promise, console, document: globalThis.document };
vm.createContext(ctx);
vm.runInContext(src.slice(start,end) + `\nthis.__api={review:_privacyPreflightReview};`, ctx);

const secret='sk-'+ 'Z'.repeat(32);
const email='private.person@example.com';
const original=`Please use ${secret} and contact ${email} \u202Ehidden\u202C`;
const promise = ctx.__api.review(original, {
  destination:'the external test endpoint', cancelLabel:'Edit', continueLabel:'Send unchanged'
});
await Promise.resolve();
ok(body.children.length === 1, 'flagged data opens one modal overlay');
const rendered = allText(body);
ok(rendered.includes('Review before continuing'), 'dialog title rendered');
ok(rendered.includes('Credential-like data'), 'credential category rendered');
ok(rendered.includes('Possible personal information'), 'personal-data category rendered');
ok(rendered.includes('Invisible / bidi controls'), 'control category rendered');
ok(rendered.includes('U+202E'), 'specific bidi codepoint rendered');
ok(!rendered.includes(secret), 'dialog never renders secret value');
ok(!rendered.includes(email), 'dialog never renders personal value');
ok(rendered.includes('Detection is advisory and incomplete'), 'dialog states detection limitation');
const redactBtn = findText(body, 'Redact & continue');
ok(!!redactBtn, 'redact action exists');
await redactBtn.click();
const decision = await promise;
ok(decision.action === 'redact', 'redact action resolves explicitly');
ok(!decision.value.includes(secret), 'redacted decision excludes secret');
ok(!decision.value.includes(email), 'redacted decision excludes email');
ok(!decision.value.includes('\u202E'), 'redacted decision excludes bidi control');
ok(body.children.length === 0, 'dialog removes itself after decision');

const cleanPromise = ctx.__api.review('ordinary question', {destination:'test'});
const cleanDecision = await cleanPromise;
ok(cleanDecision.action === 'continue', 'clean text continues without prompt');
ok(body.children.length === 0, 'clean text does not create modal');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
