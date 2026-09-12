import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(process.argv[2], 'utf8');
let pass = 0, fail = 0;
const ok = (cond, name) => { if (cond) pass++; else { fail++; console.log('FAIL ' + name); } };
const eq = (got, want, name) => { if (got === want) pass++; else { fail++; console.log(`FAIL ${name}\n  got: ${got}\n  want: ${want}`); } };

const start = src.indexOf('var _EP = (function () {');
const endMarker = '\n    }());';
const end = src.indexOf(endMarker, start);
if (start < 0 || end < 0) throw new Error('Could not locate _EP registry');
const block = src.slice(start, end + endMarker.length);

const warnings = [];
const store = new Map();
store.set('ai-assistant-ep-custom', JSON.stringify({
  _v: 1,
  profiles: {
    stale_private: { label: 'stale', base: 'https://127.0.0.1/private' },
    stale_safe: { label: 'safe', base: 'https://safe.example.com', share: 'v1/share' },
  },
  meta: {},
}));
const context = {
  URL,
  decodeURIComponent,
  console: { warn: (...args) => warnings.push(args.join(' ')), log() {}, error() {} },
  CustomEvent: function(type, init) { this.type = type; this.detail = init && init.detail; },
  document: { dispatchEvent() {} },
  localStorage: {
    getItem(k) { return store.has(k) ? store.get(k) : null; },
    setItem(k, v) { store.set(k, String(v)); },
    removeItem(k) { store.delete(k); },
  },
  window: {
    AI_ASSISTANT_ENDPOINT_DEFAULT: 'default',
    AI_ASSISTANT_ENDPOINTS: {
      default: { label: 'Default', base: 'https://proxy.example.com' },
    },
  },
};
context.window.window = context.window;
vm.createContext(context);
vm.runInContext(block, context);
const EP = context._EP;

ok(!!EP && typeof EP.validateUrl === 'function' && typeof EP.validateEndpoint === 'function', 'security validators are public');
eq(EP.validateUrl('  https://EXAMPLE.com:443/api/  ').url, 'https://example.com/api', 'absolute URL canonicalises host/default port/trailing slash');
ok(!EP.validateUrl('https://user:secret@example.com/api').ok, 'embedded URL credentials rejected');
ok(!EP.validateUrl('https://good.example.com/api#fragment').ok, 'URL fragments rejected');
ok(!EP.validateUrl('https://good.example.com/base?tenant=x').ok, 'Base query strings rejected');
ok(EP.validateEndpoint('https://good.example.com/v1/chat?tenant=x').ok, 'absolute feature endpoint may carry non-secret bounded query');
ok(!EP.validateEndpoint('https://good.example.com/v1/chat?api_key=SECRET').ok, 'absolute endpoint rejects secret-like query key');
ok(!EP.validateEndpoint('v1/chat?access_token=SECRET').ok, 'relative endpoint rejects secret-like query key');
ok(!EP.validateUrl('https://127.0.0.1/api').ok, 'IPv4 loopback rejected');
ok(!EP.validateUrl('http://2130706433/api').ok, 'non-canonical IPv4 loopback spelling rejected after URL canonicalisation');
ok(!EP.validateUrl('https://[::1]/api').ok, 'IPv6 loopback rejected');
ok(!EP.validateUrl('https://[::ffff:127.0.0.1]/api').ok, 'IPv4-mapped IPv6 rejected');
ok(!EP.validateUrl('https://metadata.google.internal/compute').ok, 'cloud metadata hostname rejected');
ok(!EP.validateUrl('https://service.internal/v1').ok, 'internal DNS suffix rejected');
ok(!EP.validateUrl('https://kubernetes/v1').ok, 'bare internal hostname rejected');
ok(!EP.validateUrl('https://good.example.com/\\evil').ok, 'absolute backslash ambiguity rejected');
ok(!EP.validateUrl('https://good.example.com/%2e%2e/admin').ok, 'absolute encoded parent traversal rejected before URL normalization');
ok(!EP.validateUrl('https://good.example.com/v1/%2Fadmin').ok, 'absolute encoded path separator rejected');
ok(!EP.validateUrl('https://good.example.com/v1?x=%0d%0aevil').ok, 'absolute encoded control character rejected');
ok(!EP.validateUrl('https://good.example.com/\u202Eevil').ok, 'bidi control rejected');
ok(!EP.validateEndpoint('//evil.example.com/v1').ok, 'protocol-relative endpoint rejected');
ok(!EP.validateEndpoint('../admin').ok, 'literal parent traversal rejected');
ok(!EP.validateEndpoint('%2e%2e/admin').ok, 'encoded parent traversal rejected');
ok(!EP.validateEndpoint('v1/%2Fadmin').ok, 'encoded path separator rejected');
ok(!EP.validateEndpoint('v1/%252e%252e/admin').ok, 'nested traversal encoding rejected');
ok(!EP.validateEndpoint('v1/share#frag').ok, 'relative fragment rejected');
ok(!EP.validateEndpoint('v1/' + 'a'.repeat(1100)).ok, 'overlong relative route rejected');
ok(!EP.validateUrl('https://good.example.com/' + 'a'.repeat(2100)).ok, 'overlong absolute URL rejected');

const added = EP.addProfile('malicious', { base: 'https://safe.example.com@127.0.0.1/api' });
ok(!added.ok, 'malicious runtime profile is rejected');
ok(warnings.some(x => x.includes('[endpoint-security]')), 'rejection emits privacy-safe security diagnostic');
ok(!warnings.some(x => x.includes('127.0.0.1') || x.includes('safe.example.com@')), 'security diagnostics never echo rejected URL');

const listed = EP.list().map(x => x.key);
ok(!listed.includes('stale_private'), 'unsafe persisted profile is discarded on load');
ok(listed.includes('stale_safe'), 'safe persisted profile survives re-sanitization');
EP.setActive('stale_safe');
eq(EP.resolveEndpoint('share'), 'https://safe.example.com/v1/share', 're-sanitized persisted relative route resolves safely');

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
