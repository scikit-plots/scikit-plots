import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(process.argv[2], 'utf8');
let pass = 0, fail = 0;
function ok(cond, name) {
  if (cond) pass++;
  else { fail++; console.log('FAIL ' + name); }
}

const start = src.indexOf('    var _EP = (function () {');
const endMarker = '\n    /**\n     * Normalise a legacy flat endpoint setting';
const end = src.indexOf(endMarker, start);
ok(start >= 0 && end > start, 'endpoint registry IIFE is extractable');
const chunk = src.slice(start, end);

// Source contract: the persistence serializer itself must not mention secret
// fields. This catches future refactors that keep runtime tests green by
// clearing token values but still write secret-shaped storage fields.
const persistStart = chunk.indexOf('function _persistCustom()');
const persistEnd = chunk.indexOf('/** Dispatch ai-assistant:profile-changed', persistStart);
const persistBody = chunk.slice(persistStart, persistEnd);
ok(persistStart >= 0 && persistEnd > persistStart, 'persistence function is extractable');
ok(!/shareToken\s*:/.test(persistBody), 'shareToken field absent from persistent payload');
ok(!/feedbackToken\s*:/.test(persistBody), 'feedbackToken field absent from persistent payload');
ok(/var _SCHEMA_VER\s*=\s*3;/.test(chunk), 'endpoint storage schema is v3');
ok(/if \(needsRewrite\) _persistCustom\(\);/.test(chunk), 'legacy storage is actively rewritten');

const storage = new Map();
const customKey = 'ai-assistant-ep-custom';
storage.set(customKey, JSON.stringify({
  _v: 2,
  profiles: {
    legacy: {
      label: 'Legacy',
      base: 'https://example.com',
      chat: '', share: '', feedback: '', training: '', datasetRepo: '',
      shareToken: 'legacy-share-secret',
      feedbackToken: 'legacy-feedback-secret',
      ttlDays: 30,
    },
  },
  meta: {},
}));

const localStorage = {
  getItem(k) { return storage.has(k) ? storage.get(k) : null; },
  setItem(k, v) { storage.set(k, String(v)); },
  removeItem(k) { storage.delete(k); },
};

const context = {
  window: {
    AI_ASSISTANT_ENDPOINTS: {},
    AI_ASSISTANT_ENDPOINT_DEFAULT: '',
    AI_ASSISTANT_CONFIG: { allowRuntimeTokens: true },
  },
  localStorage,
  document: { dispatchEvent() {} },
  CustomEvent: function CustomEvent(type, init) { this.type = type; this.detail = init?.detail; },
  URL,
  console: { warn() {}, log() {}, error() {} },
};
context.window.localStorage = localStorage;
vm.createContext(context);
vm.runInContext(chunk, context, { filename: 'endpoint-registry-under-test.js' });

ok(context._EP && typeof context._EP.addProfile === 'function', 'real endpoint registry executes');

const migratedRaw = storage.get(customKey) || '';
ok(!migratedRaw.includes('legacy-share-secret'), 'legacy share token value removed from raw localStorage');
ok(!migratedRaw.includes('legacy-feedback-secret'), 'legacy feedback token value removed from raw localStorage');
ok(!/"shareToken"\s*:/.test(migratedRaw), 'legacy shareToken field removed from raw localStorage');
ok(!/"feedbackToken"\s*:/.test(migratedRaw), 'legacy feedbackToken field removed from raw localStorage');
ok(JSON.parse(migratedRaw)._v === 3, 'legacy storage rewritten as v3');

const added = context._EP.addProfile('fresh', {
  label: 'Fresh',
  base: 'https://fresh.example.com',
  chat: '', share: '', feedback: '', training: '', datasetRepo: '',
  shareToken: 'runtime-share-secret',
  feedbackToken: 'runtime-feedback-secret',
  ttlDays: 30,
});
ok(added && added.ok, 'runtime profile with session-only tokens can be added');
ok(context._EP.setActive('fresh') === true, 'runtime profile can become active');
ok(context._EP.resolveToken('shareToken') === 'runtime-share-secret', 'share token remains usable in memory');
ok(context._EP.resolveToken('feedbackToken') === 'runtime-feedback-secret', 'feedback token remains usable in memory');

const freshRaw = storage.get(customKey) || '';
ok(!freshRaw.includes('runtime-share-secret'), 'runtime share token value never persisted');
ok(!freshRaw.includes('runtime-feedback-secret'), 'runtime feedback token value never persisted');
ok(!/"shareToken"\s*:/.test(freshRaw), 'runtime shareToken field never persisted');
ok(!/"feedbackToken"\s*:/.test(freshRaw), 'runtime feedbackToken field never persisted');

const exported = context._EP.exportCustom();
ok(!exported.includes('runtime-share-secret'), 'profile export omits share token value');
ok(!exported.includes('runtime-feedback-secret'), 'profile export omits feedback token value');
ok(!/"shareToken"\s*:/.test(exported), 'profile export omits shareToken field');
ok(!/"feedbackToken"\s*:/.test(exported), 'profile export omits feedbackToken field');

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
