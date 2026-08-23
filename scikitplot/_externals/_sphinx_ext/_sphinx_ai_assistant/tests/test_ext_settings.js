/* Zero-dependency test harness for _ext_settings.js.
 *
 * Runs the EXACT delivered file (no test-only build) against stubbed browser
 * globals, exercising validation, prototype-pollution resistance, versioned
 * export/import, behaviour-neutral legacy migration, pub/sub, and the
 * storage-unavailable fallback. Run: `node test_ext_settings.js`.
 */
'use strict';
const path = require('path');
const MODULE_PATH = path.join(__dirname, '_ext_settings.js');

let passed = 0, failed = 0;
const failures = [];
function ok(cond, msg) {
    if (cond) { passed++; }
    else { failed++; failures.push(msg); console.log('  ✗ ' + msg); }
}
function eq(a, b, msg) { ok(JSON.stringify(a) === JSON.stringify(b), msg + `  (got ${JSON.stringify(a)}, want ${JSON.stringify(b)})`); }

// A minimal in-memory localStorage that can also simulate being unavailable.
function makeStorage(initial, opts) {
    opts = opts || {};
    const store = Object.assign(Object.create(null), initial || {});
    return {
        getItem(k) { if (opts.throwOnGet) throw new Error('blocked'); return k in store ? store[k] : null; },
        setItem(k, v) { if (opts.throwOnSet) throw new Error('blocked'); store[k] = String(v); },
        removeItem(k) { delete store[k]; },
        _dump() { return Object.assign({}, store); },
    };
}

// Build a fresh window, install as globals, bust require cache, load the module.
function freshModule(env) {
    env = env || {};
    const events = [];
    const win = {
        AI_ASSISTANT: undefined,
        localStorage: env.noStorage ? undefined : (env.storage || makeStorage()),
        document: {
            dispatchEvent(ev) { events.push(ev); return true; },
        },
        CustomEvent: function (type, init) { this.type = type; this.detail = init && init.detail; },
    };
    if (env.noDocument) { win.document = undefined; }
    global.window = win;
    global.module = require.cache; // placeholder; real module object set by require
    delete require.cache[MODULE_PATH];
    const api = require(MODULE_PATH);
    return { api, events, storage: win.localStorage, win };
}

function section(name, fn) { console.log('\n### ' + name); fn(); }

// ── 1. Defaults when storage is empty ────────────────────────────────────────
section('defaults', () => {
    const { api } = freshModule();
    eq(api.get('streaming'), true, 'streaming defaults true');
    eq(api.get('shareLinkMode'), true, 'shareLinkMode defaults true');
    eq(api.get('feedbackPersist'), true, 'feedbackPersist defaults true');
    eq(api.get('datasetRepo'), '', 'datasetRepo defaults empty string');
    eq(api.get('nope'), undefined, 'unknown key returns undefined');
    eq(api.has('streaming'), true, 'has() true for defined');
    eq(api.has('nope'), false, 'has() false for undefined');
});

// ── 2. Validated writes ──────────────────────────────────────────────────────
section('validated writes', () => {
    const { api } = freshModule();
    ok(api.set('streaming', false).ok, 'set boolean false ok');
    eq(api.get('streaming'), false, 'value updated');
    const bad = api.set('streaming', 'yes');
    ok(!bad.ok, 'set boolean with junk string rejected');
    eq(api.get('streaming'), false, 'value unchanged after rejected write');
    ok(api.set('streaming', 'true').ok, "coerce 'true' string → boolean true");
    eq(api.get('streaming'), true, "coerced value stored as real boolean");
    ok(!api.set('nope', 1).ok, 'set unknown key rejected');
});

// ── 3. Custom types: enum / integer / string pattern ─────────────────────────
section('type system', () => {
    const { api } = freshModule();
    ok(api.define('effort', { type: 'enum', values: ['low', 'medium', 'high'], default: 'medium' }).ok, 'define enum');
    ok(api.define('tokens', { type: 'integer', min: 0, max: 8192, default: 1024 }).ok, 'define integer');
    ok(api.set('effort', 'high').ok, 'enum accepts allowed');
    ok(!api.set('effort', 'ultra').ok, 'enum rejects out-of-set');
    ok(!api.set('tokens', 99999).ok, 'integer rejects above max');
    ok(!api.set('tokens', 3.5).ok, 'integer rejects non-integer');
    ok(api.set('tokens', '2048').ok, 'integer coerces numeric string');
    eq(api.get('tokens'), 2048, 'coerced int stored as number');
    // invalid default refused at define time
    ok(!api.define('bad', { type: 'enum', values: ['a'], default: 'z' }).ok, 'reject invalid default');
    ok(api.define('newBool', { type: 'boolean', default: true }).ok, 'define a fresh boolean');
    ok(!api.define('effort', { type: 'boolean', default: true }).ok, 'redefining existing key rejected');
    ok(!api.define('constructor', { type: 'boolean', default: true }).ok, 'define(constructor) rejected (reserved)');
});

// ── 4. datasetRepo pattern ───────────────────────────────────────────────────
section('datasetRepo validation', () => {
    const { api } = freshModule();
    ok(api.set('datasetRepo', 'scikit-plots/ai-dataset').ok, 'valid owner/name accepted');
    ok(!api.set('datasetRepo', 'no-slash').ok, 'missing slash rejected');
    ok(!api.set('datasetRepo', 'a/b/c').ok, 'double slash rejected');
    ok(!api.set('datasetRepo', 'javascript:alert(1)/x').ok, 'scheme-ish junk rejected');
    ok(api.set('datasetRepo', '').ok, 'empty (unset) accepted');
    eq(api.get('datasetRepo'), '', 'empty stored');
});

// ── 5. Prototype-pollution resistance ────────────────────────────────────────
section('prototype pollution', () => {
    const { api } = freshModule();
    ok(!api.define('__proto__', { type: 'boolean', default: true }).ok, 'define(__proto__) rejected');
    ok(!api.define('constructor', { type: 'boolean', default: true }).ok, 'define(constructor) rejected');
    // Raw string: a `{__proto__: …}` object LITERAL sets the prototype, so it
    // would never survive JSON.stringify. JSON.parse of this raw text DOES
    // create an own `__proto__` key, which is the real attack surface.
    const r = api.importJSON('{"_v":1,"settings":{"__proto__":{"polluted":1},"constructor":{"x":1},"streaming":false}}');
    ok(r.ok, 'import runs');
    ok(r.applied.indexOf('streaming') !== -1, 'safe key applied');
    ok(r.skipped.some(s => s.key === '__proto__'), '__proto__ key skipped');
    eq(({}).polluted, undefined, 'Object.prototype NOT polluted');
    eq(api.get('streaming'), false, 'legit key from same import applied');
});

// ── 6. Export / import round-trip ────────────────────────────────────────────
section('export/import round-trip', () => {
    const { api } = freshModule();
    api.set('streaming', false);
    api.set('datasetRepo', 'org/repo');
    const json = api.exportJSON();
    const { api: api2 } = freshModule();      // clean instance
    const res = api2.importJSON(json);
    ok(res.ok, 'import ok');
    eq(res.skipped.length, 0, 'nothing skipped on clean round-trip');
    eq(api2.get('streaming'), false, 'streaming round-tripped');
    eq(api2.get('datasetRepo'), 'org/repo', 'datasetRepo round-tripped');
});

// ── 7. Import re-validates every entry ───────────────────────────────────────
section('import re-validation', () => {
    const { api } = freshModule();
    const res = api.importJSON(JSON.stringify({
        _v: 1, settings: { streaming: false, feedbackPersist: 'garbage', unknownKey: 1, datasetRepo: 'ok/repo' },
    }));
    ok(res.applied.indexOf('streaming') !== -1, 'valid boolean applied');
    ok(res.applied.indexOf('datasetRepo') !== -1, 'valid repo applied');
    ok(res.skipped.some(s => s.key === 'feedbackPersist'), 'invalid value skipped');
    ok(res.skipped.some(s => s.key === 'unknownKey'), 'unknown key skipped');
    eq(api.get('feedbackPersist'), true, 'rejected value left at default');
    ok(!api.importJSON('not json').ok, 'non-JSON rejected');
    ok(!api.importJSON(JSON.stringify({ _v: 1 })).ok, 'missing settings rejected');
    ok(!api.importJSON(JSON.stringify([])).ok, 'array rejected');
});

// ── 8. Behaviour-neutral legacy migration ────────────────────────────────────
section('legacy migration', () => {
    // Legacy keys with their historical string encodings; NO new blob yet.
    const storage = makeStorage({
        'ai-assistant-streaming-on': 'false',        // → false
        'ai-assistant-export-link-mode': 'false',    // stored!=='true' → false
        'ai-assistant-feedback-persist': 'false',    // → false
        'ai-assistant-custom-dataset-repo': 'owner/name',
    });
    const { api } = freshModule({ storage });
    eq(api.get('streaming'), false, 'streaming migrated from legacy false');
    eq(api.get('shareLinkMode'), false, 'shareLinkMode migrated (exact === "true" rule)');
    eq(api.get('feedbackPersist'), false, 'feedbackPersist migrated from legacy false');
    eq(api.get('datasetRepo'), 'owner/name', 'datasetRepo migrated');
    // legacy keys are NOT deleted (rollback-safe)
    const dump = storage._dump();
    ok('ai-assistant-streaming-on' in dump, 'legacy key preserved (rollback-safe)');
    // new blob written
    ok('ai-assistant-ext' in dump, 'consolidated blob written on migration');
    const blob = JSON.parse(dump['ai-assistant-ext']);
    eq(blob._v, 1, 'blob carries schema version');
    eq(blob.settings.streaming, false, 'blob holds migrated value');

    // Exact-semantics edge: export-link-mode 'anything' → false (only 'true' is true)
    const storage2 = makeStorage({ 'ai-assistant-export-link-mode': 'weird' });
    const { api: api2 } = freshModule({ storage: storage2 });
    eq(api2.get('shareLinkMode'), false, "shareLinkMode: non-'true' legacy → false (exact rule)");

    // Invalid legacy repo id → '' (not a broken value)
    const storage3 = makeStorage({ 'ai-assistant-custom-dataset-repo': 'not a repo id' });
    const { api: api3 } = freshModule({ storage: storage3 });
    eq(api3.get('datasetRepo'), '', 'invalid legacy repo id normalised to empty');
});

// ── 9. Blob precedence over legacy; future version untouched ──────────────────
section('precedence & forward-compat', () => {
    // Blob says streaming:true, legacy says false → blob wins.
    const storage = makeStorage({
        'ai-assistant-ext': JSON.stringify({ _v: 1, settings: { streaming: true } }),
        'ai-assistant-streaming-on': 'false',
    });
    const { api } = freshModule({ storage });
    eq(api.get('streaming'), true, 'versioned blob wins over legacy key');

    // Future schema version: leave intact, use defaults in memory.
    const futureBlob = JSON.stringify({ _v: 99, settings: { streaming: false } });
    const storage2 = makeStorage({ 'ai-assistant-ext': futureBlob });
    const { api: api2 } = freshModule({ storage: storage2 });
    eq(api2.get('streaming'), true, 'future-version blob ignored → default used');
    eq(storage2._dump()['ai-assistant-ext'], futureBlob, 'future-version blob left byte-identical');
});

// ── 10. reset / resetAll ─────────────────────────────────────────────────────
section('reset', () => {
    const { api } = freshModule();
    api.set('streaming', false); api.set('datasetRepo', 'a/b');
    api.reset('streaming');
    eq(api.get('streaming'), true, 'reset() restores default');
    eq(api.get('datasetRepo'), 'a/b', 'reset() only touches its key');
    api.resetAll();
    eq(api.get('datasetRepo'), '', 'resetAll() restores all defaults');
});

// ── 11. Pub/sub ──────────────────────────────────────────────────────────────
section('pub/sub', () => {
    const { api, events } = freshModule();
    let calls = [];
    const off = api.onChange(s => calls.push(s));
    api.set('streaming', false);
    eq(calls.length, 1, 'listener fired on change');
    eq(calls[0], { key: 'streaming', value: false }, 'listener got {key,value}');
    api.set('streaming', false);           // no-op
    eq(calls.length, 1, 'listener NOT fired on no-op write');
    ok(events.some(e => e.type === 'ai-assistant:setting-changed'), 'document CustomEvent dispatched');
    off();
    api.set('streaming', true);
    eq(calls.length, 1, 'unsubscribe stops delivery');
});

// ── 12. Storage-unavailable fallback (private mode) ──────────────────────────
section('no-storage fallback', () => {
    const { api } = freshModule({ noStorage: true });
    eq(api.get('streaming'), true, 'defaults available with no storage');
    ok(api.set('streaming', false).ok, 'set works in-memory with no storage');
    eq(api.get('streaming'), false, 'in-memory value retained');
    ok(api.exportJSON().indexOf('streaming') !== -1, 'export works with no storage');
    // throwing storage must not crash
    const throwing = makeStorage({}, { throwOnGet: true, throwOnSet: true });
    const { api: api2 } = freshModule({ storage: throwing });
    ok(api2.set('streaming', false).ok, 'set tolerates throwing storage');
});

// ── 13. Frozen snapshots ─────────────────────────────────────────────────────
section('immutability', () => {
    const { api } = freshModule();
    const snap = api.all();
    try { snap.streaming = 'mutated'; } catch (_) {}
    eq(api.get('streaming'), true, 'all() snapshot is frozen — no write-through');
    const d = api.describe('datasetRepo');
    ok(d && d.type === 'string' && d.def === '', 'describe() exposes schema for UI generation');
});

// ── Summary ──────────────────────────────────────────────────────────────────
console.log(`\n${'='.repeat(60)}\n${passed} passed, ${failed} failed\n${'='.repeat(60)}`);
if (failed) { failures.forEach(f => console.log('FAIL: ' + f)); process.exit(1); }
