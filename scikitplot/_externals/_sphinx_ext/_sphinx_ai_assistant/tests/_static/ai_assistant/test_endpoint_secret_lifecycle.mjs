// Endpoint bearer tokens never rest in browser storage (AIA-021).
//
// Two properties, both about the raw bytes under 'ai-assistant-ep-custom':
//
//   1. Nothing written at runtime contains a token. A token may live in the
//      profile object for this page only.
//   2. Nothing read at load is left behind containing one. A blob from an
//      earlier schema is removed; a current-schema blob that carries a field
//      outside the persisted set is rewritten before load returns.
//
// The second matters as much as the first. A loader that only ignores a stored
// token in memory leaves the credential readable by any same-origin script for
// as long as the visitor does not happen to edit a profile.
//
// Usage: node <this file> <path to ai-assistant.js>   (defaults to the runtime)
import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(
  process.argv[2] || new URL('../../../_static/ai-assistant.js', import.meta.url),
  'utf8',
);
let pass = 0, fail = 0;
const ok = (cond, name) => { if (cond) pass++; else { fail++; console.log('FAIL ' + name); } };
const eq = (got, want, name) => {
  if (got === want) pass++;
  else { fail++; console.log(`FAIL ${name}\n  got: ${got}\n  want: ${want}`); }
};

const start = src.indexOf('var _EP = (function () {');
const endMarker = '\n    }());';
const end = src.indexOf(endMarker, start);
if (start < 0 || end < 0) throw new Error('Could not locate _EP registry');
const block = src.slice(start, end + endMarker.length);

const KEY = 'ai-assistant-ep-custom';
const SCHEMA = Number((/var _SCHEMA_VER\s*=\s*(\d+);/.exec(block) || [])[1]);
ok(Number.isInteger(SCHEMA) && SCHEMA >= 9, 'registry declares its storage schema version');

// Built from pieces so this file holds no credential-shaped literal.
const LEGACY_SECRET = ['legacy', 'bearer', 'value', '0001'].join('-');
const RUNTIME_SECRET = ['runtime', 'bearer', 'value', '0002'].join('-');

/** Load the registry over the given storage; return it with the storage map. */
function boot(initial, { allowRuntimeTokens = false, writes = [] } = {}) {
  const store = new Map(Object.entries(initial || {}));
  const context = {
    URL,
    decodeURIComponent,
    console: { warn() {}, log() {}, error() {} },
    CustomEvent: function (type, init) { this.type = type; this.detail = init && init.detail; },
    document: { dispatchEvent() {} },
    localStorage: {
      getItem(k) { return store.has(k) ? store.get(k) : null; },
      setItem(k, v) { writes.push(['set', k]); store.set(k, String(v)); },
      removeItem(k) { writes.push(['remove', k]); store.delete(k); },
    },
    window: {
      AI_ASSISTANT_CONFIG: { allowRuntimeTokens },
      AI_ASSISTANT_ENDPOINT_DEFAULT: 'default',
      AI_ASSISTANT_ENDPOINTS: { default: { label: 'Default', base: 'https://proxy.example.com' } },
    },
  };
  context.window.window = context.window;
  vm.createContext(context);
  vm.runInContext(block, context);
  return { EP: context._EP, store, writes };
}

const customKeys = EP => EP.listCustom().map(row => row.key);
const blob = payload => JSON.stringify(payload);
const profile = extra => Object.assign({ label: 'Mine', base: 'https://safe.example.com', share: 'v1/share' }, extra);

// ── 1. An earlier schema is removed, not migrated and not left ──────────────
{
  const { EP, store } = boot({
    [KEY]: blob({ _v: SCHEMA - 1, profiles: { mine: profile({ shareToken: LEGACY_SECRET, feedbackToken: LEGACY_SECRET }) }, meta: {} }),
  });
  ok(!store.has(KEY), 'earlier-schema blob is removed from storage on load');
  ok(![...store.values()].some(v => v.includes(LEGACY_SECRET)), 'no stored value still contains the legacy token');
  eq(customKeys(EP).length, 0, 'earlier-schema profiles are not migrated into the registry');
}
{
  const { store } = boot({ [KEY]: blob({ profiles: { mine: profile({ shareToken: LEGACY_SECRET }) } }) });
  ok(!store.has(KEY), 'unversioned blob is removed from storage on load');
}

// ── 2. Anything unreadable as the current schema is removed ─────────────────
{
  const { store } = boot({ [KEY]: '{"_v":' + SCHEMA + ',"profiles":{"mine":{"shareToken":"' + LEGACY_SECRET });
  ok(!store.has(KEY), 'truncated JSON is removed rather than left with its contents');
}
{
  const { store } = boot({ [KEY]: blob({ _v: SCHEMA, profiles: [profile({ shareToken: LEGACY_SECRET })], meta: {} }) });
  ok(!store.has(KEY), 'current-schema blob with a non-object profile table is removed');
}
{
  const { store } = boot({ [KEY]: blob([{ shareToken: LEGACY_SECRET }]) });
  ok(!store.has(KEY), 'non-object blob is removed');
}

// ── 3. A current-schema blob carrying a token is rewritten at load ──────────
{
  const { EP, store } = boot({
    [KEY]: blob({ _v: SCHEMA, profiles: { mine: profile({ shareToken: LEGACY_SECRET }) }, meta: {} }),
  }, { allowRuntimeTokens: true });
  const stored = store.get(KEY) || '';
  ok(customKeys(EP).includes('mine'), 'the profile itself survives');
  ok(!stored.includes(LEGACY_SECRET), 'stored blob no longer contains the token after load');
  ok(!stored.includes('shareToken'), 'stored blob has no token field at all, not an empty one');
  eq(JSON.parse(stored)._v, SCHEMA, 'rewritten blob is the current schema');
  EP.setActive('mine');
  eq(EP.resolveToken('shareToken'), '', 'a token found in storage is not adopted into memory either');
}

// ── 4. Any field outside the persisted set triggers the rewrite ─────────────
{
  const { store } = boot({
    [KEY]: blob({ _v: SCHEMA, profiles: { mine: profile({ feedback: 'v1/feedback', feedbackToken: LEGACY_SECRET }) }, meta: {} }),
  });
  const stored = store.get(KEY) || '';
  ok(!stored.includes(LEGACY_SECRET) && !stored.includes('feedbackToken'), 'retired token field is scrubbed');
  ok(!stored.includes('"feedback"'), 'retired non-token field is scrubbed with it');
}

// ── 5. A rejected entry does not stay in storage beside the accepted ones ───
{
  const { EP, store } = boot({
    [KEY]: blob({ _v: SCHEMA, profiles: {
      private_one: { label: 'x', base: 'https://127.0.0.1/private', shareToken: LEGACY_SECRET },
      mine: profile(),
    }, meta: {} }),
  });
  const stored = store.get(KEY) || '';
  ok(customKeys(EP).includes('mine') && !customKeys(EP).includes('private_one'), 'rejected entry is not loaded');
  ok(!stored.includes('private_one') && !stored.includes(LEGACY_SECRET), 'rejected entry and its token are gone from storage');
}

// ── 6. A clean blob is left exactly as it is ────────────────────────────────
{
  const first = boot({}, { allowRuntimeTokens: true });
  ok(first.EP.addProfile('mine', profile({ shareToken: RUNTIME_SECRET })).ok, 'runtime profile with a token is accepted');
  const written = first.store.get(KEY);
  const writes = [];
  const second = boot({ [KEY]: written }, { writes });
  eq(second.store.get(KEY), written, 'reloading what the registry wrote changes no byte');
  eq(writes.filter(w => w[1] === KEY).length, 0, 'and performs no write to the profile key');
}

// ── 7. A runtime token is page memory only ──────────────────────────────────
{
  const { EP, store } = boot({}, { allowRuntimeTokens: true });
  ok(EP.addProfile('mine', profile({ shareToken: RUNTIME_SECRET })).ok, 'profile added');
  EP.setActive('mine');
  eq(EP.resolveToken('shareToken'), RUNTIME_SECRET, 'the token is usable on this page');
  const everything = [...store.values()].join('\n');
  ok(!everything.includes(RUNTIME_SECRET), 'the token is in no stored value');
  ok(!(store.get(KEY) || '').includes('shareToken'), 'the stored profile has no token field');
  ok(!EP.exportCustomJson().includes(RUNTIME_SECRET), 'the export does not carry the token');

  ok(EP.addProfile('mine', profile({ label: 'Renamed', shareToken: RUNTIME_SECRET })).ok, 'profile updated');
  ok(![...store.values()].join('\n').includes(RUNTIME_SECRET), 'an update does not persist the token either');

  const reloaded = boot(Object.fromEntries(store), { allowRuntimeTokens: true });
  reloaded.EP.setActive('mine');
  ok(customKeys(reloaded.EP).includes('mine'), 'the profile is there after a reload');
  eq(reloaded.EP.resolveToken('shareToken'), '', 'the token is not: it did not outlive the page');
}

// ── 8. With runtime tokens disallowed, a token is not kept even in memory ───
{
  const { EP, store } = boot({}, { allowRuntimeTokens: false });
  ok(EP.addProfile('mine', profile({ shareToken: RUNTIME_SECRET })).ok, 'profile added with tokens disallowed');
  EP.setActive('mine');
  eq(EP.resolveToken('shareToken'), '', 'token is not resolvable');
  ok(![...store.values()].join('\n').includes(RUNTIME_SECRET), 'and not stored');
}

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
