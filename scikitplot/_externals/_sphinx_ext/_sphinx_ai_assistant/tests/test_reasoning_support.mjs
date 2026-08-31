// Contract harness for reasoning-parameter capability resolution.
//
//   node tests/test_reasoning_support.mjs _static/ai-assistant.js
//
// The invariant that matters most: a deployment that has NOT declared support
// must send a request body byte-identical to the one the panel sent before
// these controls could reach the wire. Enabling the UI must not be able to
// break an existing site.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');

function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('not found: ' + name);
  let depth = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { depth++; started = true; }
    else if (src[j] === '}') { depth--; if (started && depth === 0) return src.slice(i, j + 1); }
  }
  throw new Error('unbalanced: ' + name);
}

function extractArray(name) {
  const i = src.indexOf('var ' + name + ' = [');
  if (i < 0) throw new Error('not found: ' + name);
  const start = src.indexOf('[', i);
  let depth = 0;
  for (let j = start; j < src.length; j++) {
    if (src[j] === '[') depth++;
    else if (src[j] === ']') { depth--; if (depth === 0) return src.slice(start, j + 1); }
  }
  throw new Error('unbalanced: ' + name);
}

function extractObject(name) {
  const i = src.indexOf('var ' + name + ' = {');
  if (i < 0) throw new Error('not found: ' + name);
  const start = src.indexOf('{', i);
  let depth = 0;
  for (let j = start; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') { depth--; if (depth === 0) return src.slice(start, j + 1); }
  }
  throw new Error('unbalanced: ' + name);
}

// ── fakes ──────────────────────────────────────────────────────────────────
let CFG = {};
globalThis._cfg = () => CFG;
globalThis._safeInt = (v, min, max, fb) => {
  const n = parseInt(v, 10);
  return (isFinite(n) && n >= min && n <= max) ? n : fb;
};
globalThis._REASONING_WIRE_DEFAULTS = (0, eval)('(' + extractObject('_REASONING_WIRE_DEFAULTS') + ')');
globalThis._EFFORT_LEVELS = (0, eval)('(' + extractArray('_EFFORT_LEVELS') + ')');
globalThis._effortMapCoversCurrentScale =
  (0, eval)('(' + extract('_effortMapCoversCurrentScale') + ')');
globalThis._THINKING_BUDGET_MIN = 500;
globalThis._THINKING_BUDGET_MAX = 16000;

let EFFORT = 'high', THINK_ON = false, BUDGET = 5000;
globalThis._getEffortLevel = () => EFFORT;
globalThis._getThinkingOn = () => THINK_ON;
globalThis._getThinkingBudget = () => BUDGET;

// _reasoningSupport now consults the discovery cache. Wire the real helpers
// (not stubs) so the precedence cases below exercise the shipped chain, and
// give the cache an empty session store so "nothing discovered" is the
// starting state.
let SS = {};
globalThis._ssGet = (k) => (k in SS ? SS[k] : null);
globalThis._ssSet = (k, v) => { SS[k] = String(v); };
globalThis._CAPS_KEY_PREFIX = 'ai-assistant-caps:';
globalThis._CAPS_TTL_MS = 15 * 60 * 1000;
globalThis.window = { location: { href: 'https://docs.example.org/page.html' } };
globalThis._EP = { hasProfiles: () => false, resolve: () => '' };
globalThis._capsOrigin = (0, eval)('(' + extract('_capsOrigin') + ')');
globalThis._capsCached = (0, eval)('(' + extract('_capsCached') + ')');
globalThis._reasoningEndpoint = (0, eval)('(' + extract('_reasoningEndpoint') + ')');

const _reasoningSupport = (0, eval)('(' + extract('_reasoningSupport') + ')');
globalThis._reasoningSupport = _reasoningSupport;
const _applyReasoningParams = (0, eval)('(' + extract('_applyReasoningParams') + ')');

let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) { pass++; }
  else { fail++; console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`); }
};
const reset = (cfg) => { CFG = cfg || {}; EFFORT = 'high'; THINK_ON = false; BUDGET = 5000; };

const openaiBody = () => ({
  model: 'm', max_tokens: 1000, stream: false,
  messages: [{ role: 'system', content: 's' }, { role: 'user', content: 'q' }],
});
const anthropicBody = () => ({
  model: 'm', max_tokens: 1000, system: 's',
  messages: [{ role: 'user', content: 'q' }],
});

// ── default is OFF, from every direction ───────────────────────────────────
reset();
t('no config, no model -> unsupported', _reasoningSupport(null).supported, false);
t('no declaration -> unsupported',
  _reasoningSupport({ provider: 'huggingface' }).supported, false);
t('explicit false on the model -> unsupported',
  _reasoningSupport({ provider: 'anthropic', reasoning: false }).supported, false);

reset({ panelReasoning: false });
t('global false -> unsupported',
  _reasoningSupport({ provider: 'openai' }).supported, false);

reset({ panelReasoning: 'yes' });
t('non-object non-bool config -> unsupported',
  _reasoningSupport({ provider: 'openai' }).supported, false);

// The scenario named in the request: a direct HuggingFace endpoint that was
// never configured for these settings.
reset();
{
  const support = _reasoningSupport({ provider: 'huggingface', id: 'hf' });
  t('huggingface undeclared -> unsupported', support.supported, false);
  t('huggingface undeclared -> no effort', support.effort, false);
  t('huggingface undeclared -> no thinking', support.thinking, false);
}

// ── the body must be untouched when unsupported ────────────────────────────
reset();
{
  const before = JSON.stringify(openaiBody());
  const after = JSON.stringify(
    _applyReasoningParams(openaiBody(), _reasoningSupport({ provider: 'huggingface' })));
  t('undeclared endpoint: body is byte-identical', after, before);
}
reset();
{
  THINK_ON = true; EFFORT = 'max';
  const before = JSON.stringify(anthropicBody());
  const after = JSON.stringify(
    _applyReasoningParams(anthropicBody(), _reasoningSupport(null)));
  t('undeclared endpoint: body identical even with both toggles on',
    after, before);
}
t('apply is a no-op on a null support object',
  JSON.stringify(_applyReasoningParams(openaiBody(), null)),
  JSON.stringify(openaiBody()));

// ── declared: OpenAI-compatible shape ──────────────────────────────────────
reset();
{
  const support = _reasoningSupport({ provider: 'groq', reasoning: true });
  t('declared openai-shape is supported', support.supported, true);
  t('declared openai-shape has effort', support.effort, true);
  t('openai-shape has no thinking field', support.thinking, false);
  t('effort param name', support.effortParam, 'reasoning_effort');

  EFFORT = 'medium';
  const b1 = _applyReasoningParams(openaiBody(), support);
  t('effort reaches the body', b1.reasoning_effort, 'medium');

  // Five panel levels onto a three-value wire scale: the two above High
  // collapse upward, so the label never promises less than it delivers.
  EFFORT = 'extra';
  t('extra maps upward',
    _applyReasoningParams(openaiBody(), support).reasoning_effort, 'high');
  EFFORT = 'max';
  t('max maps upward',
    _applyReasoningParams(openaiBody(), support).reasoning_effort, 'high');
  EFFORT = 'low';
  t('low maps through',
    _applyReasoningParams(openaiBody(), support).reasoning_effort, 'low');

  // No thinking field on this shape, even with the toggle on.
  THINK_ON = true;
  t('openai shape sends no thinking object',
    'thinking' in _applyReasoningParams(openaiBody(), support), false);
}

// ── runtime-edited scale must be fully mapped ─────────────────────────────
{
  const original = globalThis._EFFORT_LEVELS;
  globalThis._EFFORT_LEVELS = [
    { id: 'instant' }, { id: 'medium' }, { id: 'high' }, { id: 'pro' },
  ];
  let support = _reasoningSupport({ provider: 'openai', reasoning: true });
  t('OpenAI four-level preset remains mapped', support.effort, true);
  EFFORT = 'instant';
  t('Instant preset maps to low wire effort',
    _applyReasoningParams(openaiBody(), support).reasoning_effort, 'low');
  EFFORT = 'pro';
  t('Pro preset maps upward on standard OpenAI shape',
    _applyReasoningParams(openaiBody(), support).reasoning_effort, 'high');

  globalThis._EFFORT_LEVELS = [{ id: 'fast' }, { id: 'deep' }];
  support = _reasoningSupport({ provider: 'openai', reasoning: true });
  t('arbitrary custom ids are not falsely reported as mapped', support.effort, false);

  globalThis._EFFORT_LEVELS = original;
  EFFORT = 'high';
}

// ── declared: Anthropic shape ──────────────────────────────────────────────
reset();
{
  const support = _reasoningSupport({ provider: 'anthropic', reasoning: true });
  t('declared anthropic is supported', support.supported, true);
  t('anthropic has thinking', support.thinking, true);
  // Anthropic has no dedicated wire field for "effort" — it never did, and
  // still doesn't (effortParam/effortValues stay null). What changed is that
  // the Effort control is no longer permanently inert for this shape: a level
  // now resolves to a preset thinking budget (effortBudgets), so the control
  // has a real effect without requiring the separate Extended Reasoning
  // toggle. This is the fix for "Effort must enable like Thinking for
  // Anthropic" — effort=true here is the intended outcome, not a regression.
  t('anthropic effort is backed by a thinking-budget preset', support.effort, true);
  t('anthropic effort has no dedicated wire field', support.effortParam, null);
  t('anthropic effort budgets are declared', !!support.effortBudgets, true);

  // With Extended Reasoning off, the active effort level (default: 'high')
  // still drives a thinking budget on its own — that is the whole point of
  // "enable like Thinking": picking a level has an effect with no second
  // control to flip.
  // Null-safe: a mutation that reverts the effort/thinking wiring must be
  // reported as a failed assertion, never crash the run (Rule 2).
  const budgetTokensOf = (body) =>
    (body && body.thinking) ? body.thinking.budget_tokens : null;

  THINK_ON = false;
  const effortOnly = _applyReasoningParams(anthropicBody(), support);
  t('effort-derived thinking is sent with the toggle off',
    'thinking' in effortOnly, true);
  // anthropicBody() sets max_tokens: 1000, so the 8000-token "high" preset is
  // clamped below it (the same clamp exercised for the explicit-toggle case
  // below) rather than sent as-is.
  t('effort-derived budget is clamped below max_tokens',
    budgetTokensOf(effortOnly), 999);

  THINK_ON = true; BUDGET = 800;
  const b = _applyReasoningParams(anthropicBody(), support);
  t('thinking object type', b.thinking.type, 'enabled');
  t('thinking budget reaches the body', b.thinking.budget_tokens, 800);

  // Anthropic requires max_tokens > budget_tokens; a budget at or above the
  // cap would make EVERY request fail, so it is clamped rather than sent.
  //
  // Read through a helper: a regression that drops the clamp must be reported
  // as a failed assertion, not crash the run and take every later case with
  // it (Rule 2).
  const budgetOf = (b) => (b && b.thinking ? b.thinking.budget_tokens : null);

  BUDGET = 16000;
  const clamped = budgetOf(_applyReasoningParams(anthropicBody(), support));
  t('budget is clamped below max_tokens', clamped !== null && clamped < 1000, true);
  t('clamped budget is still positive', clamped !== null && clamped > 0, true);

  // A budget outside the declared range falls back to the minimum rather
  // than being sent as-is.
  BUDGET = 99999;
  const over = budgetOf(_applyReasoningParams(anthropicBody(), support));
  t('out-of-range budget does not escape',
    over !== null && over <= 16000, true);
}

// ── dict declaration: non-standard field names ─────────────────────────────
reset();
{
  const support = _reasoningSupport({
    provider: 'custom',
    reasoning: {
      effortParam: 'x_effort',
      effortValues: { low: 'l', medium: 'm', high: 'h', extra: 'h', max: 'h' },
      thinkingParam: 'x_think',
      budgetMin: 1024, budgetMax: 8192,
    },
  });
  t('custom effort param', support.effortParam, 'x_effort');
  t('custom thinking param', support.thinkingParam, 'x_think');
  t('custom budget min', support.budgetMin, 1024);
  t('custom budget max', support.budgetMax, 8192);

  EFFORT = 'high'; THINK_ON = true; BUDGET = 2000;
  const b = _applyReasoningParams(openaiBody(), support);
  t('custom effort field is used', b.x_effort, 'h');
  t('standard field is not also sent', 'reasoning_effort' in b, false);
}

// A declaration that switches everything off resolves to unsupported rather
// than to a half-enabled state.
reset();
t('declaration with no usable fields -> unsupported',
  _reasoningSupport({ provider: 'custom',
    reasoning: { effortParam: null, thinkingParam: null } }).supported, false);

// ── precedence: model beats config ─────────────────────────────────────────
reset({ panelReasoning: true });
t('model false overrides config true',
  _reasoningSupport({ provider: 'openai', reasoning: false }).supported, false);
t('config true applies when model is silent',
  _reasoningSupport({ provider: 'openai' }).supported, true);
t('source names the model when the model declared',
  _reasoningSupport({ provider: 'openai', reasoning: true }).source, 'model');
t('source names the config when the config declared',
  _reasoningSupport({ provider: 'openai' }).source, 'config');

reset({ panelReasoning: false });
t('model true overrides config false',
  _reasoningSupport({ provider: 'openai', reasoning: true }).supported, true);

// ── the UI reports Default rather than a level it will not send ────────────
{
  const chip = extract('_syncEffortChip');
  t('chip consults support', /_reasoningSupport\(/.test(chip), true);
  t('chip says Default', /'Default'/.test(chip), true);
  t('chip marks the default state for styling',
    /dataset\.effort = 'default'/.test(chip), true);

  const aria = extract('_modelBtnAccessibleLabel');
  t('accessible name says Default too', /'Default'/.test(aria), true);
  t('accessible name consults support', /_reasoningSupport\(/.test(aria), true);

  // Support depends on the ACTIVE MODEL, so the chip must re-resolve when the
  // model changes, not only when the effort level does.
  const attach = extract('_attachEffortChip');
  t('chip re-syncs on model change',
    /ai-assistant-model-change/.test(attach), true);
  t('chip re-syncs on effort change',
    /ai-assistant-effort-change/.test(attach), true);
}

// ── the sheet controls go inert, and say why ───────────────────────────────
{
  t('an unsupported note exists',
    /var _REASONING_UNSUPPORTED_NOTE =/.test(src), true);
  const m = src.match(/var _REASONING_UNSUPPORTED_NOTE =\s*([\s\S]*?);\n/);
  const NOTE = (0, eval)('(' + m[1] + ')');
  t('note states the consequence', /defaults/i.test(NOTE), true);
  t('note states the controls are inactive', /inactive/i.test(NOTE), true);
  t('note names who can change it', /conf\.py/.test(NOTE), true);
  // The panel only knows what was declared; it must not assert a fact about
  // the model itself.
  t('note does not blame the model',
    /model does not support/i.test(NOTE), false);

  t('effort section picks its note by support',
    /_support\.effort \? _EFFORT_NOTE : _REASONING_UNSUPPORTED_NOTE/.test(src), true);
  t('thinking section picks its note by support',
    /_support\.thinking\s*\?\s*_THINKING_NOTE\s*:\s*_REASONING_UNSUPPORTED_NOTE/
      .test(src), true);

  // Written as a ternary now, not a one-way assignment: the flag has to be
  // CLEARED when a supporting model is selected, not only set when an
  // unsupporting one is. A one-way write was correct while support was
  // resolved once and became a latch the moment it could change.
  t('effort control flag tracks support both ways',
    /effortSeg\.dataset\.unsupported = support\.effort \? 'false' : 'true'/
      .test(src), true);
  t('thinking row flag tracks support both ways',
    /thinkingRow\.dataset\.unsupported = support\.thinking \? 'false' : 'true'/
      .test(src), true);
  t('effort clicks are refused',
    /if \(!_support\.effort\) return;/.test(src), true);
  t('thinking clicks are refused',
    /if \(!_support\.thinking\) return;/.test(src), true);

  // Support must be resolved once per sheet build so the two sections and
  // their notes cannot disagree.
  t('support resolved once per sheet',
    (src.match(/var _support = _reasoningSupport\(/g) || []).length, 1);
}

// ── the token budget slider follows BOTH gates ─────────────────────────────
{
  // The reader's stored "thinking on" preference survives a model switch, so
  // an unsupported endpoint can be reached with thinkingOn still true. The
  // slider must be dead in that state, not merely hidden behind a toggle that
  // is itself inert.
  const gate = extract('_syncBudgetEnabled');
  t('gate reads the stored preference', /thinkingOn/.test(gate), true);
  t('gate reads endpoint support', /_support\.thinking/.test(gate), true);
  t('gate requires both', /thinkingOn && _support\.thinking/.test(gate), true);
  t('gate disables the input', /budgetRange\.disabled/.test(gate), true);
  t('gate marks the area inert', /budgetArea\.dataset\.inert/.test(gate), true);
  t('disabled slider explains itself', /not sent/.test(gate), true);

  // The expression must exist ONCE. It was previously written twice and the
  // second copy omitted the support term, staying correct only because the
  // toggle handler early-returns — correct by accident, not by construction.
  t('the disable expression has one home',
    (src.match(/budgetRange\.disabled\s*=/g) || []).length, 1);
  t('no caller re-derives the gate',
    /budgetRange\.disabled = !thinkingOn;/.test(src), false);

  // Every path that can change either gate must go through the one function:
  // the initial paint, the thinking toggle, and now the model-change
  // re-resolution. Counting them keeps a fourth caller from appearing that
  // re-derives the gate instead of calling it.
  t('every gate change goes through the one function',
    (src.match(/_syncBudgetEnabled\(\);/g) || []).length, 3);
}

// ── capability discovery: the document is UNTRUSTED input ──────────────────
//
// What a discovery document influences is the shape of every later chat
// request. These cases are the security boundary, not shape checks: each one
// is a body-injection attempt a compromised or buggy proxy could mount.
{
  globalThis._EFFORT_LEVELS = (0, eval)('(' + extractArray('_EFFORT_LEVELS') + ')');
  {
    const m = src.match(/var _CAPS_RESERVED_PARAMS = \[([\s\S]*?)\];/);
    globalThis._CAPS_RESERVED_PARAMS = (0, eval)('([' + m[1] + '])');
    const re = src.match(/var _CAPS_PARAM_RE = (\/.*\/);/);
    globalThis._CAPS_PARAM_RE = (0, eval)('(' + re[1] + ')');
  }
  const _capsSafeParam = (0, eval)('(' + extract('_capsSafeParam') + ')');
  globalThis._capsSafeParam = _capsSafeParam;
  const _capsParse = (0, eval)('(' + extract('_capsParse') + ')');

  const good = (over) => ({
    capabilities: {
      reasoning: Object.assign({
        enabled: true,
        effort_param: 'reasoning_effort',
        effort_values: { low: 'low', medium: 'medium', high: 'high',
                         extra: 'high', max: 'high' },
        budget_min: 500, budget_max: 16000,
      }, over || {}),
    },
  });

  // Happy path first, so the rejections below mean something.
  {
    const d = _capsParse(good());
    t('valid document is accepted', !!d, true);
    t('accepted effort param', d.effortParam, 'reasoning_effort');
    t('accepted every level',
      Object.keys(d.effortValues).length, _EFFORT_LEVELS.length);
  }

  // ---- field-name injection ----------------------------------------------
  // A proxy may introduce a field. It may never name one that decides what is
  // sent, to whom, or how the reply is read.
  for (const evil of ['messages', 'model', 'system', 'max_tokens', 'stream',
                      'tools', 'api_key', 'authorization', 'endpoint', 'url']) {
    t('reserved name rejected: ' + evil, _capsSafeParam(evil), null);
    t('document naming ' + evil + ' is refused',
      _capsParse(good({ effort_param: evil })), null);
  }
  // Prototype keys, in the same register.
  for (const evil of ['__proto__', 'constructor', 'prototype']) {
    t('prototype key rejected: ' + evil, _capsSafeParam(evil), null);
  }
  // Shape violations: anything that is not a plain lowercase identifier.
  for (const bad of ['Reasoning_Effort', 'reasoning-effort', 'reasoning effort',
                     '1effort', '_effort', '', 'a'.repeat(41), 'effort;drop',
                     'effort.nested', 'effort[0]']) {
    t('malformed name rejected: ' + JSON.stringify(bad),
      _capsSafeParam(bad), null);
  }
  for (const bad of [null, undefined, 42, {}, [], true]) {
    t('non-string name rejected: ' + JSON.stringify(bad),
      _capsSafeParam(bad), null);
  }

  // ---- prototype pollution ------------------------------------------------
  {
    const doc = good();
    doc.capabilities.reasoning.effort_values = JSON.parse(
      '{"low":"low","medium":"medium","high":"high","extra":"high","max":"high","__proto__":{"polluted":true}}');
    _capsParse(doc);
    t('parsing cannot pollute Object.prototype', ({}).polluted, undefined);
  }
  t('a polluted prototype is not read as a document',
    _capsParse(JSON.parse('{"__proto__":{"capabilities":{"reasoning":{"enabled":true}}}}')),
    null);

  // ---- value injection ----------------------------------------------------
  {
    // A partial map would leave unmapped levels silently sending nothing.
    t('partial effort map is refused',
      _capsParse(good({ effort_values: { low: 'low' } })), null);
    t('missing effort map is refused',
      _capsParse(good({ effort_values: undefined })), null);
    // Values must be short plain strings, never payloads.
    t('object value refused',
      _capsParse(good({ effort_values: { low: {}, medium: 'm', high: 'h',
                                         extra: 'h', max: 'h' } })), null);
    t('oversized value refused',
      _capsParse(good({ effort_values: { low: 'x'.repeat(33), medium: 'm',
                                         high: 'h', extra: 'h', max: 'h' } })),
      null);
    t('empty value refused',
      _capsParse(good({ effort_values: { low: '', medium: 'm', high: 'h',
                                         extra: 'h', max: 'h' } })), null);
  }

  // ---- budget bounds ------------------------------------------------------
  {
    // A proxy may narrow the range. It may never widen it past the panel's
    // own hard bounds.
    const wide = _capsParse(good({ budget_min: 1, budget_max: 999999 }));
    t('budget min cannot go below the floor', wide.budgetMin >= 500, true);
    t('budget max cannot exceed the ceiling', wide.budgetMax <= 16000, true);

    const narrow = _capsParse(good({ budget_min: 1024, budget_max: 8192 }));
    t('a narrower range is honoured', narrow.budgetMin, 1024);
    t('a narrower ceiling is honoured', narrow.budgetMax, 8192);

    const junk = _capsParse(good({ budget_min: 'abc', budget_max: null }));
    t('non-numeric budgets fall back', junk.budgetMin, 500);
  }

  // ---- fail-closed on every malformed shape -------------------------------
  for (const [name, doc] of [
    ['null', null], ['undefined', undefined], ['string', 'ok'], ['number', 7],
    ['array', []], ['empty object', {}],
    ['no capabilities', { status: 'ok' }],
    ['capabilities not an object', { capabilities: 'yes' }],
    ['no reasoning key', { capabilities: {} }],
    ['reasoning not an object', { capabilities: { reasoning: true } }],
    ['enabled missing', { capabilities: { reasoning: {} } }],
    // Truthiness is not consent. The string "false" is truthy in JS, so a
    // document that literally says enabled:"false" would ENABLE the controls
    // under a loose check — the worst possible reading of the clearest
    // possible refusal. Only the boolean true counts.
    ['enabled is the number 1', good({ enabled: 1 })],
    ['enabled is the string "true"', good({ enabled: 'true' })],
    ['enabled is the string "false"', good({ enabled: 'false' })],
    ['enabled is an object', good({ enabled: {} })],
    ['enabled with no usable field',
      { capabilities: { reasoning: { enabled: true } } }],
  ]) {
    t('malformed document rejected: ' + name, _capsParse(doc), null);
  }

  // An explicit "no" is a real answer, distinct from "I could not tell".
  t('explicit disable is honoured, not ignored',
    _capsParse({ capabilities: { reasoning: { enabled: false } } }), false);

  // ---- thinking-only declarations are valid -------------------------------
  {
    const d = _capsParse({ capabilities: { reasoning: {
      enabled: true, thinking_param: 'thinking' } } });
    t('thinking-only document accepted', !!d, true);
    t('thinking-only has no effort param', 'effortParam' in d, false);
    t('thinking-only param carried', d.thinkingParam, 'thinking');
  }
}

// ── discovery transport is fail-closed and non-blocking ────────────────────
{
  const disco = extract('_capsDiscover');
  t('discovery has a timeout', /_CAPS_TIMEOUT_MS/.test(disco), true);
  t('discovery aborts on timeout', /AbortController/.test(disco), true);
  t('discovery sends no credentials', /credentials: 'omit'/.test(disco), true);
  t('discovery bypasses the cache', /cache: 'no-store'/.test(disco), true);
  t('discovery caps the response size', /_CAPS_MAX_BYTES/.test(disco), true);
  t('discovery swallows every failure', /catch \(_\)/.test(disco), true);
  t('discovery checks the status', /res\.ok/.test(disco), true);

  const origin = extract('_capsOrigin');
  t('only http(s) origins are probed',
    /protocol !== 'https:' && .*protocol !== 'http:'/.test(origin), true);
  t('a malformed endpoint yields no origin', /catch \(_\) \{ return ''/.test(origin), true);

  const cached = extract('_capsCached');
  t('cached answers expire', /_CAPS_TTL_MS/.test(cached), true);
  t('a corrupt cache entry is discarded', /catch \(_\) \{ return null/.test(cached), true);

  // The request path must never await discovery.
  t('discovery is never awaited in the request path',
    /await _capsDiscover/.test(src), false);

  // Discovery must probe the origin that will receive the chat request.
  t('endpoint resolution is shared',
    (src.match(/function _reasoningEndpoint\(/g) || []).length, 1);
  t('discovery uses the shared resolver',
    /_capsOrigin\(_reasoningEndpoint\(/.test(src), true);
}

// ── precedence: model > discovery > config > off ───────────────────────────
{
  const resolver = extract('_reasoningSupport');
  const iModel = resolver.indexOf('activeModel.reasoning');
  const iDisco = resolver.indexOf('_capsCached(');
  const iCfg = resolver.indexOf('cfg.panelReasoning');
  t('model is consulted first', iModel < iDisco, true);
  t('discovery outranks the static config', iDisco < iCfg, true);
  t('discovery is a named source', /source = 'discovery'/.test(resolver), true);
}

// ── per-model scenario matrix ──────────────────────────────────────────────
//
// The controls must correspond to the ACTIVE MODEL at all times, not to a
// value resolved once. A build-wide setting is a DEFAULT for models that stay
// silent; any model may override it in either direction.
//
// Truth table (cfg = ai_assistant_panel_reasoning):
//
//   cfg      model declares    -> effective
//   false    (silent)             OFF
//   false    true                 ON      <- one custom model lights up alone
//   false    false                OFF
//   true     (silent)             ON      <- inherits the build default
//   true     true                 ON
//   true     false                OFF     <- one model opts out alone
{
  const eff = (cfgVal, modelDecl) => {
    reset(cfgVal === undefined ? {} : { panelReasoning: cfgVal });
    const model = { provider: 'openai' };
    if (modelDecl !== undefined) model.reasoning = modelDecl;
    return _reasoningSupport(model).supported;
  };

  // Scenario 1 — nothing declared anywhere: always off.
  t('cfg false + silent model -> off', eff(false, undefined), false);
  t('cfg absent + silent model -> off', eff(undefined, undefined), false);

  // Scenario 2 — cfg false, one custom model declares support. That model
  // alone is live; every other model stays inert.
  t('cfg false + model true  -> ON', eff(false, true), true);
  t('cfg false + sibling silent stays off', eff(false, undefined), false);

  // Scenario 3 — cfg true. Silent models INHERIT the build default; that is
  // what a default is for. A model that wants to opt out says so (scenario 4).
  t('cfg true + silent model -> ON (inherits)', eff(true, undefined), true);
  t('cfg true + model true   -> ON', eff(true, true), true);

  // Scenario 4 — cfg true, one model opts out. That model alone is inert.
  t('cfg true + model false  -> OFF', eff(true, false), false);
  t('cfg true + sibling silent stays ON', eff(true, undefined), true);

  // A per-model dict declaration also overrides a false build default.
  reset({ panelReasoning: false });
  t('cfg false + model dict  -> ON',
    _reasoningSupport({ provider: 'custom',
      reasoning: { effortParam: 'x_effort',
                   effortValues: { low: 'l', medium: 'm', high: 'h',
                                   extra: 'h', max: 'h' } } }).supported, true);

  // Switching models must change the answer with no other state changing.
  reset({ panelReasoning: false });
  const a = _reasoningSupport({ provider: 'openai', id: 'plain' }).supported;
  const b = _reasoningSupport({ provider: 'openai', id: 'custom', reasoning: true }).supported;
  t('two models, same config, different answers', a === b, false);
  t('the declaring one is the enabled one', b, true);
}

// ── the sheet re-resolves; it does not decide once ─────────────────────────
{
  // The defect: _support was captured at sheet-build time, so switching models
  // left Effort and Thinking showing the PREVIOUS model's state — a model that
  // accepts these settings could look inert, and one that does not could look
  // live and silently discard them.
  const sections = extract('_appendModelSheetSections');

  t('support is re-resolved, not captured once',
    /function _applyReasoningUI\(\)/.test(sections), true);
  t('re-resolution reassigns support',
    /_support = _reasoningSupport\(_getActiveModel/.test(sections), true);
  t('re-resolution runs on model change',
    /addEventListener\('ai-assistant-model-change', _applyReasoningUI\)/
      .test(sections), true);
  t('it also paints once at build time',
    /_applyReasoningUI\(\);\n\s*\(typeof _assistantEvents/.test(sections), true);

  // Every dependent control must register a sink, or it silently keeps a
  // stale value while its neighbours update — worse than all of them being
  // stale, because the sheet then contradicts itself.
  t('controls register sinks',
    (sections.match(/_supportSinks\.push\(/g) || []).length >= 3, true);
  t('every sink is driven from one loop',
    (sections.match(/_supportSinks\[i\]\(_support\)/g) || []).length, 1);
  t('the budget slider is driven too',
    /_applyReasoningUI[\s\S]{0,600}_syncBudgetEnabled\(\)/.test(sections), true);

  // No control may read a support value captured before the event.
  t('no sink hardcodes a build-time decision',
    /_supportSinks\.push\(function \(support\)/.test(sections), true);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
