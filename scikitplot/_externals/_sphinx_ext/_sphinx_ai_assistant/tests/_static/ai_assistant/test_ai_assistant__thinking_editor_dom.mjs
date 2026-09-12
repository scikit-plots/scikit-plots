// Contract harness for Thinking customization + fail-soft request fallback.
//
//   node tests/test_thinking_editor_dom.mjs _static/ai-assistant.js
//
// The model form owns capability booleans; the Thinking sheet owns validated
// wire configuration. Optional reasoning must never turn a bad mapping into a
// permanently broken chat request or leak provider error bodies.
import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');

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

let pass = 0, fail = 0;
const t = (name, got, want) => {
  if (got === want) pass++;
  else {
    fail++;
    console.log(`  FAIL ${name}\n       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`);
  }
};

const editor = extract('_buildThinkingEditorSection');
const sections = extract('_appendModelSheetSections');
const custom = extract('_appendModelCustomSection');
const support = extract('_reasoningSupport');
const apply = extract('_applyReasoningParams');
const api = extract('_panelApiCall');
const stream = extract('_panelApiCallStreaming');
const fallback = extract('_fetchWithReasoningFallback');

// ── placement and ownership ────────────────────────────────────────────────
t('Thinking editor is its own sheet section',
  /_buildSheetSection\(\s*'Customize thinking'/.test(editor), true);
t('Thinking editor is appended next to Thinking section',
  /sheet\.appendChild\(thinkingSection\);\s*sheet\.appendChild\(thinkingEditor\.section\);/.test(sections), true);
t('custom-model form no longer exposes raw Thinking field',
  /_frow\('Thinking field'/.test(custom), false);
t('custom-model form exposes boolean Thinking support',
  /_frow\('Thinking support', thinkingSel\)/.test(custom), true);

// ── validated future-proof wire modes ─────────────────────────────────────
t('Thinking field uses safe top-level key validator', /_capsSafeParam\(param\)/.test(editor), true);
t('Boolean mode is offered', /\['boolean', 'Boolean true'\]/.test(editor), true);
t('Adaptive mode is offered', /\['adaptive', 'Adaptive object'\]/.test(editor), true);
t('Budget mode is offered', /\['budget', 'Budget object'\]/.test(editor), true);
t('Claude adaptive preset exists', /Claude[^\n]*Adaptive/.test(editor), true);
t('Claude budget preset exists', /Claude[^\n]*Budget/.test(editor), true);
t('generic Boolean preset exists', /'Boolean'/.test(editor), true);
t('saving wire config preserves the model capability boolean',
  /if \(typeof spec\.thinking !== 'boolean'\)[\s\S]{0,180}_reasoningSupport/.test(editor), true);
t('invalid budget range is rejected in-editor', /min < 500[\s\S]*max > 16000[\s\S]*min > max/.test(editor), true);
t('reset removes only Thinking wire fields',
  /key !== 'thinkingParam'[\s\S]*key !== 'thinkingMode'[\s\S]*key !== 'budgetMin'[\s\S]*key !== 'budgetMax'/.test(editor), true);
t('reset preserves capability and Effort settings', /Model capability and Effort settings were preserved/.test(editor), true);

// Budget UI remains present but is live only for a declared budget wire mode.
t('budget UI gates on budget payload mode', /_support\.thinkingMode === 'budget'/.test(sections), true);
t('non-budget modes keep slider inert', /_support\.thinking && !budgetMode/.test(sections), true);
t('support returns the selected thinking mode', /thinkingMode: thinkingMode/.test(support), true);
t('Boolean payload emits only true', /thinkingMode === 'boolean'[\s\S]{0,180}= true/.test(apply), true);
t('Adaptive payload uses adaptive object', /thinkingMode === 'adaptive'[\s\S]{0,180}type: 'adaptive'/.test(apply), true);
t('Budget payload is explicit and bounded', /type: 'enabled', budget_tokens: budget/.test(apply), true);

// ── privacy-safe fail-soft request path ───────────────────────────────────
t('request snapshots provider-default body before reasoning',
  /var providerDefaultBody = JSON\.stringify\(bodyObj\);[\s\S]*_applyReasoningParams/.test(api), true);
t('non-streaming request uses one fallback helper', /_fetchWithReasoningFallback\(/.test(api), true);
t('streaming request receives a provider-default fallback body', /streamFallbackBody[\s\S]*_panelApiCallStreaming/.test(api), true);
t('streaming path uses fallback helper too', /_fetchWithReasoningFallback\(/.test(stream), true);
t('streaming retry is forbidden after partial visible output', /!accumulated && fallbackBodyStr/.test(stream), true);
t('server SSE payload is not echoed to the reader', /The AI server reported an error:/.test(stream), false);
t('HTTP error bodies are not read into errBody', /errBody/.test(api + stream), false);
t('reasoning fallback diagnostic does not log endpoint or model id',
  /reasoning-fallback[^\n]*(endpoint|model id)/i.test(extract('_openReasoningCircuit')), false);

// Execute the core fallback helper with deterministic fake fetches.
async function runFallbackScenario(sequence, fallbackBody = '{"model":"m"}') {
  const calls = [];
  const opened = [];
  let index = 0;
  const fn = (0, eval)(
    '(function(_fetch,_openReasoningCircuit){ return (async ' + fallback + '); })'
  )(
    async (_endpoint, options) => {
      calls.push(options.body);
      const item = sequence[index++];
      if (item instanceof Error) throw item;
      return item;
    },
    (_model, reason) => { opened.push(reason); },
  );
  let value, error;
  try {
    value = await fn('/chat', { method: 'POST', body: '{"model":"m","thinking":true}' }, fallbackBody, { id: 'x' });
  } catch (e) { error = e; }
  return { calls, opened, value, error };
}

{
  const r = await runFallbackScenario([
    { ok: false, status: 400 },
    { ok: true, status: 200 },
  ]);
  t('400 reasoning rejection retries exactly once', r.calls.length, 2);
  t('400 retry strips optional reasoning body', r.calls[1], '{"model":"m"}');
  t('successful HTTP fallback opens model circuit', r.opened.length, 1);
}
{
  const pipe = new TypeError('connection closed');
  const r = await runFallbackScenario([pipe, { ok: true, status: 200 }]);
  t('pre-response pipe close retries exactly once', r.calls.length, 2);
  t('successful pipe fallback opens model circuit', r.opened.length, 1);
}
{
  const abort = new Error('cancelled'); abort.name = 'AbortError';
  const r = await runFallbackScenario([abort, { ok: true, status: 200 }]);
  t('AbortError is never retried', r.calls.length, 1);
  t('AbortError is preserved', r.error && r.error.name, 'AbortError');
}
{
  const r = await runFallbackScenario([{ ok: false, status: 500 }]);
  t('unrelated HTTP 500 is not auto-retried', r.calls.length, 1);
}
{
  const r = await runFallbackScenario([
    { ok: false, status: 422 },
    { ok: false, status: 422 },
    { ok: true, status: 200 },
  ]);
  t('failed fallback never triggers a third request', r.calls.length, 2);
}
{
  const r = await runFallbackScenario([{ ok: false, status: 400 }], null);
  t('no optional reasoning means no fallback request', r.calls.length, 1);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
