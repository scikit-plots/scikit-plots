// Compose the full outbound context pipeline and print the result as JSON.
//
//   node tests/compose_context.mjs _static/ai-assistant.js
//
// Not a harness — it asserts nothing. It is the browser half of an end-to-end
// test whose other half is Python (tests/test_end_to_end_context.py), which
// feeds this output to the stub responder exactly as the proxy would receive
// it.
//
// Why this exists
// ---------------
// Every guard is unit-tested in isolation: invisible characters, redaction,
// fencing, detection. Nothing verified they COMPOSE — that the four applied in
// the shipped order, to one adversarial page, produce a system prompt where
// the fence actually closed, the key actually left, and the hidden payload
// actually vanished. Four correct parts wired in the wrong order is a defect
// no per-part test can see.
//
// The order below is copied from the request path deliberately, and a test in
// the Python half asserts that this file's order still matches the source. A
// composition test that composes differently from production tests a pipeline
// nobody ships.
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

function extractArrayLiteral(name) {
  const i = src.indexOf('var ' + name + ' = [');
  const start = src.indexOf('[', i);
  let depth = 0;
  for (let j = start; j < src.length; j++) {
    if (src[j] === '[') depth++;
    else if (src[j] === ']') { depth--; if (depth === 0) return src.slice(start, j + 1); }
  }
  throw new Error('unbalanced: ' + name);
}

// ── minimal environment ────────────────────────────────────────────────────
globalThis.window = {
  crypto: {
    getRandomValues(buf) {
      for (let i = 0; i < buf.length; i++) buf[i] = (i * 37 + 11) % 256;
      return buf;
    },
  },
};
globalThis.document = { getElementById: () => null };   // no transcript to notify
globalThis._cfg = () => ({ panelInjectionNotice: true });

{
  const m = src.match(/var _INVISIBLE_CHARS_RE =\s*([\s\S]*?);\n/);
  globalThis._INVISIBLE_CHARS_RE = (0, eval)('(' + m[1] + ')');
}
globalThis._SECRET_PATTERNS = (0, eval)('(' + extractArrayLiteral('_SECRET_PATTERNS') + ')');
globalThis._INJECTION_PATTERNS = (0, eval)('(' + extractArrayLiteral('_INJECTION_PATTERNS') + ')');
globalThis._INJECTION_THRESHOLD =
  Number(src.match(/var _INJECTION_THRESHOLD = (\d+);/)[1]);

for (const name of [
  '_stripInvisibleChars', '_untrustedNonce', '_fenceUntrusted',
  '_redactSecrets', '_redactionSummary', '_announceRedaction',
  '_scanInjection', '_injectionSummary', '_announceInjection',
]) {
  globalThis[name] = (0, eval)('(' + extract(name) + ')');
}

// ── the shipped order, mirrored ────────────────────────────────────────────
function composeSystemPrompt(pageMarkdown, contextLimit) {
  const cleaned = _stripInvisibleChars(pageMarkdown);
  const redacted = _redactSecrets(cleaned.text);
  const injection = _scanInjection(redacted.text);
  const fenced = _fenceUntrusted(
    'the documentation page the user is reading',
    redacted.text,
    contextLimit,
    _redactionSummary(redacted.findings)
  );
  return {
    systemPrompt:
      'You are a helpful documentation assistant. Answer questions ' +
      'about the documentation page below.\n\n' + fenced,
    invisibleRemoved: cleaned.removed,
    redactionFindings: redacted.findings,
    injectionKinds: injection.kinds,
    injectionFlagged: injection.flagged,
  };
}

const page = fs.readFileSync(process.argv[3], 'utf8');
const limit = Number(process.argv[4] || 100000);
process.stdout.write(JSON.stringify(composeSystemPrompt(page, limit)));
