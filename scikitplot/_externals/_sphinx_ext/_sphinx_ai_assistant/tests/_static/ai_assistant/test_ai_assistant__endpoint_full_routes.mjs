import fs from 'node:fs';
import vm from 'node:vm';

const src = fs.readFileSync(process.argv[2], 'utf8');
let pass = 0, fail = 0;
const eq = (got, want, name) => {
  if (got === want) pass++;
  else { fail++; console.log(`FAIL ${name}\n  got:  ${got}\n  want: ${want}`); }
};
const ok = (cond, name) => {
  if (cond) pass++;
  else { fail++; console.log('FAIL ' + name); }
};

const start = src.indexOf('var _EP = (function () {');
const endMarker = '\n    }());';
const end = src.indexOf(endMarker, start);
if (start < 0 || end < 0) throw new Error('Could not locate _EP registry');
const block = src.slice(start, end + endMarker.length);

const store = new Map();
const context = {
  URL,
  console,
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
      default: {
        label: 'Default',
        base: 'https://proxy.example.com',
        chat: 'https://proxy.example.com',           // legacy base-style explicit
        share: 'https://proxy.example.com/v1/share', // complete standard endpoint
        feedback: 'https://feedback.example.com/custom/ingest', // arbitrary route
        training: '',
      },
      hostonly: {
        label: 'Host only legacy',
        chat: 'https://legacy.example.com',
      },
      prefixed: {
        label: 'Path-prefixed base',
        base: 'https://proxy.example.com/api',
      },
      relative: {
        label: 'Relative endpoints',
        base: 'https://proxy.example.com',
        chat: 'v1/chat/completions',
        share: '/v1/share',
        feedback: 'hooks/feedback',
        training: '/custom/contribute/',
      },
      inheritnull: {
        label: 'Null and empty inherit',
        base: 'https://proxy.example.com',
        chat: null,
        share: '',
        feedback: null,
        training: '',
      },
      prefixedrelative: {
        label: 'Path-prefixed base plus relative route',
        base: 'https://proxy.example.com/api',
        share: '/v1/share',
      },
      spaced: {
        label: 'Whitespace trimmed absolute',
        base: 'https://proxy.example.com',
        chat: '  https://other.example.com/custom/chat  ',
      },
    },
  },
};
context.window.window = context.window;
vm.createContext(context);
vm.runInContext(block, context);
const EP = context._EP;

ok(!!EP && typeof EP.resolveEndpoint === 'function', 'resolveEndpoint is executable');
eq(EP.resolveEndpoint('chat'), 'https://proxy.example.com/v1/chat/completions', 'explicit value equal to base keeps legacy base semantics');
eq(EP.resolveEndpoint('share'), 'https://proxy.example.com/v1/share', 'full standard Share endpoint is not double-suffixed');
eq(EP.resolveEndpoint('feedback'), 'https://feedback.example.com/custom/ingest', 'arbitrary path-bearing endpoint is used verbatim');
eq(EP.resolveEndpoint('training'), 'https://proxy.example.com/v1/contribute', 'blank route derives default endpoint from base');

EP.setActive('hostonly');
eq(EP.resolveEndpoint('chat'), 'https://legacy.example.com/v1/chat/completions', 'host-only legacy feature value still gains default route');

EP.setActive('prefixed');
eq(EP.resolveEndpoint('share'), 'https://proxy.example.com/api/v1/share', 'path-prefixed Base derives default route under prefix');

EP.setActive('relative');
eq(EP.resolveEndpoint('chat'), 'https://proxy.example.com/v1/chat/completions', 'relative route without slash joins Base');
eq(EP.resolveEndpoint('share'), 'https://proxy.example.com/v1/share', 'relative route with slash joins Base identically');
eq(EP.resolveEndpoint('feedback'), 'https://proxy.example.com/hooks/feedback', 'arbitrary relative provider route joins Base');
eq(EP.resolveEndpoint('training'), 'https://proxy.example.com/custom/contribute', 'relative trailing slash is normalised at resolution');

EP.setActive('inheritnull');
eq(EP.resolveEndpoint('chat'), 'https://proxy.example.com/v1/chat/completions', 'null Chat inherits default route');
eq(EP.resolveEndpoint('share'), 'https://proxy.example.com/v1/share', 'empty Share inherits default route');
eq(EP.resolveEndpoint('feedback'), 'https://proxy.example.com/v1/feedback', 'null Feedback inherits default route');
eq(EP.getProfile('inheritnull').chat, '', 'getProfile exposes null endpoint as empty string');

EP.setActive('prefixedrelative');
eq(EP.resolveEndpoint('share'), 'https://proxy.example.com/api/v1/share', 'leading-slash relative route stays beneath path-prefixed Base');

EP.setActive('spaced');
eq(EP.resolveEndpoint('chat'), 'https://other.example.com/custom/chat', 'absolute endpoint surrounding whitespace is trimmed');

ok(typeof EP.validateEndpoint === 'function', 'feature endpoint validator is public');
eq(EP.validateEndpoint('/v1/share').url, 'v1/share', 'runtime validator canonicalises leading slash');
eq(EP.validateEndpoint('v1/share').url, 'v1/share', 'runtime validator accepts slashless relative route');
ok(!EP.validateEndpoint('//evil.example/x').ok, 'runtime validator rejects protocol-relative authority');
ok(!EP.validateEndpoint('../escape').ok, 'runtime validator rejects parent traversal');

let runtimeAdd = EP.addProfile('runtime_relative', {
  label: 'Runtime relative',
  base: '  https://runtime.example.com/base/  ',
  chat: '/v2/generate/', share: 'v2/share', feedback: null, training: '', ttlDays: 30,
});
ok(runtimeAdd.ok, 'runtime profile accepts absolute Base plus relative endpoint overrides');
EP.setActive('runtime_relative');
eq(EP.getProfile('runtime_relative').base, 'https://runtime.example.com/base', 'runtime Base whitespace/trailing slash normalised');
eq(EP.getProfile('runtime_relative').chat, 'v2/generate', 'runtime relative route stored canonically');
eq(EP.resolveEndpoint('chat'), 'https://runtime.example.com/base/v2/generate', 'runtime relative route resolves below path-prefixed Base');
eq(EP.resolveEndpoint('feedback'), 'https://runtime.example.com/base/v1/feedback', 'runtime null endpoint inherits default route');

// Network consumers must use complete endpoint resolution rather than append
// their own fixed paths after the registry has resolved the route.
ok(/_EP\.resolveEndpoint \? _EP\.resolveEndpoint\('feedback'\)/.test(src), 'feedback path uses complete endpoint resolver');
ok(/_EP\.resolveEndpoint \? _EP\.resolveEndpoint\('share'\)/.test(src), 'share path uses complete endpoint resolver');
ok(/_EP\.resolveEndpoint \? _EP\.resolveEndpoint\('training'\)/.test(src), 'training path uses complete endpoint resolver');
ok(/var _epChatUrl = _EP\.hasProfiles\(\)[\s\S]{0,120}_EP\.resolveEndpoint/.test(src), 'chat path uses complete endpoint resolver');
ok(!/_fbBase \+ '\/v1\/feedback'/.test(src), 'feedback endpoint is not double-suffixed');
ok(!/base \+ '\/v1\/share'/.test(src), 'share collection endpoint is not double-suffixed');
ok(!/_trBase\.replace\([^\n]+\) \+ '\/v1\/contribute'/.test(src), 'training endpoint is not double-suffixed');

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
