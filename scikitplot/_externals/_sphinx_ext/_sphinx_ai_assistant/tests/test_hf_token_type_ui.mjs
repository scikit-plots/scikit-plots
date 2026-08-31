import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
let pass = 0, fail = 0;
const ok = (cond, name) => {
  if (cond) pass++;
  else { fail++; console.log('FAIL ' + name); }
};

ok(/_tokenTypeLabel\.textContent = 'Inference token type'/.test(src), 'simple endpoint UI exposes inference token type');
ok(/fine-grained/.test(src) && /Read/.test(src) && /Write/.test(src), 'three HF token roles are represented');
ok(/Server-managed · auto-discovered/.test(src), 'token role is clearly server-managed');
ok(/data-token-type/.test(src), 'token roles use structured status chips');
ok(/_renderSimpleTokenType/.test(src), 'simple token role has a single renderer');
ok(/info\.tokenType/.test(src), 'simple role uses discovered server posture');
ok(/_fetchProxyDatasetInfo\(canonicalBase/.test(src), 'service discovery drives simple token posture');
ok(/if \(normalizedConfigured\) \{ return; \}/.test(src), 'dataset override does not block token discovery');
ok(/Inference token type/.test(src), 'advanced view expands inference token role');
ok(/Dataset write token type/.test(src), 'advanced view expands dataset write token role');
ok(/Review permissions/.test(src), 'write inference token warns about broad permissions');
ok(/Insufficient for writes/.test(src), 'read dataset token is flagged as insufficient');
ok(!/token\.value|hf_token\s*=/.test(src), 'UI does not introduce raw token value handling');

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
