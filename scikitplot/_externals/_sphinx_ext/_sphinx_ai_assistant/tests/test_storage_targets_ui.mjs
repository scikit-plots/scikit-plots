import fs from 'node:fs';

const src = fs.readFileSync(process.argv[2], 'utf8');
let pass = 0, fail = 0;
const ok = (cond, name) => {
  if (cond) pass++;
  else { fail++; console.log('FAIL ' + name); }
};

ok(src.includes('storage:                (data && data.storage) || null'), 'discovery reads provider-neutral storage manifest');
ok(src.includes('function _renderStorageTargets('), 'storage-target renderer exists');
ok(src.includes("target.role === 'primary' ? 'Primary' : 'Mirror'"), 'primary and mirror roles are visible');
ok(src.includes("target.provider === 'huggingface' ? 'Dataset root' : 'Repository root'"), 'root link label adapts to provider while URL comes from server manifest');
ok(src.includes("['Feedback records', links.feedback]"), 'feedback link comes from server manifest');
ok(src.includes("['Contributions', links.contributions]"), 'contributions link comes from server manifest');
ok(src.includes("target.provider === 'huggingface' && target.token"), 'HF token capability UI is provider-specific');
ok(src.includes("['fine-grained', 'Fine-grained', 'Repo-scoped write is preferred']"), 'fine-grained dataset token role is rendered');
ok(src.includes("['read', 'Read', 'Read-only; persistence is blocked']"), 'read dataset token role is rendered');
ok(src.includes("['write', 'Write', 'Broad repository write access']"), 'write dataset token role is rendered');
ok(src.includes("write_capability || 'unknown'"), 'write capability is rendered separately from token type');
ok(src.includes("Number(target.pending_retries || 0) > 0 ? '\\u25CF Retry queued'"), 'mirror retry state is surfaced');
ok(!src.includes('AI_RECORD_STORAGE_TOKEN_'), 'browser source never contains storage secret env names');

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
