import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { importCfWorkerForNode } from './_import_cf_worker_for_node.mjs';

const testsDir = path.dirname(fileURLToPath(import.meta.url));
const workerPath = path.join(path.dirname(testsDir), '_cf_worker', 'index.js');
const mod = await importCfWorkerForNode(workerPath, 'run15-rate');
const worker = mod.default;
const RateLimitBucket = mod.RateLimitBucket;
let passed = 0, failed = 0;
function ok(v, n) { if (v) { passed++; console.log('PASS', n); } else { failed++; console.log('FAIL', n); } }

class FakeStorage {
  constructor() { this.map = new Map(); this.alarmAt = null; }
  async transaction(fn) {
    const txn = {
      get: async (k) => this.map.get(k),
      put: async (k, v) => { this.map.set(k, structuredClone(v)); },
    };
    return fn(txn);
  }
  async setAlarm(ts) { this.alarmAt = ts; }
  async deleteAll() { this.map.clear(); this.alarmAt = null; }
}
function makeBucket() {
  const storage = new FakeStorage();
  const bucket = new RateLimitBucket({ storage }, {});
  return { bucket, storage };
}

const direct = makeBucket();
const d1 = await direct.bucket.consume({ limit: 2, windowSeconds: 60 });
const d2 = await direct.bucket.consume({ limit: 2, windowSeconds: 60 });
const d3 = await direct.bucket.consume({ limit: 2, windowSeconds: 60 });
const d4 = await direct.bucket.consume({ limit: 2, windowSeconds: 60 });
ok(d1.allowed && d1.count === 1, 'Durable bucket admits first request');
ok(d2.allowed && d2.count === 2, 'Durable bucket admits request at limit');
ok(!d3.allowed && d3.count === 3, 'Durable bucket rejects first request beyond limit');
ok(!d4.allowed && d4.count === 3, 'Durable bucket saturates stored denied count');
ok(Number.isFinite(d3.retryAfter) && d3.retryAfter > 0, 'Durable bucket returns bounded retry-after');
ok(Number.isFinite(direct.storage.alarmAt), 'Durable bucket schedules state cleanup alarm');
await direct.bucket.alarm();
ok(direct.storage.map.size === 0, 'Durable bucket alarm removes expired counter state');

class FakeNamespace {
  constructor() { this.buckets = new Map(); }
  getByName(name) {
    if (!this.buckets.has(name)) this.buckets.set(name, makeBucket().bucket);
    return this.buckets.get(name);
  }
}
const ns = new FakeNamespace();
const kv = { async list() { throw new Error('KV limiter must not be used when Durable Object is bound'); }, async put() {} };
const baseEnv = {
  HF_TOKEN: 'hf-test-secret', ALLOWED_MODELS: 'Qwen/safe-model',
  CHAT_RATE_LIMIT_PER_HOUR: '2', SHARE_KV: kv, RATE_LIMIT_DO: ns, RATE_LIMIT_IDENTITY_SECRET: 'r'.repeat(48),
};
const realFetch = globalThis.fetch;
let upstreamCalls = 0;
globalThis.fetch = async () => {
  upstreamCalls++;
  return new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), { status: 200, headers: { 'content-type': 'application/json' } });
};
function chatReq(ip) {
  return new Request('https://worker.example/v1/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'CF-Connecting-IP': ip },
    body: JSON.stringify({ contract: 'scikitplot-chat-v1', model: 'Qwen/safe-model', user_message: 'q', context: {}, max_tokens: 10, stream: false }),
  });
}
try {
  const envPopA = { ...baseEnv };
  const envPopB = { ...baseEnv };
  const r1 = await worker.fetch(chatReq('203.0.113.9'), envPopA);
  const r2 = await worker.fetch(chatReq('203.0.113.9'), envPopB);
  const r3 = await worker.fetch(chatReq('203.0.113.9'), envPopA);
  ok(r1.status === 200 && r2.status === 200, 'two simulated PoPs share allowed identity budget');
  ok(r3.status === 429, 'third request is denied across simulated PoP environments');
  ok(upstreamCalls === 2, 'distributed denial occurs before provider spend');
  const other = await worker.fetch(chatReq('203.0.113.10'), envPopB);
  ok(other.status === 200, 'different identity maps to independent Durable Object shard');
  const health = await worker.fetch(new Request('https://worker.example/health'), envPopA);
  const healthDoc = await health.json();
  ok(healthDoc.rate_limit.authoritative === true && healthDoc.rate_limit.backend === 'durable_object', 'Worker health truthfully advertises bound authority');

  const missingSecret = { ...baseEnv }; delete missingSecret.RATE_LIMIT_IDENTITY_SECRET;
  const missing = await worker.fetch(chatReq('192.0.2.54'), missingSecret);
  ok(missing.status === 503, 'Durable Object authority fails closed without identity HMAC secret');

  const missingBinding = { ...baseEnv, RATE_LIMIT_REQUIRE_AUTHORITATIVE: 'true' }; delete missingBinding.RATE_LIMIT_DO;
  const missingDo = await worker.fetch(chatReq('192.0.2.53'), missingBinding);
  ok(missingDo.status === 503, 'bundled authoritative policy fails closed without Durable Object binding');

  const brokenEnv = { ...baseEnv, RATE_LIMIT_DO: { getByName() { return { async consume() { throw new Error('down'); } }; } } };
  const broken = await worker.fetch(chatReq('192.0.2.55'), brokenEnv);
  ok(broken.status === 503, 'authoritative limiter failure is fail-closed');
} finally {
  globalThis.fetch = realFetch;
}

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
