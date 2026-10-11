// Reference Cloudflare Worker receiver for the page-feedback webhook provider.
//
// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// What it is
// ----------
// The feedback service (Python) validates every browser request and then, with
// the `webhook` provider, POSTs the durable event to this Worker, signed with
// HMAC-SHA256. The Worker verifies the signature and stores the event in D1
// exactly once. Browsers never talk to this Worker; only the service does.
//
//   browser --(page.feedback-request.v1)--> feedback service --(signed)--> Worker --> D1
//
// Why D1: the primary key on feedback_id makes "store once, answer replay or
// conflict" atomic. Workers KV is eventually consistent, so two retries racing
// could both believe they stored first; it is not used for authority.
//
// Deploy
// ------
//   wrangler d1 create page-feedback
//   # save the SCHEMA constant below as schema.sql, then:
//   wrangler d1 execute page-feedback --remote --file schema.sql
//   wrangler secret put FEEDBACK_WEBHOOK_TOKEN      # same value as the service
//
// wrangler.toml:
//   name = "page-feedback-ingest"
//   main = "cloudflare_worker.js"
//   compatibility_date = "2024-09-23"
//   [[d1_databases]]
//   binding = "FEEDBACK_DB"
//   database_name = "page-feedback"
//   database_id = "<from wrangler d1 create>"
//
// Service side:
//   FEEDBACK_REVIEW_MODE=custom
//   FEEDBACK_WEBHOOK_URL=https://page-feedback-ingest.<account>.workers.dev/v1/events
//   FEEDBACK_WEBHOOK_TOKEN=<secret>
//
// Export the rows for a counter snapshot with
//   wrangler d1 execute page-feedback --remote --json \
//     --command "SELECT event_json FROM feedback_events WHERE site_id = 'my-docs'"
// then write each event_json to a file in the provider layout, or feed the
// events to `write_aggregate` from Python.
//
// Contract: see _service/_webhook.py (page.feedback-delivery.v1). The Worker
// keeps no IP address, header or timestamp; the event already carries only
// page-reaction data.

export const SCHEMA = `CREATE TABLE IF NOT EXISTS feedback_events (
  feedback_id TEXT PRIMARY KEY,
  request_hash TEXT NOT NULL,
  site_id TEXT NOT NULL,
  page_id TEXT NOT NULL,
  event_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_events_site_page ON feedback_events(site_id, page_id);`;

export const PATH = "/v1/events";
export const MAX_BODY_BYTES = 65536;
const FEEDBACK_ID = /^feedback-[0-9a-f]{48}$/;
const HASH = /^[0-9a-f]{64}$/;
const SIGNATURE = /^sha256=([0-9a-f]{64})$/;

function json(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}

function hexToBytes(hex) {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i += 1) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

/**
 * Verify X-Feedback-Signature-256 over the exact body bytes (constant time).
 * @param {string} secret shared secret
 * @param {ArrayBuffer} body raw request body
 * @param {string|null} header signature header value
 * @returns {Promise<boolean>}
 */
export async function verifySignature(secret, body, header) {
  const match = SIGNATURE.exec(String(header || ""));
  if (!secret || !match) return false;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["verify"],
  );
  return crypto.subtle.verify("HMAC", key, hexToBytes(match[1]), body);
}

async function readBounded(request) {
  const declared = Number(request.headers.get("content-length") || "0");
  if (declared > MAX_BODY_BYTES) return null;
  const body = await request.arrayBuffer();
  return body.byteLength > MAX_BODY_BYTES ? null : body;
}

/**
 * Handle one delivery. Exported separately from the Worker entry for tests.
 * @param {Request} request
 * @param {{FEEDBACK_DB: D1Database, FEEDBACK_WEBHOOK_TOKEN: string}} env
 * @returns {Promise<Response>}
 */
export async function handle(request, env) {
  const url = new URL(request.url);
  if (url.pathname !== PATH) return json(404, { error: "not_found" });
  if (request.method !== "POST") return json(405, { error: "method_not_allowed" });
  const body = await readBounded(request);
  if (body === null) return json(413, { error: "too_large" });
  const signature = request.headers.get("x-feedback-signature-256");
  if (!(await verifySignature(env.FEEDBACK_WEBHOOK_TOKEN, body, signature))) {
    return json(401, { error: "bad_signature" });
  }
  let envelope;
  try {
    envelope = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(body));
  } catch {
    return json(400, { error: "invalid_json" });
  }
  const event = envelope && envelope.event;
  if (
    !envelope ||
    envelope.contract !== "page.feedback-delivery.v1" ||
    !FEEDBACK_ID.test(String(envelope.feedback_id)) ||
    !HASH.test(String(envelope.request_hash)) ||
    !event ||
    typeof event !== "object" ||
    !event.feedback ||
    event.feedback.id !== envelope.feedback_id ||
    typeof event.site_id !== "string" ||
    typeof event.page_id !== "string"
  ) {
    return json(422, { error: "invalid_delivery" });
  }
  const db = env.FEEDBACK_DB;
  const inserted = await db
    .prepare(
      "INSERT INTO feedback_events (feedback_id, request_hash, site_id, page_id, event_json) " +
        "VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(feedback_id) DO NOTHING",
    )
    .bind(envelope.feedback_id, envelope.request_hash, event.site_id, event.page_id, JSON.stringify(event))
    .run();
  let status = "accepted";
  if (!inserted.meta || inserted.meta.changes !== 1) {
    const row = await db
      .prepare("SELECT request_hash FROM feedback_events WHERE feedback_id = ?1")
      .bind(envelope.feedback_id)
      .first();
    if (!row || row.request_hash !== envelope.request_hash) {
      return json(409, { error: "conflict" });
    }
    status = "replay";
  }
  return json(202, {
    status,
    feedback_id: envelope.feedback_id,
    request_hash: envelope.request_hash,
  });
}

export default {
  async fetch(request, env) {
    try {
      return await handle(request, env);
    } catch {
      // Never echo internals; the service retries with the same feedback_id.
      return json(503, { error: "unavailable" });
    }
  },
};
