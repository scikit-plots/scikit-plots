/**
 * @fileoverview Cloudflare Worker: HuggingFace Inference API Proxy
 *
 * @description
 * Accepts POST /v1/chat/completions from the browser (no auth header).
 * Adds Authorization: Bearer $HF_TOKEN from the Worker's encrypted secrets.
 * Forwards the request to the HuggingFace Serverless Inference API.
 * Returns the response (JSON or SSE stream) with CORS headers.
 *
 * @remarks
 * **Limitations (free tier):**
 * - 100,000 requests/day
 * - 10 ms CPU time per request (network I/O wait does NOT count toward CPU)
 * - 30-second wall-clock limit per request (adequate for most completions)
 *
 * **Model IDs:**
 * Only works with models that have a registered HF Inference Provider.
 * Use original repo IDs, NOT scikit-plots/* mirrors:
 *   ✓  Qwen/Qwen2.5-Coder-32B-Instruct   (default — confirmed provider on router)
 *   ✓  Qwen/Qwen2.5-72B-Instruct
 *   ✗  scikit-plots/Qwen2.5-Coder-32B-Instruct  (mirror — no provider → 404/503)
 *      scikit-plots/* models require the full HF Spaces proxy with Path-2 routing.
 *
 * @setup
 * 1. `npm create cloudflare@latest -- hf-proxy` (NOT `wrangler init` — removed in v3)
 * 2. Replace src/index.js with this file.
 * 3. `wrangler secret put HF_TOKEN`   (interactive — paste token; never in source)
 * 4. `wrangler deploy`
 * 5. Note the deployed URL for conf.py.
 *
 * @see {@link https://developers.cloudflare.com/workers/platform/limits/}
 * @see {@link https://developers.cloudflare.com/workers/wrangler/commands/}
 */

/** @constant {string} Default model ID (must have a registered Inference Provider). */
const DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct";

/** @constant {string} HuggingFace Serverless Inference API base URL (v6.0.0+: router). */
const HF_BASE = "https://router.huggingface.co";

/**
 * Build the standard CORS response-header object.
 *
 * @returns {Object} CORS headers permitting cross-origin GET/POST from any origin.
 *
 * @remarks
 * Developer: `Authorization` is required for write endpoints (POST /v1/share,
 * POST /v1/feedback) that validate a Bearer token.  Without it the browser
 * preflight blocks the request before the handler runs.
 *
 * Developer: `GET` is required for `GET /v1/share/:uuid` share retrieval.
 * The original `POST, OPTIONS` only list caused share link opens to fail.
 */
function corsHeaders() {
  return {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  };
}

/**
 * Extract the `model` field from a JSON string body.
 *
 * @param {string} bodyText - Raw request body text (expected to be JSON).
 * @returns {string} The `model` value, or {@link DEFAULT_MODEL} on any error.
 *
 * @remarks
 * Never throws.  A malformed body falls back to `DEFAULT_MODEL` so the
 * upstream call proceeds and the HF API error message reaches the browser.
 */
function parseModel(bodyText) {
  try {
    const parsed = JSON.parse(bodyText);
    const candidate = (parsed.model ?? "").trim();
    return candidate || DEFAULT_MODEL;
  } catch {
    return DEFAULT_MODEL;
  }
}

/**
 * Build the upstream HuggingFace Inference API URL.
 *
 * @returns {string} Fully-qualified upstream endpoint URL.
 *
 * @remarks
 * `router.huggingface.co` uses a **flat** endpoint — the model is selected
 * via the `"model"` field in the request body, not the URL path.
 * Contrast with the legacy `api-inference.huggingface.co/models/{model}/...`
 * pattern that embedded the model ID in the path.
 */
function buildUpstreamUrl() {
  return `${HF_BASE}/v1/chat/completions`;
}

/**
 * Emit a structured JSON log entry to the Cloudflare Worker log stream.
 *
 * @param {'info'|'warn'|'error'} level - Log severity.
 * @param {string} event                - Short machine-readable event name.
 * @param {Object} [fields]             - Additional structured fields.
 *
 * @remarks
 * Developer: JSON format is required for Cloudflare Logpush integration
 * (R2, S3, Datadog).  Text-format lines require regex in log queries;
 * JSON fields are natively queryable.
 *
 * Developer: `ts` is milliseconds epoch so log consumers can correlate
 * across time zones without format ambiguity.
 *
 * @example
 * _log('info',  'share.write',    { uuid, bytes: body.length, ttlDays });
 * _log('warn',  'share.ratelimit',{ ip: ipHash, count });
 * _log('error', 'kv.write_fail',  { uuid, error: err.message });
 */
function _log(level, event, fields) {
  const entry = Object.assign({ level, event, ts: Date.now() }, fields || {});
  const line = JSON.stringify(entry);
  if (level === 'error') {
    console.error(line);
  } else if (level === 'warn') {
    console.warn(line);
  } else {
    console.log(line);
  }
}

/**
 * Compute a short non-cryptographic hash of a string for use in KV keys.
 *
 * @param {string} str - Input string (typically a client IP address).
 * @returns {string}   Unsigned hex string.
 *
 * @remarks
 * Developer: This is used for rate-limit KV keys only — not for security.
 * The real client IP comes from CF-Connecting-IP (injected by Cloudflare
 * edge; not spoofable from the internet unlike X-Forwarded-For).
 * We hash before storing so raw IP addresses never appear in KV keys.
 */
function _ipHash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
  }
  return (h >>> 0).toString(16);
}

/**
 * Enforce a per-IP sliding-window rate limit using KV counters.
 *
 * @param {KVNamespace} kv      - Worker KV namespace.
 * @param {string}      prefix  - Key prefix, e.g. 'rl:fb'.
 * @param {string}      ipHash  - Hashed client IP from {@link _ipHash}.
 * @param {number}      limit   - Maximum allowed requests per window.
 * @param {number}      windowS - Window size in seconds (TTL on the counter key).
 * @returns {Promise<{allowed: boolean, count: number}>}
 *
 * @remarks
 * Developer: KV TTL resets on every write.  This is a "fixed window" limiter,
 * not a sliding window, but it is sufficient for the abuse prevention goal
 * here.  The counter key expires naturally — no cleanup job needed.
 */
async function _rateLimit(kv, prefix, ipHash, limit, windowS) {
  const key = `${prefix}:${ipHash}`;
  const raw = await kv.get(key);
  const count = raw ? parseInt(raw, 10) + 1 : 1;
  await kv.put(key, String(count), { expirationTtl: windowS });
  return { allowed: count <= limit, count };
}

/**
 * Attempt a KV put with one retry on failure.
 *
 * @param {KVNamespace} kv      - Worker KV namespace.
 * @param {string}      key     - KV key.
 * @param {string}      value   - KV value (JSON string).
 * @param {Object}      [opts]  - KV put options (e.g. expirationTtl).
 * @param {number}      [retries=2] - Maximum attempts.
 * @returns {Promise<void>}
 * @throws {Error} After all retries are exhausted.
 *
 * @remarks
 * Developer: Cloudflare KV writes occasionally fail at the edge storage layer.
 * A single retry with a 100 ms delay recovers from transient failures without
 * meaningfully increasing latency for successful writes.
 */
async function _kvPut(kv, key, value, opts, retries = 2) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      await kv.put(key, value, opts || {});
      return;
    } catch (err) {
      if (attempt === retries) throw err;
      await new Promise(r => setTimeout(r, 100 * attempt));
    }
  }
}

/**
 * Main Worker fetch handler.
 *
 * @param {Request} request - Incoming HTTP request from the browser.
 * @param {Object}  env     - Worker environment.  `HF_TOKEN` lives here as
 *                            a Wrangler secret (encrypted; never in source).
 * @returns {Promise<Response>} Proxied response from HuggingFace.
 *
 * @remarks
 * **CORS preflight** — Browsers send OPTIONS before every cross-origin POST.
 * A 204 response with CORS headers is required or the POST is blocked.
 *
 * **Token guard** — Fails fast with 500 when `HF_TOKEN` is not set, so the
 * operator sees a clear error in the Worker logs rather than a cryptic 401.
 *
 * **Body parsing** — Body is read once as text and passed both to
 * {@link parseModel} (for model extraction) and to the upstream fetch (as
 * the forwarded body).  This avoids double-reading the body stream.
 */
export default {
  async fetch(request, env) {
    // URL parsed once — needed by all route handlers below.
    const url = new URL(request.url);

    // ── CORS Preflight ──────────────────────────────────────────────────────
    // Every cross-origin POST is preceded by a browser OPTIONS preflight.
    // Without this handler, the preflight gets a 404 and the POST is blocked.
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(),
      });
    }

    // ── POST /v1/feedback ────────────────────────────────────────────────────
    if (request.method === 'POST' && url.pathname === '/v1/feedback') {
      // Token guard
      const providedToken = (request.headers.get('Authorization') || '').replace(/^Bearer\s+/, '');
      if (!env.FEEDBACK_WRITE_TOKEN || providedToken !== env.FEEDBACK_WRITE_TOKEN) {
        _log('warn', 'feedback.auth_fail', { path: '/v1/feedback' });
        return new Response(JSON.stringify({ error: 'Unauthorized.' }), {
          status: 401, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      // Body size guard (64 KB)
      const MAX_FB_BYTES = 65536;
      const fbText = await request.text();
      if (fbText.length > MAX_FB_BYTES) {
        return new Response(JSON.stringify({ error: 'Payload too large. Maximum 64 KB.' }), {
          status: 413, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      let fb;
      try { fb = JSON.parse(fbText); } catch {
        return new Response(JSON.stringify({ error: 'Invalid JSON body.' }), {
          status: 400, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      // Rate limit: 100 feedback posts per IP per hour
      const fbIpHash = _ipHash(request.headers.get('CF-Connecting-IP') || 'unknown');
      const fbRl = await _rateLimit(env.SHARE_KV, 'rl:fb', fbIpHash, 100, 3600);
      if (!fbRl.allowed) {
        _log('warn', 'feedback.ratelimit', { ip: fbIpHash, count: fbRl.count });
        return new Response(JSON.stringify({ error: 'Rate limit exceeded. Try again in an hour.' }), {
          status: 429,
          headers: { 'Content-Type': 'application/json', 'Retry-After': '3600', ...corsHeaders() },
        });
      }
      // Deduplication on (sessionId, answerIndex)
      const sid = String(fb.sessionId || '');
      const aidx = String(fb.answerIndex ?? '');
      if (sid && aidx !== '') {
        const dedupKey = `fb:dedup:${sid}:${aidx}`;
        const existing = await env.SHARE_KV.get(dedupKey);
        if (existing) {
          _log('info', 'feedback.duplicate', { sessionId: sid, answerIndex: aidx });
          return new Response(JSON.stringify({ stored: false, duplicate: true }), {
            status: 200, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
          });
        }
        // Mark dedup key (TTL: 90 days)
        await env.SHARE_KV.put(dedupKey, '1', { expirationTtl: 7776000 });
      }
      // Store feedback entry
      const fbUuid = crypto.randomUUID();
      const fbEntry = JSON.stringify({ ...fb, _uuid: fbUuid, _ts: Date.now() });
      try {
        await _kvPut(env.SHARE_KV, `fb:${fbUuid}`, fbEntry, { expirationTtl: 7776000 });
      } catch (err) {
        _log('error', 'feedback.kv_fail', { uuid: fbUuid, error: err.message });
        return new Response(JSON.stringify({ error: 'Storage error. Please try again.' }), {
          status: 503, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      _log('info', 'feedback.write', { uuid: fbUuid, sessionId: sid, answerIndex: aidx, ip: fbIpHash });
      return new Response(JSON.stringify({ stored: true, uuid: fbUuid }), {
        status: 200, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
      });
    }

    // ── POST /v1/share ───────────────────────────────────────────────────────
    if (request.method === 'POST' && url.pathname === '/v1/share') {
      // Token guard
      const shToken = (request.headers.get('Authorization') || '').replace(/^Bearer\s+/, '');
      if (!env.SHARE_WRITE_TOKEN || shToken !== env.SHARE_WRITE_TOKEN) {
        _log('warn', 'share.auth_fail', {});
        return new Response(JSON.stringify({ error: 'Unauthorized.' }), {
          status: 401, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      // Body size guard (500 KB — HTML exports can be large)
      const MAX_SHARE_BYTES = 512000;
      const shareText = await request.text();
      if (shareText.length > MAX_SHARE_BYTES) {
        return new Response(JSON.stringify({ error: 'Payload too large. Maximum 500 KB.' }), {
          status: 413, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      let shPayload;
      try { shPayload = JSON.parse(shareText); } catch {
        return new Response(JSON.stringify({ error: 'Invalid JSON body.' }), {
          status: 400, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      // Rate limit: 20 shares per IP per hour
      const shIpHash = _ipHash(request.headers.get('CF-Connecting-IP') || 'unknown');
      const shRl = await _rateLimit(env.SHARE_KV, 'rl:sh', shIpHash, 20, 3600);
      if (!shRl.allowed) {
        _log('warn', 'share.ratelimit', { ip: shIpHash, count: shRl.count });
        return new Response(JSON.stringify({ error: 'Rate limit exceeded. Try again in an hour.' }), {
          status: 429,
          headers: { 'Content-Type': 'application/json', 'Retry-After': '3600', ...corsHeaders() },
        });
      }
      // Validate required fields
      const content  = String(shPayload.content  || '');
      const mimeType = String(shPayload.mimeType || 'text/plain;charset=utf-8');
      const ext      = String(shPayload.ext      || '.txt');
      const title    = String(shPayload.title    || 'Shared conversation');
      const ttlDays  = Math.max(1, Math.min(365, parseInt(shPayload.ttlDays, 10) || 30));
      const ttlS     = ttlDays * 86400;
      if (!content.trim()) {
        return new Response(JSON.stringify({ error: 'content field is required and must be non-empty.' }), {
          status: 422, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      // Server-generated UUID — client must never choose its own
      const shareUuid = crypto.randomUUID();
      const expiresAt = new Date(Date.now() + ttlS * 1000).toISOString();
      const shEntry = JSON.stringify({ content, mimeType, ext, title, ts: Date.now(), expiresAt });
      try {
        await _kvPut(env.SHARE_KV, `sh:${shareUuid}`, shEntry, { expirationTtl: ttlS });
      } catch (err) {
        _log('error', 'share.kv_fail', { uuid: shareUuid, error: err.message });
        return new Response(JSON.stringify({ error: 'Storage error. Please try again.' }), {
          status: 503, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      const shareUrl = `${url.origin}/v1/share/${shareUuid}`;
      _log('info', 'share.write', { uuid: shareUuid, bytes: content.length, ttlDays, ip: shIpHash });
      return new Response(JSON.stringify({ uuid: shareUuid, url: shareUrl, expiresAt }), {
        status: 200, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
      });
    }

    // ── GET /v1/share/:uuid ──────────────────────────────────────────────────
    if (request.method === 'GET' && url.pathname.startsWith('/v1/share/')) {
      const shareUuid = url.pathname.slice('/v1/share/'.length);
      if (!shareUuid || shareUuid.includes('/')) {
        return new Response(JSON.stringify({ error: 'Invalid share UUID.' }), {
          status: 400, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      const raw = await env.SHARE_KV.get(`sh:${shareUuid}`);
      if (!raw) {
        _log('info', 'share.miss', { uuid: shareUuid });
        return new Response(JSON.stringify({ error: 'Share not found or expired.' }), {
          status: 404, headers: { 'Content-Type': 'application/json', ...corsHeaders() },
        });
      }
      let shEntry;
      try { shEntry = JSON.parse(raw); } catch {
        return new Response('Internal error: corrupt share entry.', { status: 500 });
      }
      // Security headers — prevent XSS when serving stored HTML
      const secHeaders = {
        'Content-Type':              shEntry.mimeType || 'text/plain;charset=utf-8',
        'Content-Security-Policy':   "default-src 'none'; style-src 'unsafe-inline'; img-src data:;",
        'X-Content-Type-Options':    'nosniff',
        'X-Frame-Options':           'DENY',
        'Referrer-Policy':           'no-referrer',
        'Cache-Control':             'public, max-age=3600, stale-while-revalidate=86400',
        ...corsHeaders(),
      };
      _log('info', 'share.read', { uuid: shareUuid });
      return new Response(shEntry.content, { status: 200, headers: secHeaders });
    }

    // ── Method Guard ────────────────────────────────────────────────────────
    if (request.method !== "POST") {
      return new Response(
        JSON.stringify({
          error: "Method Not Allowed.  Use POST /v1/chat/completions.",
        }),
        {
          status: 405,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders(),
          },
        },
      );
    }

    // ── Token Guard ─────────────────────────────────────────────────────────
    // Fail fast with a clear message rather than a cryptic 401 from HF.
    if (!env.HF_TOKEN) {
      return new Response(
        JSON.stringify({
          error:
            "Server configuration error: HF_TOKEN secret is not set.  "
            + "Run: wrangler secret put HF_TOKEN",
        }),
        {
          status: 500,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders(),
          },
        },
      );
    }

    // ── Parse Body ──────────────────────────────────────────────────────────
    let bodyText;
    try {
      bodyText = await request.text();
    } catch {
      return new Response(
        JSON.stringify({ error: "Failed to read request body." }),
        {
          status: 400,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders(),
          },
        },
      );
    }

    // ── Resolve Upstream URL ────────────────────────────────────────────────
    // Extract model ID from the request body (falls back to DEFAULT_MODEL).
    // IMPORTANT: use original repo IDs — scikit-plots/* mirrors have no provider.
    // router.huggingface.co uses a flat endpoint; model stays in the request body.
    const model = parseModel(bodyText);
    const upstreamUrl = buildUpstreamUrl();

    // ── Forward to HuggingFace ──────────────────────────────────────────────
    // env.HF_TOKEN is a Worker secret — encrypted at rest, never in source.
    // Server-to-server requests are not subject to CORS restrictions.
    let hfResponse;
    try {
      hfResponse = await fetch(upstreamUrl, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${env.HF_TOKEN}`,
          "Content-Type":  "application/json",
        },
        body: bodyText,
      });
    } catch (err) {
      return new Response(
        JSON.stringify({
          error: `Failed to reach HuggingFace API: ${err.message}`,
        }),
        {
          status: 502,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders(),
          },
        },
      );
    }

    // ── Return Response ─────────────────────────────────────────────────────
    // Preserve the upstream content-type (JSON or text/event-stream for SSE).
    // Add CORS headers so the browser accepts the response.
    const contentType =
      hfResponse.headers.get("content-type") ?? "application/json";

    return new Response(hfResponse.body, {
      status:  hfResponse.status,
      headers: {
        "Content-Type": contentType,
        ...corsHeaders(),
      },
    });
  },
};
