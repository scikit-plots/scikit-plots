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
 *   ✓  openai/gpt-oss-20b
 *   ✓  Qwen/Qwen2.5-Coder-32B-Instruct
 *   ✗  scikit-plots/gpt-oss-20b   (mirror — no provider → 404/503)
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
const DEFAULT_MODEL = "openai/gpt-oss-20b";

/** @constant {string} HuggingFace Serverless Inference API base URL. */
const HF_BASE = "https://api-inference.huggingface.co/models";

/**
 * Build the standard CORS response-header object.
 *
 * @returns {Object} CORS headers permitting cross-origin POST from any origin.
 *
 * @remarks
 * The explicit `Access-Control-Allow-Headers` value matches exactly what
 * the browser sends in a preflight `Access-Control-Request-Headers` for
 * `fetch()` with `Content-Type: application/json`.
 */
function corsHeaders() {
  return {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
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
 * Build the upstream HuggingFace Inference API URL for a given model.
 *
 * @param {string} model - HuggingFace model ID (must have Inference Provider).
 * @returns {string} Fully-qualified upstream endpoint URL.
 */
function buildUpstreamUrl(model) {
  return `${HF_BASE}/${model}/v1/chat/completions`;
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

    // ── CORS Preflight ──────────────────────────────────────────────────────
    // Every cross-origin POST is preceded by a browser OPTIONS preflight.
    // Without this handler, the preflight gets a 404 and the POST is blocked.
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(),
      });
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
    const model = parseModel(bodyText);
    const upstreamUrl = buildUpstreamUrl(model);

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
