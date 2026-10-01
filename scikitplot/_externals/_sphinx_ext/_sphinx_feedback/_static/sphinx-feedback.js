/* Theme-independent, privacy-minimal page feedback controller. */
(() => {
  'use strict';

  const CONFIG_ID = 'sphinx-feedback-config';
  const PENDING_PREFIX = 'sphinx-feedback:pending:v1:';
  const ACCEPTED_PREFIX = 'sphinx-feedback:accepted:v1:';
  const MAX_RESPONSE_CHUNKS = 512;
  const MAX_RESPONSE_BYTES = 65536;
  const MAX_REQUEST_BYTES = 16 * 1024;
  const MAX_PENDING_STATE_CHARS = 12000;
  const MAX_ACCEPTED_STATE_CHARS = 4096;
  const REQUEST_TIMEOUT_MS = 20000;
  const FEEDBACK_ID_RE = /^feedback-[0-9a-f]{48}$/;
  const HASH_RE = /^[0-9a-f]{64}$/;
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const QUICK_ICONS = {
    down: {
      label: 'Not helpful',
      fallback: '👎',
      viewBox: '0 0 16 16',
      element: 'path',
      attrs: {
        d: 'M7.653 15.369a.75.75 0 0 1-.776.371l-.238-.04a3.25 3.25 0 0 1-2.591-4.099L4.506 10h-.665A3.25 3.25 0 0 1 .723 5.833l1.135-3.859A2.75 2.75 0 0 1 4.482 0H9.43c.78.003 1.538.25 2.168.702A1.752 1.752 0 0 1 12.989 0h1.272A1.75 1.75 0 0 1 16 1.75v6.5A1.75 1.75 0 0 1 14.25 10h-3.417a.25.25 0 0 0-.217.127ZM11.25 2.351l-.396-.33a2.248 2.248 0 0 0-1.44-.521H4.496a1.25 1.25 0 0 0-1.199.897L2.162 6.256A1.75 1.75 0 0 0 3.841 8.5H5.5a.75.75 0 0 1 .721.956l-.731 2.558a1.75 1.75 0 0 0 1.127 2.14L9.31 9.389a1.75 1.75 0 0 1 1.523-.889h.417Zm1.5 6.149h1.5a.25.25 0 0 0 .25-.25v-6.5a.25.25 0 0 0-.25-.25H13a.25.25 0 0 0-.25.25Z',
      },
    },
    up: {
      label: 'Helpful',
      fallback: '👍',
      viewBox: '0 0 16 16',
      element: 'path',
      attrs: {
        d: 'M8.347.631A.75.75 0 0 1 9.123.26l.238.04a3.25 3.25 0 0 1 2.591 4.098L11.494 6h.665a3.25 3.25 0 0 1 3.118 4.167l-1.135 3.859A2.751 2.751 0 0 1 11.503 16H6.586a3.75 3.75 0 0 1-2.184-.702A1.75 1.75 0 0 1 3 16H1.75A1.75 1.75 0 0 1 0 14.25v-6.5C0 6.784.784 6 1.75 6h3.417a.25.25 0 0 0 .217-.127ZM4.75 13.649l.396.33c.404.337.914.521 1.44.521h4.917a1.25 1.25 0 0 0 1.2-.897l1.135-3.859A1.75 1.75 0 0 0 12.159 7.5H10.5a.75.75 0 0 1-.721-.956l.731-2.558a1.75 1.75 0 0 0-1.127-2.14L6.69 6.611a1.75 1.75 0 0 1-1.523.889H4.75ZM3.25 7.5h-1.5a.25.25 0 0 0-.25.25v6.5c0 .138.112.25.25.25H3a.25.25 0 0 0 .25-.25Z',
      },
    },
    details: {
      label: 'Details',
      fallback: '⌄',
      viewBox: '0 0 24 24',
      element: 'polyline',
      attrs: {
        points: '6 9 12 15 18 9',
        fill: 'none',
        stroke: 'currentColor',
        'stroke-width': '2',
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
      },
    },
  };
  const RATINGS = [
    [-5, 'Terrible', '😡'],
    [-4, 'Poor', '😞'],
    [-3, 'Unsatisfied', '😟'],
    [-2, 'No', '🙁'],
    [-1, 'Not really', '😑'],
    [0, 'Neutral', '😐'],
    [1, 'Somewhat', '🙂'],
    [2, 'Mostly yes', '😊'],
    [3, 'Good', '😄'],
    [4, 'Very good', '😁'],
    [5, 'Excellent!', '🤩'],
  ];

  const one = (root, selector) => root?.querySelector?.(selector) || null;
  const all = (root, selector) =>
    root?.querySelectorAll ? [...root.querySelectorAll(selector)] : [];

  function el(tag, attrs, text) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      if (value === null || value === undefined || value === false) return;
      if (key === 'class') node.className = String(value);
      else if (key === 'hidden') node.hidden = Boolean(value);
      else node.setAttribute(key, String(value));
    });
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function quickIcon(kind) {
    const spec = QUICK_ICONS[kind];
    if (!spec) return null;
    try {
      if (typeof document.createElementNS !== 'function') {
        throw new Error('SVG DOM unavailable');
      }
      const svg = document.createElementNS(SVG_NS, 'svg');
      svg.setAttribute('class', 'sphinx-feedback-icon');
      svg.setAttribute('viewBox', spec.viewBox);
      svg.setAttribute('aria-hidden', 'true');
      svg.setAttribute('focusable', 'false');
      if (spec.element === 'path') svg.setAttribute('fill', 'currentColor');
      const shape = document.createElementNS(SVG_NS, spec.element);
      Object.entries(spec.attrs).forEach(([key, value]) => shape.setAttribute(key, value));
      svg.append(shape);
      return svg;
    } catch {
      return el(
        'span',
        {class: 'sphinx-feedback-icon-fallback', 'aria-hidden': 'true'},
        spec.fallback,
      );
    }
  }

  function fillQuickButton(button, kind, count = null) {
    const spec = QUICK_ICONS[kind];
    if (!button || !spec) return;
    const icon = quickIcon(kind);
    if (icon) button.append(icon);
    button.append(
      el(
        'span',
        {class: 'sphinx-feedback-choice-label', 'aria-hidden': 'true'},
        spec.label,
      ),
    );
    if (Number.isSafeInteger(count) && count >= 0) {
      button.append(
        el(
          'span',
          {
            class: 'sphinx-feedback-quick-count',
            'data-sphinx-feedback-quick-count': kind === 'down' ? '-1' : '1',
            'data-feedback-count': String(count),
            'aria-hidden': 'true',
          },
          formatCompactCount(count),
        ),
      );
    }
  }

  function readConfig() {
    const node = document.getElementById(CONFIG_ID);
    if (!node) return null;
    try {
      const value = JSON.parse(node.textContent || '{}');
      if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
      if (value.contract !== 'page.feedback-config.v1') return null;
      return value;
    } catch {
      return null;
    }
  }

  function secureFeedbackId() {
    const cryptoApi = globalThis.crypto;
    if (!cryptoApi?.getRandomValues) return '';
    const bytes = new Uint8Array(24);
    cryptoApi.getRandomValues(bytes);
    return `feedback-${[...bytes]
      .map((value) => value.toString(16).padStart(2, '0'))
      .join('')}`;
  }

  function codePointLength(value) {
    return [...value].length;
  }

  function normalizeCredit(value) {
    let text = String(value || '').normalize('NFC');
    if(/[\u0000-\u001f\u007f]/.test(text)) {
      throw new Error('Public credit must not contain control characters.');
    }
    text = text.replace(/ +/g, ' ').replace(/^ +| +$/g, '');
    if (codePointLength(text) > 80) {
      throw new Error('Public credit must be at most 80 characters.');
    }
    if (text.toLowerCase() === 'anonymous') {
      throw new Error('“Anonymous” is reserved; leave public credit blank instead.');
    }
    return text;
  }

  function normalizeComment(value) {
    const text = String(value || '').normalize('NFC').replace(/^[ \t\n\r]+|[ \t\n\r]+$/g, '');
    if (codePointLength(text) > 2000) {
      throw new Error('Optional details must be at most 2000 characters.');
    }
    if(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(text)) {
      throw new Error('Optional details contain unsupported control characters.');
    }
    return text;
  }

  function stateKey(prefix, config) {
    return (
      prefix +
      encodeURIComponent(config.site_id) +
      ':' +
      encodeURIComponent(config.page_id)
    );
  }

  function currentRevision(config) {
    return typeof config.page_revision === 'string' ? config.page_revision : '';
  }

  function sanitizeStoredRequest(config, request) {
    if (!request || typeof request !== 'object' || Array.isArray(request)) return null;
    if (request.contract !== 'page.feedback-request.v1' || request.action !== 'submit') return null;
    if (request.site_id !== config.site_id || request.page_id !== config.page_id) return null;
    if (!FEEDBACK_ID_RE.test(String(request.feedback_id || ''))) return null;
    if (!Number.isInteger(request.rating) || request.rating < -5 || request.rating > 5) return null;
    if (!['quick', 'detailed'].includes(request.mode)) return null;
    if (request.mode === 'quick' && config.quick_enabled !== true) return null;
    if (request.mode === 'detailed' && config.detailed_enabled !== true) return null;
    if ((request.page_revision || '') !== currentRevision(config)) return null;

    const contributorValue = request.contributor?.display_name;
    if (typeof contributorValue !== 'string') return null;
    let contributor;
    let comment;
    try {
      contributor = normalizeCredit(contributorValue);
      comment = normalizeComment(request.comment || '');
    } catch {
      return null;
    }
    if (request.mode === 'quick') {
      if (![-1, 1].includes(request.rating) || contributor || comment) return null;
    } else {
      if (comment && config.comment_enabled !== true) return null;
      if (contributor && config.contributor_enabled !== true) return null;
    }

    const clean = {
      contract: 'page.feedback-request.v1',
      action: 'submit',
      site_id: config.site_id,
      page_id: config.page_id,
      feedback_id: request.feedback_id,
      rating: request.rating,
      mode: request.mode,
      contributor: {display_name: contributor},
    };
    if (currentRevision(config)) clean.page_revision = currentRevision(config);
    if (comment) clean.comment = comment;
    return clean;
  }

  function requestFingerprint(request) {
    return JSON.stringify([
      request.rating,
      request.mode,
      request.comment || '',
      request.contributor?.display_name || '',
    ]);
  }

  function readPending(config) {
    const key = stateKey(PENDING_PREFIX, config);
    try {
      const raw = sessionStorage.getItem(key);
      if (!raw) return null;
      if (raw.length > MAX_PENDING_STATE_CHARS) {
        sessionStorage.removeItem(key);
        return null;
      }
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object' || parsed.endpoint !== config.endpoint) {
        sessionStorage.removeItem(key);
        return null;
      }
      const request = sanitizeStoredRequest(config, parsed.request);
      if (!request) {
        sessionStorage.removeItem(key);
        return null;
      }
      return {fingerprint: requestFingerprint(request), request, endpoint: config.endpoint};
    } catch {
      try { sessionStorage.removeItem(key); } catch {}
      return null;
    }
  }

  function writePending(config, value) {
    try {
      const key = stateKey(PENDING_PREFIX, config);
      if (value) sessionStorage.setItem(key, JSON.stringify({...value, endpoint: config.endpoint}));
      else sessionStorage.removeItem(key);
    } catch {
      // sessionStorage may be unavailable. In-memory retry still works.
    }
  }

  function readAccepted(config) {
    const key = stateKey(ACCEPTED_PREFIX, config);
    try {
      const raw = sessionStorage.getItem(key);
      if (!raw) return null;
      if (raw.length > MAX_ACCEPTED_STATE_CHARS) {
        sessionStorage.removeItem(key);
        return null;
      }
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') {
        sessionStorage.removeItem(key);
        return null;
      }
      if (
        (parsed.page_revision || '') !== currentRevision(config) ||
        parsed.endpoint !== config.endpoint
      ) {
        sessionStorage.removeItem(key);
        return null;
      }
      if (!Number.isInteger(parsed.rating) || parsed.rating < -5 || parsed.rating > 5) {
        sessionStorage.removeItem(key);
        return null;
      }
      if (!['quick', 'detailed'].includes(parsed.mode)) {
        sessionStorage.removeItem(key);
        return null;
      }
      return {
        rating: parsed.rating,
        mode: parsed.mode,
        page_revision: parsed.page_revision || '',
        endpoint: config.endpoint,
      };
    } catch {
      try { sessionStorage.removeItem(key); } catch {}
      return null;
    }
  }

  function writeAccepted(config, rating, mode) {
    try {
      sessionStorage.setItem(
        stateKey(ACCEPTED_PREFIX, config),
        JSON.stringify({
          rating,
          mode,
          page_revision: currentRevision(config),
          endpoint: config.endpoint,
        }),
      );
    } catch {
      // The marker is a duplicate-submit guard, not required for correctness.
    }
  }

  async function readBoundedJson(response) {
    const declared = Number(response.headers.get('content-length') || 0);
    if (Number.isFinite(declared) && declared > MAX_RESPONSE_BYTES) {
      throw new Error('Feedback service response was too large.');
    }
    if (!response.body || typeof response.body.getReader !== 'function') {
      throw new Error('Streaming feedback response verification is unavailable in this browser.');
    }

    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    let chunkCount = 0;
    try {
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        chunkCount += 1;
        if (chunkCount > MAX_RESPONSE_CHUNKS) {
          try { await reader.cancel('response too fragmented'); } catch {}
          throw new Error('Feedback service response was too fragmented.');
        }
        size += value.byteLength;
        if (size > MAX_RESPONSE_BYTES) {
          try { await reader.cancel('response too large'); } catch {}
          throw new Error('Feedback service response was too large.');
        }
        chunks.push(value);
      }
    } finally {
      try {
        reader.releaseLock();
      } catch {
        // Nothing to release.
      }
    }
    const merged = new Uint8Array(size);
    let offset = 0;
    chunks.forEach((chunk) => {
      merged.set(chunk, offset);
      offset += chunk.byteLength;
    });
    const text = new TextDecoder('utf-8', {fatal: true}).decode(merged);
    return text ? JSON.parse(text) : {};
  }

  function canonicalJson(value) {
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
    if (value && typeof value === 'object') {
      return `{${Object.keys(value)
        .sort()
        .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
        .join(',')}}`;
    }
    return JSON.stringify(value);
  }

  async function requestCommitment(request) {
    if (!globalThis.crypto?.subtle?.digest) {
      throw new Error('Secure request verification is unavailable in this browser.');
    }
    const bytes = new TextEncoder().encode(canonicalJson(request));
    if (bytes.byteLength > MAX_REQUEST_BYTES) {
      throw new Error('Feedback request exceeds the safety limit.');
    }
    const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes));
    return [...digest].map((value) => value.toString(16).padStart(2, '0')).join('');
  }

  async function validateReceipt(receipt, request) {
    if (!receipt || typeof receipt !== 'object' || Array.isArray(receipt)) {
      throw new Error('Feedback service returned an invalid receipt.');
    }
    if (
      receipt.ok !== true ||
      receipt.contract !== 'page.feedback-receipt.v1' ||
      receipt.feedback_id !== request.feedback_id ||
      !['accepted', 'replay'].includes(receipt.status) ||
      !HASH_RE.test(String(receipt.request_hash || ''))
    ) {
      throw new Error('Feedback service returned an unverifiable receipt.');
    }
    const expected = await requestCommitment(request);
    if (receipt.request_hash !== expected) {
      throw new Error('Feedback service receipt does not match the submitted request.');
    }
    return receipt;
  }

  class FeedbackHttpError extends Error {
    constructor(message, status) {
      super(message);
      this.name = 'FeedbackHttpError';
      this.status = Number(status) || 0;
      this.definitive =
        this.status >= 400 &&
        this.status < 500 &&
        ![408, 425, 429].includes(this.status);
    }
  }

  async function submitRequest(config, request) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      await requestCommitment(request);
      const response = await fetch(config.endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json', Accept: 'application/json'},
        body: JSON.stringify(request),
        credentials: 'omit',
        cache: 'no-store',
        redirect: 'error',
        referrerPolicy: 'no-referrer',
        signal: controller.signal,
      });
      let payload;
      try {
        payload = await readBoundedJson(response);
      } catch (error) {
        if (!response.ok) {
          throw new FeedbackHttpError(
            `Feedback service returned HTTP ${response.status}.`,
            response.status,
          );
        }
        throw error;
      }
      if (!response.ok) {
        const detail =
          payload && typeof payload.detail === 'string'
            ? payload.detail
            : `Feedback service returned HTTP ${response.status}.`;
        throw new FeedbackHttpError(detail, response.status);
      }
      return await validateReceipt(payload, request);
    } catch (error) {
      if (error?.name === 'AbortError') throw new Error('Feedback request timed out.');
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  function findFirst(selectors) {
    for (const selector of selectors || []) {
      try {
        const node = document.querySelector(selector);
        if (node) return node;
      } catch {
        // A project-supplied invalid selector should not break universal fallback.
      }
    }
    return null;
  }

  function makeMount(extra = '') {
    return el('div', {
      class: `sphinx-feedback-mount ${extra}`.trim(),
      'data-sphinx-feedback-mount': '',
      'data-sphinx-feedback-layout': 'compact',
    });
  }

  function autoMounts(config) {
    const mounts = [];
    all(document, '[data-sphinx-feedback-mount]').forEach((node) => mounts.push(node));
    const discoveredMain = findFirst(config.main_selectors || []);
    const main = discoveredMain || document.body || null;
    const sidebar = findFirst(config.sidebar_selectors || []);

    function appendUnique(host, kind) {
      if (!host) return null;
      const existing = one(host, `[data-sphinx-feedback-auto="${kind}"]`);
      if (existing) {
        if (!mounts.includes(existing)) mounts.push(existing);
        return existing;
      }
      const mount = makeMount();
      mount.dataset.sphinxFeedbackAuto = kind;
      host.appendChild(mount);
      mounts.push(mount);
      return mount;
    }

    if (config.position !== 'none') {
      if (config.position === 'floating') {
        const existing = one(document, '[data-sphinx-feedback-auto="floating"]');
        if (existing) mounts.push(existing);
        else {
          const mount = makeMount('sphinx-feedback-mount--floating');
          mount.dataset.sphinxFeedbackAuto = 'floating';
          document.body.appendChild(mount);
          mounts.push(mount);
        }
      } else if (config.position === 'main-bottom') {
        appendUnique(main, 'main');
      } else if (config.position === 'sidebar' || config.position === 'auto') {
        if (sidebar) appendUnique(sidebar, 'sidebar');
        else if (config.fallback === 'main-bottom') appendUnique(main, 'main');
      }
    }
    if (config.page_main && main) appendUnique(main, 'main');
    return [...new Set(mounts)];
  }

  /*
   * Developer helper: compact community-count scale. Keep this deterministic
   * rather than locale-dependent so narrow sidebars have stable geometry.
   *
   * Abbreviation   Full number
   * 1K             1,000
   * 10K            10,000
   * 100K           100,000
   * 1M             1,000,000
   * 10M            10,000,000
   * 1B             1,000,000,000
   * 1T             1,000,000,000,000
   *
   * 0..999 stay exact. K/M/B/T use at most one decimal below 100 units,
   * trimming trailing .0 (1K, 1.5K, 2.3M, 10K). Rounding that reaches
   * 1000 promotes to the next unit so 999,500 is 1M, never 1000K.
   */
  function formatCompactCount(count) {
    if (!Number.isSafeInteger(count) || count < 0) return '';
    if (count < 1000) return String(count);
    const units = [
      [1e3, 'K'],
      [1e6, 'M'],
      [1e9, 'B'],
      [1e12, 'T'],
    ];
    let index = units.length - 1;
    while (index > 0 && count < units[index][0]) index -= 1;
    let divisor = units[index][0];
    let suffix = units[index][1];
    let scaled = count / divisor;
    let decimals = scaled < 100 ? 1 : 0;
    let factor = decimals ? 10 : 1;
    let rounded = Math.round((scaled + Number.EPSILON) * factor) / factor;
    if (rounded >= 1000 && index < units.length - 1) {
      index += 1;
      divisor = units[index][0];
      suffix = units[index][1];
      scaled = count / divisor;
      decimals = scaled < 100 ? 1 : 0;
      factor = decimals ? 10 : 1;
      rounded = Math.round((scaled + Number.EPSILON) * factor) / factor;
    }
    const text = decimals ? rounded.toFixed(1).replace(/\.0$/, '') : String(rounded);
    return text + suffix;
  }

  class PageFeedbackController {
    constructor(config) {
      this.config = config;
      this.views = new Set();
      this.busy = false;
      this.selected = null;
      this.status = '';
      this.statusTone = 'neutral';
      this.quick = null;
      this.detailsOpen = false;
      this.pending = readPending(config);
      this.accepted = readAccepted(config);
      this.draftComment = '';
      this.draftContributor = '';

      if (this.accepted) {
        if (this.accepted.mode === 'quick') this.quick = this.accepted.rating;
        else this.selected = this.accepted.rating;
        this.status = 'Feedback already submitted in this tab for this page revision.';
        this.statusTone = 'success';
      } else if (this.pending) {
        const request = this.pending.request;
        if (request.mode === 'quick') {
          this.quick = request.rating;
        } else {
          this.selected = request.rating;
          this.detailsOpen = true;
          this.draftComment = request.comment || '';
          this.draftContributor = request.contributor?.display_name || '';
        }
        this.status = 'A previous feedback attempt may not have completed. Retry will reuse the same feedback id.';
        this.statusTone = 'warning';
      }
    }

    addView(mount) {
      if (!mount) return;
      for (const view of this.views) {
        if (view.mount === mount) return;
      }
      mount.dataset.sphinxFeedbackBound = 'true';
      if (mount.dataset.sphinxFeedbackLayout === 'full' && this.config.detailed_enabled) {
        this.detailsOpen = true;
      }
      const view = this.renderView(mount);
      this.views.add(view);
      this.sync();
    }

    renderView(mount) {
      mount.replaceChildren();
      const root = el('div', {
        class: 'sphinx-feedback',
        'data-sphinx-feedback': '',
        'data-feedback-tone': 'neutral',
      });
      const row = el('div', {class: 'sphinx-feedback-row'});
      const prompt = el('span', {class: 'sphinx-feedback-label'}, 'Was This Helpful?');
      const actions = el('div', {
        class: 'sphinx-feedback-quick',
        role: 'group',
        'aria-label': 'Feedback actions',
      });

      const counter = this.config.counter;
      const hasDistribution =
        counter &&
        Number.isSafeInteger(counter.count) &&
        counter.count >= 0 &&
        Number.isSafeInteger(counter.negative_count) &&
        counter.negative_count >= 0 &&
        Number.isSafeInteger(counter.positive_count) &&
        counter.positive_count >= 0 &&
        Number.isSafeInteger(counter.neutral_count) &&
        counter.neutral_count >= 0 &&
        counter.negative_count + counter.positive_count + counter.neutral_count === counter.count;
      const negativeCount = hasDistribution ? counter.negative_count : null;
      const positiveCount = hasDistribution ? counter.positive_count : null;

      let down = null;
      let up = null;
      if (this.config.quick_enabled) {
        down = el('button', {
          type: 'button',
          class: 'sphinx-feedback-quick-btn',
          'data-rating': '-1',
          'data-tone': 'negative',
          'data-feedback-count': hasDistribution ? String(negativeCount) : null,
          'aria-label': 'Not helpful (-1)',
          'aria-pressed': 'false',
          title: hasDistribution
            ? `Not helpful (-1) · ${negativeCount} negative rating${negativeCount === 1 ? '' : 's'}`
            : 'Not helpful (-1)',
        });
        fillQuickButton(down, 'down', negativeCount);
        up = el('button', {
          type: 'button',
          class: 'sphinx-feedback-quick-btn',
          'data-rating': '1',
          'data-tone': 'positive',
          'data-feedback-count': hasDistribution ? String(positiveCount) : null,
          'aria-label': 'Helpful (+1)',
          'aria-pressed': 'false',
          title: hasDistribution
            ? `Helpful (+1) · ${positiveCount} positive rating${positiveCount === 1 ? '' : 's'}`
            : 'Helpful (+1)',
        });
        fillQuickButton(up, 'up', positiveCount);
        actions.append(down, up);
      }

      let expand = null;
      if (this.config.detailed_enabled) {
        expand = el('button', {
          type: 'button',
          class: 'sphinx-feedback-expand',
          'aria-label': 'Detailed feedback options',
          'aria-expanded': 'false',
          title: 'Detailed feedback options',
        });
        fillQuickButton(expand, 'details');
        actions.append(expand);
      }

      const summary = el('div', {class: 'sphinx-feedback-summary'});
      if (
        counter &&
        Number.isSafeInteger(counter.count) &&
        counter.count >= 0 &&
        Number.isSafeInteger(counter.score)
      ) {
        const count = counter.count;
        const score = counter.score;
        summary.setAttribute(
          'aria-label',
          `Community feedback score ${score} from ${count} rating${count === 1 ? '' : 's'}`,
        );
        summary.append(
          el('span', {class: 'sphinx-feedback-sr-only'}, String(score)),
          el(
            'span',
            {class: 'sphinx-feedback-count'},
            `${formatCompactCount(count)} rating${count === 1 ? '' : 's'}`,
          ),
        );
      } else {
        summary.hidden = true;
      }
      row.append(prompt, actions, summary);
      root.append(row);

      let detail = null;
      let submit = null;
      let comment = null;
      let contributor = null;
      const ratingButtons = [];
      if (this.config.detailed_enabled) {
        detail = el('div', {
          class: 'sphinx-feedback-detail',
          role: 'group',
          'aria-label': 'Detailed feedback',
          hidden: true,
        });
        detail.append(
          el(
            'span',
            {class: 'sphinx-feedback-sr-only'},
            'Choose a detailed feedback rating from minus five to plus five.',
          ),
        );
        const options = el('div', {
          class: 'sphinx-feedback-options',
          role: 'group',
          'aria-label': 'Feedback rating from minus five to plus five',
        });
        RATINGS.forEach(([value, label, emoji]) => {
          const signed = value > 0 ? `+${value}` : String(value);
          const button = el('button', {
            type: 'button',
            class: 'sphinx-feedback-rating',
            'data-rating': String(value),
            'data-tone': value < 0 ? 'negative' : value > 0 ? 'positive' : 'neutral',
            'aria-pressed': 'false',
            'aria-label': `${label} (${signed})`,
            title: `${label} (${signed})`,
          });
          button.append(
            el('span', {class: 'sphinx-feedback-rating-emoji', 'aria-hidden': 'true'}, emoji),
            el('span', {class: 'sphinx-feedback-rating-value', 'aria-hidden': 'true'}, signed),
          );
          options.append(button);
          ratingButtons.push(button);
        });
        detail.append(options);

        if (this.config.comment_enabled) {
          const field = el('label', {class: 'sphinx-feedback-field'}, 'Optional details');
          comment = el('textarea', {
            maxlength: '2000',
            rows: '3',
            placeholder: 'What worked, what did not, or what should improve?',
          });
          comment.value = this.draftComment;
          comment.addEventListener('input', () => {
            this.draftComment = comment.value;
            this.sync();
          });
          field.append(comment);
          detail.append(field);
        }
        if (this.config.contributor_enabled) {
          const field = el(
            'label',
            {class: 'sphinx-feedback-field'},
            'Public credit · optional',
          );
          contributor = el('input', {
            type: 'text',
            maxlength: '80',
            autocomplete: 'off',
            placeholder: 'Nickname or name',
          });
          contributor.value = this.draftContributor;
          contributor.addEventListener('input', () => {
            this.draftContributor = contributor.value;
            this.sync();
          });
          field.append(contributor);
          detail.append(field);
        }
        detail.append(
          el(
            'p',
            {class: 'sphinx-feedback-meta'},
            'Anonymous by default. Only the rating and page routing identifiers are required. Optional details and public credit are sent only when you enter them. No page-view telemetry is sent by this widget.',
          ),
        );
        const submitActions = el('div', {class: 'sphinx-feedback-actions'});
        submit = el(
          'button',
          {type: 'button', class: 'sphinx-feedback-submit', disabled: 'disabled'},
          'Send feedback',
        );
        submitActions.append(submit);
        detail.append(submitActions);
        root.append(detail);
      }

      const status = el('p', {
        class: 'sphinx-feedback-status',
        role: 'status',
        'aria-live': 'polite',
        hidden: true,
      });
      root.append(status);
      mount.append(root);

      const view = {
        mount,
        root,
        down,
        up,
        expand,
        summary,
        detail,
        submit,
        comment,
        contributor,
        ratingButtons,
        status,
      };

      down?.addEventListener('click', () => this.send(-1, 'quick', '', ''));
      up?.addEventListener('click', () => this.send(1, 'quick', '', ''));
      expand?.addEventListener('click', () => {
        this.detailsOpen = !this.detailsOpen;
        this.sync();
        if (this.detailsOpen) ratingButtons[5]?.focus();
      });
      ratingButtons.forEach((button) =>
        button.addEventListener('click', () => {
          this.selected = Number(button.dataset.rating);
          this.sync();
        }),
      );
      submit?.addEventListener('click', () => {
        if (this.selected === null) return;
        this.send(
          this.selected,
          'detailed',
          this.draftComment,
          this.draftContributor,
        );
      });
      return view;
    }

    announce(message, tone = 'neutral') {
      this.status = String(message || '');
      this.statusTone = tone;
      this.sync();
    }

    sync() {
      for (const view of [...this.views]) {
        if (view.mount?.isConnected === false) this.views.delete(view);
      }
      const transportLocked = this.busy || Boolean(this.accepted) || !this.config.endpoint;
      const pendingRequest = this.pending?.request || null;
      const hasPending = Boolean(pendingRequest) && !this.accepted;
      this.views.forEach((view) => {
        view.root.dataset.feedbackTone = this.statusTone;
        if (view.down) {
          view.down.setAttribute('aria-pressed', String(this.quick === -1));
          view.down.disabled =
            transportLocked ||
            (hasPending && !(pendingRequest.mode === 'quick' && pendingRequest.rating === -1));
        }
        if (view.up) {
          view.up.setAttribute('aria-pressed', String(this.quick === 1));
          view.up.disabled =
            transportLocked ||
            (hasPending && !(pendingRequest.mode === 'quick' && pendingRequest.rating === 1));
        }
        if (view.expand) {
          view.expand.setAttribute('aria-expanded', String(this.detailsOpen));
          view.expand.disabled = this.busy || Boolean(this.accepted) || hasPending;
        }
        if (view.detail) view.detail.hidden = !this.detailsOpen;
        view.ratingButtons.forEach((button) => {
          button.setAttribute(
            'aria-pressed',
            String(Number(button.dataset.rating) === this.selected),
          );
          button.disabled = transportLocked || hasPending;
        });
        if (view.submit) {
          const retryingDetail = hasPending && pendingRequest.mode === 'detailed';
          view.submit.disabled =
            transportLocked ||
            this.selected === null ||
            (hasPending && !retryingDetail);
        }
        if (view.comment) {
          if (view.comment.value !== this.draftComment) view.comment.value = this.draftComment;
          view.comment.disabled = transportLocked || hasPending;
        }
        if (view.contributor) {
          if (view.contributor.value !== this.draftContributor) {
            view.contributor.value = this.draftContributor;
          }
          view.contributor.disabled = transportLocked || hasPending;
        }
        if (view.status) {
          view.status.textContent = this.status;
          view.status.hidden = !this.status;
        }
      });
    }

    async send(rating, mode, rawComment, rawContributor) {
      if (this.busy || this.accepted) return;
      if (mode === 'quick' && !this.config.quick_enabled) return;
      if (mode === 'detailed' && !this.config.detailed_enabled) return;
      if (!this.config.endpoint) {
        this.announce('Feedback service is not configured for this site.', 'error');
        return;
      }

      let comment;
      let contributor;
      try {
        comment = mode === 'detailed' ? normalizeComment(rawComment) : '';
        contributor = mode === 'detailed' ? normalizeCredit(rawContributor) : '';
      } catch (error) {
        this.announce(String(error?.message || error), 'error');
        return;
      }

      const fingerprint = JSON.stringify([rating, mode, comment, contributor]);
      let pending = this.pending && this.pending.fingerprint === fingerprint ? this.pending : null;
      if (!pending) {
        // If an ambiguous request is already pending, do not mint a second event
        // merely because the user changed the form. That would defeat exactly-once
        // retry semantics and could inflate counters after a lost response.
        if (this.pending) {
          this.announce(
            'A previous feedback attempt is pending. Retry that feedback before changing the response.',
            'warning',
          );
          return;
        }
        const id = secureFeedbackId();
        if (!id) {
          this.announce(
            'Secure browser randomness is unavailable; feedback was not sent.',
            'error',
          );
          return;
        }
        const request = {
          contract: 'page.feedback-request.v1',
          action: 'submit',
          site_id: this.config.site_id,
          page_id: this.config.page_id,
          feedback_id: id,
          rating: Number(rating),
          mode,
          contributor: {display_name: contributor},
        };
        if (currentRevision(this.config)) request.page_revision = currentRevision(this.config);
        if (comment) request.comment = comment;
        pending = {fingerprint, request, endpoint: this.config.endpoint};
        this.pending = pending;
        writePending(this.config, pending);
      }

      if (pending.request.mode === 'quick') this.quick = pending.request.rating;
      this.busy = true;
      this.announce('Sending feedback…');
      try {
        const receipt = await submitRequest(this.config, pending.request);
        this.pending = null;
        writePending(this.config, null);
        this.accepted = {
          rating: pending.request.rating,
          mode: pending.request.mode,
          page_revision: pending.request.page_revision || '',
        };
        writeAccepted(this.config, pending.request.rating, pending.request.mode);
        if (pending.request.mode === 'quick') this.quick = pending.request.rating;
        else this.selected = pending.request.rating;
        this.draftComment = '';
        this.draftContributor = '';
        this.detailsOpen = false;

        const mirrorDegraded =
          receipt?.mirrors && Object.values(receipt.mirrors).some((value) => value !== 'ok');
        if (receipt?.status === 'replay') {
          this.announce(
            'Feedback was already received; the retry was handled idempotently.',
            mirrorDegraded ? 'warning' : 'success',
          );
        } else {
          this.announce(
            mirrorDegraded
              ? 'Feedback accepted; a backup mirror is temporarily degraded.'
              : 'Feedback accepted for review.',
            mirrorDegraded ? 'warning' : 'success',
          );
        }
      } catch (error) {
        if (error?.definitive) {
          if (pending.request.mode === 'quick') this.quick = null;
          this.pending = null;
          writePending(this.config, null);
          this.announce(
            `Feedback was rejected safely: ${String(error?.message || error)} You can revise it and try again.`,
            'error',
          );
        } else {
          this.announce(
            `Unable to confirm feedback delivery: ${String(error?.message || error)} Retry will reuse the same feedback id.`,
            'error',
          );
        }
      } finally {
        this.busy = false;
        this.sync();
      }
    }
  }

  function boot() {
    const config = readConfig();
    if (!config || config.enabled !== true || !config.site_id || !config.page_id) return;
    const current = window.__SPHINX_FEEDBACK_CONTROLLER__;
    if (current?.config && canonicalJson(current.config) === canonicalJson(config)) {
      autoMounts(config).forEach((mount) => current.addView(mount));
      return;
    }
    if (current) {
      all(document, '[data-sphinx-feedback-auto]').forEach((mount) => mount.remove());
    }
    const controller = new PageFeedbackController(config);
    window.__SPHINX_FEEDBACK_CONTROLLER__ = controller;
    autoMounts(config).forEach((mount) => controller.addView(mount));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, {once: true});
  } else {
    boot();
  }
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) boot();
  });
})();
