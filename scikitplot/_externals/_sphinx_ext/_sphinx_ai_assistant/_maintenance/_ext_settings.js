/* eslint-disable no-var, vars-on-top */
// =============================================================================
// _ExtSettings — centralized, validated settings registry  (Option A engine)
// =============================================================================
//
// PURPOSE
// -------
// Give the "Extended Settings" section the same architectural backbone that
// the endpoint registry (`_EP`) already gives profiles:
//
//   • ONE validated write path         — `set()` is the only mutator; every
//                                         write is schema-checked, so the
//                                         section's safety is a property of the
//                                         DESIGN, not of each author remembering
//                                         to re-implement validation.
//   • ONE storage blob                 — a single versioned localStorage key
//                                         instead of four scattered keys with
//                                         four different persistence tiers.
//   • Frozen reads                     — `get()` / `all()` never hand out a
//                                         mutable reference to internal state.
//   • Prototype-pollution safe (V-01)  — null-prototype maps + a strict key
//                                         allow-list regex block `__proto__`,
//                                         `constructor`, `prototype`, etc. on
//                                         BOTH `define()` and `importJSON()`.
//   • Schema versioning                — an unknown future blob version is left
//                                         untouched (forward-compat, no clobber).
//   • Rollback-safe migration          — the four legacy keys are read ONCE and
//                                         seeded into the blob; the old keys are
//                                         left in place so a rollback still sees
//                                         them. Each key is decoded with its OWN
//                                         historical rule so migration is
//                                         behaviour-neutral.
//   • Versioned export / import        — `exportJSON()` round-trips through
//                                         `importJSON()`, which re-validates
//                                         EVERY entry individually and never
//                                         trusts the file.
//   • Self-describing                  — `describe()` exposes each key's type,
//                                         default and constraints so the UI can
//                                         be generated from the schema rather
//                                         than hand-wired per row.
//
// HONEST SCOPE NOTE (do not overclaim)
// ------------------------------------
// This is CLIENT storage. A same-origin script or a user with devtools can
// still write the underlying localStorage directly. The value here is NOT
// tamper-proofing (that is impossible on the client and adding an HMAC would be
// security theatre); it is that every value entering the application through the
// registry is validated on the way in AND re-validated on the way out, and that
// there is exactly one place to reason about it. Server-side enforcement of
// model/parameter allow-lists remains the real control (see the security
// review's server-enforced-boundaries section).
//
// STYLE: mirrors `_EP` — ES5 (`var`, IIFE), null-prototype maps, try/catch
// around every storage call, `CustomEvent` on `document` for interop plus a
// local subscriber array for in-process listeners.
//
// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
// =============================================================================

(function (root) {
    'use strict';

    var NS = root.AI_ASSISTANT = root.AI_ASSISTANT || {};
    // Idempotent: a second injection must not replace a live registry (the
    // running one may already hold subscribers and in-memory overrides). Still
    // re-export the EXISTING live surface for any CommonJS consumer so a
    // re-require never hands back an empty object.
    if (NS.settings && NS.settings.__isExtSettings) {
        if (typeof module !== 'undefined' && module.exports) { module.exports = NS.settings; }
        return;
    }

    // ── Environment handles (resolved once; tolerant of headless/Node) ────────
    var _localStorage = (function () {
        try { return root.localStorage || null; } catch (_) { return null; }
    }());
    var _doc = (function () {
        try { return root.document || null; } catch (_) { return null; }
    }());
    var _CustomEvent = (function () {
        try { return root.CustomEvent || null; } catch (_) { return null; }
    }());

    // ── Constants ─────────────────────────────────────────────────────────────
    var _STORAGE_KEY = 'ai-assistant-ext';   // single blob for all ext settings
    var _SCHEMA_VER  = 1;                     // blob schema version
    var _MAX_STR_LEN = 4096;                  // hard cap for any string value
    var _CHANGE_EVENT = 'ai-assistant:setting-changed';

    // Key-safety contract, in the spirit of `_EP._SAFE_KEY_RE` but widened to
    // allow camelCase setting names (`shareLinkMode`, …). Must start with a
    // letter, then [A-Za-z0-9_-]. The leading-letter rule structurally blocks
    // `__proto__` (starts with `_`). `_RESERVED_KEYS` additionally refuses the
    // dangerous identifier names that ARE valid under the regex, so schema and
    // imported keys are pollution-safe by two independent checks.
    var _SAFE_KEY_RE = /^[A-Za-z][A-Za-z0-9_-]{0,63}$/;
    // Array + indexOf (not an object) so this guard itself cannot be polluted.
    var _RESERVED_KEYS = ['__proto__', 'prototype', 'constructor'];
    function _isSafeKey(k) {
        return typeof k === 'string' && _SAFE_KEY_RE.test(k) && _RESERVED_KEYS.indexOf(k) === -1;
    }

    // Hugging Face `owner/name` id — copied verbatim from `_EP` (`_HF_REPO_ID_RE`)
    // so the dataset-repo setting validates identically here and there.
    var _HF_REPO_ID_RE =
        /^[A-Za-z0-9]([A-Za-z0-9_.-]{0,94}[A-Za-z0-9])?\/[A-Za-z0-9]([A-Za-z0-9_.-]{0,94}[A-Za-z0-9])?$/;

    // ── Internal state (null-prototype: no inherited keys, ever) ──────────────
    var _schema    = Object.create(null);  // key → normalised spec
    var _values    = Object.create(null);  // key → current in-memory value
    var _order     = [];                   // definition order (stable UI output)
    var _listeners = [];                   // in-process onChange subscribers
    var _loaded    = false;                // guards one-time bootstrap

    // =========================================================================
    // Type system — each `type` is a {coerce, validate, clone} triple.
    //   coerce(raw)          → best-effort normalisation (import-friendly)
    //   validate(v, spec)    → {ok, value} | {ok:false, error}
    // Keeping the per-type logic here (rather than per-setting) is the DRY
    // guarantee: a new setting declares a type, it does not re-implement checks.
    // =========================================================================
    var _TYPES = Object.create(null);

    _TYPES.boolean = {
        coerce: function (raw) {
            if (typeof raw === 'boolean') { return raw; }
            if (raw === 'true')  { return true; }
            if (raw === 'false') { return false; }
            return raw;
        },
        validate: function (v) {
            if (typeof v !== 'boolean') {
                return { ok: false, error: 'expected a boolean' };
            }
            return { ok: true, value: v };
        },
    };

    _TYPES.enum = {
        coerce: function (raw) { return typeof raw === 'string' ? raw : raw; },
        validate: function (v, spec) {
            if (typeof v !== 'string') {
                return { ok: false, error: 'expected a string' };
            }
            var allowed = spec.values || [];
            if (allowed.indexOf(v) === -1) {
                return { ok: false, error: 'must be one of: ' + allowed.join(', ') };
            }
            return { ok: true, value: v };
        },
    };

    _TYPES.integer = {
        coerce: function (raw) {
            if (typeof raw === 'string' && /^-?\d+$/.test(raw.trim())) {
                return parseInt(raw, 10);
            }
            return raw;
        },
        validate: function (v, spec) {
            if (typeof v !== 'number' || !isFinite(v) || Math.floor(v) !== v) {
                return { ok: false, error: 'expected an integer' };
            }
            if (typeof spec.min === 'number' && v < spec.min) {
                return { ok: false, error: 'must be >= ' + spec.min };
            }
            if (typeof spec.max === 'number' && v > spec.max) {
                return { ok: false, error: 'must be <= ' + spec.max };
            }
            return { ok: true, value: v };
        },
    };

    // A validated free-form string. `spec.pattern` (RegExp) and `spec.maxLen`
    // constrain it. `spec.allowEmpty` (default true) lets '' through as the
    // "unset" sentinel — this is how the dataset-repo setting models "no
    // override". An empty string ALWAYS bypasses the pattern check.
    _TYPES.string = {
        coerce: function (raw) {
            return typeof raw === 'string' ? raw.trim() : raw;
        },
        validate: function (v, spec) {
            if (typeof v !== 'string') {
                return { ok: false, error: 'expected a string' };
            }
            var maxLen = typeof spec.maxLen === 'number' ? spec.maxLen : _MAX_STR_LEN;
            if (v.length > maxLen) {
                return { ok: false, error: 'exceeds ' + maxLen + ' characters' };
            }
            var allowEmpty = spec.allowEmpty !== false;
            if (v === '') {
                return allowEmpty
                    ? { ok: true, value: '' }
                    : { ok: false, error: 'must not be empty' };
            }
            if (spec.pattern && !spec.pattern.test(v)) {
                return { ok: false, error: 'does not match required format' };
            }
            return { ok: true, value: v };
        },
    };

    // =========================================================================
    // Schema definition
    // =========================================================================
    /**
     * Register a setting. This is the ONLY way a key enters the registry, and
     * it is the analogue of `_EP.addProfile`'s single write path — a setting
     * that is not defined cannot be `set`, `get`, imported, or migrated.
     *
     * @param {string} key   Must match `_SAFE_KEY_RE`.
     * @param {Object} spec
     *   @param {string}  spec.type      One of 'boolean' | 'enum' | 'integer' | 'string'.
     *   @param {*}       spec.default   Default value (must itself validate).
     *   @param {string=} spec.label     Human label (for UI generation).
     *   @param {string[]=} spec.values  Allowed values (enum only).
     *   @param {number=}  spec.min/max  Bounds (integer only).
     *   @param {RegExp=}  spec.pattern  Format (string only).
     *   @param {number=}  spec.maxLen   Length cap (string only).
     *   @param {boolean=} spec.allowEmpty  Whether '' is valid (string; default true).
     *   @param {string=}  spec.legacyKey     Old standalone localStorage key.
     *   @param {function=} spec.legacyDecode (rawString|null) → value, used ONCE
     *                     at migration to preserve the old key's exact semantics.
     * @returns {{ok: boolean, error?: string}}
     */
    function define(key, spec) {
        if (!_isSafeKey(key)) {
            return { ok: false, error: 'setting key must match [a-z][a-z0-9_-]{0,63}' };
        }
        if (_schema[key]) {
            return { ok: false, error: 'setting already defined: ' + key };
        }
        if (!spec || typeof spec !== 'object') {
            return { ok: false, error: 'spec must be an object' };
        }
        var type = _TYPES[spec.type];
        if (!type) {
            return { ok: false, error: 'unknown type: ' + String(spec.type) };
        }
        // The declared default must itself be valid — a schema that ships an
        // invalid default is a bug we refuse at definition time, not runtime.
        var defCheck = type.validate(spec.default, spec);
        if (!defCheck.ok) {
            return { ok: false, error: 'invalid default for ' + key + ': ' + defCheck.error };
        }
        var norm = {
            type:        spec.type,
            _type:       type,
            def:         defCheck.value,
            label:       typeof spec.label === 'string' ? spec.label : key,
            values:      Array.isArray(spec.values) ? spec.values.slice() : null,
            min:         typeof spec.min === 'number' ? spec.min : null,
            max:         typeof spec.max === 'number' ? spec.max : null,
            pattern:     spec.pattern || null,
            maxLen:      typeof spec.maxLen === 'number' ? spec.maxLen : null,
            allowEmpty:  spec.allowEmpty !== false,
            legacyKey:   typeof spec.legacyKey === 'string' ? spec.legacyKey : null,
            legacyDecode: typeof spec.legacyDecode === 'function' ? spec.legacyDecode : null,
        };
        _schema[key] = norm;
        _values[key] = defCheck.value;   // seed with default until load/migrate
        _order.push(key);
        return { ok: true };
    }

    // Internal: run a value through its type's coerce+validate against a spec.
    function _check(spec, raw) {
        var coerced = spec._type.coerce(raw);
        return spec._type.validate(coerced, spec);
    }

    // =========================================================================
    // Bootstrap: load the blob, else migrate legacy keys, else defaults.
    // =========================================================================
    function _readBlob() {
        if (!_localStorage) { return null; }
        var raw;
        try { raw = _localStorage.getItem(_STORAGE_KEY); } catch (_) { return null; }
        if (!raw) { return null; }
        var parsed;
        try { parsed = JSON.parse(raw); } catch (_) { return null; }
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) { return null; }
        return parsed;
    }

    function _writeBlob() {
        if (!_localStorage) { return; }
        var settings = {};
        for (var i = 0; i < _order.length; i++) {
            var k = _order[i];
            settings[k] = _values[k];
        }
        var payload = { _v: _SCHEMA_VER, settings: settings };
        try { _localStorage.setItem(_STORAGE_KEY, JSON.stringify(payload)); } catch (_) {}
    }

    // Seed one key from its legacy standalone localStorage key, using the key's
    // own historical decode rule. Read-only w.r.t. the legacy key (never
    // deleted) so a rollback to the old code still finds it.
    function _migrateKey(key) {
        var spec = _schema[key];
        if (!spec || !spec.legacyKey || !spec.legacyDecode || !_localStorage) { return false; }
        var raw;
        try { raw = _localStorage.getItem(spec.legacyKey); } catch (_) { return false; }
        if (raw === null || typeof raw === 'undefined') { return false; }
        var decoded;
        try { decoded = spec.legacyDecode(raw); } catch (_) { return false; }
        var res = _check(spec, decoded);
        if (res.ok) { _values[key] = res.value; return true; }
        return false;
    }

    /**
     * One-time load. Precedence per key:
     *   1. value present in the versioned blob (validated) — authoritative
     *   2. else legacy standalone key (migrated, validated)
     *   3. else schema default
     * A blob whose `_v` is newer than we understand is left completely alone
     * and we fall through to migrate/default — never clobbered.
     */
    function _load() {
        if (_loaded) { return; }
        _loaded = true;

        var blob = _readBlob();
        var blobSettings = null;
        var blobUsable = false;
        if (blob) {
            if (blob._v === _SCHEMA_VER &&
                blob.settings && typeof blob.settings === 'object' &&
                !Array.isArray(blob.settings)) {
                blobSettings = blob.settings;
                blobUsable = true;
            } else if (typeof blob._v === 'number' && blob._v > _SCHEMA_VER) {
                // Future version — do not read, do not overwrite. Use defaults
                // in memory; leave the on-disk blob intact.
                return;
            }
        }

        var migratedAny = false;
        for (var i = 0; i < _order.length; i++) {
            var key = _order[i];
            var spec = _schema[key];
            var settled = false;

            if (blobUsable && Object.prototype.hasOwnProperty.call(blobSettings, key)) {
                var res = _check(spec, blobSettings[key]);
                if (res.ok) { _values[key] = res.value; settled = true; }
            }
            if (!settled) {
                if (_migrateKey(key)) { migratedAny = true; settled = true; }
            }
            // else: _values[key] already holds the schema default from define()
        }

        // Persist if we either had no blob at all, or we pulled anything in from
        // legacy keys — so the next load is a clean single-blob read.
        if (!blobUsable || migratedAny) { _writeBlob(); }
    }

    // =========================================================================
    // Change notification (in-process array + document CustomEvent for interop)
    // =========================================================================
    function _notify(key, value) {
        var snap = { key: key, value: value };
        for (var i = 0; i < _listeners.length; i++) {
            try { _listeners[i](snap); } catch (_) {}
        }
        if (_doc && _CustomEvent) {
            try {
                _doc.dispatchEvent(new _CustomEvent(_CHANGE_EVENT, {
                    bubbles: true, cancelable: false, detail: snap,
                }));
            } catch (_) {}
        }
    }

    // =========================================================================
    // Public read/write
    // =========================================================================
    /** @returns {*} current value, or the schema default, or undefined if the
     *  key is not defined. */
    function get(key) {
        _load();
        if (!_schema[key]) { return undefined; }
        return _values[key];
    }

    /**
     * The ONLY mutator. Validates against the key's schema before persisting.
     * @returns {{ok: boolean, value?: *, error?: string}}
     */
    function set(key, value) {
        _load();
        var spec = _schema[key];
        if (!spec) { return { ok: false, error: 'unknown setting: ' + String(key) }; }
        var res = _check(spec, value);
        if (!res.ok) { return { ok: false, error: key + ': ' + res.error }; }
        var changed = _values[key] !== res.value;
        _values[key] = res.value;
        _writeBlob();
        if (changed) { _notify(key, res.value); }
        return { ok: true, value: res.value };
    }

    /** Reset one key to its schema default. */
    function reset(key) {
        _load();
        var spec = _schema[key];
        if (!spec) { return { ok: false, error: 'unknown setting: ' + String(key) }; }
        return set(key, spec.def);
    }

    /** Reset every key to its schema default. */
    function resetAll() {
        _load();
        for (var i = 0; i < _order.length; i++) {
            var k = _order[i];
            var def = _schema[k].def;
            var changed = _values[k] !== def;
            _values[k] = def;
            if (changed) { _notify(k, def); }
        }
        _writeBlob();
        return { ok: true };
    }

    /** @returns {boolean} whether the key is defined. */
    function has(key) { return !!_schema[key]; }

    /** @returns {string[]} defined keys in definition order (fresh copy). */
    function keys() { return _order.slice(); }

    /** @returns {Object} frozen snapshot of every current value. */
    function all() {
        _load();
        var out = {};
        for (var i = 0; i < _order.length; i++) { out[_order[i]] = _values[_order[i]]; }
        try { Object.freeze(out); } catch (_) {}
        return out;
    }

    /**
     * Self-description for UI generation — lets the settings sheet be built
     * from the schema instead of hand-wiring each row.
     * @returns {Object|null} frozen {type, label, def, values?, min?, max?}.
     */
    function describe(key) {
        var s = _schema[key];
        if (!s) { return null; }
        var d = {
            key:   key,
            type:  s.type,
            label: s.label,
            def:   s.def,
            value: get(key),
        };
        if (s.values) { d.values = s.values.slice(); }
        if (s.min !== null) { d.min = s.min; }
        if (s.max !== null) { d.max = s.max; }
        try { Object.freeze(d); } catch (_) {}
        return d;
    }

    // =========================================================================
    // Pub/sub
    // =========================================================================
    /**
     * Subscribe to changes. Callback receives {key, value}.
     * @returns {function} unsubscribe — call to remove the listener (no leak).
     */
    function onChange(cb) {
        if (typeof cb !== 'function') { return function () {}; }
        _listeners.push(cb);
        return function unsubscribe() {
            var idx = _listeners.indexOf(cb);
            if (idx !== -1) { _listeners.splice(idx, 1); }
        };
    }

    // =========================================================================
    // Export / import (versioned; import re-validates EVERY entry individually)
    // =========================================================================
    /** @returns {string} pretty-printed `{_v, settings:{…}}` for user download. */
    function exportJSON() {
        _load();
        var settings = {};
        for (var i = 0; i < _order.length; i++) { settings[_order[i]] = _values[_order[i]]; }
        try { return JSON.stringify({ _v: _SCHEMA_VER, settings: settings }, null, 2); }
        catch (_) { return '{"_v":' + _SCHEMA_VER + ',"settings":{}}'; }
    }

    /**
     * Import from a JSON string produced by `exportJSON` (or hand-edited).
     * Unknown keys and values that fail their schema are SKIPPED, never thrown
     * on and never trusted — matching `_EP`'s per-entry import discipline.
     * @returns {{ok: boolean, applied: string[], skipped: Array<{key, reason}>, error?: string}}
     */
    function importJSON(text) {
        _load();
        var parsed;
        try { parsed = JSON.parse(text); }
        catch (_) { return { ok: false, applied: [], skipped: [], error: 'not valid JSON' }; }
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            return { ok: false, applied: [], skipped: [], error: 'expected a JSON object' };
        }
        var settings = parsed.settings;
        if (!settings || typeof settings !== 'object' || Array.isArray(settings)) {
            return { ok: false, applied: [], skipped: [], error: 'missing "settings" object' };
        }
        var applied = [];
        var skipped = [];
        var incoming = Object.keys(settings);
        for (var i = 0; i < incoming.length; i++) {
            var k = incoming[i];
            // Prototype-pollution guard on the incoming key itself.
            if (!_isSafeKey(k)) { skipped.push({ key: k, reason: 'unsafe key' }); continue; }
            if (!Object.prototype.hasOwnProperty.call(settings, k)) { continue; }
            if (!_schema[k]) { skipped.push({ key: k, reason: 'unknown setting' }); continue; }
            var res = set(k, settings[k]);
            if (res.ok) { applied.push(k); }
            else { skipped.push({ key: k, reason: res.error }); }
        }
        return { ok: true, applied: applied, skipped: skipped };
    }

    // =========================================================================
    // Built-in schema: the four functional Extended-Settings today.
    // Each carries a `legacyKey` + `legacyDecode` that reproduces its EXACT
    // historical read semantics so first-load migration changes no behaviour.
    // (Effort/Thinking are intentionally NOT folded in here — that is the
    // A-minimal session-vs-local decision from the report's §6 Q2, which is a
    // deliberate product call, not something to silently change. Folding them
    // in later is a two-line `define()` each; see the integration guide.)
    // =========================================================================
    define('streaming', {
        type: 'boolean', default: true, label: 'Streaming responses',
        legacyKey: 'ai-assistant-streaming-on',
        // historical: getItem(...) !== 'false'  → absent/anything-but-'false' == true
        legacyDecode: function (raw) { return raw !== 'false'; },
    });
    define('shareLinkMode', {
        type: 'boolean', default: true, label: 'Share as link',
        legacyKey: 'ai-assistant-export-link-mode',
        // historical: stored === null ? true : stored === 'true'
        legacyDecode: function (raw) { return raw === null ? true : raw === 'true'; },
    });
    define('feedbackPersist', {
        type: 'boolean', default: true, label: 'Store ratings permanently',
        legacyKey: 'ai-assistant-feedback-persist',
        // historical: getItem(...) !== 'false'
        legacyDecode: function (raw) { return raw !== 'false'; },
    });
    define('datasetRepo', {
        type: 'string', default: '', label: 'Custom dataset repo',
        pattern: _HF_REPO_ID_RE, maxLen: 200, allowEmpty: true,
        legacyKey: 'ai-assistant-custom-dataset-repo',
        // historical: value trusted only if it matched _HF_REPO_ID_RE, else ''
        legacyDecode: function (raw) {
            if (typeof raw !== 'string') { return ''; }
            var t = raw.trim();
            return _HF_REPO_ID_RE.test(t) ? t : '';
        },
    });

    // ── Public surface (frozen so callers cannot swap methods) ────────────────
    var api = {
        __isExtSettings: true,
        define:      define,
        get:         get,
        set:         set,
        reset:       reset,
        resetAll:    resetAll,
        has:         has,
        keys:        keys,
        all:         all,
        describe:    describe,
        onChange:    onChange,
        exportJSON:  exportJSON,
        importJSON:  importJSON,
        SCHEMA_VER:  _SCHEMA_VER,
        STORAGE_KEY: _STORAGE_KEY,
        CHANGE_EVENT: _CHANGE_EVENT,
    };
    try { Object.freeze(api); } catch (_) {}
    NS.settings = api;

    // CommonJS hook so the Node test harness can require this exact file with no
    // production-only branches (browsers ignore `module`).
    if (typeof module !== 'undefined' && module.exports) { module.exports = NS.settings; }

}(typeof window !== 'undefined' ? window
   : typeof globalThis !== 'undefined' ? globalThis
   : this));
