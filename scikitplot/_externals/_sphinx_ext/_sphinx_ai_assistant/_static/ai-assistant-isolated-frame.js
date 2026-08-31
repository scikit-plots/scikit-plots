/** B42 isolated-frame bootstrap and capability client. */
(function () {
    'use strict';
    var PROTOCOL = '2.0.0';
    var MAX_MESSAGE_CHARS = 262144;
    var MAX_POLICY_CHARS = 16384;
    var FORBIDDEN_KEY = /^(?:__proto__|prototype|constructor)$/i;
    var SECRET_KEY = /(?:^|[_-])(?:token|secret|password|authorization|cookie|api[_-]?key|access[_-]?(?:token|key)|client[_-]?secret|credential|bearer|session[_-]?key)(?:$|[_-])|(?:Token|Secret|Password|Authorization|Cookie|ApiKey|APIKey|AccessToken|AccessKey|ClientSecret|Credential|Bearer|SessionKey)$/;
    var params;
    try { params = new URLSearchParams(location.hash.slice(1)); } catch (_) { params = new URLSearchParams(); }
    var parentOrigin = String(params.get('parentOrigin') || '');
    var version = String(params.get('v') || '');
    var channelId = '';
    var port = null;
    var txSeq = 0;
    var rxSeq = 0;
    var requestSeq = 0;
    var pending = Object.create(null);
    var ready = false;

    function validOrigin(raw) {
        try {
            var u = new URL(raw);
            var localhost = /^(?:localhost|127(?:\.\d{1,3}){3}|\[::1\])$/i.test(u.hostname || '');
            if (u.protocol !== 'https:' && !(u.protocol === 'http:' && localhost)) return '';
            if (u.username || u.password || u.search || u.hash || (u.pathname && u.pathname !== '/')) return '';
            return u.origin === raw.replace(/\/$/, '') ? u.origin : '';
        } catch (_) { return ''; }
    }

    function frameOrigin() {
        try { return new URL(location.href).origin; } catch (_) { return ''; }
    }

    parentOrigin = validOrigin(parentOrigin);
    if (version !== PROTOCOL || !parentOrigin || !validOrigin(frameOrigin())) {
        document.body.textContent = 'AI isolation bootstrap rejected invalid origin/protocol parameters.';
        return;
    }

    function bounded(value, maxChars) {
        try { var s = JSON.stringify(value); return typeof s === 'string' && s.length <= (maxChars || MAX_MESSAGE_CHARS); }
        catch (_) { return false; }
    }

    function randomHex(bytes) {
        var c = window.crypto || (typeof crypto !== 'undefined' ? crypto : null);
        if (!c || typeof c.getRandomValues !== 'function') throw new Error('CRYPTO_UNAVAILABLE');
        var a = new Uint8Array(bytes);
        c.getRandomValues(a);
        return Array.prototype.map.call(a, function (b) {
            return ('0' + b.toString(16)).slice(-2);
        }).join('');
    }

    function scrubConfig(value, depth) {
        if (depth > 6) return null;
        if (value == null || typeof value === 'boolean' || typeof value === 'number') return value;
        if (typeof value === 'string') return value.slice(0, 16384);
        if (Array.isArray(value)) return value.slice(0, 256).map(function (v) { return scrubConfig(v, depth + 1); });
        if (typeof value !== 'object') return null;
        var out = Object.create(null);
        Object.keys(value).slice(0, 512).forEach(function (k) {
            if (FORBIDDEN_KEY.test(k) || SECRET_KEY.test(k)) return;
            out[k] = scrubConfig(value[k], depth + 1);
        });
        return out;
    }

    function exactPolicy(value) {
        if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
        var keys = Object.keys(value).sort();
        var expected = ['allowedParentOrigins','isolationOrigin','protocolVersion','schemaVersion'];
        if (keys.length !== expected.length || keys.some(function (k, i) { return k !== expected[i]; })) return null;
        if (value.schemaVersion !== 1 || value.protocolVersion !== PROTOCOL) return null;
        if (validOrigin(String(value.isolationOrigin || '')) !== frameOrigin()) return null;
        if (!Array.isArray(value.allowedParentOrigins) || value.allowedParentOrigins.length > 32) return null;
        var parents = [];
        for (var i = 0; i < value.allowedParentOrigins.length; i++) {
            var p = validOrigin(String(value.allowedParentOrigins[i] || ''));
            if (!p || parents.indexOf(p) !== -1) return null;
            parents.push(p);
        }
        return parents.indexOf(parentOrigin) !== -1 ? value : null;
    }

    async function readPolicyText(response) {
        var declaredRaw = response.headers && response.headers.get && response.headers.get('content-length');
        if (declaredRaw !== null && declaredRaw !== undefined && String(declaredRaw).trim() !== '') {
            if (!/^\d+$/.test(String(declaredRaw).trim())) throw new Error('POLICY_INVALID_LENGTH');
            if (Number(declaredRaw) > MAX_POLICY_CHARS) throw new Error('POLICY_TOO_LARGE');
        }
        if (response.body && typeof response.body.getReader === 'function' && typeof TextDecoder === 'function') {
            var reader = response.body.getReader();
            var decoder = new TextDecoder('utf-8', { fatal:false });
            var out = ''; var bytes = 0;
            try {
                while (true) {
                    var part = await reader.read();
                    if (part.done) break;
                    var value = part.value || new Uint8Array(0);
                    bytes += Number(value.byteLength || value.length || 0);
                    if (bytes > MAX_POLICY_CHARS) throw new Error('POLICY_TOO_LARGE');
                    out += decoder.decode(value, { stream:true });
                    if (out.length > MAX_POLICY_CHARS) throw new Error('POLICY_TOO_LARGE');
                }
                out += decoder.decode();
            } catch (err) {
                try { await reader.cancel(); } catch (_) {}
                throw err;
            } finally { try { reader.releaseLock(); } catch (_) {} }
            return out;
        }
        throw new Error('POLICY_STREAM_UNAVAILABLE');
    }

    async function loadPolicy() {
        if (typeof fetch !== 'function') throw new Error('POLICY_FETCH_UNAVAILABLE');
        var url = new URL('ai-assistant-isolation-policy.json', location.href);
        url.hash = ''; url.search = '';
        var response = await fetch(url.href, { credentials:'omit', redirect:'error', cache:'no-store' });
        if (!response.ok) throw new Error('POLICY_HTTP_' + response.status);
        var text = await readPolicyText(response);
        if (!text) throw new Error('POLICY_INVALID_JSON');
        var policy;
        try { policy = JSON.parse(text); } catch (_) { throw new Error('POLICY_INVALID_JSON'); }
        if (!exactPolicy(policy)) throw new Error('POLICY_PARENT_DENIED');
        return policy;
    }

    function send(cap, payload, expectResponse) {
        if (!port || !ready) return Promise.reject(new Error('ISOLATION_NOT_READY'));
        var id = expectResponse ? ('r' + (++requestSeq)) : '';
        var msg = { v:PROTOCOL, channel:channelId, seq:++txSeq, type:'request', id:id, cap:String(cap || '').slice(0,64), payload:payload || null };
        if (!bounded(msg)) return Promise.reject(new Error('ISOLATION_MESSAGE_TOO_LARGE'));
        if (!expectResponse) { port.postMessage(msg); return Promise.resolve(null); }
        return new Promise(function (resolve, reject) {
            var timer = setTimeout(function () { delete pending[id]; reject(new Error('ISOLATION_REQUEST_TIMEOUT')); }, 6000);
            pending[id] = { resolve:resolve, reject:reject, timer:timer };
            port.postMessage(msg);
        });
    }

    function onPortMessage(event) {
        var msg = event && event.data;
        if (!msg || msg.v !== PROTOCOL || msg.channel !== channelId) return;
        if (!Number.isSafeInteger(msg.seq) || msg.seq !== rxSeq + 1 || !bounded(msg)) return;
        if (msg.type !== 'response' || !msg.id || !pending[msg.id]) return;
        rxSeq = msg.seq;
        var p = pending[msg.id]; delete pending[msg.id]; clearTimeout(p.timer);
        if (msg.ok === true) p.resolve(msg.payload); else p.reject(new Error(String(msg.error || 'ISOLATION_REQUEST_FAILED')));
    }

    function notifyResize() {
        var mode = 'idle';
        try {
            var panel = document.querySelector('.ai-assistant-panel');
            var full = panel && panel.getAttribute('data-maximized') === 'true' && panel.classList.contains('ai-assistant-panel--open');
            var open = panel && panel.classList.contains('ai-assistant-panel--open') && panel.style.display !== 'none';
            var dropdown = document.getElementById('ai-assistant-dropdown');
            var menu = dropdown && getComputedStyle(dropdown).display !== 'none';
            mode = full ? 'full' : (open ? 'panel' : (menu ? 'menu' : 'idle'));
        } catch (_) {}
        send('ui.resize', { mode:mode }, false).catch(function () {});
    }

    function storageScope(page) {
        var path = '/';
        try {
            var root = new URL(page && page.docsRootUrl || '/', parentOrigin);
            if (root.origin === parentOrigin) path = root.pathname || '/';
        } catch (_) {}
        path = ('/' + String(path).replace(/^\/+|\/+$/g, '') + '/').replace(/\/{2,}/g, '/').slice(0, 1024);
        return 'host-site:' + parentOrigin + '|' + path;
    }

    function installNavigationGuard(cfg) {
        document.addEventListener('click', function (event) {
            var target = event && event.target;
            var a = target && typeof target.closest === 'function' ? target.closest('a[href]') : null;
            if (!a || a.hasAttribute('download')) return;
            var raw = String(a.getAttribute('href') || '').trim();
            if (!raw || raw.charAt(0) === '#') return;
            var destination;
            try { destination = new URL(raw, cfg.hostPageUrl || parentOrigin); } catch (_) { return; }
            if (destination.protocol !== 'http:' && destination.protocol !== 'https:') return;
            event.preventDefault();
            if (typeof event.stopImmediatePropagation === 'function') event.stopImmediatePropagation();
            // HTTP(S) navigation must never replace the isolated frame. A frame
            // navigated onto the docs origin while keeping allow-same-origin +
            // allow-scripts would collapse the SOP boundary.
            var opened = window.open(destination.href, '_blank', 'noopener,noreferrer');
            try { if (opened) opened.opener = null; } catch (_) {}
        }, true);
    }

    function startAssistant(init) {
        var cfg = scrubConfig(init.config || {}, 0) || {};
        cfg.isolatedFrameRuntime = true;
        cfg.hostPageUrl = init.page && init.page.url || cfg.hostPageUrl || '';
        cfg.hostPageTitle = init.page && init.page.title || cfg.hostPageTitle || '';
        cfg.hostPageName = init.page && init.page.pageName || cfg.hostPageName || 'index';
        cfg.hostDocsRootUrl = init.page && init.page.docsRootUrl || cfg.hostDocsRootUrl || '';
        cfg.isolationStorageScope = storageScope(init.page || {});
        cfg.staticPath = new URL('./', location.href).href;
        window.AI_ASSISTANT_CONFIG = cfg;
        window.AI_ASSISTANT_ENDPOINTS = scrubConfig(init.endpoints || {}, 0) || {};
        window.AI_ASSISTANT_ENDPOINT_DEFAULT = String(init.endpointDefault || '').slice(0, 128);
        window.SphinxAIAssistantIsolationFrame = true;
        window.AI_ASSISTANT_ISOLATION_BRIDGE = Object.freeze({
            protocolVersion: PROTOCOL,
            page: Object.freeze({ url:cfg.hostPageUrl, title:cfg.hostPageTitle, pageName:cfg.hostPageName }),
            request: function (cap, payload) { return send(cap, payload, true); },
            notify: function (cap, payload) { return send(cap, payload, false); }
        });
        installNavigationGuard(cfg);

        var script = document.createElement('script');
        script.src = new URL('ai-assistant.js', location.href).href;
        script.defer = true;
        script.onload = function () {
            notifyResize();
            try {
                var observer = new MutationObserver(function () { notifyResize(); });
                observer.observe(document.body, { subtree:true, childList:true, attributes:true, attributeFilter:['class','style','data-maximized'] });
                window.addEventListener('resize', notifyResize);
            } catch (_) {}
        };
        script.onerror = function () { document.body.textContent = 'AI isolation runtime failed to load.'; };
        document.head.appendChild(script);
    }

    function onWindowMessage(event) {
        if (ready || event.source !== parent || event.origin !== parentOrigin) return;
        var data = event.data || {};
        if (data.type !== 'AI_ASSISTANT_ISOLATION_INIT' || data.v !== PROTOCOL || data.channel !== channelId) return;
        if (!event.ports || event.ports.length !== 1 || !bounded(data)) return;
        port = event.ports[0];
        port.onmessage = onPortMessage;
        port.start();
        ready = true;
        window.removeEventListener('message', onWindowMessage);
        startAssistant(data);
    }

    async function bootstrap() {
        try {
            await loadPolicy();
            channelId = randomHex(16);
            window.addEventListener('message', onWindowMessage);
            parent.postMessage({ type:'AI_ASSISTANT_ISOLATION_HELLO', v:PROTOCOL, channel:channelId }, parentOrigin);
            setTimeout(function () {
                if (!ready) document.body.textContent = 'AI isolation handshake timed out.';
            }, 8000);
        } catch (_) {
            document.body.textContent = 'AI isolation policy/handshake failed closed.';
        }
    }
    bootstrap();
}());
