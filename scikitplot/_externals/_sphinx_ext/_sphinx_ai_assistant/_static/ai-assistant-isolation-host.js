/**
 * B41 separate-origin host bridge.
 *
 * This file runs on the documentation origin. When isolationOrigin is empty it
 * is a strict no-op. When configured, it owns only the page-side capability
 * adapter; the assistant UI/transcript/runtime state live in a cross-origin
 * iframe and ai-assistant.js is suppressed in the parent page.
 */
(function () {
    'use strict';

    var _SNAPSHOT_FORBIDDEN_KEY = /^(?:__proto__|prototype|constructor)$/i;
    var _SNAPSHOT_SECRET_KEY = /(?:^|[_-])(?:token|secret|password|authorization|cookie|api[_-]?key|access[_-]?(?:token|key)|client[_-]?secret|credential|bearer|session[_-]?key)(?:$|[_-])|(?:Token|Secret|Password|Authorization|Cookie|ApiKey|APIKey|AccessToken|AccessKey|ClientSecret|Credential|Bearer|SessionKey)$/;
    // Snapshot security-critical browser primitives before any asynchronous
    // handshake step. Later same-origin monkeypatching must not redirect the
    // bridge after it has established its boundary. Compromise before this
    // script executes remains the explicit SEC-P1-42 residual.
    var _NativeMessageChannel = window.MessageChannel || (typeof MessageChannel === 'function' ? MessageChannel : null);
    var _NativeURL = window.URL || URL;
    var _NativeURLSearchParams = window.URLSearchParams || URLSearchParams;
    var _nativeFetch = (typeof window.fetch === 'function') ? window.fetch.bind(window)
        : ((typeof fetch === 'function') ? fetch : null);
    var _nativeAddMessageListener = (typeof window.addEventListener === 'function')
        ? window.addEventListener.bind(window) : null;
    var _nativeRemoveMessageListener = (typeof window.removeEventListener === 'function')
        ? window.removeEventListener.bind(window) : null;
    var _nativeSetTimeout = (typeof window.setTimeout === 'function')
        ? window.setTimeout.bind(window) : ((typeof setTimeout === 'function') ? setTimeout : null);
    var _nativeWindowPostMessage = (typeof window.postMessage === 'function') ? window.postMessage : null;
    var _nativeStopImmediatePropagation = (window.Event && window.Event.prototype &&
            typeof window.Event.prototype.stopImmediatePropagation === 'function')
        ? window.Event.prototype.stopImmediatePropagation : null;

    function _snapshotBridgeValue(value, depth, seen) {
        if (depth > 8) return null;
        if (value == null || typeof value === 'boolean' || typeof value === 'number') return value;
        if (typeof value === 'string') return value.slice(0, 16384);
        if (typeof value !== 'object') return null;
        if (seen.indexOf(value) !== -1) return null;
        seen.push(value);
        var out;
        if (Array.isArray(value)) {
            out = value.slice(0, 256).map(function (item) {
                return _snapshotBridgeValue(item, depth + 1, seen);
            });
        } else {
            out = Object.create(null);
            Object.keys(value).slice(0, 512).forEach(function (key) {
                if (_SNAPSHOT_FORBIDDEN_KEY.test(key) || _SNAPSHOT_SECRET_KEY.test(key)) return;
                out[key] = _snapshotBridgeValue(value[key], depth + 1, seen);
            });
        }
        seen.pop();
        return out;
    }

    // Snapshot the bootstrap contract immediately. A later same-origin script
    // can no longer mutate global config/endpoints between host startup and the
    // cross-origin HELLO. A parent already compromised before this point remains
    // an explicit architectural residual (SEC-P1-42).
    var cfg = _snapshotBridgeValue(window.AI_ASSISTANT_CONFIG || {}, 0, []) || Object.create(null);
    var _endpointSnapshot = _snapshotBridgeValue(window.AI_ASSISTANT_ENDPOINTS || {}, 0, []) || Object.create(null);
    var _endpointDefaultSnapshot = String(window.AI_ASSISTANT_ENDPOINT_DEFAULT || '').slice(0, 128);
    var _hostScriptUrl = (document.currentScript && document.currentScript.src) ? String(document.currentScript.src) : '';
    var rawOrigin = typeof cfg.isolationOrigin === 'string' ? cfg.isolationOrigin.trim() : '';
    if (!rawOrigin) return;

    // Mark the parent before ai-assistant.js executes. There is deliberately no
    // silent same-origin fallback when isolated mode was requested.
    window.SphinxAIAssistantIsolationHostActive = true;

    var PROTOCOL = '2.0.0';
    var MAX_MESSAGE_CHARS = 262144;
    var MAX_CANONICAL_CHARS = 1048576;
    var MAX_PUBLIC_EVENT_CHARS = 8192;
    var _frame = null;
    var _port = null;
    var _txSeq = 0;
    var _rxSeq = 0;
    var _ready = false;
    var _failed = false;
    // B42: the frame, not the parent, creates the cryptographic channel nonce.
    // It is never present in iframe.src where arbitrary parent scripts could read it.
    var _channelId = '';

    function _postToFrame(target, message, targetOrigin, ports) {
        if (!target || typeof target.postMessage !== 'function') throw new Error('POSTMESSAGE_UNAVAILABLE');
        if (_nativeWindowPostMessage) return _nativeWindowPostMessage.call(target, message, targetOrigin, ports || []);
        return target.postMessage(message, targetOrigin, ports || []);
    }

    function _parseIsolationOrigin(raw) {
        try {
            var u = new _NativeURL(raw);
            var localhost = /^(?:localhost|127(?:\.\d{1,3}){3}|\[::1\])$/i.test(u.hostname || '');
            if (u.protocol !== 'https:' && !(u.protocol === 'http:' && localhost)) return '';
            if (u.username || u.password || u.search || u.hash) return '';
            if (u.pathname && u.pathname !== '/') return '';
            return u.origin;
        } catch (_) { return ''; }
    }

    var isolationOrigin = _parseIsolationOrigin(rawOrigin);
    if (!isolationOrigin || isolationOrigin === location.origin) {
        _renderFailure('AI isolation unavailable: configure a distinct HTTPS origin.');
        return;
    }
    if (!/^https?:\/\//i.test(String(location.origin || ''))) {
        _renderFailure('AI isolation unavailable: the documentation page needs an HTTP(S) origin.');
        return;
    }

    function _safeFramePath(raw) {
        var p = typeof raw === 'string' && raw ? raw : '/ai-assistant-isolated.html';
        if (!/^\/[A-Za-z0-9._~!$&'()*+,;=:@%\/-]{1,512}$/.test(p) || p.indexOf('..') !== -1) {
            return '/ai-assistant-isolated.html';
        }
        return p;
    }

    function _pageUrl() {
        try {
            var u = new _NativeURL(location.href);
            // Query strings and fragments frequently contain search terms,
            // tokens, or user-specific navigation state. They do not cross the
            // separate-origin boundary.
            u.search = '';
            u.hash = '';
            return u.href;
        } catch (_) { return ''; }
    }

    function _docsRootUrl() {
        try {
            var options = window.DOCUMENTATION_OPTIONS || {};
            if (typeof options.URL_ROOT === 'string' && options.URL_ROOT.trim()) {
                var root = new _NativeURL(options.URL_ROOT.trim(), document.baseURI);
                if (root.origin === location.origin && !root.username && !root.password) {
                    root.search = ''; root.hash = '';
                    return root.href.slice(0, 4096);
                }
            }
            if (_hostScriptUrl) {
                var scriptRoot = new _NativeURL('../', new _NativeURL(_hostScriptUrl));
                if (scriptRoot.origin === location.origin) {
                    scriptRoot.search = ''; scriptRoot.hash = '';
                    return scriptRoot.href.slice(0, 4096);
                }
            }
        } catch (_) {}
        try { return new _NativeURL('/', location.origin).href; } catch (_) { return ''; }
    }

    function _pageName(url) {
        try {
            var options = window.DOCUMENTATION_OPTIONS || {};
            if (typeof options.pagename === 'string' && options.pagename.trim()) {
                return options.pagename.trim().replace(/^\/+|\/+$/g, '').slice(0, 1024) || 'index';
            }
            var p = new _NativeURL(url).pathname.replace(/^\/+|\/+$/g, '');
            p = p.replace(/\/index\.html?$/i, '').replace(/\.html?$/i, '');
            return p || 'index';
        } catch (_) { return 'index'; }
    }

    function _pageDescriptor() {
        var url = _pageUrl();
        return {
            url: url.slice(0, 4096),
            title: String(document.title || '').slice(0, 512),
            pageName: _pageName(url).slice(0, 1024),
            docsRootUrl: _docsRootUrl()
        };
    }

    function _boundedJson(value, maxChars) {
        try {
            var s = JSON.stringify(value);
            return typeof s === 'string' && s.length <= maxChars;
        } catch (_) { return false; }
    }

    function _stripContextNode(clone) {
        var remove = [
            'script','style','noscript','template','iframe','object','embed','base','meta','link','form',
            'input','textarea','select','button','.ai-assistant-container',
            '.ai-assistant-panel','nav','.sidebar','[hidden]','[aria-hidden="true"]'
        ];
        remove.forEach(function (sel) {
            try { clone.querySelectorAll(sel).forEach(function (el) { el.remove(); }); } catch (_) {}
        });
        try {
            clone.querySelectorAll('*').forEach(function (el) {
                Array.prototype.slice.call(el.attributes || []).forEach(function (attr) {
                    var n = String(attr.name || '').toLowerCase();
                    if (/^on/.test(n) || n === 'srcdoc' || n === 'style' || n === 'nonce' ||
                            n === 'integrity' || n === 'crossorigin' || n === 'formaction') {
                        el.removeAttribute(attr.name);
                    }
                });
            });
        } catch (_) {}
    }

    function _readPageContext() {
        var selector = typeof cfg.content_selector === 'string' && cfg.content_selector
            ? cfg.content_selector : 'article';
        var content = null;
        try { content = document.querySelector(selector); } catch (_) {}
        if (!content) content = document.querySelector('main, article, [role="main"]') || document.body;
        if (!content) throw new Error('PAGE_CONTEXT_UNAVAILABLE');
        var clone = content.cloneNode(true);
        // The live rendered DOM is the visibility authority. Detached clones
        // lose stylesheet/layout/scroll geometry and cannot safely decide what
        // a human reader could perceive.
        try {
            if (typeof window.getComputedStyle === 'function') {
                var originals = [content].concat(Array.prototype.slice.call(content.querySelectorAll('*')));
                var copies = [clone].concat(Array.prototype.slice.call(clone.querySelectorAll('*')));
                var de = document.documentElement || {}, body = document.body || {};
                var maxX = Math.max(Number(de.scrollWidth) || 0, Number(body.scrollWidth) || 0,
                                    Number(de.clientWidth) || 0, Number(window.innerWidth) || 0);
                var maxY = Math.max(Number(de.scrollHeight) || 0, Number(body.scrollHeight) || 0,
                                    Number(de.clientHeight) || 0, Number(window.innerHeight) || 0);
                var sx = Number(window.scrollX || window.pageXOffset || 0);
                var sy = Number(window.scrollY || window.pageYOffset || 0);
                var doomed = [];
                for (var vi = 0; vi < originals.length && vi < copies.length; vi++) {
                    var el = originals[vi], cp = copies[vi];
                    var cs = window.getComputedStyle(el);
                    if (!cs || !cp) continue;
                    var hidden = cs.display === 'none' || cs.visibility === 'hidden' ||
                        cs.visibility === 'collapse' || cs.contentVisibility === 'hidden' ||
                        Number.parseFloat(cs.opacity) === 0;
                    var rect = el.getBoundingClientRect ? el.getBoundingClientRect() : null;
                    var text = String(el.textContent || '').trim();
                    var leafText = !!text && !(el.children && el.children.length);
                    var zeroLeaf = leafText && rect &&
                        ((rect.width <= 0 || rect.height <= 0) || Number.parseFloat(cs.fontSize) <= 0);
                    var clip = String(cs.clip || '').replace(/\s+/g, '').toLowerCase();
                    var clipPath = String(cs.clipPath || cs.webkitClipPath || '').replace(/\s+/g, '').toLowerCase();
                    var classicallyClipped = leafText && (
                        clip === 'rect(0px,0px,0px,0px)' || clip === 'rect(0,0,0,0)' ||
                        clipPath === 'inset(50%)' || clipPath === 'inset(100%)');
                    var indent = Number.parseFloat(cs.textIndent);
                    var extremeIndent = leafText && cs.overflow === 'hidden' && Number.isFinite(indent) &&
                        Math.abs(indent) >= 10000;
                    var unreachable = false;
                    if (leafText && rect && maxX > 0 && maxY > 0 && cs.position !== 'fixed') {
                        var left = rect.left + sx, right = rect.right + sx;
                        var top = rect.top + sy, bottom = rect.bottom + sy;
                        unreachable = right < 0 || bottom < 0 || left > maxX || top > maxY;
                    }
                    if (hidden || zeroLeaf || classicallyClipped || extremeIndent || unreachable) doomed.push(cp);
                }
                for (var di = doomed.length - 1; di >= 0; di--) {
                    if (doomed[di] !== clone && doomed[di] && doomed[di].parentNode) doomed[di].parentNode.removeChild(doomed[di]);
                }
            }
        } catch (_) {}
        _stripContextNode(clone);
        var max = Number(cfg.isolationContextMaxChars);
        if (!Number.isFinite(max)) max = 200000;
        max = Math.max(1000, Math.min(500000, Math.floor(max)));
        var html = String(clone.innerHTML || '');
        if (html.length <= max) {
            return { format: 'html', content: html, truncated: false, page: _pageDescriptor() };
        }
        // Never cut an HTML token in half. Oversize pages cross the boundary as
        // bounded plain text instead.
        var text = String(clone.textContent || '').replace(/\u0000/g, '');
        return {
            format: 'text',
            content: text.slice(0, max),
            truncated: text.length > max,
            page: _pageDescriptor()
        };
    }

    function _markdownUrl() {
        var bare = _pageUrl();
        if (!bare) throw new Error('PAGE_URL_UNAVAILABLE');
        if (/\.html$/i.test(bare)) return bare.replace(/\.html$/i, '.md');
        if (/\/$/.test(bare)) return bare + 'index.md';
        return bare + '.md';
    }

    async function _readBoundedResponseText(response, maxChars) {
        var rawDeclared = response.headers && response.headers.get && response.headers.get('content-length');
        if (rawDeclared !== null && rawDeclared !== undefined && String(rawDeclared).trim() !== '') {
            var cleanDeclared = String(rawDeclared).trim();
            if (!/^\d+$/.test(cleanDeclared)) throw new Error('CANONICAL_INVALID_LENGTH');
            var declared = Number(cleanDeclared);
            if (!Number.isSafeInteger(declared) || declared > maxChars * 4) throw new Error('CANONICAL_TOO_LARGE');
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
                    if (bytes > maxChars * 4) throw new Error('CANONICAL_TOO_LARGE');
                    out += decoder.decode(value, { stream:true });
                    if (out.length > maxChars) throw new Error('CANONICAL_TOO_LARGE');
                }
                out += decoder.decode();
            } catch (err) {
                try { await reader.cancel(); } catch (_) {}
                throw err;
            }
            if (out.length > maxChars) throw new Error('CANONICAL_TOO_LARGE');
            return out;
        }
        throw new Error('CANONICAL_STREAM_UNAVAILABLE');
    }

    async function _readCanonicalMarkdown() {
        var url = _markdownUrl();
        if (!_nativeFetch) throw new Error('CANONICAL_FETCH_UNAVAILABLE');
        var response = await _nativeFetch(url, { credentials: 'same-origin', redirect: 'error', cache:'no-store' });
        if (!response.ok) throw new Error('CANONICAL_HTTP_' + response.status);
        var text = await _readBoundedResponseText(response, MAX_CANONICAL_CHARS);
        if (!text || !text.trim()) throw new Error('CANONICAL_EMPTY');
        return { text: text, url: url };
    }

    var PUBLIC_EVENT_TYPES = Object.freeze({
        'ai-assistant-feedback': true,
        'ai-assistant-conversation-reset': true,
        'ai-assistant-open-contribution': true,
        'ai-assistant-effort-change': true,
        'ai-assistant-thinking-change': true,
        'ai-assistant-thinking-budget-change': true,
        'ai-assistant-model-change': true,
        'ai-assistant-model-edit': true,
        'ai-assistant-model-removed': true,
        'ai-assistant:profile-changed': true
    });
    var FORBIDDEN_PUBLIC_KEY = /(?:token|secret|password|authorization|cookie|endpoint|url|query|answer|message|content|conversationid|modelobject|raw)/i;

    function _publicDetailSafe(value, depth) {
        if (depth > 3) return false;
        if (value == null || typeof value === 'boolean' || typeof value === 'number') return true;
        if (typeof value === 'string') return value.length <= 1024;
        if (Array.isArray(value)) {
            if (value.length > 32) return false;
            return value.every(function (x) { return _publicDetailSafe(x, depth + 1); });
        }
        if (typeof value === 'object') {
            var keys = Object.keys(value);
            if (keys.length > 32) return false;
            return keys.every(function (k) {
                return !FORBIDDEN_PUBLIC_KEY.test(k) && _publicDetailSafe(value[k], depth + 1);
            });
        }
        return false;
    }

    function _emitPublicIntegration(payload) {
        if (!payload || !PUBLIC_EVENT_TYPES[payload.eventType]) throw new Error('PUBLIC_EVENT_TYPE_DENIED');
        var detail = payload.detail == null ? {} : payload.detail;
        if (!_publicDetailSafe(detail, 0) || !_boundedJson(detail, MAX_PUBLIC_EVENT_CHARS)) {
            throw new Error('PUBLIC_EVENT_DETAIL_DENIED');
        }
        document.dispatchEvent(new CustomEvent(payload.eventType, {
            bubbles: false, cancelable: false, detail: detail
        }));
        return { emitted: true };
    }

    function _resize(mode) {
        if (!_frame) return;
        var m = String(mode || 'idle');
        // Use stylesheet classes instead of dynamic inline CSS so strict
        // documentation CSPs do not need style-src 'unsafe-inline'.
        _frame.classList.remove(
            'ai-assistant-isolated-host-frame--idle',
            'ai-assistant-isolated-host-frame--panel',
            'ai-assistant-isolated-host-frame--menu',
            'ai-assistant-isolated-host-frame--full'
        );
        _frame.classList.add('ai-assistant-isolated-host-frame--' +
            (m === 'full' ? 'full' : (m === 'panel' ? 'panel' : (m === 'menu' ? 'menu' : 'idle'))));
    }

    function _send(type, id, ok, payload, error) {
        if (!_port) return;
        var msg = {
            v: PROTOCOL, channel: _channelId, seq: ++_txSeq,
            type: type, id: id || '', ok: ok === true,
            payload: payload == null ? null : payload,
            error: error ? String(error).slice(0, 128) : ''
        };
        if (_boundedJson(msg, MAX_MESSAGE_CHARS)) _port.postMessage(msg);
    }

    async function _handlePortMessage(event) {
        var msg = event && event.data;
        if (!msg || msg.v !== PROTOCOL || msg.channel !== _channelId) return;
        if (!Number.isSafeInteger(msg.seq) || msg.seq !== _rxSeq + 1) return;
        if (!_boundedJson(msg, MAX_MESSAGE_CHARS)) return;
        _rxSeq = msg.seq;
        if (msg.type !== 'request') return;
        var id = typeof msg.id === 'string' ? msg.id.slice(0, 128) : '';
        var cap = String(msg.cap || '').slice(0, 64);
        if (!id && cap !== 'ui.resize' && cap !== 'page.integration.emit') return;
        try {
            var result;
            if (cap === 'page.context.read') result = _readPageContext();
            else if (cap === 'page.canonical.read') result = await _readCanonicalMarkdown();
            else if (cap === 'page.print') { window.print(); result = { requested: true }; }
            else if (cap === 'ui.resize') { _resize(msg.payload && msg.payload.mode); result = { resized: true }; }
            else if (cap === 'page.integration.emit') result = _emitPublicIntegration(msg.payload);
            else throw new Error('CAPABILITY_DENIED');
            _send('response', id, true, result, '');
        } catch (err) {
            _send('response', id, false, null, err && err.message || 'REQUEST_FAILED');
        }
    }

    function _renderFailure(text) {
        if (_failed) return;
        _failed = true;
        var el = document.createElement('div');
        el.id = 'ai-assistant-isolation-error';
        el.setAttribute('role', 'status');
        el.textContent = String(text || 'AI isolation failed.');
        el.className = 'ai-assistant-isolation-error';
        (document.body || document.documentElement).appendChild(el);
    }

    function _start() {
        if (!_NativeMessageChannel || !_nativeAddMessageListener || !_nativeSetTimeout) {
            _renderFailure('AI isolation unavailable: required secure messaging primitives are missing.');
            return;
        }
        var frameUrl = new _NativeURL(_safeFramePath(cfg.isolationFramePath), isolationOrigin);
        var fragment = new _NativeURLSearchParams({
            v: PROTOCOL,
            parentOrigin: location.origin
        });
        frameUrl.hash = fragment.toString();

        var iframe = document.createElement('iframe');
        _frame = iframe;
        iframe.id = 'ai-assistant-isolated-frame';
        iframe.title = String(cfg.panelTitle || 'AI Assistant');
        iframe.referrerPolicy = 'no-referrer';
        // allow-same-origin is required for isolated-origin storage/CORS. B42
        // removes popup escape and installs a frame-side navigation guard so an
        // HTTP(S) link cannot navigate this browsing context onto the docs origin.
        iframe.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-downloads allow-popups');
        var allow = ['clipboard-write'];
        if (cfg.isolationAllowMicrophone === true && cfg.panelSpeakBanner !== false) allow.push('microphone');
        iframe.setAttribute('allow', allow.join('; '));
        iframe.src = frameUrl.href;
        iframe.className = 'ai-assistant-isolated-host-frame ai-assistant-isolated-host-frame--idle';

        var onMessage = function (event) {
            if (_ready || event.source !== iframe.contentWindow || event.origin !== isolationOrigin) return;
            var data = event.data || {};
            if (data.type !== 'AI_ASSISTANT_ISOLATION_HELLO' || data.v !== PROTOCOL ||
                    !/^[a-f0-9]{32}$/.test(String(data.channel || ''))) return;
            // The frame-generated nonce is sensitive bootstrap authority. Stop it
            // from reaching listeners registered after this host bridge.
            if (_nativeStopImmediatePropagation) _nativeStopImmediatePropagation.call(event);
            else if (typeof event.stopImmediatePropagation === 'function') event.stopImmediatePropagation();
            _channelId = String(data.channel);
            var channel = new _NativeMessageChannel();
            _port = channel.port1;
            _port.onmessage = _handlePortMessage;
            _port.start();
            var page = _pageDescriptor();
            var initConfig = _snapshotBridgeValue(cfg, 0, []) || Object.create(null);
            initConfig.isolatedFrameRuntime = true;
            initConfig.isolationOrigin = isolationOrigin;
            initConfig.hostPageUrl = page.url;
            initConfig.hostPageTitle = page.title;
            initConfig.hostPageName = page.pageName;
            initConfig.hostDocsRootUrl = page.docsRootUrl;
            // If the embedding site did not explicitly delegate microphone to
            // the cross-origin frame, hide the voice UI rather than rendering a
            // control that can never acquire permission.
            if (cfg.isolationAllowMicrophone !== true) initConfig.panelSpeakBanner = false;
            var init = {
                type:'AI_ASSISTANT_ISOLATION_INIT', v:PROTOCOL, channel:_channelId,
                page:page,
                config:initConfig,
                endpoints:_endpointSnapshot,
                endpointDefault:_endpointDefaultSnapshot
            };
            if (!_boundedJson(init, MAX_MESSAGE_CHARS)) {
                _renderFailure('AI isolation failed: configuration is too large.');
                return;
            }
            _postToFrame(iframe.contentWindow, init, isolationOrigin, [channel.port2]);
            _ready = true;
            if (_nativeRemoveMessageListener) _nativeRemoveMessageListener('message', onMessage, true);
        };
        // Register before the iframe is attached/navigated. Capture phase +
        // stopImmediatePropagation prevents later listeners from observing the
        // frame-generated nonce. A listener installed before this bridge remains
        // part of the explicit hostile-parent residual.
        _nativeAddMessageListener('message', onMessage, true);
        (document.body || document.documentElement).appendChild(iframe);
        _nativeSetTimeout(function () {
            if (!_ready) {
                if (_frame) _frame.remove();
                _renderFailure('AI isolation failed closed: the separate-origin frame did not complete its secure handshake.');
                if (_nativeRemoveMessageListener) _nativeRemoveMessageListener('message', onMessage, true);
            }
        }, 8000);
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', _start, { once:true });
    else _start();
}());
