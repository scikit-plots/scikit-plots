(function () {
  'use strict';
  var roots = document.querySelectorAll('[data-document-generation]');
  if (!roots.length) return;

  function parse(root) {
    var n = root.querySelector('[data-document-generation-payload]');
    try {
      var v = JSON.parse(n ? n.textContent : '{}');
      return v && v.contract === 'learn.document-generation-page.v1' ? v : null;
    } catch (_) { return null; }
  }
  function stripKnownRoute(value) {
    return String(value || '').trim().replace(/\/+$/, '').replace(/\/v1\/(?:chat\/completions|share|feedback|contribute|image(?:-generations)?|video(?:-generations)?|audio(?:-generations)?|document(?:-generations)?)\/?$/i, '');
  }
  function endpoint() {
    try {
      var api = window.AI_ASSISTANT_ENDPOINT_API;
      if (api && typeof api.resolveEndpoint === 'function') {
        var direct = api.resolveEndpoint('document');
        if (direct) return String(direct).replace(/\/+$/, '');
      }
    } catch (_) {}
    try {
      var cfg = window.AI_ASSISTANT_CONFIG || {};
      var models = Array.isArray(cfg.panelApiModels) ? cfg.panelApiModels : [];
      var base = stripKnownRoute(models.length ? models[0].endpoint : '');
      if (base) return base + '/v1/document';
    } catch (_) {}
    return '';
  }
  function selectedModel() {
    var snapshot = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.assistantModelSnapshot ? window.AI_LEARN_GENERATION_UI.assistantModelSnapshot() : null;
    return snapshot ? String(snapshot.model || snapshot.id || '') : '';
  }
  function activeProfile() {
    try {
      var api = window.AI_ASSISTANT_ENDPOINT_API;
      var id = api && api.getActiveProfile && api.getActiveProfile();
      return api && api.getProfile && id ? api.getProfile(id) : null;
    } catch (_) { return null; }
  }
  function record(rows, id) { return (rows || []).find(function (x) { return x.id === id; }) || null; }
  function runtimeJson(url, init, options) {
    var fn = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.fetchJson;
    if (typeof fn !== 'function') return Promise.reject(new Error('Shared AI Learn runtime transport is unavailable.'));
    return fn(url, init, options);
  }
  function copyText(text) {
    return navigator.clipboard && navigator.clipboard.writeText ? navigator.clipboard.writeText(text) : Promise.reject(new Error('Clipboard is unavailable.'));
  }

  roots.forEach(function (root) {
    var data = parse(root), form = root.querySelector('[data-document-form]');
    if (!data || !form) return;
    var ui = window.AI_LEARN_GENERATION_UI || {};
    var activity = ui.bindGenerationStatus ? ui.bindGenerationStatus(root) : null;
    if (ui.bindPublicationCredit) ui.bindPublicationCredit(root, {storageKey:'learn-publication-credit:v1:' + String(data.site_id || 'default') + ':document'});
    var library = ui.bindPrivateLibrary ? ui.bindPrivateLibrary(root, {
      storageKey:'learn-document:v2:' + String(data.site_id || 'default') + ':library',
      singular:'document generation', plural:'document generations'
    }) : null;
    if (library) library.setRefreshEnabled(false, 'Document generation currently returns completed artifacts directly and does not expose a receipt refresh API.');
    var context = window.AI_LEARN_CONTEXT_API && window.AI_LEARN_CONTEXT_API.get ? window.AI_LEARN_CONTEXT_API.get(root) : null;
    var prompt = root.querySelector('[data-document-prompt]');
    var title = root.querySelector('[data-document-title]');
    var format = root.querySelector('[data-document-format]');
    var structure = root.querySelector('[data-document-structure]');
    var glossary = root.querySelector('[data-document-glossary]');
    var applications = root.querySelector('[data-document-applications]');
    var submit = root.querySelector('[data-document-submit]');
    var status = root.querySelector('[data-document-status]');
    var runtimeStatus = root.querySelector('[data-document-runtime-status]');
    var runtimeAuthority = root.querySelector('[data-document-runtime-authority]');
    var card = root.querySelector('[data-document-result-card]');
    var out = root.querySelector('[data-document-result]');
    var meta = root.querySelector('[data-document-result-meta]');
    var download = root.querySelector('[data-document-download]');
    var copy = root.querySelector('[data-document-copy]');
    var last = null, runtimeEnabled = false, discoverySerial = 0, unsubscribeProfile = null;

    function announce(x, state, title) { if (activity) activity.set(state || 'idle', x, title); else if (status) status.textContent = String(x || ''); }
    function announceRuntime(x) { if (runtimeStatus) runtimeStatus.textContent = String(x || ''); }
    function ready(selector, label, state) { if (ui.setReadiness) ui.setReadiness(root, selector, label, state); }
    function authority(label) { if (runtimeAuthority) runtimeAuthority.textContent = String(label || 'Unknown'); }
    function cleanReferenceUrl(value) {
      if (ui.cleanPublicReferenceUrl) return ui.cleanPublicReferenceUrl(value);
      try { var u = new URL(String(value || '').trim()); return u.protocol === 'https:' && !u.username && !u.password && !u.hash ? u.href : ''; } catch (_) { return ''; }
    }
    function mode() { return context && context.activeType ? context.activeType() : 'topic'; }
    function applyQuery() { try { context && context.applyQuery && context.applyQuery(new URLSearchParams(location.search)); } catch (_) {} }

    function advancedGuidance() {
      var lines = [];
      var value = String(structure && structure.value || 'auto');
      if (value === 'outline') lines.push('Structure: begin with a compact outline, then expand each section.');
      if (value === 'study-guide') lines.push('Structure: organize the document as a study guide with key takeaways and review cues.');
      if (value === 'reference') lines.push('Structure: organize the document as a concise reference note optimized for later lookup.');
      if (glossary && glossary.checked) lines.push('Include a short glossary for terms that materially help comprehension.');
      if (applications && applications.checked) lines.push('Include practical applications where they are supported by the supplied context.');
      return lines.join('\n');
    }
    function buildPrompt() {
      var instructions = String(prompt && prompt.value || '').trim();
      if (!instructions) return '';
      var lenses = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.readLensProfile ? window.AI_LEARN_GENERATION_UI.readLensProfile(root, 'document') : {};
      instructions = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.withLensGuidance ? window.AI_LEARN_GENERATION_UI.withLensGuidance(instructions, lenses) : instructions;
      var advanced = advancedGuidance();
      if (advanced) instructions = instructions + '\n' + advanced;
      var m = mode();
      if (m === 'prompt') {
        var primaryPrompt = String(context && context.prompt ? context.prompt() : '').trim();
        if (!primaryPrompt) return '';
        return ['Prompt: ' + primaryPrompt, 'Document instructions: ' + instructions].join('\n\n').slice(0, 48000);
      }
      if (m === 'url') {
        var referenceUrl = cleanReferenceUrl(context && context.url ? context.url() : '');
        if (!referenceUrl) return '';
        return [
          'Public reference URL (untrusted; retrieval is model/runtime-dependent): ' + referenceUrl,
          'Document instructions: ' + instructions
        ].join('\n\n').slice(0, 48000);
      }
      var row = m === 'topic' ? record(data.topics, context && context.singleId ? context.singleId('topic') : '') : m === 'source' ? record(data.sources, context && context.singleId ? context.singleId('source') : '') : null;
      if (!row) return '';
      return [
        'Title: ' + row.title,
        row.summary ? 'Context: ' + row.summary : '',
        row.url ? 'Source URL: ' + row.url : '',
        'Document instructions: ' + instructions
      ].filter(Boolean).join('\n\n').slice(0, 48000);
    }

    function draftState() {
      return {
        context:context && context.snapshot ? context.snapshot() : {},
        lenses:window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.readLensProfile ? window.AI_LEARN_GENERATION_UI.readLensProfile(root, 'document') : {},
        instructions:String(prompt && prompt.value || ''), title:String(title && title.value || ''), format:String(format && format.value || 'markdown'),
        structure:String(structure && structure.value || 'auto'), include_glossary:!!(glossary && glossary.checked),
        include_applications:!!(applications && applications.checked)
      };
    }
    function restoreDraft(state) {
      if (!state || typeof state !== 'object') return;
      if (context && context.restore) context.restore(state.context || state);
      if (window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.restoreLensProfile) window.AI_LEARN_GENERATION_UI.restoreLensProfile(root, 'document', state.lenses);
      if (prompt && typeof state.instructions === 'string') prompt.value = state.instructions;
      if (title && typeof state.title === 'string') title.value = state.title;
      if (format && typeof state.format === 'string') format.value = state.format;
      if (structure && typeof state.structure === 'string') structure.value = state.structure;
      if (glossary && typeof state.include_glossary === 'boolean') glossary.checked = state.include_glossary;
      if (applications && typeof state.include_applications === 'boolean') applications.checked = state.include_applications;
    }
    function requestBody() {
      return {
        contract:'assistant.document-generation-request.v1', prompt:buildPrompt(), selected_model:selectedModel(),
        format:String(format.value || 'markdown'), title:String(title.value || 'AI Learn document').trim() || 'AI Learn document'
      };
    }
    function validateRequest(body) {
      var lensError = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.validateLensProfile ? window.AI_LEARN_GENERATION_UI.validateLensProfile(draftState().lenses) : '';
      if (lensError) return lensError;
      if (!body || !String(body.selected_model || '').trim()) return 'Select an Assistant model before generating a document.';
      if (!String(body.prompt || '').trim()) {
        if (mode() === 'url') return 'Enter a valid public HTTPS URL and provide document instructions.';
        if (mode() === 'prompt') return 'Enter a prompt and provide document instructions.';
        return 'Choose context and provide document instructions.';
      }
      return '';
    }
    var requestActions = ui.bindRequestActions ? ui.bindRequestActions(root, {
      storageKey:'learn-document:v2:' + String(data.site_id || 'default') + ':draft',
      snapshot:draftState, restore:restoreDraft, buildRequest:requestBody, validateRequest:validateRequest, announce:announce
    }) : null;
    if (requestActions) requestActions.loadDraft();
    applyQuery();

    async function discoverRuntime() {
      var serial = ++discoverySerial;
      runtimeEnabled = false;
      var ep = endpoint();
      ready('[data-document-ready-endpoint]', 'Checking', 'pending');
      ready('[data-document-ready-generation]', 'Checking', 'pending');
      ready('[data-document-ready-publishing]', 'Checking', 'pending');
      authority('Discovering…');
      if (!ep) {
        ready('[data-document-ready-endpoint]', 'Missing', 'off');
        ready('[data-document-ready-generation]', 'Disabled', 'off');
        ready('[data-document-ready-publishing]', 'Disabled', 'off');
        authority('Unavailable');
        announceRuntime('Document generation is not configured for the active Assistant endpoint profile.');
        return;
      }
      ready('[data-document-ready-endpoint]', 'Ready', 'ready');
      var profile = activeProfile();
      var explicit = !!(profile && String(profile.document || '').trim());
      var base = ui.runtimeBase ? ui.runtimeBase(ep) : stripKnownRoute(ep);
      if (!base) {
        if (explicit) {
          runtimeEnabled = true;
          ready('[data-document-ready-generation]', 'Explicit route', 'ready');
          ready('[data-document-ready-publishing]', 'Separate step', 'ready');
          authority('Explicit route');
          announceRuntime('Document generation uses an explicit endpoint. The selected Assistant model remains the request model; publishing stays a separate reviewed step.');
        } else {
          ready('[data-document-ready-generation]', 'Unverified', 'pending');
          ready('[data-document-ready-publishing]', 'Unknown', 'pending');
          authority('Unverified');
          announceRuntime('The active runtime could not be verified for document generation. Generate Now remains available to re-check the runtime, but no request will be sent until capability discovery succeeds.');
        }
        return;
      }
      try {
        var packet = await runtimeJson(base + '/', {headers:{'Accept':'application/json'}}, {label:'Document capability discovery', timeoutMs:12000, maxBytes:512*1024});
        var res = packet.response;
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var discovery = packet.body || {};
        if (serial !== discoverySerial) return;
        var cap = discovery && discovery.capabilities && discovery.capabilities.document_generation;
        var contractOk = cap && String(cap.contract || '') === 'assistant.document-generation-request.v1';
        if (cap && cap.enabled === true && contractOk) {
          runtimeEnabled = true;
          ready('[data-document-ready-generation]', 'Ready', 'ready');
          ready('[data-document-ready-publishing]', String(cap.publication || '') === 'separate-reviewed-step' ? 'Separate step' : 'Runtime-defined', 'ready');
          authority('Chat-backed · Ready');
          announceRuntime('Document generation is available through the chat-backed runtime. The active Assistant model is used for generation; publishing remains separate from generation.');
          return;
        }
        if (explicit) {
          runtimeEnabled = true;
          ready('[data-document-ready-generation]', 'Explicit route', 'ready');
          ready('[data-document-ready-publishing]', 'Separate step', 'ready');
          authority('Explicit route');
          announceRuntime('The Document capability was not advertised, but an explicit operator-configured route is active.');
          return;
        }
        ready('[data-document-ready-generation]', cap && cap.enabled === false ? 'Disabled' : 'Contract mismatch', 'off');
        ready('[data-document-ready-publishing]', 'Disabled', 'off');
        authority(cap && cap.enabled === false ? 'Disabled' : 'Contract mismatch');
        announceRuntime('The active runtime is reachable, but compatible document generation is not enabled. Generate Now remains available to re-check after configuration changes.');
      } catch (_) {
        if (serial !== discoverySerial) return;
        if (explicit) {
          runtimeEnabled = true;
          ready('[data-document-ready-generation]', 'Explicit route', 'ready');
          ready('[data-document-ready-publishing]', 'Unknown', 'pending');
          authority('Explicit · Unverified');
          announceRuntime('The explicit Document endpoint is configured, but runtime capability discovery could not be verified.');
        } else {
          ready('[data-document-ready-generation]', 'Unverified', 'pending');
          ready('[data-document-ready-publishing]', 'Unknown', 'pending');
          authority('Unverified');
          announceRuntime('The active runtime could not be verified for document generation. Generate Now remains available to re-check the runtime, but no request will be sent until capability discovery succeeds.');
        }
      }
    }

    form.addEventListener('submit', async function (ev) {
      ev.preventDefault();
      var ep = endpoint(), request = requestBody(), validationError = validateRequest(request);
      if (validationError) { announce(validationError, 'warning'); return; }
      if (!ep || !runtimeEnabled) {
        await discoverRuntime();
        ep = endpoint();
        if (!ep || !runtimeEnabled) { announce('Document generation is not ready in the active runtime. Check Endpoint / Generation above, or save and copy the request while the runtime is configured.', 'warning'); return; }
      }
      submit.disabled = true; card.hidden = true; announce('Generating document…', 'working');
      try {
        var packet = await runtimeJson(ep, {
          method:'POST',
          headers:{'Content-Type':'application/json','Accept':'application/json'},
          body:JSON.stringify(request)
        }, {label:'Document generation', timeoutMs:120000, maxBytes:8*1024*1024});
        var res = packet.response, body = packet.body || {};
        if (!res.ok) throw new Error((body.error && body.error.message) || String(body.detail || ('HTTP ' + res.status)));
        if (body.contract !== 'assistant.document-generation-response.v1' || typeof body.content !== 'string') throw new Error('Document runtime returned an invalid response.');
        last = body;
        out.textContent = body.content;
        meta.textContent = [body.filename, body.mime_type, body.size + ' bytes', body.selected_model ? 'Model ' + body.selected_model : '', 'SHA-256 ' + body.sha256].filter(Boolean).join(' · ');
        card.hidden = false;
        if (library) library.upsert({
          id:'document:' + String(body.sha256 || (Date.now() + '-' + Math.random().toString(36).slice(2,8))),
          title:String(body.filename || request.title || 'AI Learn document'), status:'ready', state_label:'Ready', preview_label:'Document',
          meta:[body.mime_type, body.size ? body.size + ' bytes' : '', body.selected_model ? 'Model ' + body.selected_model : ''].filter(Boolean).join(' · '),
          note:'The receipt stores metadata only. Use the private result above to download or copy the generated content in this session.'
        });
        if (requestActions) requestActions.saveDraft(true);
        announce('Document is ready for review.', 'success');
      } catch (e) {
        announce('Unable to generate document: ' + String(e.message || e), 'error');
      } finally {
        submit.disabled = false;
      }
    });
    download.addEventListener('click', function () {
      if (!last) return;
      var blob = new Blob([last.content], {type:last.mime_type || 'text/plain'}), u = URL.createObjectURL(blob), a = document.createElement('a');
      a.href = u; a.download = last.filename || 'ai-learn-document.txt'; document.body.appendChild(a); a.click(); a.remove(); setTimeout(function () { URL.revokeObjectURL(u); }, 1000);
    });
    copy.addEventListener('click', function () {
      if (!last) return;
      copyText(last.content).then(function () { announce('Document text copied.', 'success'); }).catch(function () { announce('Clipboard is unavailable.', 'warning'); });
    });

    discoverRuntime();
    try {
      var endpointApi = window.AI_ASSISTANT_ENDPOINT_API;
      unsubscribeProfile = endpointApi && typeof endpointApi.onProfileChange === 'function' ? endpointApi.onProfileChange(discoverRuntime) : null;
    } catch (_) {}
    var dispose = ui.onPageDispose || function (callback) {
      window.addEventListener('pagehide', function handler(event) {
        if (event && event.persisted === true) return;
        window.removeEventListener('pagehide', handler);
        callback();
      });
    };
    dispose(function () { if (typeof unsubscribeProfile === 'function') unsubscribeProfile(); });
  });
}());
