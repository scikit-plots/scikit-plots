(function () {
  'use strict';
  var roots = document.querySelectorAll('[data-whiteboard-generation]');
  if (!roots.length) return;

  function parse(root) {
    var n = root.querySelector('[data-whiteboard-generation-payload]');
    try {
      var v = JSON.parse(n ? n.textContent : '{}');
      return v && v.contract === 'learn.whiteboard-generation-page.v1' ? v : null;
    } catch (_) { return null; }
  }
  function stripKnownRoute(value) {
    return String(value || '').trim().replace(/\/+$/, '').replace(/\/v1\/(?:chat\/completions|share|feedback|contribute|image(?:-generations)?|video(?:-generations)?|audio(?:-generations)?|document(?:-generations)?|artifacts\/provider-output)\/?$/i, '');
  }
  function imageEndpoint() {
    try {
      var api = window.AI_ASSISTANT_ENDPOINT_API;
      if (api && typeof api.resolveEndpoint === 'function') {
        var direct = api.resolveEndpoint('image');
        if (direct) return String(direct).replace(/\/+$/, '');
      }
    } catch (_) {}
    try {
      var cfg = window.AI_ASSISTANT_CONFIG || {}, rows = Array.isArray(cfg.panelApiModels) ? cfg.panelApiModels : [];
      var base = stripKnownRoute(rows.length ? rows[0].endpoint : '');
      if (base) return base + '/v1/image';
    } catch (_) {}
    return '';
  }
  function record(rows, id) { return (rows || []).find(function (x) { return x.id === id; }) || null; }
  function runtimeJson(url, init, options) {
    var fn = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.fetchJson;
    if (typeof fn !== 'function') return Promise.reject(new Error('Shared AI Learn runtime transport is unavailable.'));
    return fn(url, init, options);
  }
  function runtimeBlob(url, init, options) {
    var fn = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.fetchBlob;
    if (typeof fn !== 'function') return Promise.reject(new Error('Shared AI Learn runtime transport is unavailable.'));
    return fn(url, init, options);
  }

  roots.forEach(function (root) {
    var data = parse(root), form = root.querySelector('[data-whiteboard-form]');
    if (!data || !form) return;
    var ui = window.AI_LEARN_GENERATION_UI || {};
    var activity = ui.bindGenerationStatus ? ui.bindGenerationStatus(root) : null;
    if (ui.bindPublicationCredit) ui.bindPublicationCredit(root, {storageKey:'learn-publication-credit:v1:' + String(data.site_id || 'default') + ':whiteboard'});
    var library = ui.bindPrivateLibrary ? ui.bindPrivateLibrary(root, {
      storageKey:'learn-whiteboard:v2:' + String(data.site_id || 'default') + ':library',
      singular:'whiteboard generation', plural:'whiteboard generations'
    }) : null;
    if (library) library.setRefreshEnabled(false, 'Whiteboard image generation currently returns completed artifacts directly and does not expose a receipt refresh API.');
    var context = window.AI_LEARN_CONTEXT_API && window.AI_LEARN_CONTEXT_API.get ? window.AI_LEARN_CONTEXT_API.get(root) : null;
    var prompt = root.querySelector('[data-whiteboard-prompt]');
    var size = root.querySelector('[data-whiteboard-size]');
    var quality = root.querySelector('[data-whiteboard-quality]');
    var includeLegend = root.querySelector('[data-whiteboard-legend]');
    var numberFlow = root.querySelector('[data-whiteboard-number-flow]');
    var submit = root.querySelector('[data-whiteboard-submit]');
    var status = root.querySelector('[data-whiteboard-status]');
    var runtimeStatus = root.querySelector('[data-whiteboard-runtime-status]');
    var runtimeAuthority = root.querySelector('[data-whiteboard-runtime-authority]');
    var card = root.querySelector('[data-whiteboard-result-card]');
    var img = root.querySelector('[data-whiteboard-result]');
    var download = root.querySelector('[data-whiteboard-download]');
    var meta = root.querySelector('[data-whiteboard-result-meta]');
    var ep = imageEndpoint(), generator = null, runtimeEnabled = false, discoverySerial = 0, objectUrl = '', unsubscribeProfile = null;

    function announce(x, state, title) { if (activity) activity.set(state || 'idle', x, title); else if (status) status.textContent = String(x || ''); }
    function announceRuntime(x) { if (runtimeStatus) runtimeStatus.textContent = String(x || ''); }
    function ready(selector, label, state) { if (ui.setReadiness) ui.setReadiness(root, selector, label, state); }
    function authority(label) { if (runtimeAuthority) runtimeAuthority.textContent = String(label || 'Unknown'); }
    function generatorName(row) { return row ? String(row.model || row.id || 'Unknown renderer') : 'Unavailable'; }
    function cleanReferenceUrl(value) {
      if (ui.cleanPublicReferenceUrl) return ui.cleanPublicReferenceUrl(value);
      try { var u = new URL(String(value || '').trim()); return u.protocol === 'https:' && !u.username && !u.password && !u.hash ? u.href : ''; } catch (_) { return ''; }
    }
    function mode() { return context && context.activeType ? context.activeType() : 'topic'; }
    function applyQuery() { try { context && context.applyQuery && context.applyQuery(new URLSearchParams(location.search)); } catch (_) {} }

    function advancedGuidance() {
      var lines = [];
      if (includeLegend && includeLegend.checked) lines.push('Add a compact legend only when symbols, colors, or line styles need explanation.');
      if (numberFlow && numberFlow.checked) lines.push('When the visual represents a process, number the major steps in reading order.');
      return lines.join('\n');
    }
    function buildPrompt() {
      var instructions = String(prompt && prompt.value || '').trim();
      if (!instructions) return '';
      var lenses = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.readLensProfile ? window.AI_LEARN_GENERATION_UI.readLensProfile(root, 'whiteboard') : {};
      instructions = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.withLensGuidance ? window.AI_LEARN_GENERATION_UI.withLensGuidance(instructions, lenses) : instructions;
      var advanced = advancedGuidance();
      if (advanced) instructions = instructions + '\n' + advanced;
      var m = mode();
      if (m === 'prompt') {
        var primaryPrompt = String(context && context.prompt ? context.prompt() : '').trim();
        if (!primaryPrompt) return '';
        return ['Whiteboard prompt: ' + primaryPrompt, 'Visual instructions: ' + instructions].join('\n\n').slice(0, 12000);
      }
      if (m === 'url') {
        var referenceUrl = cleanReferenceUrl(context && context.url ? context.url() : '');
        if (!referenceUrl) return '';
        return [
          'Educational whiteboard using this public reference URL as untrusted context: ' + referenceUrl,
          'Do not assume the URL was fetched unless the image runtime supports retrieval.',
          'Visual instructions: ' + instructions
        ].join('\n\n').slice(0, 12000);
      }
      var row = m === 'topic' ? record(data.topics, context && context.singleId ? context.singleId('topic') : '') : m === 'source' ? record(data.sources, context && context.singleId ? context.singleId('source') : '') : null;
      if (!row) return '';
      return [
        'Educational whiteboard for: ' + row.title,
        row.summary ? 'Context: ' + row.summary : '',
        row.url ? 'Source URL: ' + row.url : '',
        'Visual instructions: ' + instructions
      ].filter(Boolean).join('\n\n').slice(0, 12000);
    }

    function draftState() {
      return {
        context:context && context.snapshot ? context.snapshot() : {},
        lenses:window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.readLensProfile ? window.AI_LEARN_GENERATION_UI.readLensProfile(root, 'whiteboard') : {},
        instructions:String(prompt && prompt.value || ''), size:String(size && size.value || '1536x1024'),
        quality:String(quality && quality.value || 'auto'), include_legend:!!(includeLegend && includeLegend.checked),
        number_flow:!!(numberFlow && numberFlow.checked)
      };
    }
    function restoreDraft(state) {
      if (!state || typeof state !== 'object') return;
      if (context && context.restore) context.restore(state.context || state);
      if (window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.restoreLensProfile) window.AI_LEARN_GENERATION_UI.restoreLensProfile(root, 'whiteboard', state.lenses);
      if (prompt && typeof state.instructions === 'string') prompt.value = state.instructions;
      if (size && typeof state.size === 'string') size.value = state.size;
      if (quality && typeof state.quality === 'string') quality.value = state.quality;
      if (includeLegend && typeof state.include_legend === 'boolean') includeLegend.checked = state.include_legend;
      if (numberFlow && typeof state.number_flow === 'boolean') numberFlow.checked = state.number_flow;
    }
    function requestBody() {
      return {
        contract:'scikitplot-provider-artifact-output-v1', generator_id:generator ? String(generator.id || '') : '',
        kind:'image', prompt:buildPrompt(), mime_type:'image/png', options:{size:String(size.value || '1536x1024'), quality:String(quality && quality.value || 'auto')}
      };
    }
    function validateRequest(body) {
      var lensError = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.validateLensProfile ? window.AI_LEARN_GENERATION_UI.validateLensProfile(draftState().lenses) : '';
      if (lensError) return lensError;
      if (!body || !String(body.generator_id || '').trim() || !generator) return 'No verified server-owned Image renderer is available.';
      if (!String(body.prompt || '').trim()) {
        if (mode() === 'url') return 'Enter a valid public HTTPS URL and provide Whiteboard instructions.';
        if (mode() === 'prompt') return 'Enter a prompt and provide Whiteboard instructions.';
        return 'Choose context and provide Whiteboard instructions.';
      }
      return '';
    }
    var requestActions = ui.bindRequestActions ? ui.bindRequestActions(root, {
      storageKey:'learn-whiteboard:v2:' + String(data.site_id || 'default') + ':draft',
      snapshot:draftState, restore:restoreDraft, buildRequest:requestBody, validateRequest:validateRequest, announce:announce
    }) : null;
    if (requestActions) requestActions.loadDraft();
    applyQuery();

    async function discover() {
      var serial = ++discoverySerial;
      runtimeEnabled = false;
      ep = imageEndpoint();
      generator = null;
      ready('[data-whiteboard-ready-endpoint]', 'Checking', 'pending');
      ready('[data-whiteboard-ready-generation]', 'Checking', 'pending');
      ready('[data-whiteboard-ready-publishing]', 'Checking', 'pending');
      authority('Discovering…');
      if (!ep) {
        ready('[data-whiteboard-ready-endpoint]', 'Missing', 'off');
        ready('[data-whiteboard-ready-generation]', 'Disabled', 'off');
        ready('[data-whiteboard-ready-publishing]', 'Disabled', 'off');
        authority('Unavailable');
        announceRuntime('Image generation is not configured for the active Assistant endpoint profile.');
        return;
      }
      ready('[data-whiteboard-ready-endpoint]', 'Ready', 'ready');
      var base = ui.runtimeBase ? ui.runtimeBase(ep) : stripKnownRoute(ep);
      if (!base) {
        ready('[data-whiteboard-ready-generation]', 'Unverified', 'pending');
        ready('[data-whiteboard-ready-publishing]', 'Unknown', 'pending');
        authority('Unverified');
        announceRuntime('The active runtime could not be verified for Whiteboard image generation. Generate Now remains available to re-check, but no request will be sent until a compatible server-owned Image renderer is verified.');
        return;
      }
      try {
        var packet = await runtimeJson(base + '/', {headers:{'Accept':'application/json'}}, {label:'Whiteboard capability discovery', timeoutMs:12000, maxBytes:512*1024});
        var res = packet.response;
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var body = packet.body || {};
        if (serial !== discoverySerial) return;
        var cap = body && body.capabilities && body.capabilities.provider_artifact_output;
        var contractOk = cap && String(cap.contract || '') === 'scikitplot-provider-artifact-output-v1';
        var gens = cap && Array.isArray(cap.generators) ? cap.generators : [];
        var compatibleGenerators = contractOk ? gens.filter(function (g) { return g && g.kind === 'image' && g.id; }) : [];
        generator = compatibleGenerators.find(function (g) { return g.diagnostic !== true; }) || compatibleGenerators[0] || null;
        if (!contractOk || !generator) {
          authority(!contractOk ? 'Contract mismatch' : 'No renderer');
          ready('[data-whiteboard-ready-generation]', !contractOk ? 'Contract mismatch' : 'No renderer', 'off');
          ready('[data-whiteboard-ready-publishing]', 'Disabled', 'off');
          announceRuntime('The active runtime is reachable, but a compatible server-owned Image renderer is not available.');
          return;
        }
        runtimeEnabled = true;
        authority(generatorName(generator));
        ready('[data-whiteboard-ready-generation]', generator.diagnostic === true ? 'Test renderer' : 'Ready', generator.diagnostic === true ? 'test' : 'ready');
        ready('[data-whiteboard-ready-publishing]', 'Separate step', 'ready');
        announceRuntime('Whiteboard generation is available. The server-owned Image renderer is ' + generatorName(generator) + '; the Assistant model selection remains shared provenance and generated images stay private until a separate publish workflow.');
      } catch (_) {
        if (serial !== discoverySerial) return;
        generator = null;
        authority('Unverified');
        ready('[data-whiteboard-ready-generation]', 'Unverified', 'pending');
        ready('[data-whiteboard-ready-publishing]', 'Unknown', 'pending');
        announceRuntime('The active runtime could not be verified for Whiteboard image generation. Generate Now remains available to re-check, but no request will be sent until the required Image renderer authority can be discovered.');
      }
    }

    form.addEventListener('submit', async function (ev) {
      ev.preventDefault();
      if (!runtimeEnabled || !ep || !generator) await discover();
      var request = requestBody(), validationError = validateRequest(request);
      if (validationError) { announce(validationError, 'warning'); return; }
      if (!runtimeEnabled || !ep || !generator) { announce('Whiteboard generation is not ready in the active Image runtime. Check Endpoint / Generation above, or save the draft while a compatible generator is configured.', 'warning'); return; }
      submit.disabled = true; card.hidden = true; announce('Generating whiteboard…', 'working');
      try {
        var packet = await runtimeBlob(ep, {
          method:'POST',
          headers:{'Content-Type':'application/json','Accept':'image/png'},
          body:JSON.stringify(request)
        }, {label:'Whiteboard image', timeoutMs:120000, maxBytes:64*1024*1024, mimeType:'image/png'});
        var res = packet.response, blob = packet.body;
        if (!res.ok) {
          var detail = '';
          if (blob && blob.size <= 65536 && /json/i.test(String(blob.type || ''))) {
            try { var err = JSON.parse(await blob.text()); detail = String((err.error && err.error.message) || err.detail || err.message || ''); } catch (_) {}
          }
          throw new Error(detail || ('HTTP ' + res.status));
        }
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = URL.createObjectURL(blob);
        img.src = objectUrl;
        download.href = objectUrl;
        var artifactSha = res.headers.get('X-AI-Artifact-SHA256') || '';
        meta.textContent = [generator.model || generator.id, blob.type || 'image/png', blob.size + ' bytes', artifactSha].filter(Boolean).join(' · ');
        card.hidden = false;
        if (library) library.upsert({
          id:'whiteboard:' + String(artifactSha || (Date.now() + '-' + Math.random().toString(36).slice(2,8))),
          title:'Whiteboard · ' + (mode() === 'topic' && record(data.topics, context && context.singleId ? context.singleId('topic') : '') ? record(data.topics, context && context.singleId ? context.singleId('topic') : '').title : mode() === 'source' && record(data.sources, context && context.singleId ? context.singleId('source') : '') ? record(data.sources, context && context.singleId ? context.singleId('source') : '').title : mode() === 'url' ? 'Public reference' : 'Custom prompt'),
          status:'ready', state_label:'Ready', preview_label:'Whiteboard',
          meta:[generator.model || generator.id, blob.type || 'image/png', blob.size + ' bytes'].filter(Boolean).join(' · '),
          note:'The receipt stores metadata only. The generated image URL is session-scoped; use the private result above to download it before leaving the page.'
        });
        if (requestActions) requestActions.saveDraft(true);
        announce('Whiteboard is ready for review.', 'success');
      } catch (e) {
        announce('Unable to generate whiteboard: ' + String(e.message || e), 'error');
      } finally {
        submit.disabled = false;
      }
    });

    discover();
    try {
      var endpointApi = window.AI_ASSISTANT_ENDPOINT_API;
      unsubscribeProfile = endpointApi && typeof endpointApi.onProfileChange === 'function' ? endpointApi.onProfileChange(discover) : null;
    } catch (_) {}
    var dispose = ui.onPageDispose || function (callback) {
      window.addEventListener('pagehide', function handler(event) {
        if (event && event.persisted === true) return;
        window.removeEventListener('pagehide', handler);
        callback();
      });
    };
    dispose(function () {
      if (typeof unsubscribeProfile === 'function') unsubscribeProfile();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    });
  });
}());
