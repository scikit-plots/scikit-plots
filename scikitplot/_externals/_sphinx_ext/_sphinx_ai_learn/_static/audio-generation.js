(function () {
  'use strict';

  var roots = document.querySelectorAll('[data-audio-generation]');
  if (!roots.length) return;

  function parsePayload(root) {
    var node = root.querySelector('[data-audio-generation-payload]');
    if (!node) return null;
    try {
      var data = JSON.parse(node.textContent || '{}');
      return data && data.contract === 'learn.audio-generation-page.v1' ? data : null;
    } catch (_) { return null; }
  }

  function selectedAssistantModel() {
    var snapshot = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.assistantModelSnapshot ? window.AI_LEARN_GENERATION_UI.assistantModelSnapshot() : null;
    return snapshot ? String(snapshot.model || snapshot.id || '') : '';
  }

  function stripKnownRoute(value) {
    var raw = String(value || '').trim().replace(/\/+$/, '');
    return raw.replace(/\/v1\/(?:chat\/completions|image(?:-generations)?|video(?:-generations)?|audio(?:-generations)?|document(?:-generations)?)\/?$/i, '');
  }

  function audioEndpoint() {
    try {
      var api = window.AI_ASSISTANT_ENDPOINT_API;
      if (api && typeof api.resolveEndpoint === 'function') {
        var direct = String(api.resolveEndpoint('audio') || '').trim();
        if (direct) return direct;
      }
      if (api && typeof api.getProfile === 'function' && typeof api.getActiveProfile === 'function') {
        var profile = api.getProfile(api.getActiveProfile());
        if (profile) {
          var explicit = String(profile.audio || '').trim();
          if (explicit) return explicit;
          var base = String(profile.base || '').trim().replace(/\/+$/, '');
          if (base) return base + '/v1/audio';
          var sibling = String(profile.video || profile.chat || '').trim();
          var root = stripKnownRoute(sibling);
          if (root && root !== sibling) return root + '/v1/audio';
        }
      }
    } catch (_) {}
    try {
      var cfg = window.AI_ASSISTANT_CONFIG || {};
      var models = Array.isArray(cfg.panelApiModels) ? cfg.panelApiModels : [];
      var endpoint = models.length ? String(models[0].endpoint || '') : '';
      var base = stripKnownRoute(endpoint);
      if (base && base !== endpoint) return base + '/v1/audio';
    } catch (_) {}
    return '';
  }

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

  function randomKey() {
    try { if (crypto && typeof crypto.randomUUID === 'function') return crypto.randomUUID(); } catch (_) {}
    var bytes = new Uint8Array(16);
    try { crypto.getRandomValues(bytes); } catch (_) { for (var i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256); }
    return Array.prototype.map.call(bytes, function (b) { return b.toString(16).padStart(2, '0'); }).join('');
  }

  function safeArtifactUrl(endpoint, artifactId) {
    try {
      var base = new URL(endpoint, window.location.href);
      var root = stripKnownRoute(base.toString());
      if (!root) return '';
      var url = new URL(root + '/v1/generated-artifacts/' + encodeURIComponent(artifactId));
      if (url.username || url.password || url.hash) return '';
      if (url.protocol === 'https:' && (!url.port || url.port === '443')) return url.toString();
      if (url.protocol === 'http:' && ['localhost','127.0.0.1','::1'].includes(url.hostname)) return url.toString();
      return '';
    } catch (_) { return ''; }
  }

  function labelStage(stage) {
    return ({queued:'Queued', running:'Synthesizing speech', synthesizing:'Synthesizing speech', ready:'Ready', failed:'Failed', cancelled:'Cancelled'})[stage] || String(stage || 'Working');
  }

  roots.forEach(function (root) {
    var payload = parsePayload(root);
    var form = root.querySelector('[data-audio-form]');
    var context = window.AI_LEARN_CONTEXT_API && window.AI_LEARN_CONTEXT_API.get ? window.AI_LEARN_CONTEXT_API.get(root) : null;
    var prompt = root.querySelector('[data-audio-prompt]');
    var language = root.querySelector('[data-audio-language]');
    var voice = root.querySelector('[data-audio-voice]');
    var pacing = root.querySelector('[data-audio-pacing]');
    var expandAcronyms = root.querySelector('[data-audio-expand-acronyms]');
    var verbalizeUncertainty = root.querySelector('[data-audio-verbalize-uncertainty]');
    var submit = root.querySelector('[data-audio-submit]');
    var cancel = root.querySelector('[data-audio-cancel]');
    var status = root.querySelector('[data-audio-status]');
    var runtimeStatus = root.querySelector('[data-audio-runtime-status]');
    var generatorLabel = root.querySelector('[data-audio-generator-label]');
    var card = root.querySelector('[data-audio-job]');
    var progress = root.querySelector('[data-audio-progress]');
    var stage = root.querySelector('[data-audio-stage]');
    var resultAudio = root.querySelector('[data-audio-result]');
    var resultNote = root.querySelector('[data-audio-result-note]');
    var pollTimer = 0;
    var objectUrl = '';
    var activeJob = null;
    var endpoint = audioEndpoint();
    var runtimeEnabled = false;
    var discoverySerial = 0;
    var unsubscribeProfile = null;
    var idempotencyKey = '';
    var idempotencySignature = '';

    if (!payload || !form) return;

    var ui = window.AI_LEARN_GENERATION_UI || {};
    var activity = ui.bindGenerationStatus ? ui.bindGenerationStatus(root) : null;
    if (ui.bindPublicationCredit) ui.bindPublicationCredit(root, {storageKey:'learn-publication-credit:v1:' + String(payload.site_id || 'default') + ':audio'});
    var library = ui.bindPrivateLibrary ? ui.bindPrivateLibrary(root, {
      storageKey:'learn-audio:v2:' + String(payload.site_id || 'default') + ':library',
      singular:'audio generation', plural:'audio generations'
    }) : null;
    if (library) library.setRefreshEnabled(false, 'Audio job capabilities are scoped to the active session; collection refresh is not exposed by this endpoint yet.');
    function announce(message, state, title) { if (activity) activity.set(state || 'idle', message, title); else if (status) status.textContent = String(message || ''); }
    function announceRuntime(message) { if (runtimeStatus) runtimeStatus.textContent = String(message || ''); }
    function ready(selector, label, state) { if (ui.setReadiness) ui.setReadiness(root, selector, label, state); }
    function cleanReferenceUrl(value) {
      if (ui.cleanPublicReferenceUrl) return ui.cleanPublicReferenceUrl(value);
      try { var u = new URL(String(value || '').trim()); return u.protocol === 'https:' && !u.username && !u.password && !u.hash ? u.href : ''; } catch (_) { return ''; }
    }
    function activeProfile() {
      try { var api = window.AI_ASSISTANT_ENDPOINT_API; var id = api && api.getActiveProfile && api.getActiveProfile(); return api && api.getProfile && id ? api.getProfile(id) : null; } catch (_) { return null; }
    }
    function mode() { return context && context.activeType ? context.activeType() : 'topic'; }

    function recordById(rows, id) { return (rows || []).find(function (row) { return row.id === id; }) || null; }
    function libraryTitle() {
      var value = mode(), row = null;
      if (value === 'topic') row = recordById(payload.topics, context && context.singleId ? context.singleId('topic') : '');
      if (value === 'source') row = recordById(payload.sources, context && context.singleId ? context.singleId('source') : '');
      if (row && row.title) return 'Audio · ' + row.title;
      if (value === 'url') {
        try { return 'Audio · ' + new URL(String(context && context.url ? context.url() : '')).hostname; } catch (_) {}
      }
      if (value === 'prompt') {
        var text = String(context && context.prompt ? context.prompt() : '').trim();
        if (text) return 'Audio · ' + text.slice(0, 72);
      }
      return 'Audio explanation';
    }
    function rememberJob(job) {
      if (!library || !job || !job.generation_id) return;
      var execution = job.execution || {};
      var state = String(execution.state || execution.stage || 'queued');
      library.upsert({
        id:'audio:' + String(job.generation_id), title:libraryTitle(), status:state,
        state_label:labelStage(execution.stage || execution.state), preview_label:'Audio',
        meta:selectedAssistantModel() ? 'Request model · ' + selectedAssistantModel() : '',
        note:state === 'ready' ? 'The generated audio artifact is session-scoped; use the private result above while it is available.' : ''
      });
    }
    function advancedGuidance() {
      var lines = [];
      var pace = String(pacing && pacing.value || 'auto');
      if (pace === 'deliberate') lines.push('Delivery: speak deliberately and explain the sequence step by step.');
      if (pace === 'concise') lines.push('Delivery: stay concise and minimize repetition.');
      if (expandAcronyms && expandAcronyms.checked) lines.push('Pronunciation: expand acronyms on first mention when the expansion is known from context.');
      if (verbalizeUncertainty && verbalizeUncertainty.checked) lines.push('Evidence: state important uncertainty and limitations explicitly in the narration.');
      return lines.join('\n');
    }
    function buildText() {
      var instructions = String(prompt && prompt.value || '').trim();
      if (!instructions) return '';
      var lenses = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.readLensProfile ? window.AI_LEARN_GENERATION_UI.readLensProfile(root, 'audio') : {};
      instructions = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.withLensGuidance ? window.AI_LEARN_GENERATION_UI.withLensGuidance(instructions, lenses) : instructions;
      var advanced = advancedGuidance();
      if (advanced) instructions = instructions + '\n' + advanced;
      var value = mode();
      var record = null;
      if (value === 'topic') record = recordById(payload.topics, context && context.singleId ? context.singleId('topic') : '');
      if (value === 'source') record = recordById(payload.sources, context && context.singleId ? context.singleId('source') : '');
      if (value === 'prompt') {
        var primaryPrompt = String(context && context.prompt ? context.prompt() : '').trim();
        if (!primaryPrompt) return '';
        return ['Prompt: ' + primaryPrompt, 'Narration instructions: ' + instructions].join('\n\n').slice(0, 12000);
      }
      if (value === 'url') {
        var referenceUrl = cleanReferenceUrl(context && context.url ? context.url() : '');
        if (!referenceUrl) return '';
        return [
          'Public reference URL (untrusted; retrieval is runtime-dependent): ' + referenceUrl,
          'Narration instructions: ' + instructions
        ].join('\n\n').slice(0, 12000);
      }
      if (!record) return '';
      return [
        'Title: ' + record.title,
        record.summary ? 'Context: ' + record.summary : '',
        record.url ? 'Source URL: ' + record.url : '',
        'Narration instructions: ' + instructions
      ].filter(Boolean).join('\n\n').slice(0, 12000);
    }

    function prefillFromQuery() { try { context && context.applyQuery && context.applyQuery(new URLSearchParams(window.location.search)); } catch (_) {} }
    function draftState() {
      return {
        context:context && context.snapshot ? context.snapshot() : {},
        lenses:window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.readLensProfile ? window.AI_LEARN_GENERATION_UI.readLensProfile(root, 'audio') : {},
        instructions:String(prompt && prompt.value || ''), language:String(language && language.value || 'auto'), voice:String(voice && voice.value || 'auto'),
        pacing:String(pacing && pacing.value || 'auto'), expand_acronyms:!!(expandAcronyms && expandAcronyms.checked),
        verbalize_uncertainty:!!(verbalizeUncertainty && verbalizeUncertainty.checked)
      };
    }
    function restoreDraft(state) {
      if (!state || typeof state !== 'object') return;
      if (context && context.restore) context.restore(state.context || state);
      if (window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.restoreLensProfile) window.AI_LEARN_GENERATION_UI.restoreLensProfile(root, 'audio', state.lenses);
      if (prompt && typeof state.instructions === 'string') prompt.value = state.instructions;
      if (language && typeof state.language === 'string') language.value = state.language;
      if (voice && typeof state.voice === 'string') voice.value = state.voice;
      if (pacing && typeof state.pacing === 'string') pacing.value = state.pacing;
      if (expandAcronyms && typeof state.expand_acronyms === 'boolean') expandAcronyms.checked = state.expand_acronyms;
      if (verbalizeUncertainty && typeof state.verbalize_uncertainty === 'boolean') verbalizeUncertainty.checked = state.verbalize_uncertainty;
    }
    function requestBody() {
      return {contract:'assistant.audio-generation-request.v1', text:buildText(), selected_model:selectedAssistantModel(), mode:'narration'};
    }
    function ensureIdempotency(request) {
      var signature = JSON.stringify(request || {});
      if (!idempotencyKey || idempotencySignature !== signature) {
        idempotencyKey = randomKey();
        idempotencySignature = signature;
      }
      return idempotencyKey;
    }

    function validateRequest(body) {
      var lensError = window.AI_LEARN_GENERATION_UI && window.AI_LEARN_GENERATION_UI.validateLensProfile ? window.AI_LEARN_GENERATION_UI.validateLensProfile(draftState().lenses) : '';
      if (lensError) return lensError;
      if (!body || !String(body.selected_model || '').trim()) return 'Select an Assistant model before generating audio.';
      if (!String(body.text || '').trim()) {
        if (mode() === 'url') return 'Enter a valid public HTTPS URL and provide narration instructions.';
        if (mode() === 'prompt') return 'Enter a prompt and provide narration instructions.';
        return 'Choose context and provide narration instructions.';
      }
      return '';
    }
    var requestActions = ui.bindRequestActions ? ui.bindRequestActions(root, {
      storageKey:'learn-audio:v2:' + String(payload.site_id || 'default') + ':draft',
      snapshot:draftState, restore:restoreDraft, buildRequest:requestBody, validateRequest:validateRequest, announce:announce
    }) : null;
    if (requestActions) requestActions.loadDraft();
    prefillFromQuery();

    async function discoverRuntime() {
      var serial = ++discoverySerial;
      runtimeEnabled = false;
      endpoint = audioEndpoint();
      ready('[data-audio-ready-endpoint]', 'Checking', 'pending');
      ready('[data-audio-ready-generation]', 'Checking', 'pending');
      ready('[data-audio-ready-publishing]', 'Checking', 'pending');
      if (generatorLabel) generatorLabel.textContent = 'Discovering…';
      if (!endpoint) {
        ready('[data-audio-ready-endpoint]', 'Missing', 'off');
        ready('[data-audio-ready-generation]', 'Disabled', 'off');
        ready('[data-audio-ready-publishing]', 'Disabled', 'off');
        if (generatorLabel) generatorLabel.textContent = 'Unavailable';
        announceRuntime('Audio generation is not configured for the active Assistant endpoint profile.');
        return;
      }
      ready('[data-audio-ready-endpoint]', 'Ready', 'ready');
      var profile = activeProfile();
      var explicit = !!(profile && String(profile.audio || '').trim());
      var base = ui.runtimeBase ? ui.runtimeBase(endpoint) : stripKnownRoute(endpoint);
      if (!base) {
        if (explicit) {
          runtimeEnabled = true;
          ready('[data-audio-ready-generation]', 'Explicit route', 'ready');
          ready('[data-audio-ready-publishing]', 'Runtime-defined', 'pending');
          if (generatorLabel) generatorLabel.textContent = 'Server-selected';
          announceRuntime('Audio generation uses an explicit endpoint. Renderer capability discovery is unavailable, so the server remains authoritative.');
        } else {
          ready('[data-audio-ready-generation]', 'Unverified', 'pending');
          ready('[data-audio-ready-publishing]', 'Unknown', 'pending');
          if (generatorLabel) generatorLabel.textContent = 'Unverified';
          announceRuntime('The active runtime could not be verified for audio generation. Generate Now remains available to re-check the runtime, but no request will be sent until capability discovery succeeds.');
        }
        return;
      }
      try {
        var packet = await runtimeJson(base + '/', {headers:{'Accept':'application/json'}}, {label:'Audio capability discovery', timeoutMs:12000, maxBytes:512*1024});
        var res = packet.response;
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var discovery = packet.body || {};
        if (serial !== discoverySerial) return;
        var cap = discovery && discovery.capabilities && discovery.capabilities.audio_generation;
        var providerCap = discovery && discovery.capabilities && discovery.capabilities.provider_artifact_output;
        var generators = providerCap && Array.isArray(providerCap.generators) ? providerCap.generators : [];
        var renderer = generators.find(function (row) { return row && row.kind === 'audio' && row.diagnostic !== true; }) || generators.find(function (row) { return row && row.kind === 'audio'; }) || null;
        if (generatorLabel) generatorLabel.textContent = renderer ? String(renderer.model || renderer.id) : 'Server-selected';
        var contractOk = cap && String(cap.contract || '') === 'assistant.audio-generation-request.v1' && String(cap.job_contract || '') === 'scikitplot-generation-job.v2';
        if (cap && cap.enabled === true && contractOk) {
          runtimeEnabled = true;
          ready('[data-audio-ready-generation]', cap.test_mode === true ? 'Test renderer' : 'Ready', cap.test_mode === true ? 'test' : 'ready');
          ready('[data-audio-ready-publishing]', 'Separate step', 'ready');
          announceRuntime('Audio generation is available. The server-owned speech renderer is ' + (renderer ? String(renderer.model || renderer.id) : 'selected at runtime') + '; generated audio remains private until a separate publish workflow.');
          return;
        }
        if (explicit) {
          runtimeEnabled = true;
          ready('[data-audio-ready-generation]', 'Explicit route', 'ready');
          ready('[data-audio-ready-publishing]', 'Runtime-defined', 'pending');
          announceRuntime('The Audio capability was not advertised, but an explicit operator-configured route is active. The server remains authoritative for rendering.');
          return;
        }
        ready('[data-audio-ready-generation]', cap && cap.enabled === false ? 'Disabled' : 'Contract mismatch', 'off');
        ready('[data-audio-ready-publishing]', 'Disabled', 'off');
        announceRuntime('The active runtime is reachable, but compatible audio generation is not enabled. Generate Now remains available to re-check after configuration changes.');
      } catch (_) {
        if (serial !== discoverySerial) return;
        if (explicit) {
          runtimeEnabled = true;
          ready('[data-audio-ready-generation]', 'Explicit route', 'ready');
          ready('[data-audio-ready-publishing]', 'Unknown', 'pending');
          if (generatorLabel) generatorLabel.textContent = 'Server-selected';
          announceRuntime('The explicit Audio endpoint is configured, but runtime capability discovery could not be verified. The server remains authoritative for rendering.');
        } else {
          ready('[data-audio-ready-generation]', 'Unverified', 'pending');
          ready('[data-audio-ready-publishing]', 'Unknown', 'pending');
          if (generatorLabel) generatorLabel.textContent = 'Unverified';
          announceRuntime('The active runtime could not be verified for audio generation. Generate Now remains available to re-check the runtime, but no request will be sent until capability discovery succeeds.');
        }
      }
    }

    async function fetchArtifact(job) {
      var artifact = job.artifacts && job.artifacts[0];
      if (!artifact || !artifact.artifact_id || !artifact.artifact_capability) return false;
      var url = safeArtifactUrl(endpoint, artifact.artifact_id);
      if (!url) throw new Error('Artifact delivery endpoint is unavailable.');
      var packet = await runtimeBlob(url, {
        method:'GET',
        headers:{'Accept': artifact.mime_type || 'audio/*', 'X-Artifact-Capability': artifact.artifact_capability}
      }, {label:'Audio artifact', timeoutMs:60000, maxBytes:64*1024*1024, mimeType:artifact.mime_type || 'audio/*'});
      var res = packet.response;
      if (!res.ok) throw new Error('Unable to retrieve generated audio.');
      var blob = packet.body;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(blob);
      resultAudio.src = objectUrl;
      resultAudio.hidden = false;
      resultNote.hidden = true;
      return true;
    }

    function stopPolling() { if (pollTimer) { window.clearTimeout(pollTimer); pollTimer = 0; } }

    async function poll(job) {
      if (!job || !job.generation_id || !job.generation_capability) { submit.disabled = false; announce('Audio runtime returned an invalid generation receipt.', 'error'); return; }
      activeJob = job;
      card.hidden = false;
      cancel.hidden = false;
      var state = job.execution || {};
      rememberJob(job);
      if (progress) progress.value = typeof state.progress === 'number' ? state.progress : 0;
      if (stage) stage.textContent = labelStage(state.stage || state.state);
      if (state.state !== 'ready' && state.state !== 'failed' && state.state !== 'cancelled') {
        var percent = typeof state.progress === 'number' ? ' · ' + Math.round(Math.max(0, Math.min(1, state.progress)) * 100) + '%' : '';
        announce(labelStage(state.stage || state.state) + percent, 'working');
      }
      if (state.state === 'ready') {
        stopPolling(); cancel.hidden = true; submit.disabled = false;
        try {
          var ok = await fetchArtifact(job);
          announce(ok ? 'Audio is ready.' : 'Audio completed without a published artifact.', ok ? 'success' : 'warning');
        } catch (error) { announce(String(error.message || error), 'error'); }
        return;
      }
      if (state.state === 'failed' || state.state === 'cancelled') {
        stopPolling(); cancel.hidden = true; submit.disabled = false;
        announce(job.error && job.error.message ? job.error.message : 'Audio generation ' + state.state + '.', state.state === 'failed' ? 'error' : 'warning');
        return;
      }
      stopPolling();
      pollTimer = window.setTimeout(async function () {
        try {
          var packet = await runtimeJson(endpoint + '/' + encodeURIComponent(job.generation_id), {
            headers:{'Accept':'application/json', 'X-Generation-Capability': job.generation_capability}
          }, {label:'Audio status', timeoutMs:15000, maxBytes:256*1024});
          var res = packet.response, body = packet.body || {};
          if (!res.ok) throw new Error((body.error && body.error.message) || ('HTTP ' + res.status));
          poll(body);
        } catch (error) {
          announce('Audio status is temporarily unavailable: ' + String(error.message || error), 'warning');
          pollTimer = window.setTimeout(function () { poll(job); }, 4000);
        }
      }, 2000);
    }

    form.addEventListener('submit', async function (event) {
      event.preventDefault();
      endpoint = audioEndpoint();
      var request = requestBody();
      var validationError = validateRequest(request);
      if (validationError) { announce(validationError, 'warning'); return; }
      if (!endpoint || !runtimeEnabled) {
        await discoverRuntime();
        endpoint = audioEndpoint();
        if (!endpoint || !runtimeEnabled) { announce('Audio generation is not ready in the active runtime. Check Endpoint / Generation above, or save and copy the request while the runtime is configured.', 'warning'); return; }
      }
      submit.disabled = true; cancel.hidden = true; resultAudio.hidden = true; resultNote.hidden = true;
      announce('Submitting audio generation…', 'working');
      try {
        var packet = await runtimeJson(endpoint, {
          method:'POST',
          headers:{'Content-Type':'application/json','Accept':'application/json','Idempotency-Key':ensureIdempotency(request)},
          body:JSON.stringify(request)
        }, {label:'Audio generation', timeoutMs:45000, maxBytes:512*1024});
        var res = packet.response, body = packet.body || {};
        if (!res.ok) throw new Error((body.error && body.error.message) || String(body.detail || ('HTTP ' + res.status)));
        if (requestActions) requestActions.saveDraft(true);
        announce('Audio generation accepted.', 'working', 'Queued');
        rememberJob(body);
        poll(body);
      } catch (error) {
        submit.disabled = false;
        announce('Unable to start audio generation: ' + String(error.message || error), 'error');
      }
    });

    cancel.addEventListener('click', async function () {
      if (!activeJob || !activeJob.generation_id || !activeJob.generation_capability) return;
      try {
        var packet = await runtimeJson(endpoint + '/' + encodeURIComponent(activeJob.generation_id) + '/cancel', {
          method:'POST', headers:{'Accept':'application/json','X-Generation-Capability':activeJob.generation_capability}
        }, {label:'Audio cancellation', timeoutMs:20000, maxBytes:256*1024});
        var res = packet.response, body = packet.body || {};
        if (!res.ok) throw new Error((body.error && body.error.message) || ('HTTP ' + res.status));
        poll(body);
      } catch (error) { announce('Unable to cancel audio generation: ' + String(error.message || error), 'error'); }
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
    dispose(function () {
      stopPolling();
      if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = ''; }
      if (typeof unsubscribeProfile === 'function') unsubscribeProfile();
    });
  });
}());
