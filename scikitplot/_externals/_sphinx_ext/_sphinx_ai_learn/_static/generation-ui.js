(function () {
  'use strict';

  var roots = document.querySelectorAll('.learn-generation-shell');

  function appendInstruction(field, phrase) {
    var text = String(field.value || '').trim();
    var addition = String(phrase || '').trim();
    if (!addition || text.indexOf(addition) !== -1) return;
    field.value = text ? text.replace(/[.\s]+$/, '') + '. ' + addition + '.' : addition + '.';
    field.dispatchEvent(new Event('input', {bubbles:true}));
    field.focus();
  }

  function updateCounter(field) {
    var card = field.closest('.learn-generation-card');
    var counter = card && card.querySelector('[data-generation-counter]');
    if (!counter) return;
    var max = Number(field.getAttribute('maxlength') || 0);
    var used = String(field.value || '').length;
    counter.textContent = max ? used.toLocaleString() + ' / ' + max.toLocaleString() : used.toLocaleString() + ' characters';
  }


  function cleanPublicReferenceUrl(value) {
    var raw = String(value || '').trim();
    if (!raw) return '';
    try {
      var url = new URL(raw);
      if (url.protocol !== 'https:' || !url.hostname || url.username || url.password || url.hash) return '';
      var host = String(url.hostname || '').replace(/^\[|\]$/g, '').replace(/\.$/, '').toLowerCase();
      if (host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local')) return '';
      if (/^(?:127\.|10\.|0\.|169\.254\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)/.test(host)) return '';
      if (host.indexOf(':') !== -1 && (host === '::1' || host.startsWith('fe80:') || /^f[cd][0-9a-f]*:/i.test(host))) return '';
      return url.href;
    } catch (_) { return ''; }
  }

  function runtimeBase(value) {
    return String(value || '').trim().replace(/\/+$/, '').replace(/\/v1\/(?:chat\/completions|share|feedback|contribute|image(?:-generations)?|video(?:-generations)?|audio(?:-generations)?|document(?:-generations)?|artifacts\/provider-output)\/?$/i, '');
  }

  function assistantModelState() {
    try {
      var api = window.AI_ASSISTANT_MODEL_API;
      var state = api && typeof api.getState === 'function' ? api.getState() : null;
      if (state && state.active) return state;
    } catch (_) {}
    try {
      var cfg = window.AI_ASSISTANT_CONFIG || {};
      var rows = Array.isArray(cfg.panelApiModels) ? cfg.panelApiModels : [];
      if (rows.length) return {active:rows[0], effort:null};
    } catch (_) {}
    return null;
  }

  function assistantModelSnapshot() {
    var state = assistantModelState();
    var active = state && state.active;
    if (!active || !(active.id || active.model)) return null;
    var id = String(active.id || active.model);
    var effort = state.effort || {};
    return {
      source:'ai-assistant',
      id:id,
      label:String(active.label || active.model || id),
      provider:String(active.provider || 'custom'),
      model:String(active.model || active.id || id),
      effort:{
        id:String(effort.id || 'default'),
        label:String(effort.label || 'Default'),
        supported:!!effort.supported
      }
    };
  }

  function lensAttribute(scope, group) {
    var suffix = group === 'audiences' ? 'audience' : group === 'purposes' ? 'purpose' : group === 'skills' ? 'skill' : 'role';
    return 'data-' + String(scope || '').trim() + '-' + suffix;
  }

  function readLensProfile(root, scope) {
    var profile = {audiences:[], purposes:[], skills:[], roles:[]};
    if (!root || !scope) return profile;
    Object.keys(profile).forEach(function (group) {
      var attr = lensAttribute(scope, group);
      profile[group] = Array.prototype.slice.call(root.querySelectorAll('input[' + attr + ']:checked')).map(function (node) {
        return String(node.getAttribute(attr) || '').trim();
      }).filter(Boolean);
    });
    return profile;
  }

  function restoreLensProfile(root, scope, profile) {
    if (!root || !scope || !profile || typeof profile !== 'object') return;
    ['audiences','purposes','skills','roles'].forEach(function (group) {
      if (!Array.isArray(profile[group])) return;
      var attr = lensAttribute(scope, group);
      var selected = new Set(profile[group].map(String));
      root.querySelectorAll('input[' + attr + ']').forEach(function (node) {
        node.checked = selected.has(String(node.getAttribute(attr) || ''));
      });
    });
  }

  function validateLensProfile(profile) {
    profile = profile || {};
    if (!Array.isArray(profile.audiences) || !profile.audiences.length) return 'Select at least one Audience lens.';
    if (!Array.isArray(profile.purposes) || !profile.purposes.length) return 'Select at least one Purpose lens.';
    return '';
  }

  function lensGuidance(profile) {
    profile = profile || {};
    function values(key, fallback) {
      var rows = Array.isArray(profile[key]) ? profile[key].map(String).filter(Boolean) : [];
      return rows.length ? rows.join(', ') : fallback;
    }
    return [
      'AI lenses (combined in this one generation request; not separate autonomous agents):',
      'Audience: ' + values('audiences', 'general') + '.',
      'Purpose: ' + values('purposes', 'understand') + '.',
      'Supporting skills: ' + values('skills', 'none') + '.',
      'Role lenses: ' + values('roles', 'none') + '.'
    ].join('\n');
  }

  function withLensGuidance(text, profile, limit) {
    var base = String(text || '').trim();
    var guidance = lensGuidance(profile);
    var max = Number(limit || 0);
    if (max > 0) {
      if (guidance.length >= max) return guidance.slice(0, max);
      var room = Math.max(0, max - guidance.length - (base ? 2 : 0));
      if (base.length > room) base = base.slice(0, room).trimEnd();
    }
    return [base, guidance].filter(Boolean).join('\n\n');
  }

  function setFlowStage(root, scope, order, active) {
    if (!root || !scope || !Array.isArray(order) || !order.length) return;
    var index = order.indexOf(active);
    if (index < 0) index = 0;
    var attr = 'data-' + String(scope).trim() + '-stage';
    root.querySelectorAll('[' + attr + ']').forEach(function (node) {
      var at = order.indexOf(String(node.getAttribute(attr) || ''));
      node.dataset.state = at < index ? 'complete' : at === index ? 'current' : 'pending';
    });
  }

  function setReadiness(root, selector, label, state) {
    var node = root && root.querySelector(selector);
    if (!node) return;
    node.dataset.state = String(state || 'pending');
    var strong = node.querySelector('strong');
    if (strong) strong.textContent = String(label || 'Unknown');
  }

  function bindGenerationStatus(root, options) {
    options = options || {};
    if (!root) return null;
    var node = root.querySelector('[data-generation-status]');
    if (!node) return null;
    if (node._learnGenerationStatus) return node._learnGenerationStatus;
    var label = node.querySelector('[data-generation-status-label]');
    var message = node.querySelector('[data-generation-status-message]');
    var labels = Object.assign({
      idle:'Ready', working:'Working', success:'Complete', warning:'Attention', error:'Error'
    }, options.labels || {});
    var allowed = new Set(['idle','working','success','warning','error']);

    function set(state, text, title) {
      state = allowed.has(String(state || '')) ? String(state) : 'idle';
      node.dataset.state = state;
      if (label) label.textContent = String(title || labels[state] || labels.idle);
      if (message) message.textContent = String(text || '');
      return api;
    }
    function announce(text, state, title) { return set(state || 'idle', text, title); }
    var api = {
      node:node,
      set:set,
      announce:announce,
      idle:function (text, title) { return set('idle', text, title); },
      working:function (text, title) { return set('working', text, title); },
      success:function (text, title) { return set('success', text, title); },
      warning:function (text, title) { return set('warning', text, title); },
      error:function (text, title) { return set('error', text, title); }
    };
    node._learnGenerationStatus = api;
    return api;
  }

  function pulseAction(button) {
    if (!button) return;
    if (button._learnActionTimer) window.clearTimeout(button._learnActionTimer);
    button.dataset.actionState = 'success';
    button._learnActionTimer = window.setTimeout(function () {
      if (button.dataset.actionState === 'success') delete button.dataset.actionState;
      button._learnActionTimer = 0;
    }, 1200);
  }

  function clipboardWrite(text) {
    if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
      return Promise.reject(new Error('Clipboard is unavailable.'));
    }
    return navigator.clipboard.writeText(String(text || ''));
  }

  function bindRequestActions(root, options) {
    options = options || {};
    if (!root || root.dataset.generationActionsBound === 'true') return null;
    var save = root.querySelector('[data-generation-save-draft]');
    var copy = root.querySelector('[data-generation-copy-request]');
    if (!save || !copy) return null;
    root.dataset.generationActionsBound = 'true';
    var storageKey = String(options.storageKey || '').trim();
    var announce = typeof options.announce === 'function' ? options.announce : function () {};
    var snapshot = typeof options.snapshot === 'function' ? options.snapshot : function () { return {}; };
    var restore = typeof options.restore === 'function' ? options.restore : function () {};
    var buildRequest = typeof options.buildRequest === 'function' ? options.buildRequest : function () { return null; };
    var validateRequest = typeof options.validateRequest === 'function' ? options.validateRequest : function () { return ''; };

    function saveDraft(silent) {
      if (!storageKey) { if (!silent) announce('Draft storage is unavailable for this page.', 'warning'); return false; }
      try {
        var payload = JSON.stringify({version:1,saved_at:new Date().toISOString(),state:snapshot()});
        if (payload.length > 60000) throw new Error('Draft is too large for browser storage.');
        localStorage.setItem(storageKey, payload);
        pulseAction(save);
        if (!silent) announce('Draft saved in this browser only.', 'success');
        return true;
      } catch (_) {
        if (!silent) announce('Browser storage is unavailable; the draft was not saved.', 'warning');
        return false;
      }
    }

    function loadDraft() {
      if (!storageKey) return false;
      try {
        var raw = localStorage.getItem(storageKey);
        if (!raw || raw.length > 60000) return false;
        var parsed = JSON.parse(raw);
        if (!parsed || parsed.version !== 1 || !parsed.state || typeof parsed.state !== 'object') return false;
        restore(parsed.state);
        root.querySelectorAll('[data-generation-countable]').forEach(updateCounter);
        return true;
      } catch (_) { return false; }
    }

    async function copyRequest() {
      try {
        var request = buildRequest();
        var error = String(validateRequest(request) || '').trim();
        if (error) { announce(error, 'warning'); return false; }
        await clipboardWrite(JSON.stringify(request, null, 2));
        pulseAction(copy);
        announce('Generation request copied. No network request was sent.', 'success');
        return true;
      } catch (_) {
        announce('Clipboard unavailable. Save the draft or inspect the form values instead.', 'warning');
        return false;
      }
    }

    save.addEventListener('click', function () { saveDraft(false); });
    copy.addEventListener('click', function () { copyRequest(); });
    return {saveDraft:saveDraft, loadDraft:loadDraft, copyRequest:copyRequest};
  }


  function onPageDispose(callback) {
    if (typeof callback !== 'function') return function () {};
    var active = true;
    function handler(event) {
      // BFCache pagehide is a freeze, not a disposal. Keep listeners, object
      // URLs, and subscriptions alive so Back/Forward restores a live page.
      if (event && event.persisted === true) return;
      if (!active) return;
      active = false;
      window.removeEventListener('pagehide', handler);
      callback(event || null);
    }
    window.addEventListener('pagehide', handler);
    return function () {
      if (!active) return;
      active = false;
      window.removeEventListener('pagehide', handler);
    };
  }

  function bindPrivateLibrary(root, options) {
    options = options || {};
    if (!root) return null;
    var section = root.querySelector('[data-generation-library]');
    if (!section) return null;
    if (section._learnPrivateLibrary) return section._learnPrivateLibrary;
    var filter = section.querySelector('[data-generation-library-filter]');
    var refresh = section.querySelector('[data-generation-library-refresh]');
    var grid = section.querySelector('[data-generation-library-grid]');
    var empty = section.querySelector('[data-generation-library-empty]');
    var status = section.querySelector('[data-generation-library-status]');
    if (!filter || !refresh || !grid || !empty || !status) return null;
    var storageKey = String(options.storageKey || '').trim();
    var singular = String(options.singular || section.dataset.generationLibrarySingular || 'generation');
    var plural = String(options.plural || section.dataset.generationLibraryPlural || singular + 's');
    var renderCard = typeof options.renderCard === 'function' ? options.renderCard : null;
    var refreshHandler = typeof options.refresh === 'function' ? options.refresh : null;
    var maxRows = boundedInteger(options.maxRows, 50, 1, 100);
    var maxStorage = boundedInteger(options.maxStorage, 250000, 10000, 1000000);
    var volatileRows = [];

    function announce(message) { status.textContent = String(message || ''); }
    function libraryText(value, limit) {
      return String(value == null ? '' : value).replace(/[\u0000-\u001f\u007f]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, limit);
    }
    function safeLibraryRow(row) {
      if (!row || typeof row !== 'object' || Array.isArray(row)) return null;
      var id = libraryText(row.id, 256);
      var title = libraryText(row.title, 300);
      if (!id || !title) return null;
      // Persist display metadata only. Bearer capabilities, endpoint authority,
      // request bodies and any future unknown fields are intentionally dropped.
      return {
        id:id, title:title,
        status:libraryText(row.status || 'ready', 40) || 'ready',
        state_label:libraryText(row.state_label, 120),
        preview_label:libraryText(row.preview_label, 40),
        meta:libraryText(row.meta, 600),
        note:libraryText(row.note, 1200),
        created_at:libraryText(row.created_at, 64),
        updated_at:libraryText(row.updated_at, 64),
        previous_status:libraryText(row.previous_status, 40)
      };
    }
    function safeRows(value) {
      return Array.isArray(value) ? value.map(safeLibraryRow).filter(Boolean).slice(0, maxRows) : [];
    }
    function read() {
      if (!storageKey) return volatileRows.slice();
      try {
        var raw = localStorage.getItem(storageKey);
        if (!raw || raw.length > maxStorage) return volatileRows.slice();
        var parsed = safeRows(JSON.parse(raw));
        volatileRows = parsed;
        return parsed.slice();
      } catch (_) { return volatileRows.slice(); }
    }
    function write(rows) {
      var normalized = safeRows(rows);
      volatileRows = normalized;
      if (!storageKey) return false;
      try {
        var text = JSON.stringify(normalized);
        if (text.length > maxStorage) {
          var trimmed = normalized.slice(0, Math.max(1, Math.floor(maxRows / 2)));
          text = JSON.stringify(trimmed);
          if (text.length > maxStorage) return false;
          volatileRows = trimmed;
        }
        localStorage.setItem(storageKey, text);
        return true;
      } catch (_) { return false; }
    }
    function nowIso() { return new Date().toISOString(); }
    function dateLabel(value) {
      try { return new Date(value).toLocaleString([], {dateStyle:'medium', timeStyle:'short'}); } catch (_) { return ''; }
    }
    function visibleRows() {
      var mode = String(filter.value || 'active');
      return read().filter(function (row) {
        return mode === 'all' || (mode === 'archived' ? row.status === 'archived' : row.status !== 'archived');
      });
    }
    function button(label, handler, className) {
      var node = document.createElement('button');
      node.type = 'button';
      node.textContent = label;
      if (className) node.className = className;
      node.addEventListener('click', handler);
      return node;
    }
    function updateStatus(count) {
      var noun = count === 1 ? singular : plural;
      announce(count + ' ' + noun + ' shown from this browser.');
    }
    function archive(id, archived) {
      var rows = read();
      var changed = false;
      rows = rows.map(function (row) {
        if (row.id !== id) return row;
        changed = true;
        return Object.assign({}, row, {
          status: archived ? 'archived' : String(row.previous_status || 'ready'),
          previous_status: archived ? String(row.status || 'ready') : '',
          updated_at: nowIso()
        });
      });
      if (!changed) {
        render();
        announce('That ' + singular + ' is no longer available in this browser. Refresh the library and try again.');
        return false;
      }
      var persisted = write(rows);
      render();
      if (!persisted) {
        announce((archived ? 'Archived ' : 'Restored ') + singular + ' for this tab. Browser storage is unavailable, so the change will not survive reload.');
        return false;
      }
      announce((archived ? 'Archived ' : 'Restored ') + singular + '.');
      return true;
    }
    function defaultCard(row) {
      var card = document.createElement('article');
      card.className = 'learn-generation-library-card';
      card.dataset.status = String(row.status || 'ready');
      card.dataset.generationLibraryId = row.id;
      var preview = document.createElement('div');
      preview.className = 'learn-generation-library-preview';
      var mark = document.createElement('span');
      mark.className = 'learn-generation-library-preview-mark';
      mark.setAttribute('aria-hidden', 'true');
      mark.textContent = String(row.preview_label || section.dataset.generationLibraryKind || 'AI').slice(0, 14).toUpperCase();
      preview.append(mark);
      var body = document.createElement('div');
      body.className = 'learn-generation-library-body';
      var heading = document.createElement('h3');
      heading.textContent = row.title;
      var state = document.createElement('p');
      state.className = 'learn-generation-library-state learn-meta';
      state.textContent = String(row.state_label || row.status || 'Ready');
      body.append(heading, state);
      if (row.meta) {
        var meta = document.createElement('p');
        meta.className = 'learn-meta';
        meta.textContent = String(row.meta);
        body.append(meta);
      }
      if (row.note) {
        var note = document.createElement('p');
        note.className = 'learn-generation-library-note';
        note.textContent = String(row.note);
        body.append(note);
      }
      var date = dateLabel(row.updated_at || row.created_at);
      if (date) {
        var time = document.createElement('p');
        time.className = 'learn-generation-library-time learn-meta';
        time.textContent = date;
        body.append(time);
      }
      var menu = document.createElement('details');
      menu.className = 'learn-generation-library-menu';
      var summary = document.createElement('summary');
      summary.textContent = '⋯';
      summary.setAttribute('aria-label', 'More options for ' + row.title);
      summary.title = 'More options';
      var menuBody = document.createElement('div');
      menuBody.className = 'learn-generation-library-menu-body';
      var confirm = null;
      if (row.status === 'archived') {
        menuBody.append(button('Restore', function () { archive(row.id, false); }));
      } else if (['submitted','queued','running','synthesizing'].indexOf(String(row.status || '')) === -1) {
        confirm = document.createElement('div');
        confirm.className = 'learn-generation-library-archive-confirmation';
        confirm.hidden = true;
        confirm.setAttribute('role', 'group');
        confirm.setAttribute('aria-label', 'Archive this ' + singular + '?');
        var question = document.createElement('p');
        question.textContent = 'Archive this ' + singular + '?';
        var confirmActions = document.createElement('div');
        confirmActions.className = 'learn-actions';
        confirmActions.append(
          button('Archive', function () { archive(row.id, true); }, 'learn-destructive'),
          button('Cancel', function () { confirm.hidden = true; menu.open = true; summary.focus(); })
        );
        confirm.append(question, confirmActions);
        menuBody.append(button('Archive', function () { menu.open = false; confirm.hidden = false; }));
      } else {
        var pending = document.createElement('span');
        pending.className = 'learn-meta';
        pending.textContent = 'Archive is available when generation finishes.';
        menuBody.append(pending);
      }
      menu.append(summary, menuBody);
      card.append(preview, body, menu);
      if (confirm) card.append(confirm);
      return card;
    }
    function render() {
      var rows = visibleRows();
      var cards = rows.map(function (row) { return renderCard ? renderCard(row, api) : defaultCard(row); }).filter(Boolean);
      grid.replaceChildren.apply(grid, cards);
      empty.hidden = cards.length > 0;
      updateStatus(cards.length);
    }
    function upsert(row) {
      row = row && typeof row === 'object' ? row : {};
      var id = String(row.id || '').trim();
      var title = String(row.title || '').trim();
      if (!id || !title) return false;
      var rows = read();
      var existing = rows.find(function (item) { return item.id === id; }) || null;
      var next = Object.assign({}, existing || {}, row, {
        id:id,
        title:title,
        status:String(row.status || (existing && existing.status) || 'ready'),
        created_at:String((existing && existing.created_at) || row.created_at || nowIso()),
        updated_at:String(row.updated_at || nowIso())
      });
      rows = rows.filter(function (item) { return item.id !== id; });
      rows.unshift(next);
      var ok = write(rows);
      render();
      if (!ok) announce('Browser storage is unavailable. This ' + singular + ' receipt is kept only in this tab and will not survive reload.');
      return ok;
    }
    async function runRefresh() {
      if (!refreshHandler || refresh.disabled) return;
      refresh.disabled = true;
      refresh.dataset.loading = 'true';
      announce('Refreshing ' + plural + '…');
      try { await refreshHandler(api); render(); }
      catch (_) { announce('Unable to refresh ' + plural + ' right now.'); }
      finally { delete refresh.dataset.loading; refresh.disabled = false; }
    }
    function setRefreshEnabled(enabled, title) {
      refresh.disabled = !enabled;
      if (title) refresh.title = String(title);
      refresh.setAttribute('aria-disabled', enabled ? 'false' : 'true');
    }
    var api = {
      read:read,
      write:write,
      render:render,
      upsert:upsert,
      archive:function (id) { return archive(id, true); },
      restore:function (id) { return archive(id, false); },
      announce:announce,
      setRefreshEnabled:setRefreshEnabled,
      section:section
    };
    section._learnPrivateLibrary = api;
    function onStorage(event) {
      if (!storageKey || event.key !== storageKey) return;
      volatileRows = [];
      render();
    }
    window.addEventListener('storage', onStorage);
    onPageDispose(function () { window.removeEventListener('storage', onStorage); });
    filter.addEventListener('change', render);
    refresh.addEventListener('click', runRefresh);
    setRefreshEnabled(!!refreshHandler, refresh.title);
    render();
    return api;
  }

  function normalizePublicationCredit(value) {
    value = String(value || '');
    if (/[\u0000-\u001f\u007f]/.test(value)) throw new Error('Contributor credit must not contain control characters.');
    value = value.replace(/\s+/g, ' ').trim();
    if (value.length > 80) throw new Error('Contributor credit must be plain text of at most 80 characters.');
    return value;
  }

  function publicationCreditValue(root) {
    var input = root && root.querySelector ? root.querySelector('[data-publication-credit]') : null;
    return normalizePublicationCredit(input && input.value || '');
  }

  function restorePublicationCredit(root, value) {
    var input = root && root.querySelector ? root.querySelector('[data-publication-credit]') : null;
    if (!input) return false;
    try {
      input.value = normalizePublicationCredit(value || '');
      input.setCustomValidity('');
      return true;
    } catch (_) { return false; }
  }

  function bindPublicationCredit(root, options) {
    options = options || {};
    var input = root && root.querySelector ? root.querySelector('[data-publication-credit]') : null;
    if (!input) return null;
    if (input._learnPublicationCredit) return input._learnPublicationCredit;
    var storageKey = String(options.storageKey || '').trim();

    function persist() {
      try {
        var value = normalizePublicationCredit(input.value || '');
        input.setCustomValidity('');
        if (storageKey) {
          if (value) localStorage.setItem(storageKey, value);
          else localStorage.removeItem(storageKey);
        }
        return value;
      } catch (error) {
        input.setCustomValidity(String(error && error.message || 'Invalid contributor credit.'));
        return null;
      }
    }
    if (storageKey) {
      try {
        var saved = localStorage.getItem(storageKey);
        if (saved && saved.length <= 80) restorePublicationCredit(root, saved);
      } catch (_) {}
    }
    input.addEventListener('input', persist);
    input.addEventListener('change', persist);
    var binding = {input:input, persist:persist, value:function () { return publicationCreditValue(root); }};
    input._learnPublicationCredit = binding;
    return binding;
  }

  function publicationContributor(root) {
    var value = publicationCreditValue(root);
    return {display_name:value || 'Anonymous'};
  }

  function publicationEndpoint() {
    try {
      var api = window.AI_ASSISTANT_ENDPOINT_API;
      if (api && typeof api.resolveEndpoint === 'function') {
        return String(api.resolveEndpoint('publication') || '').trim().replace(/\/+$/, '');
      }
    } catch (_) {}
    return '';
  }

  function boundedInteger(value, fallback, minimum, maximum) {
    var number = Number(value);
    if (!Number.isFinite(number) || !Number.isInteger(number)) number = fallback;
    return Math.max(minimum, Math.min(maximum, number));
  }

  async function readBoundedResponse(response, maxBytes, label) {
    maxBytes = boundedInteger(maxBytes, 256 * 1024, 1, 128 * 1024 * 1024);
    label = String(label || 'Runtime response');
    var declared = Number(response && response.headers && response.headers.get ? response.headers.get('Content-Length') : 0);
    if (Number.isFinite(declared) && declared > maxBytes) throw new Error(label + ' exceeded the ' + maxBytes.toLocaleString() + '-byte response limit.');
    var body = response && response.body;
    if (!body || typeof body.getReader !== 'function') {
      var fallback = new Uint8Array(await response.arrayBuffer());
      if (fallback.byteLength > maxBytes) throw new Error(label + ' exceeded the ' + maxBytes.toLocaleString() + '-byte response limit.');
      return [fallback];
    }
    var reader = body.getReader(), chunks = [], total = 0;
    try {
      while (true) {
        var part = await reader.read();
        if (part.done) break;
        var chunk = part.value instanceof Uint8Array ? part.value : new Uint8Array(part.value || []);
        total += chunk.byteLength;
        if (total > maxBytes) {
          try { await reader.cancel(); } catch (_) {}
          throw new Error(label + ' exceeded the ' + maxBytes.toLocaleString() + '-byte response limit.');
        }
        chunks.push(chunk);
      }
    } finally {
      try { reader.releaseLock(); } catch (_) {}
    }
    return chunks;
  }

  function decodeChunks(chunks) {
    var decoder = new TextDecoder('utf-8'), text = '';
    chunks.forEach(function (chunk) { text += decoder.decode(chunk, {stream:true}); });
    return text + decoder.decode();
  }

  async function runtimeFetch(url, init, options, reader) {
    var requestUrl;
    try {
      requestUrl = new URL(String(url || '').trim(), window.location.href);
      if (!/^https?:$/.test(requestUrl.protocol) || requestUrl.username || requestUrl.password || requestUrl.hash) throw new Error();
      var host = String(requestUrl.hostname || '').toLowerCase();
      var loopback = host === 'localhost' || host === '127.0.0.1' || host === '::1' || host === '[::1]';
      if (requestUrl.protocol === 'http:' && !loopback) throw new Error();
    } catch (_) { throw new Error('Runtime request URL must use HTTPS (or loopback HTTP for local development) without credentials or fragments.'); }
    init = init && typeof init === 'object' ? init : {};
    options = options && typeof options === 'object' ? options : {};
    var label = String(options.label || 'Runtime request');
    var timeoutMs = boundedInteger(options.timeoutMs, 30000, 1000, 180000);
    var externalSignal = init.signal || null;
    if (typeof AbortController !== 'function') throw new Error('This browser does not support bounded runtime requests.');
    var controller = new AbortController();
    var timedOut = false, abortListener = null;
    if (controller && externalSignal) {
      if (externalSignal.aborted) controller.abort();
      else { abortListener = function () { controller.abort(); }; externalSignal.addEventListener('abort', abortListener, {once:true}); }
    }
    var timer = controller ? window.setTimeout(function () { timedOut = true; controller.abort(); }, timeoutMs) : 0;
    var requestInit = Object.assign({}, init, {credentials:'omit', cache:'no-store', redirect:'error', referrerPolicy:'no-referrer'});
    if (controller) requestInit.signal = controller.signal;
    try {
      var response = await fetch(requestUrl.href, requestInit);
      var body = await reader(response, options);
      return {response:response, body:body};
    } catch (error) {
      if (timedOut) throw new Error(label + ' timed out after ' + Math.round(timeoutMs / 1000) + ' seconds.');
      throw error;
    } finally {
      if (timer) window.clearTimeout(timer);
      if (externalSignal && abortListener) { try { externalSignal.removeEventListener('abort', abortListener); } catch (_) {} }
    }
  }

  async function fetchJson(url, init, options) {
    return runtimeFetch(url, init, options, async function (response, settings) {
      var chunks = await readBoundedResponse(response, settings.maxBytes || 256 * 1024, settings.label || 'Runtime JSON response');
      var raw = decodeChunks(chunks).trim();
      if (!raw) return {};
      try {
        var value = JSON.parse(raw);
        if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error();
        return value;
      } catch (_) {
        if (!response.ok) return {};
        throw new Error(String(settings.label || 'Runtime') + ' returned invalid JSON.');
      }
    });
  }

  function normalizedMime(value) {
    return String(value || '').split(';', 1)[0].trim().toLowerCase();
  }

  function mimeMatches(expected, actual) {
    expected = normalizedMime(expected);
    actual = normalizedMime(actual);
    if (!expected) return true;
    if (!actual) return false;
    if (expected.endsWith('/*')) return actual.startsWith(expected.slice(0, -1));
    return actual === expected;
  }

  async function fetchBlob(url, init, options) {
    return runtimeFetch(url, init, options, async function (response, settings) {
      var label = String(settings.label || 'Runtime artifact');
      var expectedMime = normalizedMime(settings.mimeType);
      var actualMime = normalizedMime(response.headers.get('Content-Type'));
      if (response.ok && expectedMime && !mimeMatches(expectedMime, actualMime)) {
        throw new Error(label + ' returned an unexpected content type' + (actualMime ? ': ' + actualMime : '.'));
      }
      var chunks = await readBoundedResponse(response, settings.maxBytes || 64 * 1024 * 1024, label);
      return new Blob(chunks, {type:actualMime || expectedMime || 'application/octet-stream'});
    });
  }

  function safeWorkflowUrl(value) {
    try {
      var url = new URL(String(value || '').trim());
      if (url.protocol !== 'https:' || url.username || url.password || url.hash || (url.port && url.port !== '443')) return '';
      if (url.hostname !== 'github.com') return '';
      if (!/^\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\/actions\/runs\/[0-9]+(?:\/.*)?$/.test(url.pathname)) return '';
      return url.href;
    } catch (_) { return ''; }
  }

  function appendPublicationReceiptLink(container, receipt) {
    if (!container) return false;
    container.hidden = true;
    container.replaceChildren();
    var href = safeWorkflowUrl(receipt && receipt.workflow_url);
    if (!href) return false;
    var link = document.createElement('a');
    link.href = href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = 'Review GitHub Actions run';
    container.appendChild(link);
    container.hidden = false;
    return true;
  }

  async function submitPublication(request, options) {
    options = options || {};
    var endpoint = publicationEndpoint();
    if (!endpoint) throw new Error('AI Learn publication transport is not configured for the active endpoint profile.');
    var raw;
    try { raw = JSON.stringify(request); }
    catch (_) { throw new Error('Publication request could not be encoded.'); }
    if (!raw || raw.length > 60000) throw new Error('Publication request is too large for the reviewed handoff.');
    var result = await fetchJson(endpoint, {
      method:'POST',
      headers:{'Content-Type':'application/json','Accept':'application/json'},
      body:raw
    }, {
      label:'Publication service',
      timeoutMs:Number(options.timeoutMs || 30000),
      maxBytes:65536
    });
    var response = result.response, doc = result.body || {};
    if (!response.ok) {
      var detail = doc && (doc.detail || doc.message);
      throw new Error(String(detail || ('Publication service returned HTTP ' + response.status + '.')));
    }
    if (!doc || doc.contract !== 'learn.publication-receipt.v1') throw new Error('Publication service returned an unexpected receipt.');
    if (doc.workflow_url) {
      var workflowUrl = safeWorkflowUrl(doc.workflow_url);
      if (workflowUrl) doc.workflow_url = workflowUrl;
      else delete doc.workflow_url;
    }
    return doc;
  }

  function publicationReceiptMessage(receipt) {
    if (!receipt || typeof receipt !== 'object') return 'Publication request completed.';
    if (receipt.mode === 'stub' || receipt.state === 'simulated') return 'Publication validated in stub mode. No GitHub pull request was created.';
    if (receipt.workflow_url) return 'Publication queued for repository validation. Review the GitHub Actions run before the pull request is merged.';
    return String(receipt.message || 'Publication queued for repository validation.');
  }

  window.AI_LEARN_GENERATION_UI = Object.assign({}, window.AI_LEARN_GENERATION_UI || {}, {
    cleanPublicReferenceUrl: cleanPublicReferenceUrl,
    runtimeBase: runtimeBase,
    setReadiness: setReadiness,
    setFlowStage: setFlowStage,
    bindGenerationStatus: bindGenerationStatus,
    bindRequestActions: bindRequestActions,
    bindPrivateLibrary: bindPrivateLibrary,
    readLensProfile: readLensProfile,
    restoreLensProfile: restoreLensProfile,
    validateLensProfile: validateLensProfile,
    lensGuidance: lensGuidance,
    withLensGuidance: withLensGuidance,
    assistantModelState: assistantModelState,
    assistantModelSnapshot: assistantModelSnapshot,
    publicationEndpoint: publicationEndpoint,
    fetchJson: fetchJson,
    fetchBlob: fetchBlob,
    onPageDispose: onPageDispose,
    safeWorkflowUrl: safeWorkflowUrl,
    appendPublicationReceiptLink: appendPublicationReceiptLink,
    normalizePublicationCredit: normalizePublicationCredit,
    publicationCreditValue: publicationCreditValue,
    restorePublicationCredit: restorePublicationCredit,
    bindPublicationCredit: bindPublicationCredit,
    publicationContributor: publicationContributor,
    submitPublication: submitPublication,
    publicationReceiptMessage: publicationReceiptMessage
  });

  var assistantAuthorityBindings = [];
  var assistantAuthorityState = null;
  var assistantAuthorityUnsubscribe = null;

  function assistantAuthorityApi() { return window.AI_ASSISTANT_MODEL_API || null; }
  function assistantAuthorityModels() {
    var current = assistantAuthorityApi();
    if (current && typeof current.listModels === 'function') return current.listModels() || [];
    var cfg = window.AI_ASSISTANT_CONFIG || {};
    return Array.isArray(cfg.panelApiModels) ? cfg.panelApiModels : [];
  }

  function connectAssistantAuthorityPicker(picker) {
    if (!picker || picker.dataset.generationAuthorityBound === 'true') return;
    if (picker.getAttribute('data-generation-authority-kind') !== 'assistant' || picker.getAttribute('data-generation-authority-managed') !== 'shared') return;
    var open = picker.querySelector('[data-generation-authority-open]');
    var more = picker.querySelector('[data-generation-authority-more]');
    var menu = picker.querySelector('[data-generation-authority-menu]');
    var label = picker.querySelector('[data-generation-authority-label]');
    var badge = picker.querySelector('[data-generation-authority-badge]');
    var container = picker.closest('.learn-generation-authority') || picker.parentElement;
    var status = container && container.querySelector('[data-generation-authority-status]');
    if (!open || !more || !menu || !label || !badge) return;
    picker.dataset.generationAuthorityBound = 'true';
    var state = assistantAuthorityState;

    var authorityContext = status ? String(status.getAttribute('data-generation-authority-context') || '').trim() : '';
    function announce(message) {
      if (!status) return;
      status.textContent = [String(message || '').trim(), authorityContext].filter(Boolean).join(' ');
    }
    function api() { return assistantAuthorityApi(); }
    function models() { return assistantAuthorityModels(); }
    function activeRow() {
      if (state && state.active) return state.active;
      var rows = models();
      return rows.length ? rows[0] : null;
    }
    function closeMenu(focus) {
      menu.hidden = true;
      more.setAttribute('aria-expanded', 'false');
      picker.classList.remove('is-open');
      if (focus) more.focus();
    }
    function renderMenu() {
      var rows = models();
      var active = activeRow();
      menu.replaceChildren();
      if (!rows.length) {
        var empty = document.createElement('p');
        empty.className = 'learn-meta';
        empty.textContent = 'No selectable Assistant models are available.';
        menu.append(empty);
        return;
      }
      rows.forEach(function (row) {
        var id = String(row && (row.id || row.model) || '');
        if (!id) return;
        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'learn-generation-authority-menu-item';
        button.setAttribute('role', 'menuitem');
        button.dataset.modelId = id;
        var marker = document.createElement('span');
        marker.className = 'learn-generation-authority-menu-check';
        marker.setAttribute('aria-hidden', 'true');
        marker.textContent = active && String(active.id || active.model || '') === id ? '✓' : '';
        var body = document.createElement('span');
        body.className = 'learn-generation-authority-menu-text';
        var title = document.createElement('span');
        title.className = 'learn-generation-authority-menu-label';
        title.textContent = String(row.label || row.model || row.id || id);
        var hint = document.createElement('span');
        hint.className = 'learn-generation-authority-menu-hint';
        hint.textContent = marker.textContent ? 'Current model' : String(row.provider || 'custom');
        body.append(title, hint);
        button.append(marker, body);
        button.addEventListener('click', function () {
          var current = api();
          if (current && typeof current.selectModel === 'function' && current.selectModel(id)) {
            closeMenu(true);
            syncAssistantAuthorities(typeof current.getState === 'function' ? current.getState() : null);
            announce('Model changed to ' + title.textContent + '. This selection is shared with AI Assistant.');
          } else {
            announce('That model is no longer selectable. Refresh the model list or open Model Configuration.');
          }
        });
        menu.append(button);
      });
    }
    function render() {
      var active = activeRow();
      var effort = state && state.effort;
      var name = active ? String(active.label || active.model || active.id || 'Runtime default') : 'Runtime default';
      var effortLabel = effort ? String(effort.label || effort.id || 'Default') : 'Default';
      label.textContent = name;
      badge.textContent = effortLabel;
      badge.dataset.effort = effort && effort.id ? String(effort.id) : 'default';
      open.dataset.modelText = name;
      open.setAttribute('aria-label', 'Model Configuration — current: ' + name + ', effort: ' + effortLabel);
      open.title = name;
      renderMenu();
    }
    function sync(nextState) { state = nextState || null; render(); }
    function connect() {
      var current = api();
      state = assistantAuthorityState;
      if (!current) {
        render();
        open.disabled = true;
        more.disabled = true;
        announce(models().length ? 'Using the first configured AI Assistant model; Model Configuration is unavailable on this page.' : 'AI Assistant model discovery is unavailable on this page; runtime model provenance cannot be changed here.');
        return;
      }
      open.disabled = typeof current.openPicker !== 'function';
      more.disabled = !(typeof current.selectModel === 'function' && models().length);
      render();
      announce(open.disabled ? 'Model selection is synchronized with AI Assistant.' : 'Model selection is synchronized with AI Assistant. Open Model Configuration or use the quick model menu to change it.');
    }

    open.addEventListener('click', function () {
      var current = api();
      if (current && typeof current.openPicker === 'function' && current.openPicker(open)) {
        closeMenu(false);
        announce('AI Assistant Model Configuration opened. Changes remain synchronized with this generation request.');
      } else if (!more.disabled) {
        renderMenu();
        menu.hidden = false;
        more.setAttribute('aria-expanded', 'true');
        picker.classList.add('is-open');
        var first = menu.querySelector('button');
        if (first) first.focus();
      }
    });
    more.addEventListener('click', function (event) {
      event.stopPropagation();
      if (menu.hidden === false) { closeMenu(true); return; }
      renderMenu();
      menu.hidden = false;
      more.setAttribute('aria-expanded', 'true');
      picker.classList.add('is-open');
      var first = menu.querySelector('button');
      if (first) first.focus();
    });
    menu.addEventListener('keydown', function (event) {
      var items = Array.prototype.slice.call(menu.querySelectorAll('button:not([disabled])'));
      if (!items.length) return;
      var index = items.indexOf(document.activeElement);
      if (event.key === 'Escape') { event.preventDefault(); closeMenu(true); return; }
      if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp' && event.key !== 'Home' && event.key !== 'End') return;
      event.preventDefault();
      if (event.key === 'Home') index = 0;
      else if (event.key === 'End') index = items.length - 1;
      else if (event.key === 'ArrowDown') index = (index + 1 + items.length) % items.length;
      else index = (index - 1 + items.length) % items.length;
      items[index].focus();
    });
    document.addEventListener('click', function (event) {
      if (!picker.contains(event.target)) closeMenu(false);
    });
    var binding = {picker:picker, sync:sync, connect:connect};
    assistantAuthorityBindings.push(binding);
    connect();
  }

  function syncAssistantAuthorities(nextState) {
    assistantAuthorityState = nextState || assistantModelState();
    assistantAuthorityBindings.forEach(function (binding) { binding.sync(assistantAuthorityState); });
  }

  function connectAssistantAuthorityApi() {
    if (typeof assistantAuthorityUnsubscribe === 'function') assistantAuthorityUnsubscribe();
    assistantAuthorityUnsubscribe = null;
    var current = assistantAuthorityApi();
    assistantAuthorityState = current && typeof current.getState === 'function' ? current.getState() : assistantModelState();
    assistantAuthorityBindings.forEach(function (binding) { binding.connect(); binding.sync(assistantAuthorityState); });
    if (current && typeof current.onChange === 'function') assistantAuthorityUnsubscribe = current.onChange(syncAssistantAuthorities);
  }

  function bindAssistantAuthorities(root) {
    if (!root || !root.querySelectorAll) return;
    root.querySelectorAll('[data-generation-authority-picker][data-generation-authority-kind="assistant"][data-generation-authority-managed="shared"]').forEach(connectAssistantAuthorityPicker);
    connectAssistantAuthorityApi();
  }

  window.AI_LEARN_GENERATION_UI.bindAssistantAuthorities = bindAssistantAuthorities;

  roots.forEach(function (root) {
    root.querySelectorAll('[data-generation-chip]').forEach(function (button) {
      button.addEventListener('click', function () {
        var selector = String(button.getAttribute('data-generation-target') || '').trim();
        var field = selector ? root.querySelector(selector) : null;
        if (!field || !('value' in field)) return;
        appendInstruction(field, button.getAttribute('data-generation-chip'));
      });
    });

    root.querySelectorAll('[data-generation-countable]').forEach(function (field) {
      updateCounter(field);
      field.addEventListener('input', function () { updateCounter(field); });
    });

  });

  bindAssistantAuthorities(document);
  window.addEventListener('ai-assistant-model-api-ready', connectAssistantAuthorityApi);
  onPageDispose(function () {
    if (typeof assistantAuthorityUnsubscribe === 'function') assistantAuthorityUnsubscribe();
    assistantAuthorityUnsubscribe = null;
    window.removeEventListener('ai-assistant-model-api-ready', connectAssistantAuthorityApi);
  });
}());
