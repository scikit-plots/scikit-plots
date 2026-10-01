(function () {
  'use strict';

  var profiles = document.querySelectorAll('[data-ai-lens-profile]');
  if (!profiles.length) return;

  function checkboxLabel(input) {
    var label = input.closest('label');
    return label ? String(label.textContent || '').replace(/\s+/g, ' ').trim() : '';
  }

  function plural(count, singular, pluralValue) {
    return count === 1 ? singular : pluralValue;
  }

  function updateGroup(fieldset) {
    var summary = fieldset.querySelector('[data-ai-choice-summary]');
    var countNode = fieldset.querySelector('[data-ai-choice-count]');
    var previewNode = fieldset.querySelector('[data-ai-choice-preview]');
    var selected = Array.prototype.slice.call(fieldset.querySelectorAll('.learn-ai-choice-grid input[type="checkbox"]:checked'));
    var labels = selected.map(checkboxLabel).filter(Boolean);
    var count = labels.length;
    var required = fieldset.getAttribute('data-ai-lens-required') === 'true';
    var state = count ? 'selected' : (required ? 'required-empty' : 'empty');

    fieldset.dataset.selectionCount = String(count);
    fieldset.dataset.selectionState = state;
    if (summary) {
      summary.dataset.state = state;
      summary.setAttribute('aria-label', count
        ? count + ' selected: ' + labels.join(', ')
        : (required ? '0 selected. Choose at least one.' : '0 selected. None.'));
    }
    if (countNode) countNode.textContent = count + ' selected';
    if (previewNode) {
      if (!count) {
        previewNode.textContent = required ? 'Choose at least one' : 'None';
        previewNode.removeAttribute('title');
      } else {
        var visible = labels.slice(0, 2);
        var remaining = labels.length - visible.length;
        previewNode.textContent = visible.join(' · ') + (remaining > 0 ? ' · +' + remaining + ' more' : '');
        previewNode.title = labels.join(', ');
      }
    }
    return count;
  }

  function updateProfile(profile) {
    var groups = Array.prototype.slice.call(profile.querySelectorAll('[data-ai-lens-group]'));
    var total = 0;
    var activeGroups = 0;
    var missingRequired = 0;
    groups.forEach(function (group) {
      var count = updateGroup(group);
      total += count;
      if (count) activeGroups += 1;
      if (!count && group.getAttribute('data-ai-lens-required') === 'true') missingRequired += 1;
    });
    var summary = profile.querySelector('[data-ai-lens-profile-summary]');
    var countNode = profile.querySelector('[data-ai-lens-profile-count]');
    var detailNode = profile.querySelector('[data-ai-lens-profile-detail]');
    if (summary) summary.dataset.state = missingRequired ? 'attention' : 'ready';
    if (countNode) countNode.textContent = total + ' ' + plural(total, 'selection', 'selections');
    if (detailNode) {
      detailNode.textContent = missingRequired
        ? missingRequired + ' required ' + plural(missingRequired, 'group needs', 'groups need') + ' a choice'
        : 'Across ' + activeGroups + ' ' + plural(activeGroups, 'lens group', 'lens groups');
    }
  }

  profiles.forEach(function (profile) {
    if (profile.dataset.aiLensSummaryBound === 'true') return;
    profile.dataset.aiLensSummaryBound = 'true';
    profile.addEventListener('change', function (event) {
      if (event.target && event.target.matches('.learn-ai-choice-grid input[type="checkbox"]')) updateProfile(profile);
    });
    updateProfile(profile);
  });
}());
