// R173T92 — collapsible answer sections must look and read like disclosures.
//
// Usage:
//   node test_ai_assistant__section_disclosure_discoverability.mjs ai-assistant.js ai-assistant.css
//
// This is intentionally a source contract. Native <details>/<summary> owns the
// expanded state in the browser; the regression risk here is that presentation
// drifts back to a plain heading whose interactivity is only visible on hover.
import fs from 'node:fs';

const js = fs.readFileSync(process.argv[2], 'utf8');
const css = fs.readFileSync(process.argv[3], 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
let pass = 0, fail = 0;
function t(name, actual, expected) {
  if (actual === expected) { pass++; }
  else { fail++; console.error(`FAIL ${name}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`); }
}

// DOM: title + redundant visual action hint, while native details remains the
// accessibility/state authority.
t('section title gets a dedicated class', js.includes("titleSpan.className = 'ai-md-section-title';"), true);
t('disclosure gets a visual action hint', js.includes("actionHint.className = 'ai-md-section-action-hint';"), true);
t('visual action hint is appended to the summary', js.includes('summary.appendChild(actionHint);'), true);
t('action hint is hidden from accessibility name', js.includes("actionHint.setAttribute('aria-hidden', 'true');"), true);
t('open-state action says Hide section', js.includes("hideHint.textContent = 'Hide section';"), true);
t('closed-state action says Show section', js.includes("showHint.textContent = 'Show section';"), true);
t('native details still defaults open', /details\.open\s*=\s*true;/.test(js), true);

// Persistent interactive surface: discoverability cannot depend on hover.
const summary = css.match(/\.ai-assistant-panel-bubble--assistant summary\.ai-md-section-summary \{([^}]*)\}/)?.[1] || '';
t('summary has persistent background', /background:\s*var\(--ai-section-summary-bg\)/.test(summary), true);
t('summary has persistent border', /border:\s*1px solid var\(--ai-section-summary-border\)/.test(summary), true);
t('summary has a minimum touch/readable height', /min-height:\s*2\.35rem/.test(summary), true);
t('summary uses explicit line-height', /line-height:\s*1\.35/.test(summary), true);

// Conventional disclosure direction: right when closed, down when open.
const chev = css.match(/\.ai-md-section-chevron \{([^}]*)\}/)?.[1] || '';
t('closed chevron points right via -90deg rotation', /transform:\s*rotate\(-90deg\)/.test(chev), true);
t('chevron has a persistent badge ground', /background:\s*var\(--ai-section-chevron-bg\)/.test(chev), true);
t('chevron has a persistent badge border', /border:\s*1px solid var\(--ai-section-chevron-border\)/.test(chev), true);
t('open chevron returns to down orientation', /details\.ai-md-section\[open\] > summary \.ai-md-section-chevron \{\s*transform:\s*rotate\(0deg\);/.test(css), true);

// State hint follows [open] in CSS; there is no second JS toggle state.
t('closed sections hide the Hide label', /details\.ai-md-section:not\(\[open\]\) > summary \.ai-md-section-action--hide[\s\S]{0,120}?display:\s*none/.test(css), true);
t('open sections hide the Show label', /details\.ai-md-section\[open\] > summary \.ai-md-section-action--show[\s\S]{0,120}?display:\s*none/.test(css), true);
t('action hint is styled as a compact pill', /\.ai-md-section-action-hint \{[\s\S]{0,500}?border-radius:\s*999px/.test(css), true);

// Theme-aware persistent surfaces; dark mode must not reuse a light literal.
t('light theme defines disclosure surface from theme tokens', /:root \{[\s\S]{0,900}?--ai-section-summary-bg:\s*color-mix\(in srgb, var\(--pst-color-surface/.test(css), true);
t('dark theme has its own disclosure surface mix', /\[data-bs-theme="dark"\][\s\S]{0,1000}?--ai-section-summary-bg:\s*color-mix\(in srgb, var\(--pst-color-surface, #1f1f1f\)/.test(css), true);
t('forced colors restores system button surface', /forced-colors: active[\s\S]{0,500}?summary\.ai-md-section-summary[\s\S]{0,180}?background:\s*ButtonFace/.test(css), true);
t('reduced motion disables disclosure transitions', /prefers-reduced-motion: reduce[\s\S]{0,240}?ai-md-section-summary[\s\S]{0,160}?transition:\s*none/.test(css), true);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
