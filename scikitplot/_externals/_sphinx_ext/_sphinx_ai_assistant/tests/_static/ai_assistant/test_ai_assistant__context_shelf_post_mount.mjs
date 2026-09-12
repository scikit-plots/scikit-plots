// Run 95 regression: persisted pinned-page cards must render on the first panel
// mount even when automatic current-page context is OFF. The renderer resolves
// its tray through document.getElementById(), therefore any pre-mount render is
// necessarily a no-op and must not be the only initial projection path.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed = 0, failed = 0;
function ok(cond, name) { if (cond) passed++; else { failed++; console.error('FAIL ' + name); } }
function extract(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('missing ' + name);
  let d = 0, started = false;
  for (let j = i; j < src.length; j++) {
    if (src[j] === '{') { d++; started = true; }
    else if (src[j] === '}') { d--; if (started && d === 0) return src.slice(i, j + 1); }
  }
  throw new Error('unbalanced ' + name);
}

const createPanel = extract('createAIPanel');
const render = extract('_renderComposerAttachments');
const refresh = extract('_refreshPinnedPageContextShelf');
const appendAt = createPanel.indexOf('document.body.appendChild(panel)');
const refreshAt = createPanel.indexOf('_refreshPinnedPageContextShelf()', appendAt);
const trayAt = createPanel.indexOf("attachmentTray.id = 'ai-assistant-panel-attachments'");
const preMountSegment = createPanel.slice(trayAt, appendAt);

ok(appendAt >= 0, 'panel has an explicit document mount point');
ok(refreshAt > appendAt, 'canonical pinned-shelf refresh runs after panel is connected');
ok(!createPanel.includes("inputGroup.appendChild(attachmentTray);\n        _renderComposerAttachments();"), 'detached tray is not synchronously rendered before panel mount');
ok(!preMountSegment.includes('_loadPinnedPageContexts(true)'), 'initial forced pin hydration is owned by the post-mount refresh');
ok(refresh.includes('_loadPinnedPageContexts(true)'), 'post-mount refresh force-hydrates persisted pinned pages');
ok(refresh.includes('_renderComposerAttachments()'), 'post-mount refresh projects hydrated pins into the visible shelf');
ok(render.includes("document.getElementById('ai-assistant-panel-attachments')"), 'renderer still resolves the mounted tray from document');
ok(createPanel.includes('if (_currentPageContextActive())'), 'automatic current page remains an independent optional source with per-page exclusion support');
ok(createPanel.indexOf('if (_currentPageContextActive())') < appendAt, 'current-page preparation can begin before mount without owning initial shelf visibility');

// Minimal runtime model of the exact bug. Before mount, document lookup cannot
// see the detached tray. After mount, the canonical refresh hydrates A and the
// first real render immediately projects it, without needing a current-page pin.
let mounted = false;
let persistenceReads = 0;
let rendered = [];
const persisted = [{ name: 'Page A', contextRole: 'pinned' }];
function fakeGetElementById(id) { return mounted && id === 'ai-assistant-panel-attachments' ? {} : null; }
function hydrate() { persistenceReads++; return persisted.slice(); }
function project(items) {
  const tray = fakeGetElementById('ai-assistant-panel-attachments');
  if (!tray) return false;
  rendered = items.slice();
  return true;
}
ok(project(hydrate()) === false, 'runtime: detached initial render cannot visualize persisted Page A');
ok(rendered.length === 0, 'runtime: pre-mount no-op leaves no visible card');
mounted = true;
const restored = hydrate();
ok(project(restored) === true, 'runtime: post-mount refresh can project persisted pins');
ok(rendered.length === 1 && rendered[0].name === 'Page A', 'runtime: Page A is visible immediately on Page B');
rendered.push({ name: 'Page B', contextRole: 'pinned' });
ok(rendered.map(x => x.name).join(',') === 'Page A,Page B', 'runtime: pinning current Page B adds to, rather than reveals, previous Page A');
ok(persistenceReads === 2, 'runtime model distinguishes failed detached projection from successful mounted refresh');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
