/* Run with Node, Playwright, and a built examples directory.
 * LEARN_HTML=/absolute/build PLAYWRIGHT_MODULE=/path/to/playwright
 * Optional CHROMIUM_EXECUTABLE and CHROMIUM_ARGS (JSON array).
 * Provider handoff is a mock contract test, not a live AI integration.
 */
const {test, before, after} = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
let server, browser, origin;
const html = path.resolve(process.env.LEARN_HTML);
before(async () => {
    server = http.createServer((req, res) => {
        const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
        const target = path.resolve(html, '.' + pathname);
        if (!target.startsWith(html + path.sep) || !fs.existsSync(target) || !fs.statSync(target).isFile()) {
            res.writeHead(404); res.end(); return;
        }
        const ext = path.extname(target);
        res.setHeader('Content-Type', {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json'}[ext] || 'application/octet-stream');
        res.end(fs.readFileSync(target));
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    origin = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE,
        args: JSON.parse(process.env.CHROMIUM_ARGS || '[]')});
});
after(async () => { if (browser) await browser.close(); if (server) await new Promise(resolve => server.close(resolve)); });
async function pageFor(t, options={}) {
    const context = await browser.newContext(options);
    t.after(() => context.close());
    const page = await context.newPage();
    page.setDefaultTimeout(8000);
    page.failures = [];
    page.on('pageerror', error => page.failures.push(error.message));
    t.after(() => assert.deepEqual(page.failures, []));
    await page.goto(origin + '/hub.html');
    if (options.javaScriptEnabled !== false) await page.locator('.la-app').waitFor();
    return page;
}
async function newDraft(page, title='A generic topic') {
    await page.getByRole('button', {name:'Create topic', exact:true}).click();
    await page.getByLabel('Subject title', {exact:true}).fill(title);
    await page.getByLabel('Short description', {exact:true}).fill('A reader-owned explanation.');
    await page.getByLabel('Explanation (plain text)', {exact:true}).fill('A concrete observation.');
}
async function mockBridge(page) {
    await page.evaluate(() => {
        const root = document.querySelector('[data-skplt-learn-ai-mount]');
        const data = root.querySelector('.la-data');
        const config = JSON.parse(data.textContent);
        config.runtime = 'assistant'; data.textContent = JSON.stringify(config);
        window.AI_ASSISTANT = {tasks:{contract:'assistant.tasks.v1',run(request, {signal}) {
            window.taskRequest = request; window.taskSignal = signal;
            return new Promise(resolve => { window.finishTask = body => resolve({contract:'assistant.task-result.v1',request_id:request.request_id,selected:true,body}); });
        }}};
        window.SPHINX_AI_LEARN.destroy(root); window.SPHINX_AI_LEARN.mountAll();
    });
}

test('static fallback and real RST navigation', async t => {
    const page = await pageFor(t, {javaScriptEnabled:false});
    assert.match(await page.locator('.la-static').innerText(), /Understanding probability calibration/);
    assert.equal(await page.locator('.la-app').count(), 0);
    await page.getByRole('link', {name:'Check a reference before reusing a claim', exact:true}).first().click();
    assert.match(page.url(), /\/skills\/20260916T000000Z-[a-f0-9]{16}\.html/);
});

test('search sources and all kinds; browse does not save', async t => {
    const page = await pageFor(t);
    assert.equal(await page.evaluate(() => Object.keys(localStorage).filter(k => k.startsWith('skplt-learn-ai:')).length), 0);
    await page.getByRole('button', {name:'All', exact:true}).click();
    await page.getByRole('searchbox', {name:'Search learning content'}).fill('research-methods');
    assert.equal(await page.locator('.la-card').count(), 1);
    await page.locator('.la-card a').click();
    assert.match(page.url(), /\/skills\//);
    assert.match(await page.locator('.la-content').innerText(), /Instructions/);
});

test('custom draft survives reload and preserves plain text', async t => {
    const page = await pageFor(t);
    await newDraft(page, '<img src=x onerror=window.INJECTED=true>');
    await page.getByLabel('Section', {exact:true}).selectOption('custom');
    await page.getByLabel('Section title', {exact:true}).fill('My observation');
    await page.getByLabel('Authorship', {exact:true}).selectOption('ai-assisted');
    await page.getByRole('button', {name:'Save on this browser', exact:true}).click();
    await page.reload();
    await page.getByRole('button', {name:'My drafts', exact:true}).click();
    assert.equal(await page.locator('.la-card').count(), 1);
    assert.equal(await page.locator('.la-card img').count(), 0);
    assert.equal(await page.evaluate(() => window.INJECTED), undefined);
    await page.getByRole('button', {name:'Continue editing', exact:true}).click();
    assert.equal(await page.getByLabel('Section title', {exact:true}).inputValue(), 'My observation');
    assert.equal(await page.getByLabel('Authorship', {exact:true}).inputValue(), 'ai-assisted');
    await page.getByLabel('Subject title', {exact:true}).fill('Updated generic topic');
    await page.getByRole('button', {name:'Save on this browser', exact:true}).click();
    assert.match(await page.locator('.la-status').innerText(), /Saved on this browser/);
});

test('newer storage revision prevents overwrite', async t => {
    const page = await pageFor(t);
    await newDraft(page);
    await page.getByRole('button', {name:'Save on this browser', exact:true}).click();
    await page.evaluate(() => {
        const key = Object.keys(localStorage).find(k=>k.startsWith('skplt-learn-ai:'));
        const row = JSON.parse(localStorage.getItem(key)); row.version = 'newer';
        localStorage.setItem(key, JSON.stringify(row));
    });
    await page.getByLabel('Explanation (plain text)', {exact:true}).fill('My competing change');
    await page.getByRole('button', {name:'Save on this browser', exact:true}).click();
    assert.match(await page.locator('.la-status').innerText(), /changed in another tab/);
});

test('selected AI result records context and provenance; edits are preserved', async t => {
    const page = await pageFor(t);
    await mockBridge(page);
    await newDraft(page);
    await page.getByRole('button', {name:'Generate with assistant', exact:true}).click();
    await page.evaluate(() => window.finishTask('Selected AI explanation'));
    await page.waitForFunction(() => document.querySelector('.la-draft-body').value === 'Selected AI explanation');
    assert.match(await page.locator('.la-activity').innerText(), /ai result selected/);
    await page.getByRole('button', {name:'Save on this browser', exact:true}).click();
    const record = await page.evaluate(() => JSON.parse(localStorage.getItem(Object.keys(localStorage).find(k=>k.startsWith('skplt-learn-ai:')))).draft);
    assert.equal(record.authorship, 'ai-assisted');
    assert.equal(record.interactions[0].section_id, record.section.id);
    assert.equal(record.interactions[0].workflow_id, 'learn.explanation.v1');
    await page.getByRole('button', {name:'Generate with assistant', exact:true}).click();
    await page.getByLabel('Explanation (plain text)', {exact:true}).fill('New human edit');
    await page.evaluate(() => window.finishTask('Should not overwrite'));
    await page.waitForFunction(() => document.querySelector('.la-status').textContent.includes('edits were kept'));
    assert.equal(await page.getByLabel('Explanation (plain text)', {exact:true}).inputValue(), 'New human edit');
});

test('cancelled task ignores stale result and permits another request', async t => {
    const page = await pageFor(t);
    await mockBridge(page); await newDraft(page);
    await page.getByRole('button', {name:'Generate with assistant', exact:true}).click();
    await page.getByRole('button', {name:'Sources', exact:true}).click();
    assert.equal(await page.evaluate(() => window.taskSignal.aborted), true);
    await page.evaluate(() => window.finishTask('Stale result'));
    assert.equal(await page.getByLabel('Explanation (plain text)', {exact:true}).inputValue(), 'A concrete observation.');
    assert.equal(await page.getByRole('button', {name:'Generate with assistant', exact:true}).isEnabled(), true);
    await page.evaluate(() => window.SPHINX_AI_LEARN.mountAll());
    assert.equal(await page.locator('.la-app').count(), 1);
});

test('media requests require explicit load and use fixed YouTube host', async t => {
    const page = await pageFor(t);
    const external = [];
    await page.route('https://**/*', route => {external.push(route.request().url()); return route.abort();});
    await page.evaluate(() => {
        const root=document.querySelector('[data-skplt-learn-ai-mount]');
        const data=root.querySelector('.la-data'), config=JSON.parse(data.textContent);
        config.catalog.subjects.push({id:'video-test',kind:'video',title:'Test video',summary:'Test only',domains:[],related:[],sections:[],url:'https://youtu.be/abcdefghijk'});
        config.initial_subject='video-test'; data.textContent=JSON.stringify(config);
        history.replaceState({}, '', location.pathname);
        window.SPHINX_AI_LEARN.destroy(root); window.SPHINX_AI_LEARN.mountAll();
    });
    assert.equal(await page.locator('iframe.la-media').count(), 0);
    assert.deepEqual(external, []);
    await page.getByRole('button', {name:'Load YouTube player', exact:true}).click();
    assert.equal(await page.locator('iframe.la-media').getAttribute('src'), 'https://www.youtube-nocookie.com/embed/abcdefghijk');
    await page.waitForFunction(() => !!document.querySelector('iframe.la-media'));
});

test('mobile layout fits and destroy restores readable fallback', async t => {
    const page = await pageFor(t, {viewport:{width:390,height:844}});
    if (process.env.LEARN_SCREENSHOT) await page.screenshot({path:process.env.LEARN_SCREENSHOT,fullPage:true});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true, JSON.stringify(await page.evaluate(() => Array.from(document.querySelectorAll('body *')).filter(e=>e.getBoundingClientRect().right>innerWidth+1).slice(0,10).map(e=>({tag:e.tagName,class:e.className,right:e.getBoundingClientRect().right})))));
    await newDraft(page);
    if (process.env.LEARN_SCREENSHOT) await page.screenshot({path:process.env.LEARN_SCREENSHOT,fullPage:true});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true, JSON.stringify(await page.evaluate(() => Array.from(document.querySelectorAll('body *')).filter(e=>e.getBoundingClientRect().right>innerWidth+1).slice(0,10).map(e=>({tag:e.tagName,class:e.className,right:e.getBoundingClientRect().right})))));
    if (process.env.LEARN_SCREENSHOT) await page.screenshot({path:process.env.LEARN_SCREENSHOT,fullPage:true});
    await page.evaluate(() => window.SPHINX_AI_LEARN.destroy(document.querySelector('[data-skplt-learn-ai-mount]')));
    assert.equal(await page.locator('.la-app').count(), 0);
    assert.equal(await page.locator('.la-static').isVisible(), true);
});
