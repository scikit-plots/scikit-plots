import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, '..', '_static', 'ai-assistant.js'), 'utf8');

assert.match(source, /method:\s*'PUT'/, 'pending review updates must use PUT');
assert.ok(source.includes('ai-assistant-active-contribution-review-v1'));
assert.ok(source.includes('Update existing review'));
assert.ok(source.includes('Already in review · no changes'));
assert.ok(source.includes('No duplicate review was opened'));
assert.ok(source.includes('Changed reviewed content updates the same repository review'));
assert.ok(source.includes('identical content creates no new commit'));
assert.ok(source.includes('_forgetActiveContributionReview'));
assert.ok(source.includes("status === 409 || status === 404 || status === 410"));
assert.ok(source.includes('No new review was opened; retry keeps the same review identity.'));
assert.ok(source.includes('made identical resubmits look like new revisions'));

assert.ok(source.includes('Recover withdrawal access'));
assert.ok(source.includes('Copy private withdrawal code'));
assert.ok(source.includes('Copy support reference'));
assert.ok(source.includes('Copy maintainer removal request'));
assert.ok(source.includes('aicm2.'));
assert.ok(source.includes('Save private receipt'));
assert.ok(source.includes("if (activeReview && !result.firstChild)"));
assert.ok(source.includes('Never publish the private receipt or withdrawal code.'));
assert.ok(source.includes('This reference contains no withdrawal capability.'));
assert.ok(source.includes('reviewPath'));
assert.ok(source.includes('_managementReceiptMatchesEndpoint'));
assert.ok(source.includes('belongs to a different contribution service'));
console.log('22 passed, 0 failed');
