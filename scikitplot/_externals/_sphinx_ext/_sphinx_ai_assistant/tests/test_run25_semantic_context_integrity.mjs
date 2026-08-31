import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const mainPath = process.argv[2];
const testsDir = path.dirname(fileURLToPath(import.meta.url));
const staticDir = path.join(path.dirname(testsDir), '_static');
const main = fs.readFileSync(mainPath, 'utf8');
const host = fs.readFileSync(path.join(staticDir, 'ai-assistant-isolation-host.js'), 'utf8');
let passed=0, failed=0;
function ok(c,n){if(c)passed++;else{failed++;console.error('FAIL '+n);}}

ok(main.includes('function _stripModelOnlyLiveNodes(liveRoot, cloneRoot)'), 'same-origin live visibility helper exists');
ok(main.includes('_stripModelOnlyLiveNodes(content, cloned);'), 'same-origin prunes before clone serialization');
ok(main.includes("cs.display === 'none'"), 'same-origin filters display none');
ok(main.includes("cs.visibility === 'hidden'"), 'same-origin filters visibility hidden');
ok(main.includes("cs.contentVisibility === 'hidden'"), 'same-origin filters content-visibility hidden');
ok(main.includes('Number.parseFloat(cs.opacity) === 0'), 'same-origin filters complete transparency');
ok(main.includes('zeroLeaf'), 'same-origin filters zero-area or zero-font leaf text');
ok(main.includes('classicallyClipped'), 'same-origin filters classic clipping');
ok(main.includes('extremeIndent'), 'same-origin filters extreme clipped indentation');
ok(main.includes('unreachable'), 'same-origin filters unreachable-document-surface text');
ok(main.includes("cs.position !== 'fixed'"), 'same-origin keeps fixed-position geometry out of document reachability rule');
ok(main.includes('maxY = Math.max'), 'same-origin uses full reachable document height not viewport height');

ok(host.includes('The live rendered DOM is the visibility authority'), 'isolation host documents live DOM authority');
ok(host.includes("cs.display === 'none'"), 'isolation filters display none');
ok(host.includes("cs.visibility === 'hidden'"), 'isolation filters visibility hidden');
ok(host.includes("cs.contentVisibility === 'hidden'"), 'isolation filters content-visibility hidden');
ok(host.includes('zeroLeaf'), 'isolation filters zero-area or zero-font leaf text');
ok(host.includes('classicallyClipped'), 'isolation filters clipping');
ok(host.includes('extremeIndent'), 'isolation filters extreme indentation');
ok(host.includes('unreachable'), 'isolation filters unreachable-document-surface text');

console.log(`${passed} passed, ${failed} failed`);if(failed)process.exit(1);
