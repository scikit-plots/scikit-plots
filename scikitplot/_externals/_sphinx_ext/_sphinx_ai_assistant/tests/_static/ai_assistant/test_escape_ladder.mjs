// Run 173 T53 - what Escape does, and what it says it does, must agree.
//
// A reader pressing Escape to leave a filtered sheet saw the panel stay open
// and read it as Escape not working. It had cleared a search filter they were
// not looking at -- a rung on the ladder that the shortcuts sheet never
// mentioned.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let n = 0, f = 0;
const ok = (c, m) => { c ? n++ : (f++, console.error('FAIL ' + m)); };

const ladder = (src.match(/'Escape first stops[^']*'/) || [''])[0];
ok(ladder.length > 0,'the shortcuts sheet documents an Escape ladder');
['microphone capture','live response','search filter','menu, popup, or sheet','closes the AI Assistant']
  .forEach(function (step) {
    ok(ladder.includes(step), 'the ladder names: ' + step);
  });

// Every rung that swallows Escape must be a documented rung. The filter clear
// stops propagation, so without its entry the description was incomplete in
// exactly the way that made the behaviour look broken.
ok(/clears an active search filter/.test(ladder),'the filter-clearing rung is documented');
ok(src.includes("if ((e.key === 'Escape' || e.keyCode === 27) && _query) {"),'and it only fires when there is a filter to clear');
ok(src.includes("// Escape clears an active search filter, and stops there."),'the handler states that it terminates the ladder');

// With no query the handler must not swallow the key, or an unfiltered sheet
// becomes impossible to leave from its search box.
const clearHandler = (src.match(/_input\.addEventListener\('keydown', function \(e\) \{[\s\S]*?\n        \}\);/) || [''])[0];
ok(/&& _query\)/.test(clearHandler),'the guard is on the query, not on the key alone');
ok(!/else\s*\{[\s\S]*stopPropagation/.test(clearHandler),'nothing stops propagation when there is no query');

// The dispatcher is the last rung and must still exist.
ok(/Nothing lighter owns Escape: close the assistant/.test(src),'the final rung closes the assistant');

console.log(`${n} passed, ${f} failed`); if (f) process.exit(1);
