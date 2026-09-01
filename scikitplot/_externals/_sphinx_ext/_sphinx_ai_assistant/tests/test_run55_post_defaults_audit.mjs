import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
let passed=0, failed=0;
function ok(cond,name){if(cond){passed++;}else{failed++;console.error('FAIL '+name);}}
function extract(name) {
  const i=src.indexOf('function '+name+'('); if(i<0) throw new Error('missing '+name);
  let d=0, started=false, q=null, line=false, block=false;
  for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1];
    if(line){if(c==='\n')line=false;continue} if(block){if(c==='*'&&n==='/'){block=false;j++}continue}
    if(q){if(c==='\\'){j++;continue}if(c===q)q=null;continue}
    if(c==='/'&&n==='/'){line=true;j++;continue} if(c==='/'&&n==='*'){block=true;j++;continue}
    if(c==='"'||c==="'"||c==='`'){q=c;continue} if(c==='{'){d++;started=true}else if(c==='}'){d--;if(started&&d===0)return src.slice(i,j+1)}
  } throw new Error('unbalanced '+name);
}
const fingerprint=(0,eval)('('+extract('_feedbackReviewFingerprint')+')');
const base={schemaVersion:1,consentFlag:true,consentVersion:'2.0.0',consentAt:100,trainingConsentFlag:true,feedbackId:'f2',feedbackChainId:'f1',prevFeedbackId:'f1',prevFeedbackIds:['f1'],editCount:1,ratingValue:1,message:'useful',query:'Q',answer:'A',page:'https://docs.example/page',ts:50};
const reloaded={...base,consentAt:999999};
ok(fingerprint(base)===fingerprint(reloaded),'volatile site-default consentAt cannot create a false review revision after reload');
ok(fingerprint(base)!==fingerprint({...base,message:'changed'}),'meaningful feedback content still changes review fingerprint');
ok(fingerprint(base)!==fingerprint({...base,page:''}),'Content & privacy source-page changes still change review fingerprint');
ok(fingerprint(base)!==fingerprint({...base,ts:null}),'Content & privacy timestamp changes still change review fingerprint');
ok(src.includes('absent state uses the configured site default (built-in: OFF)'), 'telemetry precedence comment matches configurable default behavior');
ok(!src.includes('prevSessionId'), 'browser feedback contract contains no retired prevSessionId alias');
console.log(`${passed} passed, ${failed} failed`);
if(failed)process.exit(1);
