// Run 16.2.2 — remote POST result/error visibility contract.
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
function extract(name){
  const i=src.indexOf('function '+name+'('); if(i<0) throw new Error('not found '+name);
  let d=0,started=false,q=null,line=false,block=false;
  for(let j=i;j<src.length;j++){const c=src[j],n=src[j+1];
    if(line){if(c==='\n')line=false;continue} if(block){if(c==='*'&&n==='/'){block=false;j++;}continue}
    if(q){if(c==='\\'){j++;continue}if(c===q)q=null;continue}
    if(c==='/'&&n==='/'){line=true;j++;continue} if(c==='/'&&n==='*'){block=true;j++;continue}
    if(c==='"'||c==="'"||c==='`'){q=c;continue} if(c==='{'){d++;started=true}else if(c==='}'){d--;if(started&&d===0)return src.slice(i,j+1)}
  } throw new Error('unbalanced '+name);
}
globalThis._log=()=>{};
globalThis.location={href:'https://docs.example.test/'};
globalThis._safeUrlForLog=(0,eval)('('+extract('_safeUrlForLog')+')');
const boundedBegin=src.indexOf('    // B43: remote response ceilings');
const boundedEnd=src.indexOf('    /**\n     * Stable radio-group name',boundedBegin);
(0,eval)(src.slice(boundedBegin,boundedEnd));
const _remotePost=(0,eval)('('+extract('_remotePost')+')');
let pass=0,fail=0; function t(name,got,want=true){if(got===want)pass++;else{fail++;console.log(`FAIL ${name}\n  got ${JSON.stringify(got)}\n want ${JSON.stringify(want)}`)}}
async function run(fetchImpl, body={x:1}){
  globalThis._fetch=fetchImpl;
  let success=null,error=null;
  _remotePost('https://share.example.test/v1/share','',body,{keepalive:false,onSuccess:v=>{success=v},onError:e=>{error=e}});
  await new Promise(r=>setTimeout(r,0));
  await new Promise(r=>setTimeout(r,0));
  return {success,error};
}
let r=await run(()=>Promise.resolve(new Response(JSON.stringify({uuid:'a'.repeat(32)}),{status:200,headers:{'content-type':'application/json'}})));
t('valid JSON success reaches success callback',r.success?.uuid,'a'.repeat(32));t('valid JSON success has no error',r.error,null);
r=await run(()=>Promise.resolve(new Response('',{status:200})));
t('empty 2xx becomes visible protocol error',r.error?.status,502);t('empty 2xx explains empty response',r.error?.message,'The service returned an empty success response.');
r=await run(()=>Promise.resolve(new Response('<html>proxy page</html>',{status:200,headers:{'content-type':'text/html'}})));
t('HTML 2xx becomes visible protocol error',r.error?.status,502);t('HTML 2xx is not reflected into UI',r.error?.message,'The service returned a non-JSON success response.');
r=await run(()=>Promise.resolve(new Response(JSON.stringify({detail:'Origin is not allowed'}),{status:403,statusText:'Forbidden',headers:{'content-type':'application/json'}})));
t('HTTP error status preserved',r.error?.status,403);t('safe JSON error reason surfaced',r.error?.message,'Origin is not allowed');
r=await run(()=>Promise.reject(new TypeError('Failed to fetch')));
t('fetch/CORS rejection becomes status zero',r.error?.status,0);t('fetch/CORS rejection gets stable message',r.error?.message,'Network/CORS request failed.');
const circular={};circular.self=circular;globalThis._fetch=()=>{throw new Error('must not fetch')};let serialError=null;_remotePost('https://share.example.test/v1/share','',circular,{onError:e=>{serialError=e}});t('serialization failure invokes error callback',serialError?.message,'Could not serialize the request body.');
let missingError=null;_remotePost('', '', {x:1},{onError:e=>{missingError=e}});t('missing endpoint invokes error callback',missingError?.message,'Endpoint is not configured.');
console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
