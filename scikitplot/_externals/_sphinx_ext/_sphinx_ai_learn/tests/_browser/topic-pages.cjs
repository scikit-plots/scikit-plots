const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(process.env.LEARN_HTML);
let server,browser,origin;
const empty='/learn-ai/topics/20260919T121000Z-82e02eb2493f5d87.html';
const filled='/learn-ai/topics/20260919T121000Z-4eebdf3057d15bf8.html';
const bayesian='/learn-ai/topics/20260919T121000Z-3034d9c9e8aa7e29.html';
const whiteboard='/learn-ai/whiteboards/20260919T124000Z-139990bca17c1d80.html';
const video4k='/learn-ai/videos/20260918T213000Z-579852f9b66e3892.html';
const rickrollWhiteboard='/learn-ai/whiteboards/20260918T213500Z-db3c643106cffc0e.html';
const sharedDetails=[
 ['/learn-ai/topics/20260919T121000Z-4eebdf3057d15bf8.html','topic-lasso'],
 ['/learn-ai/open-problems/20260919T122000Z-191926f99f6ef116.html','problem-regularization-correlated-predictors'],
 ['/learn-ai/sources/20260919T120000Z-4556aa2d3c18d8e9.html','source-sklearn-linear-models'],
 ['/learn-ai/skills/20260916T000000Z-a7495a1e347e1fb9.html','skill-check-reference'],
 [whiteboard,'whiteboard-lasso-regularization'],
 [video4k,'video-rickroll-4k'],
];
before(async()=>{
 server=http.createServer((req,res)=>{const target=path.resolve(root,'.'+new URL(req.url,'http://localhost').pathname);if(!target.startsWith(root+path.sep)||!fs.existsSync(target)||!fs.statSync(target).isFile()){res.writeHead(404);res.end();return;}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml'})[path.extname(target)]||'application/octet-stream');res.end(fs.readFileSync(target));});
 await new Promise(r=>server.listen(0,'127.0.0.1',r));origin='http://127.0.0.1:'+server.address().port;
 browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE,args:JSON.parse(process.env.CHROMIUM_ARGS||'[]')});
});
after(async()=>{if(browser)await browser.close();if(server)await new Promise(r=>server.close(r));});
async function open(t,url=empty,options={}){const ctx=await browser.newContext(options);t.after(()=>ctx.close());const p=await ctx.newPage();p.setDefaultTimeout(7000);const errors=[];p.on('pageerror',e=>errors.push(e.message));t.after(()=>assert.deepEqual(errors,[]));await p.route('**/*',route=>route.request().url().startsWith(origin)?route.continue():route.abort());await p.goto(origin+url);return p;}
async function seedAiDraft(p,sectionId,body='AI-generated local draft'){await p.evaluate(({sectionId,body})=>{const d=JSON.parse(document.querySelector('.learn-page-data').textContent);const key='learn-ai-section:v2:'+d.site_id+':'+d.subject.id+':'+d.revision+':'+sectionId;localStorage.setItem(key,JSON.stringify({contract:'learn.section-draft.v2',title:document.querySelector('[data-section="'+sectionId+'"]')?.closest('section')?.querySelector('h2,h3,h4')?.textContent?.replace('¶','').trim()||sectionId,body,instructions:'',expanded:true,provenance:{authorship:'ai-generated',workflow_id:'learn.section-generation.v2',skill:'browser-test',model:'stub/test',generated_at:'2026-09-23T00:00:00.000Z',base_revision:d.revision,audience:'general',purpose:'understand',depth:'balanced',request_id:''}}));},{sectionId,body});await p.reload();}
test('empty page is complete without JavaScript and has real section anchors',async t=>{
 const p=await open(t,empty,{javaScriptEnabled:false});assert.equal(await p.locator('.learn-section').count(),21);assert.equal(await p.locator('.learn-toc a').count(),23);
 for(const href of await p.locator('.learn-toc a').evaluateAll(a=>a.map(n=>n.getAttribute('href'))))assert.equal(await p.locator(href).count(),1);
 assert.ok(await p.locator('[data-generate]:visible').count()>=1);assert.match(await p.locator('article').innerText(),/No one has generated a summary of this topic yet/);assert.match(await p.locator('article').innerText(),/We haven't generated follow-up questions for this topic yet/);
});
test('video empty state exposes generation, all-videos, and subscription actions',async t=>{
 const p=await open(t);const s=p.locator('[data-section="video"]');assert.match(await p.locator('article').innerText(),/Topic to Video \(Beta\)/);assert.match(await s.innerText(),/No one has generated a video about this topic yet/);
 assert.equal(await s.getByRole('link',{name:'Generate Now',exact:true}).count(),1);assert.match(await s.getByRole('link',{name:'Generate Now',exact:true}).getAttribute('href'),/videos\/new\.html\?mode=topic/);assert.match(await s.getByRole('link',{name:'All videos',exact:true}).getAttribute('href'),/videos/);assert.equal(await s.getByRole('button',{name:'Subscribe on YouTube',exact:true}).isDisabled(),true);
});
test('published text stays read-only until an AI draft exists; AI-assisted edits preserve provenance',async t=>{
 const p=await open(t,filled);const s=p.locator('[data-section="summary"]');assert.equal(await s.getByRole('button',{name:'Edit section',exact:true}).isHidden(),true);assert.equal(await s.getByRole('button',{name:'Generate Now',exact:true}).count(),1);
 await seedAiDraft(p,'summary','<img src=x onerror=window.BAD=true> A generated explanation.');assert.equal(await s.getByRole('button',{name:'Regenerate Now',exact:true}).count(),1);assert.equal(await s.getByRole('button',{name:'Edit section',exact:true}).isVisible(),true);assert.equal(await s.getByRole('button',{name:'Discard AI draft',exact:true}).isVisible(),true);
 await s.getByRole('button',{name:'Edit section',exact:true}).click();await s.getByLabel('AI draft text',{exact:true}).fill('<img src=x onerror=window.BAD=true> An AI-assisted explanation.');await s.getByLabel('Start expanded',{exact:true}).uncheck();await s.getByRole('button',{name:'Save changes',exact:true}).click();
 assert.equal(await s.locator('.learn-section-content').isVisible(),false);await s.getByRole('button',{name:'Show content',exact:true}).click();assert.match(await s.locator('.learn-prose').innerText(),/<img src=x/);assert.equal(await s.locator('.learn-prose img').count(),0);assert.equal(await p.evaluate(()=>window.BAD),undefined);assert.match(await s.locator('[data-state-label]').innerText(),/AI-assisted draft/);
 const stored=await p.evaluate(()=>{const d=JSON.parse(document.querySelector('.learn-page-data').textContent);return JSON.parse(localStorage.getItem('learn-ai-section:v2:'+d.site_id+':'+d.subject.id+':'+d.revision+':summary'));});assert.equal(stored.provenance.authorship,'ai-assisted');
});

test('accepted generation feedback has retry-safe quick votes and the full eleven-point detailed scale',async t=>{
 const p=await open(t,bayesian);const section=p.locator('[data-section="eli14"]');const feedback=section.locator('[data-learn-generation-feedback]');assert.equal(await feedback.count(),1);
 const beforeScore=await feedback.locator('[data-generation-feedback-score]').innerText();
 await p.evaluate(()=>{window.__learnFeedbackCalls=[];let loseFirst=true;const ui=window.AI_LEARN_GENERATION_UI;ui.publicationReceiptMessage=()=> 'Feedback queued for review.';ui.submitPublication=async request=>{window.__learnFeedbackCalls.push(JSON.parse(JSON.stringify(request)));if(loseFirst){loseFirst=false;throw new Error('simulated lost response');}return{contract:'learn.publication-receipt.v1',mode:'github',status:'queued',message:'queued'};};});
 const positive=feedback.getByRole('button',{name:'Helpful (+1)',exact:true});await positive.click();await feedback.locator('[data-learn-feedback-status]').filter({hasText:'Retry will reuse the same feedback id'}).waitFor();await positive.click();await feedback.locator('[data-learn-feedback-status]').filter({hasText:'Feedback queued for review'}).waitFor();
 const quick=await p.evaluate(()=>window.__learnFeedbackCalls.slice(0,2));assert.equal(quick.length,2);assert.equal(quick[0].rating,1);assert.equal(quick[1].rating,1);assert.equal(quick[0].feedback_id,quick[1].feedback_id);assert.match(quick[0].feedback_id,/^feedback-[0-9a-f]{48}$/);assert.equal(quick[0].generation_id,quick[1].generation_id);assert.deepEqual(Object.keys(quick[0]).sort(),['action','base_revision','contract','contributor','feedback_id','feedback_mode','generation_id','rating','section_id','subject_id'].sort());assert.equal(await feedback.locator('[data-generation-feedback-score]').innerText(),beforeScore);
 const negative=feedback.getByRole('button',{name:'Not helpful (-1)',exact:true});await negative.click();await feedback.locator('[data-learn-feedback-status]').filter({hasText:'Feedback queued for review'}).waitFor();const negativeCall=await p.evaluate(()=>window.__learnFeedbackCalls.at(-1));assert.equal(negativeCall.rating,-1);assert.notEqual(negativeCall.feedback_id,quick[1].feedback_id);assert.equal(await feedback.locator('[data-generation-feedback-score]').innerText(),beforeScore);
 await feedback.getByRole('button',{name:'Detailed feedback options',exact:true}).click();const detail=feedback.locator('[data-learn-feedback-detail]');assert.equal(await detail.isVisible(),true);const ratings=detail.locator('[data-learn-feedback-rating]');assert.equal(await ratings.count(),11);assert.deepEqual(await ratings.evaluateAll(nodes=>nodes.map(node=>Number(node.dataset.learnFeedbackRating))),[-5,-4,-3,-2,-1,0,1,2,3,4,5]);
 const neutral=detail.getByRole('button',{name:'Neutral (0)',exact:true});await neutral.click();assert.equal(await neutral.getAttribute('aria-pressed'),'true');await detail.locator('[data-learn-feedback-comment]').fill('Useful context, neutral rating.');await detail.locator('[data-learn-feedback-contributor]').fill('DataFox');await detail.getByRole('button',{name:'Send feedback',exact:true}).click();await feedback.locator('[data-learn-feedback-status]').filter({hasText:'Feedback queued for review'}).waitFor();
 const detailed=await p.evaluate(()=>window.__learnFeedbackCalls.at(-1));assert.equal(detailed.rating,0);assert.equal(detailed.comment,'Useful context, neutral rating.');assert.deepEqual(detailed.contributor,{display_name:'DataFox'});assert.equal(detailed.action,'feedback');assert.equal(detailed.contract,'learn.publication-request.v1');assert.equal(await feedback.locator('[data-generation-feedback-score]').innerText(),beforeScore);
});
test('feedback retries preserve the original anonymous event envelope across unrelated catalog rebuilds',async t=>{
 const p=await open(t,bayesian);const seed=await p.evaluate(()=>{const d=JSON.parse(document.querySelector('.learn-page-data').textContent),root=document.querySelector('[data-section="eli14"] [data-learn-generation-feedback]'),generationId=root.dataset.generationId,sectionId=root.dataset.feedbackSectionId,key='learn-ai-feedback-pending:v1:'+d.subject.id+':'+sectionId+':'+generationId,id='feedback-'+'ab'.repeat(24),request={contract:'learn.publication-request.v1',action:'feedback',base_revision:'tree-aaaaaaaaaaaaaaaa',subject_id:d.subject.id,section_id:sectionId,generation_id:generationId,feedback_id:id,rating:1,feedback_mode:'quick',contributor:{display_name:''}},fingerprint=JSON.stringify([1,'','','quick']);sessionStorage.setItem(key,JSON.stringify({fingerprint,request}));return{id};});
 await p.reload();await p.evaluate(()=>{window.__learnFeedbackCalls=[];window.AI_LEARN_GENERATION_UI.submitPublication=async request=>{window.__learnFeedbackCalls.push(JSON.parse(JSON.stringify(request)));return{contract:'learn.publication-receipt.v1',mode:'github',status:'queued'};};});
 const feedback=p.locator('[data-section="eli14"] [data-learn-generation-feedback]');await feedback.getByRole('button',{name:'Helpful (+1)',exact:true}).click();await feedback.locator('[data-learn-feedback-status]').filter({hasText:'Feedback queued for review'}).waitFor();const calls=await p.evaluate(()=>window.__learnFeedbackCalls);assert.equal(calls.length,1);assert.equal(calls[0].feedback_id,seed.id);assert.equal(calls[0].base_revision,'tree-aaaaaaaaaaaaaaaa');assert.deepEqual(Object.keys(calls[0]).sort(),['action','base_revision','contract','contributor','feedback_id','feedback_mode','generation_id','rating','section_id','subject_id'].sort());
});
test('generation feedback keeps icon-first controls responsive while selected labels and tones stay truthful',async t=>{
 const p=await open(t,bayesian);const feedback=p.locator('[data-section="eli14"] [data-learn-generation-feedback]');
 assert.equal((await feedback.locator('.learn-generation-feedback-label').innerText()).trim(),'Was This Helpful?');
 const labels=feedback.locator('.learn-generation-feedback-quick .learn-generation-feedback-choice-label');for(let i=0;i<await labels.count();i++)assert.equal(await labels.nth(i).isVisible(),false);
 const geometry=async()=>p.evaluate(()=>{const root=document.querySelector('[data-section="eli14"] [data-learn-generation-feedback]'),prompt=root.querySelector('.learn-generation-feedback-prompt').getBoundingClientRect(),quick=root.querySelector('.learn-generation-feedback-quick').getBoundingClientRect(),summary=root.querySelector('.learn-generation-feedback-summary').getBoundingClientRect(),score=root.querySelector('[data-generation-feedback-score]'),count=root.querySelector('[data-generation-feedback-count]'),buttons=[...root.querySelectorAll('.learn-generation-feedback-quick > button')].map(node=>node.getBoundingClientRect());return{prompt:{x:prompt.x,y:prompt.y,w:prompt.width,h:prompt.height},quick:{x:quick.x,y:quick.y,w:quick.width,h:quick.height},summary:{x:summary.x,y:summary.y,w:summary.width,h:summary.height},scoreHidden:(()=>{const cs=getComputedStyle(score),b=score.getBoundingClientRect();return cs.position==='absolute'&&b.width<=1&&b.height<=1&&(cs.clip!=='auto'||cs.clipPath!=='none'||cs.overflow==='hidden');})(),countText:count.textContent.trim(),buttons:buttons.map(b=>({y:b.y,h:b.height}))};});
 await p.setViewportSize({width:1200,height:900});let g=await geometry();const centers=[g.prompt.y+g.prompt.h/2,g.quick.y+g.quick.h/2,g.summary.y+g.summary.h/2];assert.ok(Math.max(...centers)-Math.min(...centers)<3);assert.ok(g.quick.x-(g.prompt.x+g.prompt.w)<24);assert.ok(g.summary.x>g.quick.x+g.quick.w);assert.equal(g.scoreHidden,true);assert.match(g.countText,/^\d+ ratings?$/);
 const compact=await feedback.evaluate(root=>{const status=root.querySelector('[data-learn-feedback-status]'),before=root.getBoundingClientRect().height,display=getComputedStyle(status).display,statusHeight=status.getBoundingClientRect().height;root.querySelector('[data-generation-feedback-count]').textContent='1234 ratings';root.querySelector('.learn-generation-feedback-summary').setAttribute('aria-label','Community feedback score 42 from 1234 ratings');const noOverflow=root.scrollWidth<=root.clientWidth+1;status.textContent='Feedback queued for review. Community ratings update only after merge and rebuild.';const withStatus=root.getBoundingClientRect().height;status.textContent='';const restored=root.getBoundingClientRect().height;return{before,display,statusHeight,noOverflow,withStatus,restored,count:root.querySelector('[data-generation-feedback-count]').textContent.trim()};});assert.equal(compact.display,'none');assert.equal(compact.statusHeight,0);assert.equal(compact.count,'1234 ratings');assert.equal(compact.noOverflow,true);assert.ok(compact.withStatus>compact.before);assert.ok(Math.abs(compact.restored-compact.before)<1);
 await p.setViewportSize({width:640,height:900});g=await geometry();assert.ok(g.prompt.y<g.quick.y&&g.quick.y<g.summary.y);assert.ok(Math.abs(g.buttons[0].y-g.buttons[1].y)<3);
 await p.setViewportSize({width:390,height:1000});g=await geometry();assert.ok(g.buttons[0].y<g.buttons[1].y&&g.buttons[1].y<g.buttons[2].y&&g.buttons[2].y<g.summary.y);
 const neutral=feedback.locator('[data-learn-feedback-rating="0"]');await feedback.getByRole('button',{name:'Detailed feedback options',exact:true}).click();const detailValues=feedback.locator('[data-learn-feedback-rating]');assert.deepEqual(await detailValues.allInnerTexts(),['😡','😞','😟','🙁','😑','😐','🙂','😊','😄','😁','🤩']);assert.equal(await neutral.getAttribute('aria-label'),'Neutral (0)');assert.equal(await neutral.locator('.learn-generation-feedback-rating-value').isVisible(),false);const before=await neutral.evaluate(el=>getComputedStyle(el).backgroundColor);await neutral.click();assert.equal(await neutral.getAttribute('aria-pressed'),'true');assert.equal((await neutral.innerText()).replace(/\s+/g,' ').trim(),'😐 0');assert.equal(await neutral.locator('.learn-generation-feedback-rating-value').isVisible(),true);assert.notEqual(await neutral.evaluate(el=>getComputedStyle(el).backgroundColor),before);const negativeFive=detailValues.first(),positiveFive=detailValues.last();await negativeFive.click();assert.equal((await negativeFive.innerText()).replace(/\s+/g,' ').trim(),'😡 -5');const negBg=await negativeFive.evaluate(el=>getComputedStyle(el).backgroundColor);await positiveFive.click();assert.equal((await positiveFive.innerText()).replace(/\s+/g,' ').trim(),'🤩 +5');const posBg=await positiveFive.evaluate(el=>getComputedStyle(el).backgroundColor);assert.notEqual(negBg,posBg);
});
test('inline AI generation panel exposes multi audience purpose skill role lenses and depth without mutating published text',async t=>{
 const p=await open(t);const s=p.locator('[data-section="knowledge-gaps"]');await s.getByRole('button',{name:'Generate Now',exact:true}).click();const panel=s.locator('[data-section-ai-panel]');assert.equal(await panel.isVisible(),true);assert.ok(await panel.locator('[data-section-ai-audience]').count()>=9);assert.ok(await panel.locator('[data-section-ai-purpose]').count()>=5);assert.ok(await panel.locator('[data-section-ai-skill]').count()>=5);assert.ok(await panel.locator('[data-section-ai-role]').count()>=6);assert.equal(await panel.getByLabel('Depth',{exact:true}).count(),1);assert.match(await panel.innerText(),/Context/);assert.match(await panel.innerText(),/Human judgment/);assert.equal(await s.getByRole('button',{name:'Edit section',exact:true}).isHidden(),true);
 await panel.locator('[data-section-ai-audience="practitioner"]').check();await panel.locator('[data-section-ai-purpose="apply"]').check();await panel.locator('[data-section-ai-skill="evidence-audit"]').check();await panel.locator('[data-section-ai-role="analyst"]').check();await panel.getByLabel('Depth',{exact:true}).selectOption('deep');assert.equal(await panel.getByRole('button',{name:'Generate Now',exact:true}).count(),1);assert.equal(await panel.getByRole('button',{name:'Copy request',exact:true}).count(),1);
});
test('inline AI generation uses the configured chat authority and creates a provenance-bound private draft',async t=>{
 const p=await open(t,filled);let seen=null;await p.route('**/v1/chat/completions',async route=>{seen=route.request().postDataJSON();await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({id:'section-test-1',choices:[{message:{content:'A source-grounded browser-test explanation.'}}]})});});
 await p.evaluate(()=>{window.AI_ASSISTANT_MODEL_API={getState:()=>({active:{model:'stub-section-model',label:'Stub section model'},effort:'default'}),openPicker:()=>true,onChange:()=>()=>{}};window.AI_ASSISTANT_ENDPOINT_API={resolveEndpoint:key=>key==='chat'?location.origin+'/v1/chat/completions':''};});
 const s=p.locator('[data-section="summary"]');await s.getByRole('button',{name:'Generate Now',exact:true}).click();const panel=s.locator('[data-section-ai-panel]');await panel.locator('[data-section-ai-audience="general"]').uncheck();await panel.locator('[data-section-ai-audience="decision-maker"]').check();await panel.locator('[data-section-ai-purpose="understand"]').uncheck();await panel.locator('[data-section-ai-purpose="compare"]').check();await panel.locator('[data-section-ai-skill="evidence-audit"]').check();await panel.locator('[data-section-ai-role="analyst"]').check();await panel.getByLabel('Depth',{exact:true}).selectOption('deep');await panel.getByRole('button',{name:'Generate Now',exact:true}).click();await panel.locator('[data-section-ai-status]').filter({hasText:'AI draft ready for review'}).waitFor();
 assert.equal(seen.contract,'scikitplot-chat-v1');assert.equal(seen.model,'stub-section-model');assert.match(seen.user_message,/Role lenses:/);assert.match(seen.user_message,/decision maker/);assert.match(seen.user_message,/compare alternatives/);assert.match(seen.context.page_text,/Published baseline text/);assert.match(await s.locator('.learn-prose').innerText(),/browser-test explanation/);assert.equal(await s.getByRole('button',{name:'Regenerate Now',exact:true}).count(),1);assert.equal(await s.getByRole('button',{name:'Edit section',exact:true}).isVisible(),true);
 const stored=await p.evaluate(()=>{const d=JSON.parse(document.querySelector('.learn-page-data').textContent);return JSON.parse(localStorage.getItem('learn-ai-section:v2:'+d.site_id+':'+d.subject.id+':'+d.revision+':summary'));});assert.equal(stored.provenance.authorship,'ai-generated');assert.equal(stored.provenance.model,'stub-section-model');assert.equal(stored.provenance.agent,'learning-section-agent');assert.equal(stored.provenance.audience,'decision-maker');assert.equal(stored.provenance.purpose,'compare');assert.ok(stored.provenance.skills.includes('evidence-audit'));assert.ok(stored.provenance.roles.includes('analyst'));assert.equal(stored.provenance.depth,'deep');assert.equal(stored.provenance.request_id,'section-test-1');
});
test('prompt toggles persist across the topic page and Topic Prompts library',async t=>{
 const p=await open(t);const toggle=p.locator('[data-prompt-toggle="knowledge-gaps"]').first();const section=p.locator('[data-section="knowledge-gaps"]');
 assert.equal(await toggle.isChecked(),true);await toggle.uncheck();assert.equal(await section.locator('xpath=ancestor::section[1]').isVisible(),false);assert.equal(await p.locator('.learn-toc a[data-toc-section="knowledge-gaps"]').locator('xpath=ancestor::li[1]').isVisible(),false);
 await p.reload();assert.equal(await p.locator('[data-prompt-toggle="knowledge-gaps"]').first().isChecked(),false);
 await p.goto(origin+'/learn-ai/topic-prompts/index.html');const libraryToggle=p.locator('[data-prompt-toggle="knowledge-gaps"]').first();assert.equal(await libraryToggle.isChecked(),false);const link=p.getByRole('link',{name:'View prompt',exact:true}).first();assert.match(await link.getAttribute('href'),/topic-prompts\//);
});
test('prompt and skill switches keep fixed geometry and native accessible names across page types',async t=>{
 const urls=[empty,'/learn-ai/topic-prompts/index.html','/learn-ai/skills/index.html','/learn-ai/topic-prompts/eli14/index.html','/learn-ai/skills/skill-check-reference/index.html'];
 for(const url of urls){
  const p=await open(t,url,{viewport:{width:390,height:844}});const switches=p.locator('label.learn-switch');assert.ok(await switches.count()>=1);
  assert.equal(await switches.locator('.sr-only').count(),0);
  for(let i=0;i<await switches.count();i++){const sw=switches.nth(i),input=sw.locator('input[type="checkbox"]');const box=await sw.boundingBox();assert.ok(box);assert.ok(Math.abs(box.width-76)<0.6,`${url} switch width ${box.width}`);assert.match(await input.getAttribute('aria-label'),/^Show .+ on topic pages$/);}
 }
});

test('record-aware AI overview uses the configured chat model and selected context',async t=>{
 const p=await open(t,filled);let seen=null;await p.route('**/v1/chat/completions',async route=>{seen=route.request().postDataJSON();await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({id:'overview-test',choices:[{message:{content:'Private overview from selected catalog context.'}}]})});});await p.evaluate(()=>{window.AI_ASSISTANT_MODEL_API={getState:()=>({active:{model:'stub-overview',label:'Stub overview model'}}),openPicker:()=>true,onChange:()=>()=>{}};window.AI_ASSISTANT_ENDPOINT_API={resolveEndpoint:key=>key==='chat'?location.origin+'/v1/chat/completions':''};});await p.getByRole('button',{name:'Generate AI Overview',exact:true}).click();const panel=p.locator('.learn-overview-generation');assert.equal(await panel.isVisible(),true);const publishButtons=panel.locator('[data-overview-publish]');assert.equal(await publishButtons.count(),2);assert.equal(await publishButtons.first().isDisabled(),true);const credit=panel.locator('[data-publication-credit]');assert.equal(await credit.count(),1);await credit.fill('Overview DataFox');await panel.locator('[data-overview-purpose="research"]').check();await panel.getByRole('button',{name:'Generate AI Overview',exact:true}).click();await panel.locator('[data-overview-status]').filter({hasText:'AI overview ready'}).waitFor();assert.equal(seen.model,'stub-overview');assert.match(seen.context.page_text,/Record title:/);assert.doesNotMatch(JSON.stringify(seen),/Overview DataFox/);assert.match(await panel.locator('[data-overview-output]').innerText(),/Private overview/);assert.equal(await publishButtons.first().isDisabled(),false);assert.equal(await publishButtons.last().isDisabled(),false);
 await p.reload();await p.getByRole('button',{name:'Review AI Overview',exact:true}).click();assert.equal(await p.locator('.learn-overview-generation [data-publication-credit]').inputValue(),'Overview DataFox');
});
test('filled page evidence, native media and direct contents links',async t=>{
 const p=await open(t,filled);const s=p.locator('[data-section="summary"]');assert.match(await s.locator('.learn-prose').innerText(),/Lasso is a linear model/);await s.getByRole('button',{name:'Review sources',exact:true}).click();assert.equal(await s.locator('.learn-evidence').getAttribute('open'),'');assert.equal(await s.locator('[data-evidence-review]').isVisible(),true);assert.ok(await s.locator('[data-evidence-use]').count()>=1);
 assert.match(await s.locator('.learn-evidence a').getAttribute('href'),/scikit-learn\.org\/1\.9\/modules\/linear_model\.html/);assert.equal(await p.locator('[data-section="whiteboard"] .sd-card img').count(),1);
 await s.getByRole('button',{name:'Hide content',exact:true}).click();await p.locator('.learn-toc a[data-toc-section="summary"]').click();await s.locator('.learn-section-content').waitFor({state:'visible'});assert.equal(await s.locator('.learn-section-content').isVisible(),true);
 if(process.env.LEARN_SCREENSHOT)await p.screenshot({path:process.env.LEARN_SCREENSHOT.replace('.png','-desktop.png'),fullPage:false});
});
test('AI draft storage conflicts preserve the in-progress edit',async t=>{
 const p=await open(t,filled);await seedAiDraft(p,'summary','Existing AI draft');const s=p.locator('[data-section="summary"]');await s.getByRole('button',{name:'Edit section',exact:true}).click();await s.getByLabel('AI draft text',{exact:true}).fill('My unsaved work');await p.evaluate(()=>{const d=JSON.parse(document.querySelector('.learn-page-data').textContent);const key='learn-ai-section:v2:'+d.site_id+':'+d.subject.id+':'+d.revision+':summary';localStorage.setItem(key,JSON.stringify({...JSON.parse(localStorage.getItem(key)),body:'changed elsewhere'}));});await s.getByRole('button',{name:'Save changes',exact:true}).click();assert.match(await p.locator('.learn-message').innerText(),/another tab/);assert.equal(await s.getByLabel('AI draft text',{exact:true}).inputValue(),'My unsaved work');
});
test('search resolves a real topic and research mode remains honest',async t=>{
 const p=await open(t);await p.getByLabel('Explore a topic',{exact:true}).fill('Lasso sparse regression');await p.getByRole('button',{name:'Explore →',exact:true}).click();await p.locator('[data-search-results] a').click();assert.equal(new URL(p.url()).pathname,filled);
 await p.getByLabel('Explore a topic',{exact:true}).fill('https://scikit-learn.org/1.9/modules/linear_model.html');await p.getByRole('button',{name:'Explore →',exact:true}).click();assert.ok(await p.locator('[data-search-results] a').count()>0);
 await p.getByLabel('Mode',{exact:true}).selectOption('research');await p.getByRole('button',{name:'Explore →',exact:true}).click();assert.match(await p.locator('[data-search-results]').innerText(),/configured/);
});
test('native youtube directive works standalone and inside existing gallery grid',async t=>{
 const p=await open(t,'/media-directives.html');assert.equal(await p.locator('iframe[src*="youtube-nocookie.com/embed/JXtISpdDPNY"]').count(),2);assert.equal(await p.locator('.sd-card iframe').count(),1);
});
test('mobile reading, editing and explorer layouts stay within viewport',async t=>{
 const p=await open(t,filled,{viewport:{width:390,height:844}});await seedAiDraft(p,'summary','Mobile AI draft');const s=p.locator('[data-section="summary"]');await s.getByRole('button',{name:'Edit section',exact:true}).click();assert.equal(await s.getByLabel('AI draft text',{exact:true}).isEditable(),true);
 assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));if(process.env.LEARN_SCREENSHOT)await p.screenshot({path:process.env.LEARN_SCREENSHOT,fullPage:false});
 await p.goto(origin+'/learn-ai/whiteboards/index.html');assert.equal(await p.locator('.learn-whiteboard-card').count(),3);assert.equal(await p.locator('.learn-whiteboard-card .learn-media-card-content img').count(),3);assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
});
test('topic whiteboard image opens and supports one-click zoom in and out',async t=>{
 const p=await open(t,filled);const image=p.locator('[data-section="whiteboard"] img').first();assert.equal(await image.count(),1);await image.click();const viewer=p.locator('.learn-lightbox');assert.equal(await viewer.isVisible(),true);const reset=viewer.getByRole('button',{name:'Reset zoom',exact:true});const viewed=viewer.locator('.learn-lightbox-image');await viewed.click();assert.equal(await reset.innerText(),'200%');await viewed.click();assert.equal(await reset.innerText(),'100%');await viewer.getByRole('button',{name:'Close',exact:true}).click();
});
test('disabled prompt sections disappear from both article and PyData secondary sidebar',async t=>{
 const p=await open(t,filled);const id='future-research';const section=p.locator(`[data-section="${id}"]`).locator('xpath=ancestor::section[1]');const toc=p.locator(`[data-toc-section="${id}"]`).locator('xpath=ancestor::li[1]');assert.equal(await section.isHidden(),true);assert.equal(await toc.isHidden(),true);
});
test('whiteboard detail image opens the native zoom viewer',async t=>{
 const p=await open(t,whiteboard);const image=p.locator('[data-whiteboard-gallery] img').first();assert.equal(await image.count(),1);await image.click();
 const viewer=p.locator('.learn-lightbox');assert.equal(await viewer.isVisible(),true);assert.equal(await viewer.locator('.learn-lightbox-counter').innerText(),'1 / 1');
 assert.equal(await viewer.getByRole('button',{name:'Previous image',exact:true}).isHidden(),true);assert.equal(await viewer.getByRole('button',{name:'Next image',exact:true}).isHidden(),true);
 assert.equal(await viewer.getByRole('button',{name:'Thumbnails',exact:true}).isDisabled(),true);assert.equal(await viewer.getByRole('button',{name:'Turn on slideshow',exact:true}).isDisabled(),true);
 const reset=viewer.getByRole('button',{name:'Reset zoom',exact:true});const viewed=viewer.locator('.learn-lightbox-image');await viewed.click();assert.equal(await reset.innerText(),'200%');await viewed.click();assert.equal(await reset.innerText(),'100%');
 await viewer.getByRole('button',{name:'Zoom in',exact:true}).click();assert.equal(await reset.innerText(),'125%');await viewer.getByRole('button',{name:'Zoom out',exact:true}).click();assert.equal(await reset.innerText(),'100%');
 await viewer.press('0');assert.equal(await reset.innerText(),'100%');await viewer.getByRole('button',{name:'Close',exact:true}).click();assert.equal(await p.locator('.learn-lightbox').count(),0);
});

test('source-owned detail sections never expose AI edit or regenerate controls',async t=>{
 const checks=[[video4k,'transcript'],[video4k,'evidence'],[whiteboard,'evidence'],['/learn-ai/sources/20260919T120000Z-4556aa2d3c18d8e9.html','related'],['/learn-ai/skills/20260916T000000Z-a7495a1e347e1fb9.html','evidence']];for(const [url,id] of checks){const p=await open(t,url);const s=p.locator('[data-section="'+id+'"]');assert.equal(await s.locator('[data-generate]').count(),0);assert.equal(await s.locator('[data-edit]').count(),0);}
});
test('media indexes separate playback zoom and navigation targets',async t=>{
 const p=await open(t,'/learn-ai/videos/index.html');assert.equal(await p.locator('.learn-video-card').count(),6);assert.equal(await p.locator('.learn-video-card iframe[src*="youtube-nocookie.com"]').count(),5);
 const videoLink=p.locator('.learn-video-card h3 a').filter({hasText:'Rickroll explainer — 4K'});assert.match(await videoLink.getAttribute('href'),/videos\/20260918T213000Z-579852f9b66e3892\.html/);
 await p.goto(origin+'/learn-ai/whiteboards/index.html');const title=p.locator('.learn-whiteboard-card h3 a').first();assert.match(await title.getAttribute('href'),/whiteboards\/20260917T120000Z-3633c646b1fef0db\.html/);
 const image=p.locator('.learn-whiteboard-card .learn-media-card-content img').first();await image.click();assert.equal(await p.locator('.learn-lightbox').isVisible(),true);await p.locator('.learn-lightbox').getByRole('button',{name:'Close',exact:true}).click();
});
test('video detail is media first and routes variants into the generation lifecycle',async t=>{
 const p=await open(t,video4k);const article=p.locator('article');assert.equal(await article.locator('iframe[src*="youtube-nocookie.com/embed/dQw4w9WgXcQ"]').count(),1);const actions=p.locator('[data-media-actions]');const variant=actions.getByRole('link',{name:'Generate Variant',exact:true});assert.equal(await variant.count(),1);assert.match(await variant.getAttribute('href'),/videos\/new\.html\?from=video/);assert.equal(await actions.getByRole('button',{name:'Copy URL',exact:true}).count(),1);assert.equal(await actions.getByRole('link',{name:'View Topic',exact:true}).count(),0);assert.equal(await actions.locator('.learn-overview-actions').count(),1);assert.equal(await actions.locator('[data-media-generation]').count(),0);
});
test('shared overview action bar works on topic problem source skill whiteboard and video detail pages',async t=>{
 for(const [url,id] of sharedDetails){
  const p=await open(t,url);const actions=p.locator('.learn-overview-actions');assert.equal(await actions.isVisible(),true);
  const reading=actions.getByLabel('Reading',{exact:true});assert.deepEqual(await reading.locator('option').allTextContents(),['Unread','Want to Read','Currently Reading','Completed']);await reading.selectOption('Want to Read');
  assert.equal(await p.evaluate(({id})=>localStorage.getItem('learn-user:v1:scikit-plots-learn:reading:'+id),{id}),'Want to Read');
  const bookmark=actions.getByRole('button',{name:'Bookmark',exact:true});await bookmark.click();assert.equal(await bookmark.getAttribute('aria-pressed'),'true');assert.equal(await bookmark.innerText(),'Bookmarked');
  assert.match(await actions.getByRole('link',{name:'Collections',exact:true}).getAttribute('href'),/collections\/index\.html/);
  assert.match(await actions.getByRole('link',{name:'Bookmarks',exact:true}).getAttribute('href'),/bookmarks\/index\.html/);
  await actions.getByRole('button',{name:'Generate AI Overview',exact:true}).click();const panel=p.locator('.learn-overview-generation');assert.equal(await panel.isVisible(),true);assert.match(await panel.innerText(),/Private draft/);await panel.getByRole('button',{name:'Close',exact:true}).click();assert.equal(await panel.isHidden(),true);assert.equal(await actions.getByRole('button',{name:'Export page edits',exact:true}).count(),0);
 }
});
test('reading collections include non-topic records selected from shared detail actions',async t=>{
 const p=await open(t,sharedDetails[2][0]);await p.getByLabel('Reading',{exact:true}).selectOption('Want to Read');await p.goto(origin+'/learn-ai/collections/want-to-read/index.html');assert.match(await p.locator('[data-collection-items]').innerText(),/scikit-learn: Linear Models/);assert.match(await p.locator('[data-collection-items]').innerText(),/source/);
});
test('lasso topic connects a zoomable whiteboard while Rickroll media remains standalone',async t=>{
 const p=await open(t,filled);const image=p.locator('[data-section="whiteboard"] img').first();assert.equal(await image.count(),1);await image.click();assert.equal(await p.locator('.learn-lightbox').isVisible(),true);await p.getByRole('button',{name:'Close',exact:true}).click();
 await p.goto(origin+rickrollWhiteboard);assert.equal(await p.locator('[data-whiteboard-gallery] img').count(),3);await p.locator('[data-whiteboard-gallery] img').first().click();const viewer=p.locator('.learn-lightbox');assert.equal(await viewer.locator('.learn-lightbox-counter').innerText(),'1 / 3');await viewer.getByRole('button',{name:'Next image',exact:true}).click();assert.equal(await viewer.locator('.learn-lightbox-counter').innerText(),'2 / 3');
 await p.goto(origin+video4k);assert.equal(await p.getByRole('link',{name:'View Topic',exact:true}).count(),0);
});
test('legacy draft section selection remains editable',async t=>{
 const p=await open(t,'/hub.html');await p.getByRole('button',{name:'Create topic',exact:true}).click();await p.getByLabel('Subject title',{exact:true}).fill('Test');await p.getByLabel('Explanation (plain text)',{exact:true}).fill('A draft');await p.getByRole('button',{name:'Save on this browser',exact:true}).click();await p.reload();await p.getByRole('button',{name:'My drafts',exact:true}).click();await p.getByRole('button',{name:'Continue editing',exact:true}).click();const select=p.getByLabel('Section',{exact:true});assert.equal(await select.isEnabled(),true);await select.selectOption('glossary');await p.getByRole('button',{name:'Save on this browser',exact:true}).click();assert.match(await p.locator('.la-status').innerText(),/Saved/);
});

test('bookmarks and reading collections share stable browser-local preferences without nesting topics on collection home',async t=>{
 const p=await open(t);await p.getByRole('button',{name:'Bookmark',exact:true}).click();await p.getByLabel('Reading',{exact:true}).selectOption('Want to Read');
 const groups=p.locator('.learn-overview-action-group');assert.equal(await groups.count(),4);assert.equal(await p.getByRole('link',{name:'Generate Video',exact:true}).count(),1);
 assert.equal(await p.getByRole('link',{name:'Generate Audio',exact:true}).count(),1);assert.equal(await p.getByRole('link',{name:'Generate Document',exact:true}).count(),1);assert.equal(await p.getByRole('link',{name:'Generate Whiteboard',exact:true}).count(),1);
 await p.getByRole('link',{name:'Bookmarks',exact:true}).click();assert.match(await p.locator('[data-bookmarks-list]').innerText(),/Lasso/);
 await p.goto(origin+'/learn-ai/collections/index.html');assert.equal(await p.locator('.learn-library-item').count(),0);
 const wantCard=p.locator('[data-collection-link="Want to Read"]');assert.equal(await wantCard.locator('[data-collection-count-for="Want to Read"]').innerText(),'1');
 await wantCard.click();assert.match(new URL(p.url()).pathname,/\/collections\/want-to-read\/index\.html$/);assert.match(await p.locator('[data-collection-items]').innerText(),/Lasso/);
});
test('bookmarks and reading collections share responsive grid density preference',async t=>{
 const p=await open(t);await p.getByRole('button',{name:'Bookmark',exact:true}).click();await p.getByLabel('Reading',{exact:true}).selectOption('Want to Read');
 await p.getByRole('link',{name:'Bookmarks',exact:true}).click();const bookmarkGrid=p.locator('[data-bookmarks-list]');assert.equal(await bookmarkGrid.getAttribute('data-grid-columns'),'2');
 const four=p.getByRole('button',{name:'4',exact:true});await four.click();assert.equal(await bookmarkGrid.getAttribute('data-grid-columns'),'4');assert.equal(await four.getAttribute('aria-pressed'),'true');
 assert.equal(await p.evaluate(()=>localStorage.getItem('learn-user:v1:scikit-plots-learn:library-grid-columns')),'4');
 await p.goto(origin+'/learn-ai/collections/want-to-read/index.html');const readingGrid=p.locator('[data-collection-items]');assert.equal(await readingGrid.getAttribute('data-grid-columns'),'4');assert.equal(await p.getByRole('button',{name:'4',exact:true}).getAttribute('aria-pressed'),'true');
 await p.setViewportSize({width:390,height:900});const mobile=await readingGrid.evaluate(el=>({client:el.clientWidth,scroll:el.scrollWidth,columns:getComputedStyle(el).gridTemplateColumns.split(' ').length}));assert.ok(mobile.scroll<=mobile.client+1);assert.equal(mobile.columns,1);
});

test('canonical audio document and whiteboard generation pages are materialized',async t=>{
 const p=await open(t,'/learn-ai/audios/new.html');assert.equal(await p.locator('[data-audio-generation]').count(),1);
 await p.goto(origin+'/learn-ai/documents/new.html');assert.equal(await p.locator('[data-document-generation]').count(),1);assert.equal(await p.getByRole('button',{name:'Generate Now',exact:true}).count(),1);
 await p.goto(origin+'/learn-ai/whiteboards/new.html');assert.equal(await p.locator('[data-whiteboard-generation]').count(),1);assert.equal(await p.getByRole('button',{name:'Generate Now',exact:true}).count(),1);assert.equal(await p.locator('[data-generation-authority-picker][data-generation-authority-kind="assistant"][data-generation-authority-managed="shared"]').count(),1);assert.equal(await p.locator('[data-generation-authority-open]').count(),1);assert.equal(await p.locator('[data-generation-authority-more]').count(),1);assert.equal(await p.locator('[data-generation-authority-menu]').count(),1);assert.equal(await p.locator('[data-whiteboard-runtime-authority]').count(),1);
 await p.goto(origin+'/learn-ai/audio/new.html');assert.equal(await p.locator('[data-audio-generation]').count(),1);
});

test('generation studio shares navigation steps and prompt interaction across modalities',async t=>{
 const p=await open(t,'/learn-ai/videos/new.html');
 const allStudios=[['topics','Topics'],['sources','Sources'],['open-problems','Open Problems'],['whiteboards','Whiteboards'],['videos','Videos'],['audios','Audio'],['documents','Documents'],['skills','Skills'],['topic-prompts','Topic Prompts']];
 for(const [path,label] of allStudios){
  await p.goto(origin+'/learn-ai/'+path+'/new.html');
  const nav=p.locator('.learn-generation-studio-nav');assert.equal(await nav.getByRole('link').count(),9);
  assert.equal(await nav.locator('[aria-current="page"]').innerText(),label);
 }
 const cases=[['videos','Videos'],['audios','Audio'],['documents','Documents'],['whiteboards','Whiteboards']];
 for(const [path,label] of cases){
  await p.goto(origin+'/learn-ai/'+path+'/new.html');
  const nav=p.locator('.learn-generation-studio-nav');assert.equal(await nav.getByRole('link').count(),9);assert.equal(await nav.locator('[aria-current="page"]').innerText(),label);
  assert.equal(await p.locator('.learn-generation-step-index').count(),4);
  assert.equal(await p.locator('[data-generation-shaping]').count(),1);
  const lenses=p.locator('[data-generation-lenses]');assert.equal(await lenses.count(),1);
  assert.equal(await lenses.locator('[data-ai-lens-group]').count(),4);assert.equal(await lenses.locator('[data-ai-lens-required="true"]').count(),2);
  assert.match(await lenses.locator('[data-ai-lens-profile-summary]').innerText(),/4 selections/);
  const advanced=p.locator('[data-generation-advanced]');assert.equal(await advanced.count(),1);
  assert.equal(await advanced.locator('select').count(),1);assert.equal(await advanced.locator('input[type="checkbox"]').count(),2);
  assert.deepEqual(await p.locator('.learn-generation-prompt-chips button').allTextContents(),['Beginner-friendly','Technical','Examples','Intuition','Compare','Limitations']);
  assert.equal(await p.locator('[data-generation-counter]').count(),1);
  assert.equal(await p.locator('[role="radiogroup"]').count(),1);
  assert.equal(await p.locator('.learn-generation-authority').count(),1);
  assert.equal(await p.locator('.learn-generation-authority-picker').count(),1);
  assert.equal(await p.locator('.learn-generation-authority-main').count(),1);
  assert.equal(await p.locator('.learn-generation-authority-more').count(),1);
  assert.equal(await p.locator('.learn-generation-authority-menu').count(),1);
  assert.equal(await p.locator('.learn-generation-model-readout').count(),1);
  const credit=p.locator('[data-publication-credit]');assert.equal(await credit.count(),1);
  const actions=p.locator('.learn-generation-action-bar');assert.equal(await actions.count(),1);
  assert.equal(await p.evaluate(()=>{const credit=document.querySelector('[data-publication-credit]')?.closest('.learn-publication-credit'),actions=document.querySelector('.learn-generation-action-bar');return !!credit&&!!actions&&!!(credit.compareDocumentPosition(actions)&Node.DOCUMENT_POSITION_FOLLOWING);}),true);
  const activity=p.locator('[data-generation-status]');assert.equal(await activity.count(),1);
  assert.equal(await activity.getAttribute('data-state'),'idle');
  assert.equal(await activity.locator('[data-generation-status-label]').innerText(),'Ready');
  assert.equal(await activity.locator('[data-generation-status-message]').count(),1);
  assert.equal(await actions.getByRole('button',{name:'Generate Now',exact:true}).count(),1);assert.equal(await actions.getByRole('button',{name:'Generate Now',exact:true}).isEnabled(),true);
  assert.equal(await actions.getByRole('button',{name:'Save draft',exact:true}).count(),1);
  assert.equal(await actions.getByRole('button',{name:'Copy request',exact:true}).count(),1);
  const flowOrder=await p.evaluate(()=>{const form=document.querySelector('.learn-generation-form'),step1=[...form.querySelectorAll('.learn-generation-step-index')].find(n=>n.textContent.trim()==='1')?.closest('fieldset'),shape=form.querySelector('[data-generation-shaping]'),step3=form.querySelector('[data-generation-lenses]'),step4=[...form.querySelectorAll('.learn-generation-step-index')].find(n=>n.textContent.trim()==='4')?.closest('fieldset'),advanced=form.querySelector('[data-generation-advanced]'),authority=form.querySelector('.learn-generation-authority'),actions=form.querySelector('.learn-generation-action-bar'),nodes=[step1,shape,step3,step4,advanced,authority,actions];return nodes.every(Boolean)&&nodes.every((node,i)=>i===nodes.length-1||Boolean(node.compareDocumentPosition(nodes[i+1])&Node.DOCUMENT_POSITION_FOLLOWING));});assert.equal(flowOrder,true);
  const library=p.locator('[data-generation-library]');assert.equal(await library.count(),1);
  assert.equal(await library.locator('[data-generation-library-filter]').count(),1);
  assert.deepEqual(await library.locator('[data-generation-library-filter] option').allTextContents(),['Active','Archived','All']);
  assert.equal(await library.getByRole('button',{name:'Refresh',exact:true}).count(),1);
  assert.equal(await library.locator('[data-generation-library-grid]').count(),1);
  assert.equal(await library.locator('[data-generation-library-empty]').count(),1);
  assert.equal(await library.locator('[data-generation-library-status]').count(),1);
 }
 await p.goto(origin+'/learn-ai/documents/new.html');
 await p.locator('[data-document-form] label').filter({hasText:/^Prompt$/}).click();
 assert.equal(await p.locator('[data-document-prompt-note]').isVisible(),true);
 const instructions=p.locator('[data-document-prompt]');const before=await instructions.inputValue();
 await p.getByRole('button',{name:'Compare',exact:true}).click();assert.notEqual(await instructions.inputValue(),before);
 assert.match(await instructions.inputValue(),/alternatives and trade-offs/);
 await p.locator('[data-document-structure]').selectOption('study-guide');
 await p.locator('[data-document-glossary]').check();
 await p.locator('[data-document-audience="beginner"]').check();await p.locator('[data-document-role="analyst"]').check();
 assert.match(await p.locator('[data-generation-lenses] [data-ai-lens-profile-summary]').innerText(),/6 selections/);
 const primaryPrompt=p.locator('[data-document-free-prompt]');await primaryPrompt.fill('Persistent draft sentinel');await p.locator('[data-publication-credit]').fill('Studio DataFox');
 await p.getByRole('button',{name:'Save draft',exact:true}).click();assert.match(await p.locator('[data-document-status]').innerText(),/Draft saved/);assert.equal(await p.locator('[data-document-status]').getAttribute('data-state'),'success');
 await p.reload();assert.equal(await p.locator('[data-document-free-prompt]').inputValue(),'Persistent draft sentinel');assert.equal(await p.locator('[data-document-structure]').inputValue(),'study-guide');assert.equal(await p.locator('[data-document-glossary]').isChecked(),true);assert.equal(await p.locator('[data-document-audience="beginner"]').isChecked(),true);assert.equal(await p.locator('[data-document-role="analyst"]').isChecked(),true);assert.equal(await p.locator('[data-publication-credit]').inputValue(),'Studio DataFox');
 await p.evaluate(()=>Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{window.__learnCopiedRequest=text;}}}));
 await p.getByRole('button',{name:'Copy request',exact:true}).click();
 const copied=JSON.parse(await p.evaluate(()=>window.__learnCopiedRequest));assert.equal(copied.contract,'assistant.document-generation-request.v1');assert.match(copied.prompt,/Persistent draft sentinel/);assert.match(copied.prompt,/study guide/);assert.match(copied.prompt,/glossary/);assert.match(copied.prompt,/AI lenses/);assert.match(copied.prompt,/Audience: general, beginner/);assert.match(copied.prompt,/Role lenses: explainer, analyst/);assert.doesNotMatch(JSON.stringify(copied),/Studio DataFox/);
});

test('audio and document authority pickers use the same Assistant model transaction as Video',async t=>{
 const p=await open(t,'/learn-ai/audios/new.html');
 for(const path of ['audios','documents']){
  await p.goto(origin+'/learn-ai/'+path+'/new.html');
  const initial=await p.evaluate(()=>window.AI_ASSISTANT_MODEL_API.getState());
  const models=await p.evaluate(()=>window.AI_ASSISTANT_MODEL_API.listModels());
  const target=models.find(m=>m.id!==initial.active.id);assert.ok(target);
  assert.equal(await p.locator('[data-generation-authority-label]').innerText(),initial.active.label);
  await p.locator('[data-generation-authority-more]').click();
  assert.equal(await p.locator('[data-generation-authority-more]').getAttribute('aria-expanded'),'true');
  await p.locator('[data-generation-authority-menu] [data-model-id="'+target.id+'"]').click();
  assert.equal((await p.evaluate(()=>window.AI_ASSISTANT_MODEL_API.getState())).active.id,target.id);
  assert.equal(await p.locator('[data-generation-authority-label]').innerText(),target.label);
  await p.evaluate(id=>window.AI_ASSISTANT_MODEL_API.selectModel(id),initial.active.id);
 }
});

test('shared generation authority stays synchronized with AI Assistant model state',async t=>{
 const p=await open(t,'/learn-ai/videos/new.html');
 assert.equal(await p.evaluate(()=>!!window.AI_ASSISTANT_MODEL_API),true);
 const initial=await p.evaluate(()=>window.AI_ASSISTANT_MODEL_API.getState());
 const models=await p.evaluate(()=>window.AI_ASSISTANT_MODEL_API.listModels());
 assert.ok(initial&&initial.active&&models.length>=2);
 assert.equal(await p.locator('[data-generation-authority-label]').innerText(),initial.active.label);
 const target=models.find(m=>m.id!==initial.active.id);assert.ok(target);
 await p.locator('[data-generation-authority-more]').click();
 assert.equal(await p.locator('[data-generation-authority-more]').getAttribute('aria-expanded'),'true');
 await p.locator('[data-generation-authority-menu] [data-model-id="'+target.id+'"]') .click();
 assert.equal((await p.evaluate(()=>window.AI_ASSISTANT_MODEL_API.getState())).active.id,target.id);
 assert.equal(await p.locator('[data-generation-authority-label]').innerText(),target.label);
 await p.locator('[data-generation-authority-open]').click();
 const sheet=p.locator('#ai-assistant-panel-model-sheet');assert.equal(await sheet.count(),1);
 await p.evaluate(id=>{const radio=[...document.querySelectorAll('#ai-assistant-panel-model-sheet input[type="radio"]')].find(r=>r.value===id);if(!radio)throw new Error('original model radio missing');radio.click();},initial.active.id);
 assert.equal((await p.evaluate(()=>window.AI_ASSISTANT_MODEL_API.getState())).active.id,initial.active.id);
 assert.equal(await p.locator('[data-generation-authority-label]').innerText(),initial.active.label);
 // The Assistant footer's compact “Try a different model” menu must use the
 // same canonical selection transaction as the full sheet and Learn picker.
 // This protects the rare path that previously only persisted the model id
 // without dispatching the shared model-change event.
 const assistantMore=p.locator('.ai-assistant-panel-inline-picker-more').first();
 await assistantMore.click();
 const assistantMenu=p.locator('.ai-assistant-menu.ai-assistant-panel-changed-file-menu[data-open="true"]');
 assert.equal(await assistantMenu.count(),1);
 const targetItem=assistantMenu.getByRole('menuitem').filter({hasText:target.label}).first();
 await targetItem.click();
 assert.equal((await p.evaluate(()=>window.AI_ASSISTANT_MODEL_API.getState())).active.id,target.id);
 assert.equal(await p.locator('[data-generation-authority-label]').innerText(),target.label);
 assert.match(await p.locator('.ai-assistant-panel-inline-model-picker').first().innerText(),new RegExp(target.label.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')));
 assert.ok(models.every(m=>!Object.prototype.hasOwnProperty.call(m,'endpoint')&&!Object.prototype.hasOwnProperty.call(m,'token')));
});

test('trending topics table filters and sorts catalog-backed signals',async t=>{
 const p=await open(t,'/learn-ai/topics/index.html');const table=p.locator('[data-trending-table]');assert.equal(await table.locator('.learn-topic-row').count(),16);
 const toggle=table.getByRole('button',{name:'More search options',exact:true});assert.equal(await toggle.getAttribute('aria-expanded'),'false');
 await toggle.click();assert.equal(await toggle.getAttribute('aria-expanded'),'true');
 await table.getByLabel('Sort by',{exact:true}).selectOption('topic');const visible=table.locator('.learn-topic-row:visible');assert.ok(await visible.count()===16);
 await table.getByLabel('Category',{exact:true}).selectOption('machine-learning');assert.equal(await table.locator('.learn-topic-row:visible').count(),16);
 await table.getByLabel('Search',{exact:true}).fill('Lasso');assert.ok(await table.locator('.learn-topic-row:visible').count()>=1);assert.ok(await table.locator('.learn-topic-row:visible').count()<16);assert.match(p.url(),/[?&]q=Lasso(?:&|$)/);await table.getByRole('button',{name:'Search',exact:true}).click();assert.ok(await table.locator('.learn-topic-row:visible').count()>=1);
});
test('trending topics layout fits desktop tablet and mobile containers without horizontal scrolling',async t=>{
 const p=await open(t,'/learn-ai/topics/index.html');
 for(const width of [1440,820,390]){
  await p.setViewportSize({width,height:900});
  const fit=await p.locator('.learn-table-scroll').evaluate(el=>({client:el.clientWidth,scroll:el.scrollWidth,row:el.querySelector('.learn-topic-row').getBoundingClientRect().width,display:getComputedStyle(el.querySelector('.learn-topic-row')).display}));
  assert.ok(fit.scroll<=fit.client+1,`table overflow at ${width}px: ${JSON.stringify(fit)}`);
  assert.ok(fit.row<=fit.client+1,`row overflow at ${width}px: ${JSON.stringify(fit)}`);
  if(width===390)assert.equal(fit.display,'grid');
 }
});

test('open problems sources and skills reuse responsive catalog table logic',async t=>{
 for(const spec of [
  {path:'/learn-ai/open-problems/index.html',sort:'references',count:8},
  {path:'/learn-ai/sources/index.html',sort:'publisher',count:12},
  {path:'/learn-ai/skills/index.html',sort:'sections',count:5},
 ]){
  const p=await open(t,spec.path);const table=p.locator('[data-catalog-table]');assert.equal(await table.locator('.learn-catalog-row').count(),spec.count);
  const toggle=table.getByRole('button',{name:'More search options',exact:true});assert.equal(await toggle.getAttribute('aria-expanded'),'false');await toggle.click();
  await table.getByLabel('Sort by',{exact:true}).selectOption(spec.sort);
  for(const width of [820,390]){
   await p.setViewportSize({width,height:900});
   const fit=await table.locator('.learn-table-scroll').evaluate(el=>({client:el.clientWidth,scroll:el.scrollWidth,row:el.querySelector('.learn-catalog-row').getBoundingClientRect().width,display:getComputedStyle(el.querySelector('.learn-catalog-row')).display}));
   assert.ok(fit.scroll<=fit.client+1,`${spec.path} overflow at ${width}px: ${JSON.stringify(fit)}`);
   assert.ok(fit.row<=fit.client+1,`${spec.path} row overflow at ${width}px: ${JSON.stringify(fit)}`);
   if(width===390)assert.equal(fit.display,'grid');
  }
 }
});
test('media indexes reuse the same compact live search and advanced controls',async t=>{
 for(const spec of [
  {path:'/learn-ai/videos/index.html',kind:'video',query:'Rickroll',hasItems:true},
  {path:'/learn-ai/audios/index.html',kind:'audio',query:'Audio',hasItems:false},
  {path:'/learn-ai/documents/index.html',kind:'document',query:'Document',hasItems:false},
  {path:'/learn-ai/whiteboards/index.html',kind:'whiteboard',query:'Whiteboard',hasItems:true},
 ]){
  const p=await open(t,spec.path), explorer=p.locator(`[data-learn-explorer][data-kind="${spec.kind}"]`);
  const controls=explorer.locator('[data-explorer-controls]');assert.equal(await controls.count(),1);
  assert.equal(await controls.locator('.learn-search-field').count(),1);assert.equal(await controls.getByRole('button',{name:'Search',exact:true}).count(),1);
  assert.equal(await controls.locator('[data-explorer-search-variant="pill-overflow"]').count(),1);assert.equal(await controls.locator('.learn-filter-overflow-icon').count(),1);
  const toggle=controls.getByRole('button',{name:'More search options',exact:true});assert.equal(await toggle.getAttribute('aria-expanded'),'false');
  const sizes=await controls.evaluate(el=>{const field=el.querySelector('.learn-search-field').getBoundingClientRect(),button=el.querySelector('.learn-filter-disclosure').getBoundingClientRect();return {fieldHeight:field.height,buttonHeight:button.height,buttonWidth:button.width,fieldRadius:getComputedStyle(el.querySelector('.learn-search-field')).borderRadius,buttonRadius:getComputedStyle(el.querySelector('.learn-filter-disclosure')).borderRadius};});
  assert.ok(Math.abs(sizes.fieldHeight-sizes.buttonHeight)<.6,`search/disclosure height mismatch: ${JSON.stringify(sizes)}`);assert.ok(Math.abs(sizes.buttonHeight-sizes.buttonWidth)<.6,`overflow button is not square: ${JSON.stringify(sizes)}`);assert.ok(parseFloat(sizes.fieldRadius)>=sizes.fieldHeight/2-1,`search field is not pill-shaped: ${JSON.stringify(sizes)}`);assert.ok(parseFloat(sizes.buttonRadius)>=sizes.buttonHeight/2-1,`overflow button is not circular: ${JSON.stringify(sizes)}`);
  await toggle.click();assert.equal(await toggle.getAttribute('aria-expanded'),'true');assert.equal(await controls.getByLabel('Timeframe',{exact:true}).count(),1);assert.equal(await controls.getByLabel('Sort by',{exact:true}).count(),1);assert.equal(await controls.locator('[data-explorer-direction]').count(),1);
  if(spec.hasItems){const before=await explorer.locator('.learn-entry:visible').count();await controls.getByLabel('Search',{exact:true}).fill(spec.query);const after=await explorer.locator('.learn-entry:visible').count();assert.ok(after>=1&&after<=before);assert.match(p.url(),new RegExp('[?&]q='+spec.query+'(?:&|$)'));}
  assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 }
});

test('HackerNews follows Tweets and stays honest when no discussion is cataloged',async t=>{
 const p=await open(t,filled);const tweets=p.locator('[data-section="tweets"]'),hn=p.locator('[data-section="hackernews"]');assert.equal(await tweets.evaluate(el=>Boolean(el.compareDocumentPosition(document.querySelector('[data-section="hackernews"]'))&Node.DOCUMENT_POSITION_FOLLOWING)),true);
 assert.match(await hn.innerText(),/No relevant Hacker News discussions have been added/);
});

test('evidence review selection causally constrains the next inline AI request',async t=>{
 const p=await open(t,filled);let seen=null;
 await p.route('**/v1/chat/completions',async route=>{seen=route.request().postDataJSON();await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({id:'evidence-context-test',choices:[{message:{content:'Draft generated without excluded evidence.'}}]})});});
 await p.evaluate(()=>{window.AI_ASSISTANT_MODEL_API={getState:()=>({active:{model:'stub-evidence',label:'Stub evidence model'}}),openPicker:()=>true,onChange:()=>()=>{}};window.AI_ASSISTANT_ENDPOINT_API={resolveEndpoint:key=>key==='chat'?location.origin+'/v1/chat/completions':''};});
 const s=p.locator('[data-section="summary"]');await s.getByRole('button',{name:'Review sources',exact:true}).click();const review=s.locator('[data-evidence-review]');await review.getByRole('button',{name:'Use none',exact:true}).click();await review.getByRole('button',{name:'Save review state',exact:true}).click();await review.getByRole('button',{name:'Close',exact:true}).click();
 await s.getByRole('button',{name:'Generate Now',exact:true}).click();await s.locator('[data-section-ai-panel]').getByRole('button',{name:'Generate Now',exact:true}).click();await s.locator('[data-section-ai-status]').filter({hasText:'AI draft ready for review'}).waitFor();
 assert.match(seen.context.page_text,/explicitly selected no attached references/);assert.doesNotMatch(seen.context.page_text,/Catalog source notes:/);
});

test('record-aware overview restores its generation profile and becomes a review/regenerate lifecycle',async t=>{
 const p=await open(t,filled);await p.route('**/v1/chat/completions',async route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({id:'overview-persist',choices:[{message:{content:'Persisted private overview.'}}]})}));
 await p.evaluate(()=>{window.AI_ASSISTANT_MODEL_API={getState:()=>({active:{model:'stub-overview-persist',label:'Stub overview'}}),openPicker:()=>true,onChange:()=>()=>{}};window.AI_ASSISTANT_ENDPOINT_API={resolveEndpoint:key=>key==='chat'?location.origin+'/v1/chat/completions':''};});
 await p.getByRole('button',{name:'Generate AI Overview',exact:true}).click();const panel=p.locator('[data-overview-generation]');await panel.locator('[data-overview-audience="young-learner"]').check();await panel.locator('[data-overview-purpose="research"]').check();await panel.getByLabel('Depth',{exact:true}).selectOption('deep');await panel.locator('[data-overview-instructions]').fill('Preserve uncertainty.');let copied='';await p.evaluate(()=>{navigator.clipboard.writeText=async value=>{window.__overviewCopied=value;};});await panel.getByRole('button',{name:'Copy request',exact:true}).click();copied=await p.evaluate(()=>window.__overviewCopied||'');const copiedRequest=JSON.parse(copied);assert.equal(copiedRequest.max_tokens,2800);assert.match(copiedRequest.user_message,/Depth: deep\./);await panel.getByRole('button',{name:'Generate AI Overview',exact:true}).click();await panel.locator('[data-overview-status]').filter({hasText:'AI overview ready'}).waitFor();assert.equal(await p.getByRole('button',{name:'Review AI Overview',exact:true}).count(),1);assert.equal(await panel.getByRole('button',{name:'Regenerate AI Overview',exact:true}).count(),1);
 await p.reload();assert.equal(await p.getByRole('button',{name:'Review AI Overview',exact:true}).count(),1);await p.getByRole('button',{name:'Review AI Overview',exact:true}).click();const restored=p.locator('[data-overview-generation]');assert.equal(await restored.locator('[data-overview-audience="young-learner"]').isChecked(),true);assert.equal(await restored.locator('[data-overview-purpose="research"]').isChecked(),true);assert.equal(await restored.getByLabel('Depth',{exact:true}).inputValue(),'deep');assert.equal(await restored.locator('[data-overview-instructions]').inputValue(),'Preserve uncertainty.');
});

test('AI text creation studios are shared across record kinds and retain bounded private results',async t=>{
 const p=await open(t,'/learn-ai/topics/new.html');const studios=[['topics','topic'],['sources','source'],['open-problems','problem'],['topic-prompts','topic-prompt'],['skills','skill']];for(const [pathName,kind] of studios){await p.goto(origin+'/learn-ai/'+pathName+'/new.html');assert.equal(await p.locator('[data-record-generation][data-record-kind="'+kind+'"]').count(),1);assert.ok(await p.locator('[data-generation-context-picker][data-generation-context-policy="composable"]').count()===1);assert.ok(await p.locator('[data-generation-context-record]').count()>=1);assert.ok(await p.locator('[data-record-audience]').count()>=9);assert.ok(await p.locator('[data-record-purpose]').count()>=5);assert.ok(await p.locator('[data-record-skill]').count()>=5);assert.ok(await p.locator('[data-record-role]').count()>=6);const advanced=p.locator('[data-generation-advanced="record"]');assert.equal(await advanced.count(),1);assert.equal(await advanced.locator('select').count(),1);assert.equal(await advanced.locator('input[type="checkbox"]').count(),2);assert.equal(await p.locator('[data-generation-authority-picker][data-generation-authority-kind="assistant"][data-generation-authority-managed="shared"]').count(),1);assert.equal(await p.locator('[data-generation-authority-open]').count(),1);assert.equal(await p.locator('[data-generation-authority-more]').count(),1);assert.equal(await p.locator('[data-generation-authority-menu]').count(),1);assert.equal(await p.locator('[data-generation-runtime-authority]').count(),1);const generate=p.getByRole('button',{name:'Generate Now',exact:true});assert.equal(await generate.count(),1);assert.equal(await generate.isEnabled(),true);assert.equal(await p.getByRole('button',{name:'Save draft',exact:true}).count(),1);assert.equal(await p.getByRole('button',{name:'Copy request',exact:true}).count(),1);const order=await p.evaluate(()=>{const form=document.querySelector('[data-record-generation-form]'),lenses=[...form.querySelectorAll('.learn-generation-step-index')].find(n=>n.textContent.trim()==='3')?.closest('fieldset'),advanced=form.querySelector('[data-generation-advanced="record"]'),authority=form.querySelector('.learn-generation-authority'),actions=form.querySelector('.learn-generation-action-bar'),nodes=[lenses,advanced,authority,actions];return nodes.every(Boolean)&&nodes.every((node,i)=>i===nodes.length-1||Boolean(node.compareDocumentPosition(nodes[i+1])&Node.DOCUMENT_POSITION_FOLLOWING));});assert.equal(order,true);const library=p.locator('[data-generation-library]');assert.equal(await library.count(),1);assert.deepEqual(await library.locator('[data-generation-library-filter] option').allTextContents(),['Active','Archived','All']);assert.equal(await library.locator('[data-generation-library-refresh]').isDisabled(),true);const expectedTitle={topic:'Your Topics',source:'Your Sources',problem:'Your Open Problems','topic-prompt':'Your Topic Prompts',skill:'Your Skills'}[kind];assert.equal(await library.getByRole('heading',{name:expectedTitle,exact:true}).count(),1);assert.match(await library.innerText(),/Private AI draft library/);}
 await p.goto(origin+'/learn-ai/topics/new.html');await p.route('**/v1/chat/completions',async route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({id:'topic-create',choices:[{message:{content:JSON.stringify({title:'Generated topic',summary:'A bounded draft.',domains:['testing'],sections:[{id:'summary',title:'Summary',body:'Private generated body.'}],evidence_gaps:['Need a real source.'],related_questions:['What evidence is missing?']})}}]})}));await p.evaluate(()=>{window.AI_ASSISTANT_MODEL_API={getState:()=>({active:{model:'stub-record',label:'Stub record model'}}),openPicker:()=>true,onChange:()=>()=>{}};window.AI_ASSISTANT_ENDPOINT_API={resolveEndpoint:key=>key==='chat'?location.origin+'/v1/chat/completions':''};});await p.locator('[data-record-brief]').fill('Create a testable topic draft.');await p.locator('[data-record-audience="decision-maker"]').check();await p.getByRole('button',{name:'Generate Now',exact:true}).click();await p.locator('[data-generation-status-message]').filter({hasText:'AI draft ready'}).waitFor();assert.match(await p.locator('[data-record-result-preview]').innerText(),/Generated topic/);const topicLibrary=p.locator('[data-generation-library]');assert.equal(await topicLibrary.locator('[data-generation-library-card]').count(),1);assert.match(await topicLibrary.innerText(),/Generated topic/);assert.match(await topicLibrary.innerText(),/Review required/);await p.reload();assert.match(await p.locator('[data-record-result-preview]').innerText(),/Generated topic/);assert.equal(await p.locator('[data-generation-library-card]').count(),1);
});
