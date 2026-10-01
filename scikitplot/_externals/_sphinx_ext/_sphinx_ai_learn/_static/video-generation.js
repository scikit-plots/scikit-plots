/* Source-aware video generation shell. Network execution is capability-gated. */
(() => {
  'use strict';
  const one=(root,selector)=>root?.querySelector?.(selector)||null;
  const all=(root,selector)=>root?.querySelectorAll?[...root.querySelectorAll(selector)]:[];
  const text=(tag,value)=>{const el=document.createElement(tag);el.textContent=value;return el;};
  const REQUEST_CONTRACT='learn.video-generation-request.v1';
  const JOB_CONTRACT='learn.video-generation-job.v1';
  const STATUSES=new Set(['draft','submitted','queued','running','ready','failed','cancelled','archived']);
  const ACTIVE_STATUSES=new Set(['submitted','queued','running']);
  const STAGE_LABELS={
    submitted:'Submitted',queued:'Waiting to start',preparing:'Preparing sources',grounding:'Grounding content',
    outline:'Building outline',script:'Writing script',visuals:'Creating visuals',rendering:'Rendering video',
    verifying:'Verifying & publishing',publishing:'Publishing',ready:'Ready',failed:'Needs attention',
    cancelled:'Cancelled',archived:'Archived'
  };
  const safeJson=node=>{try{return JSON.parse(node?.textContent||'{}');}catch{return null;}};
  const httpUrl=value=>{try{const u=new URL(String(value||''),location.href);if(u.username||u.password)return '';if(u.protocol==='https:')return u.href;if(u.protocol==='http:'&&['localhost','127.0.0.1','::1'].includes(u.hostname))return u.href;return '';}catch{return '';}};
  const nowIso=()=>new Date().toISOString();
  const randomId=()=>{try{return crypto.randomUUID();}catch{return 'vg-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2,10);}};
  const runtimeJson=(url,init,options)=>{const fn=window.AI_LEARN_GENERATION_UI?.fetchJson;if(typeof fn!=='function')return Promise.reject(new Error('Shared AI Learn runtime transport is unavailable.'));return fn(url,init,options);};

  all(document,'[data-video-generator]').forEach(root=>{
    const data=safeJson(one(root,'.learn-video-generator-data'));
    if(!data||data.contract!=='learn.video-generation-page.v1')return;
    const form=one(root,'[data-video-generation-form]');
    const runtimeStatus=one(root,'[data-video-runtime-status]');
    const readyEndpoint=one(root,'[data-video-ready-endpoint]');
    const readyGeneration=one(root,'[data-video-ready-generation]');
    const readyPublishing=one(root,'[data-video-ready-publishing]');
    const formStatus=one(root,'[data-video-form-status]');
    const generationStatus=window.AI_LEARN_GENERATION_UI?.bindGenerationStatus?.(root)||null;
    const submit=one(root,'[data-video-submit]');
    const refresh=one(root,'[data-generation-library-refresh]');
    const grid=one(root,'[data-generation-library-grid]');
    const empty=one(root,'[data-generation-library-empty]');
    const jobsStatus=one(root,'[data-generation-library-status]');
    const jobFilter=one(root,'[data-generation-library-filter]');
    const derived=one(root,'[data-video-derived-context]');
    const runtimeAuthority=one(root,'[data-video-runtime-authority]');
    const topicMap=new Map((data.topics||[]).map(row=>[row.id,row]));
    const sourceMap=new Map((data.sources||[]).map(row=>[row.id,row]));
    const videoMap=new Map((data.videos||[]).map(row=>[row.id,row]));
    const context=window.AI_LEARN_CONTEXT_API?.get?.(root)||null;
    const storagePrefix='learn-video:v2:'+String(data.site_id||'default')+':';
    const draftKey=storagePrefix+'draft';
    const jobsKey=storagePrefix+'jobs';
    window.AI_LEARN_GENERATION_UI?.bindPublicationCredit?.(root,{storageKey:'learn-publication-credit:v1:'+String(data.site_id||'default')+':video'});
    let endpoint='';
    let runtimeEnabled=false;
    let runtimeTestMode=false;
    let runtimePublishesMedia=false;
    let runtimePublishProvider='';
    let runtimeActions=new Set(['cancel','retry','archive','restore']);
    let activeProfile='';
    let idempotencyKey='';
    let idempotencySignature='';
    let pollTimer=0;
    let discoverySerial=0;
    let activeGenerationId='';

    function announce(node,message){if(node)node.textContent=message;}
    function announceGeneration(message,state='idle',title){if(generationStatus)generationStatus.set(state,message,title);else announce(formStatus,message);}
    function setReadiness(node,label,state='pending'){
      if(!node)return;node.dataset.state=state;const strong=one(node,'strong');if(strong)strong.textContent=label;
    }
    function resetRuntimeUi(){
      runtimeEnabled=false;runtimeTestMode=false;runtimePublishesMedia=false;runtimePublishProvider='';runtimeActions=new Set(['cancel','retry','archive','restore']);
      endpoint='';submit.textContent='Generate Now';submit.title='Generate Now will validate the configured Video runtime when clicked.';refresh.disabled=true;
      setReadiness(readyEndpoint,'Checking','pending');setReadiness(readyGeneration,'Checking','pending');setReadiness(readyPublishing,'Checking','pending');if(runtimeAuthority)runtimeAuthority.textContent='Discovering…';
    }
    function selectedMode(){return context?.activeType?.()||'topic';}
    function selectedTopic(){return topicMap.get(context?.singleId?.('topic')||'')||null;}
    function selectedSource(){return sourceMap.get(context?.singleId?.('source')||'')||null;}
    function sourceSnapshot(row){return row?{id:row.id,title:row.title,summary:row.summary||'',publisher:row.publisher||'',format:row.format||'',url:row.url||''}:null;}
    function topicSnapshot(row){
      if(!row)return null;
      const relatedSources=(row.related||[]).map(id=>sourceMap.get(id)).filter(Boolean).map(sourceSnapshot);
      return {id:row.id,title:row.title,summary:row.summary||'',domains:row.domains||[],sources:relatedSources};
    }
    function modelSnapshot(){
      return window.AI_LEARN_GENERATION_UI?.assistantModelSnapshot?.()||null;
    }
    function modelSignature(){const snap=modelSnapshot();return snap?JSON.stringify(snap):'runtime-default';}
    function cleanUrlInput(){
      const raw=String(context?.url?.()||'').trim();
      if(!raw)return '';
      try{const u=new URL(raw);if(u.protocol!=='https:')return '';if(u.username||u.password||u.hash)return '';return u.href;}
      catch{return '';}
    }
    function formState(){
      return {
        context:context?.snapshot?.()||{},
        lenses:window.AI_LEARN_GENERATION_UI?.readLensProfile?.(root,'video')||{},
        instructions:String(form.elements.instructions?.value||''),style:form.elements.style?.value||'explainer',
        length:form.elements.length?.value||'standard',language:form.elements.language?.value||'en',voice:form.elements.voice?.value||'default',
        aspect_ratio:form.elements.aspect_ratio?.value||'16:9',captions:!!form.elements.captions?.checked,branding:!!form.elements.branding?.checked,
        derived_from_video_id:derived.dataset.videoId||''
      };
    }
    function restoreState(state){
      if(!state||typeof state!=='object')return;
      context?.restore?.(state.context||state);
      window.AI_LEARN_GENERATION_UI?.restoreLensProfile?.(root,'video',state.lenses);
      for(const name of ['instructions','style','length','language','voice','aspect_ratio']){
        const field=form.elements[name];if(field&&typeof state[name]==='string')field.value=state[name];
      }
      if(form.elements.captions&&typeof state.captions==='boolean')form.elements.captions.checked=state.captions;
      if(form.elements.branding&&typeof state.branding==='boolean')form.elements.branding.checked=state.branding;
    }
    function applyQuery(){
      const params=new URLSearchParams(location.search),id=params.get('id')||'',from=params.get('from');
      context?.applyQuery?.(params);
      if(from==='video'&&videoMap.has(id)){
        const video=videoMap.get(id);derived.hidden=false;derived.dataset.videoId=id;derived.replaceChildren();
        const strong=text('strong','Variant of: '+video.title);const link=text('a','Open existing video');link.href=video.href;derived.append(strong,text('span',' · '),link);
        const relatedTopic=(video.related||[]).find(target=>topicMap.has(target));
        const relatedSource=(video.related||[]).find(target=>sourceMap.has(target));
        if(relatedTopic)context?.selectId?.('topic',relatedTopic);
        else if(relatedSource)context?.selectId?.('source',relatedSource);
        else context?.setActive?.('prompt');
      }
    }
    function requestBody(){
      const state=formState(),ctx=state.context||{};const mode=ctx.active||selectedMode();
      const input={mode};
      if(mode==='topic')input.topic=topicSnapshot(selectedTopic());
      else if(mode==='source')input.source=sourceSnapshot(selectedSource());
      else if(mode==='url')input.url=cleanUrlInput();
      else input.prompt=String(ctx.prompt||'').trim();
      const body={
        contract:REQUEST_CONTRACT,client_request_id:idempotencyKey||randomId(),
        input,instructions:window.AI_LEARN_GENERATION_UI?.withLensGuidance?.(state.instructions,state.lenses,4000)||state.instructions.trim(),
        presentation:{style:state.style,length:state.length,language:state.language,voice:state.voice,aspect_ratio:state.aspect_ratio,captions:state.captions,branding:state.branding},
        model_selection:modelSnapshot(),
        provenance:{site_id:data.site_id,catalog_revision:data.revision,source_page:location.href.split('#')[0]},
      };
      if(state.derived_from_video_id)body.derived_from_video_id=state.derived_from_video_id;
      return body;
    }
    function requestLabel(body){
      if(body.input.topic)return body.input.topic.title;
      if(body.input.source)return body.input.source.title;
      if(body.input.url)return body.input.url;
      return (body.input.prompt||'Untitled video').slice(0,120);
    }
    function validateRequest(body){
      const lensError=window.AI_LEARN_GENERATION_UI?.validateLensProfile?.(formState().lenses)||'';if(lensError)return lensError;
      const mode=body?.input?.mode;
      if(mode==='topic'&&!body.input.topic)return 'Choose a Topic.';
      if(mode==='source'&&!body.input.source)return 'Choose a Source.';
      if(mode==='url'&&!body.input.url)return 'Enter a public HTTPS URL without credentials or fragments.';
      if(mode==='prompt'&&!String(body.input.prompt||'').trim())return 'Enter a prompt.';
      return '';
    }
    function signature(){return JSON.stringify({form:formState(),model:modelSnapshot()});}
    function ensureIdempotency(){const sig=signature();if(!idempotencyKey||sig!==idempotencySignature){idempotencyKey=randomId();idempotencySignature=sig;}return idempotencyKey;}
    const requestActions=window.AI_LEARN_GENERATION_UI?.bindRequestActions?.(root,{
      storageKey:draftKey,snapshot:formState,restore:restoreState,
      buildRequest:()=>{ensureIdempotency();return requestBody();},validateRequest,
      announce:(message,state,title)=>announceGeneration(message,state,title)
    })||null;
    form.addEventListener('input',()=>{if(signature()!==idempotencySignature)idempotencyKey='';});
    form.addEventListener('change',()=>{if(signature()!==idempotencySignature)idempotencyKey='';});
    let volatileJobs=[];
    function boundedJobs(rows){return Array.isArray(rows)?rows.map(row=>normalizeJob(row)).filter(Boolean).slice(0,50):[];}
    function readJobs(){
      try{
        const raw=localStorage.getItem(jobsKey);
        if(!raw)return volatileJobs.slice();
        if(raw.length>250000)return volatileJobs.slice();
        const parsed=JSON.parse(raw),jobs=boundedJobs(parsed);volatileJobs=jobs;return jobs.slice();
      }catch{return volatileJobs.slice();}
    }
    function writeJobs(jobs){
      const normalized=boundedJobs(jobs);volatileJobs=normalized;
      try{const encoded=JSON.stringify(normalized);if(encoded.length>250000)return false;localStorage.setItem(jobsKey,encoded);return true;}catch{return false;}
    }
    function normalizeJob(raw,fallbackTitle='Generated video'){
      if(!raw||typeof raw!=='object')return null;
      const id=String(raw.generation_id||raw.id||'').trim();if(!/^[A-Za-z0-9._:-]{1,128}$/.test(id))return null;
      const status=STATUSES.has(String(raw.status||''))?String(raw.status):'submitted';
      const progress=Number(raw.progress);const bounded=Number.isFinite(progress)?Math.max(0,Math.min(1,progress)):null;
      const result=raw.result&&typeof raw.result==='object'?{
        provider:String(raw.result.provider||''),provider_id:String(raw.result.provider_id||''),url:httpUrl(raw.result.url||''),
        thumbnail_url:httpUrl(raw.result.thumbnail_url||raw.result.poster||''),duration_seconds:Number(raw.result.duration_seconds)||0,test_mode:raw.result.test_mode===true,
      }:null;
      const execution=raw.execution&&typeof raw.execution==='object'?{pipeline_version:String(raw.execution.pipeline_version||''),planner_model:String(raw.execution.planner_model||''),renderer:String(raw.execution.renderer||''),publish_provider:String(raw.execution.publish_provider||''),fallback_used:raw.execution.fallback_used===true}:null;
      return {contract:JOB_CONTRACT,generation_id:id,status,stage:String(raw.stage||status),progress:bounded,
        title:String(raw.title||fallbackTitle).slice(0,200),created_at:String(raw.created_at||nowIso()),updated_at:String(raw.updated_at||nowIso()),
        error:raw.error&&typeof raw.error==='object'?{code:String(raw.error.code||''),message:String(raw.error.message||'Generation could not complete.').slice(0,500),retryable:raw.error.retryable!==false}:null,
        result,execution};
    }
    function mergeJob(job){if(!job)return false;const jobs=readJobs().filter(row=>row.generation_id!==job.generation_id);jobs.unshift(job);const persisted=writeJobs(jobs);renderJobs();syncGenerationStatus(job);return persisted;}
    function visibleJobs(){const filter=jobFilter?.value||'active';return readJobs().filter(job=>filter==='all'||(filter==='archived'?job.status==='archived':job.status!=='archived'));}
    function stageLabel(job){return STAGE_LABELS[job.stage]||STAGE_LABELS[job.status]||job.stage||job.status;}
    function syncGenerationStatus(job){
      if(!job||!activeGenerationId||job.generation_id!==activeGenerationId)return;
      if(job.status==='ready'){
        announceGeneration(job.result?.test_mode?'Test video generation completed. No media was published.':'Video is ready for review.','success');
        activeGenerationId='';
        return;
      }
      if(job.status==='failed'){
        announceGeneration(job.error?.message||'Video generation could not complete.','error');
        activeGenerationId='';
        return;
      }
      if(job.status==='cancelled'){
        announceGeneration('Video generation was cancelled.','warning');
        activeGenerationId='';
        return;
      }
      if(ACTIVE_STATUSES.has(job.status)){
        const progress=formatProgress(job.progress);const detail=stageLabel(job)+(progress?' · '+progress:'');
        announceGeneration(detail,'working');
      }
    }
    function formatProgress(value){return typeof value==='number'?Math.round(value*100)+'%':'';}
    function actionButton(label,handler){const b=text('button',label);b.type='button';b.addEventListener('click',handler);return b;}
    async function lifecycleAction(job,action){
      if(!runtimeEnabled||!endpoint){announce(jobsStatus,'The video runtime is not enabled.');return;}
      if(!runtimeActions.has(action)){announce(jobsStatus,'The active video runtime does not advertise the '+action+' action.');return;}
      const target=endpoint+'/'+encodeURIComponent(job.generation_id)+'/'+action;
      try{
        const packet=await runtimeJson(target,{method:'POST',headers:{'Accept':'application/json'}},{label:'Video lifecycle action',timeoutMs:20000,maxBytes:256*1024});
        const res=packet.response,body=packet.body||{};
        if(!res.ok)throw new Error(String(body.detail||body.error||('HTTP '+res.status)));
        const optimisticStatus={archive:'archived',restore:(job.result?.url?'ready':'queued'),cancel:'cancelled',retry:'queued'}[action]||job.status;
        const next=normalizeJob(body,job.title)||{...job,status:optimisticStatus,stage:optimisticStatus,updated_at:nowIso()};
        const persisted=mergeJob(next);announce(jobsStatus,action[0].toUpperCase()+action.slice(1)+' request accepted.'+(persisted?'':' This receipt is kept only in this tab because browser storage is unavailable.'));
      }
      catch(error){announce(jobsStatus,'Unable to '+action+' this generation: '+String(error.message||error));}
    }
    function renderJob(job){
      const card=document.createElement('article');card.className='learn-generation-library-card';card.dataset.status=job.status;
      const media=document.createElement('div');media.className='learn-generation-library-preview';
      if(job.result?.thumbnail_url){const img=document.createElement('img');img.src=job.result.thumbnail_url;img.alt='';img.loading='lazy';media.append(img);}
      else{const spinner=document.createElement('div');spinner.className='learn-generation-library-spinner';spinner.setAttribute('aria-hidden','true');for(let i=0;i<8;i++)spinner.append(document.createElement('span'));media.append(spinner);}
      const body=document.createElement('div');body.className='learn-generation-library-body';
      const h=text('h3',job.title);const meta=text('p',stageLabel(job)+(formatProgress(job.progress)?' · '+formatProgress(job.progress):''));meta.className='learn-meta';
      body.append(h,meta);
      if(job.progress!==null&&ACTIVE_STATUSES.has(job.status)){const progress=document.createElement('progress');progress.max=1;progress.value=job.progress;progress.setAttribute('aria-label','Generation progress');body.append(progress);}
      if(job.error){const err=text('p',job.error.message);err.className='learn-generation-library-error';body.append(err);}
      if(job.status==='ready'&&job.result?.url){const open=text('a','Open video');open.className='learn-button learn-primary';open.href=job.result.url;open.target='_blank';open.rel='noopener noreferrer';body.append(open);}
      else if(job.status==='ready'&&job.result?.test_mode){const note=text('p','Test backend completed the lifecycle. No media was published.');note.className='learn-meta';body.append(note);}
      const menu=document.createElement('details');menu.className='learn-generation-library-menu';const summary=text('summary','⋯');summary.setAttribute('aria-label','More options for '+job.title);summary.title='More options';menu.append(summary);
      const menuBody=document.createElement('div');menuBody.className='learn-generation-library-menu-body';
      if(ACTIVE_STATUSES.has(job.status)&&runtimeActions.has('cancel'))menuBody.append(actionButton('Cancel',()=>lifecycleAction(job,'cancel')));
      if(job.status==='failed'&&job.error?.retryable!==false&&runtimeActions.has('retry'))menuBody.append(actionButton('Retry',()=>lifecycleAction(job,'retry')));
      if(job.status==='archived'&&runtimeActions.has('restore'))menuBody.append(actionButton('Restore',()=>lifecycleAction(job,'restore')));
      else if(!ACTIVE_STATUSES.has(job.status)&&runtimeActions.has('archive')){
        const archive=actionButton('Archive',()=>{
          menu.open=false;confirm.hidden=false;confirm.querySelector('button')?.focus();
        });menuBody.append(archive);
      }
      if(job.result?.url){const copy=actionButton('Copy URL',async()=>{try{await navigator.clipboard.writeText(job.result.url);announce(jobsStatus,'Video URL copied.');}catch{announce(jobsStatus,'Clipboard unavailable.');}});menuBody.append(copy);}
      menu.append(menuBody);
      const confirm=document.createElement('div');confirm.className='learn-generation-library-archive-confirmation';confirm.hidden=true;confirm.setAttribute('role','group');confirm.setAttribute('aria-label','Archive this video?');confirm.append(text('p','Archive this video?'));
      const confirmActions=document.createElement('div');confirmActions.className='learn-actions';const yes=actionButton('Archive',()=>lifecycleAction(job,'archive'));yes.className='learn-destructive';const no=actionButton('Cancel',()=>{confirm.hidden=true;menu.open=true;summary.focus();});confirmActions.append(yes,no);confirm.append(confirmActions);
      card.append(media,body,menu,confirm);return card;
    }
    function renderJobs(){const jobs=visibleJobs();grid.replaceChildren(...jobs.map(renderJob));empty.hidden=jobs.length>0;announce(jobsStatus,jobs.length+' video generation'+(jobs.length===1?'':'s')+' shown from this browser.');}
    jobFilter?.addEventListener('change',renderJobs);
    const handleJobsStorage=event=>{if(event.key!==jobsKey)return;volatileJobs=[];renderJobs();};
    window.addEventListener('storage',handleJobsStorage);

    async function fetchJob(job){
      if(!runtimeEnabled||!endpoint)return null;
      try{const packet=await runtimeJson(endpoint+'/'+encodeURIComponent(job.generation_id),{headers:{'Accept':'application/json'}},{label:'Video status',timeoutMs:15000,maxBytes:256*1024});const res=packet.response;if(!res.ok)return null;return normalizeJob(packet.body,job.title);}
      catch{return null;}
    }
    async function refreshJobs(){
      if(!runtimeEnabled||!endpoint){announce(jobsStatus,'The configured runtime does not advertise video generation yet.');return;}
      refresh.disabled=true;
      try{
        const jobs=readJobs();let changed=false;
        for(const job of jobs){if(job.status==='archived')continue;const live=await fetchJob(job);if(live){const index=jobs.findIndex(row=>row.generation_id===live.generation_id);jobs[index]=live;syncGenerationStatus(live);changed=true;}}
        if(changed&&!writeJobs(jobs))announce(jobsStatus,'Updated video receipts are kept only in this tab because browser storage is unavailable.');renderJobs();
      }finally{refresh.disabled=false;}
    }
    refresh?.addEventListener('click',refreshJobs);
    function schedulePolling(){clearInterval(pollTimer);if(!runtimeEnabled)return;pollTimer=setInterval(()=>{if(document.visibilityState==='visible'&&readJobs().some(job=>ACTIVE_STATUSES.has(job.status)))refreshJobs();},5000);}

    async function discoverRuntime(){
      const serial=++discoverySerial;
      resetRuntimeUi();
      if(data.runtime!=='assistant'){setReadiness(readyEndpoint,'Disabled','off');setReadiness(readyGeneration,'Disabled','off');setReadiness(readyPublishing,'Disabled','off');if(runtimeAuthority)runtimeAuthority.textContent='Disabled';announce(runtimeStatus,'Video generation runtime is disabled for this site. You can still save or copy a request draft.');return;}
      const api=window.AI_ASSISTANT_ENDPOINT_API;
      if(!api||typeof api.resolveEndpoint!=='function'){setReadiness(readyEndpoint,'Unavailable','off');setReadiness(readyGeneration,'Unavailable','off');setReadiness(readyPublishing,'Unavailable','off');if(runtimeAuthority)runtimeAuthority.textContent='Unavailable';announce(runtimeStatus,'AI Assistant endpoint discovery is unavailable. You can still save or copy a request draft.');return;}
      endpoint=String(api.resolveEndpoint('video')||'');activeProfile=String(api.getActiveProfile?.()||'');const profile=api.getProfile?.(activeProfile)||null;
      if(!endpoint){setReadiness(readyEndpoint,'Missing','off');setReadiness(readyGeneration,'Disabled','off');setReadiness(readyPublishing,'Disabled','off');if(runtimeAuthority)runtimeAuthority.textContent='Unavailable';announce(runtimeStatus,'The active AI endpoint profile does not define a video-generation route.');return;}
      setReadiness(readyEndpoint,'Ready','ready');
      if(profile?.video){runtimeEnabled=true;refresh.disabled=false;submit.title='Generate video through the explicit endpoint in the active AI profile.';setReadiness(readyGeneration,'Explicit route','ready');setReadiness(readyPublishing,'Runtime-defined','pending');if(runtimeAuthority)runtimeAuthority.textContent='Server-selected · explicit route';announce(runtimeStatus,'Video generation is enabled by the explicit endpoint in '+(profile.label||activeProfile)+'. Capability discovery is bypassed for this operator-configured route.');schedulePolling();return;}
      const base=String(profile?.base||'').replace(/\/+$/,'');
      if(!base){setReadiness(readyGeneration,'Unverified','pending');setReadiness(readyPublishing,'Unknown','pending');if(runtimeAuthority)runtimeAuthority.textContent='Unverified';announce(runtimeStatus,'A video endpoint can be derived, but no service base is available for capability discovery.');return;}
      try{
        const packet=await runtimeJson(base+'/',{headers:{'Accept':'application/json'}},{label:'Video capability discovery',timeoutMs:12000,maxBytes:512*1024});const res=packet.response;if(!res.ok)throw new Error('HTTP '+res.status);
        const discovery=packet.body||{};if(serial!==discoverySerial)return;const cap=discovery?.capabilities?.video_generation;
        const contractOk=String(cap?.contract||'')===REQUEST_CONTRACT&&String(cap?.job_contract||'')===JOB_CONTRACT;
        if(cap?.enabled===true&&contractOk){
          runtimeEnabled=true;runtimeTestMode=cap.test_mode===true;runtimePublishesMedia=cap.publishes_media===true;runtimePublishProvider=String(cap.publish_provider||'');if(runtimeAuthority)runtimeAuthority.textContent=runtimeTestMode?'Stub lifecycle · no media':(String(cap.mode||'upstream')==='upstream'?'Upstream video service':'Server-selected video service');runtimeActions=new Set(Array.isArray(cap.actions)?cap.actions.filter(action=>['cancel','retry','archive','restore'].includes(action)):[]);
          refresh.disabled=false;setReadiness(readyGeneration,runtimeTestMode?'Test backend':'Ready',runtimeTestMode?'test':'ready');
          if(runtimeTestMode){submit.textContent='Generate Now';submit.title='Generate through the diagnostic video lifecycle without publishing media.';setReadiness(readyPublishing,'No publish','test');announce(runtimeStatus,'Video generation test mode is active. Generate Now is enabled for lifecycle testing; no video will be published.');}
          else{submit.textContent='Generate Now';submit.title='Generate a video with the active runtime.';setReadiness(readyPublishing,runtimePublishesMedia?(runtimePublishProvider||'Ready'):'Not advertised',runtimePublishesMedia?'ready':'pending');announce(runtimeStatus,'Video generation is available through '+(profile?.label||activeProfile||'the active runtime')+'.'+(runtimePublishesMedia?' Publishing: '+(runtimePublishProvider||'runtime-defined')+'.':' Publishing is not advertised by this runtime.'));}
          schedulePolling();return;
        }
        setReadiness(readyGeneration,cap?.enabled===false?'Disabled':'Contract mismatch','off');setReadiness(readyPublishing,'Disabled','off');if(runtimeAuthority)runtimeAuthority.textContent='Unavailable';announce(runtimeStatus,'The active runtime is reachable, but compatible video generation is not enabled yet. Generate Now remains available to re-check; draft and copy controls also remain available.');
      }catch(error){if(serial!==discoverySerial)return;setReadiness(readyGeneration,'Unverified','pending');setReadiness(readyPublishing,'Unknown','pending');if(runtimeAuthority)runtimeAuthority.textContent='Unverified';announce(runtimeStatus,'The active runtime could not be verified for video generation. Generate Now remains available to re-check; draft and copy controls also remain available.');}
    }


    form.addEventListener('submit',async event=>{
      event.preventDefault();ensureIdempotency();const body=requestBody();const error=validateRequest(body);if(error){announceGeneration(error,'warning');return;}
      if(!runtimeEnabled||!endpoint){await discoverRuntime();if(!runtimeEnabled||!endpoint){announceGeneration('Video generation is not ready in the active runtime. Check Endpoint / Generation above, or save and copy the request while the runtime is configured.','warning');return;}}
      submit.disabled=true;announceGeneration('Submitting video generation…','working');
      try{
        const packet=await runtimeJson(endpoint,{method:'POST',headers:{'Content-Type':'application/json','Accept':'application/json','Idempotency-Key':idempotencyKey},body:JSON.stringify(body)},{label:'Video generation',timeoutMs:45000,maxBytes:512*1024});
        const res=packet.response,raw=packet.body||{};if(!res.ok)throw new Error(String(raw.detail||raw.error||('HTTP '+res.status)));
        const job=normalizeJob(raw,requestLabel(body));if(!job)throw new Error('The runtime returned an invalid generation receipt.');
        activeGenerationId=job.generation_id;const persisted=mergeJob(job);requestActions?.saveDraft(true);if(activeGenerationId){const baseMessage=runtimeTestMode?'Test generation accepted. No media will be published; lifecycle progress appears in Your Videos below.':'Generation accepted. Progress appears in Your Videos below.';announceGeneration(baseMessage+(persisted?'':' The receipt is kept only in this tab because browser storage is unavailable; keep this page open until generation finishes.'),'working','Queued');}schedulePolling();
      }catch(error){announceGeneration('Video generation could not start: '+String(error.message||error),'error');}
      finally{submit.disabled=false;}
    });

    try{requestActions?.loadDraft();}catch{}
    applyQuery();renderJobs();discoverRuntime();
    const endpointApi=window.AI_ASSISTANT_ENDPOINT_API;
    const unsubscribeProfile=endpointApi?.onProfileChange?.(()=>discoverRuntime());
    const dispose=window.AI_LEARN_GENERATION_UI?.onPageDispose||((callback)=>window.addEventListener('pagehide',function handler(event){if(event?.persisted===true)return;window.removeEventListener('pagehide',handler);callback();}));
    dispose(()=>{window.removeEventListener('storage',handleJobsStorage);clearInterval(pollTimer);if(typeof unsubscribeProfile==='function')unsubscribeProfile();});
  });
})();
