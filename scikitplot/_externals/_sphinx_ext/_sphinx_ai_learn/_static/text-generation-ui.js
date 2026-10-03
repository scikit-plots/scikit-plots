/* Shared AI Learn text-generation runtime + lifecycle helpers. */
(() => {
  'use strict';
  const CHAT_CONTRACT='scikitplot-chat-v1';
  const modelApi=()=>window.AI_ASSISTANT_MODEL_API||null;
  const endpointApi=()=>window.AI_ASSISTANT_ENDPOINT_API||null;
  const generationUi=()=>window.AI_LEARN_GENERATION_UI||null;

  function activeModel(){
    try{const api=modelApi(),state=api&&typeof api.getState==='function'?api.getState():null,active=state&&state.active;if(active&&(active.model||active.id))return{model:String(active.model||active.id),label:String(active.label||active.model||active.id),effort:state.effort||null};}catch{}
    try{const rows=Array.isArray(window.AI_ASSISTANT_CONFIG?.panelApiModels)?window.AI_ASSISTANT_CONFIG.panelApiModels:[];if(rows.length)return{model:String(rows[0].model||rows[0].id||''),label:String(rows[0].label||rows[0].model||rows[0].id||''),effort:null};}catch{}
    return{model:'',label:'Not selected',effort:null};
  }
  function chatEndpoint(){
    try{const api=endpointApi();if(api&&typeof api.resolveEndpoint==='function'){const direct=api.resolveEndpoint('chat');if(direct)return String(direct).replace(/\/+$/,'');}}catch{}
    try{const rows=Array.isArray(window.AI_ASSISTANT_CONFIG?.panelApiModels)?window.AI_ASSISTANT_CONFIG.panelApiModels:[];if(rows.length&&rows[0].endpoint)return String(rows[0].endpoint).replace(/\/+$/,'');}catch{}
    return'';
  }
  function extractReply(data){
    data=data&&typeof data==='object'?data:{};
    if(Array.isArray(data.choices)&&data.choices.length){const msg=data.choices[0]&&data.choices[0].message;if(msg&&typeof msg.content==='string'&&msg.content.trim())return msg.content.trim();}
    if(Array.isArray(data.content)){const value=data.content.filter(x=>x&&x.type==='text'&&typeof x.text==='string').map(x=>x.text).join('\n').trim();if(value)return value;}
    for(const key of ['reply','answer','text'])if(typeof data[key]==='string'&&data[key].trim())return data[key].trim();
    return'';
  }
  function canonicalRequest(input){
    input=input&&typeof input==='object'?input:{};
    if(input.contract&&input.contract!==CHAT_CONTRACT)throw new Error('Unsupported text-generation request contract.');
    const context=input.context&&typeof input.context==='object'?input.context:{};
    const rawMaxTokens=input.max_tokens==null?1400:input.max_tokens;
    if(typeof rawMaxTokens!=='number'||!Number.isInteger(rawMaxTokens)||rawMaxTokens<256||rawMaxTokens>32000)throw new Error('max_tokens must be an integer from 256 to 32000.');
    const maxTokens=rawMaxTokens;
    return{contract:CHAT_CONTRACT,model:String(input.model||''),user_message:String(input.user_message||'').slice(0,20000),context:{page_text:String(context.page_text||'').slice(0,32000),page_descriptor:String(context.page_descriptor||'AI Learn text generation').slice(0,1000)},max_tokens:maxTokens,stream:false};
  }
  async function runRequest(input,{signal,model:knownModel}={}){
    const endpoint=chatEndpoint(),request=canonicalRequest(input);
    if(!endpoint)throw new Error('The active Assistant profile has no chat endpoint.');
    if(!request.model)throw new Error('Select an Assistant model before generating.');
    const model=knownModel&&knownModel.model===request.model?knownModel:{model:request.model,label:request.model,effort:null};
    const transport=generationUi()?.fetchJson;
    if(typeof transport!=='function')throw new Error('Shared AI Learn runtime transport is unavailable.');
    const packet=await transport(endpoint,{method:'POST',headers:{'Content-Type':'application/json','Accept':'application/json'},body:JSON.stringify(request),signal},{label:'AI text generation',timeoutMs:120000,maxBytes:1024*1024});
    const res=packet.response,payload=packet.body||{};
    if(!res.ok)throw new Error(String(payload?.error?.message||payload?.detail||('HTTP '+res.status)));
    const reply=extractReply(payload);if(!reply)throw new Error('The AI runtime returned no text.');
    return{reply,payload,request,model,endpoint};
  }
  async function run({userMessage,pageText,pageDescriptor,maxTokens=1400,signal}){
    const model=activeModel();
    return runRequest({contract:CHAT_CONTRACT,model:model.model,user_message:userMessage,context:{page_text:pageText,page_descriptor:pageDescriptor},max_tokens:maxTokens,stream:false},{signal,model});
  }

  function workflowMessage(messages,key,fallback,context){
    const value=messages&&messages[key];
    if(typeof value==='function')return String(value(context||{})||fallback||'');
    if(typeof value==='string')return value;
    return String(fallback||'');
  }

  function createWorkflow(options){
    options=options&&typeof options==='object'?options:{};
    const runButton=options.runButton||null,cancelButton=options.cancelButton||null;
    const publishButtons=Array.isArray(options.publishButtons)?options.publishButtons.filter(Boolean):[];
    const messages=options.messages||{};
    const publication=options.publication&&typeof options.publication==='object'?options.publication:null;
    let controller=null,running=false,publishing=false;

    const status=(value,state='idle')=>{if(typeof options.status==='function')options.status(String(value||''),state);};
    const stage=value=>{if(value&&typeof options.stage==='function')options.stage(value);};
    const currentDraft=()=>typeof options.getDraft==='function'?options.getDraft():null;
    const readProfile=()=>typeof options.readProfile==='function'?options.readProfile():{};
    const validateProfile=profile=>generationUi()?.validateLensProfile?.(profile)||'';

    function sync(){
      const draft=currentDraft();
      if(runButton)runButton.disabled=running||publishing;
      if(cancelButton)cancelButton.hidden=!(running&&controller);
      const unavailable=!draft;
      for(const button of publishButtons)button.disabled=running||publishing||unavailable;
      try{if(typeof options.onSync==='function')options.onSync({running,publishing,draft});}catch{}
    }

    function requestFor(profile,startState){
      const request=typeof options.buildRequest==='function'?options.buildRequest(profile,startState):null;
      if(!request||typeof request!=='object')throw new Error('Generation request is unavailable.');
      return canonicalRequest(request);
    }

    async function copyRequest(){
      const profile=readProfile(),profileError=validateProfile(profile);
      if(profileError){status(profileError,'warning');return false;}
      let request;
      try{request=requestFor(profile);}catch(error){status(String(error?.message||error),'error');return false;}
      if(options.requireModel!==false&&!request.model){status(workflowMessage(messages,'copyModelMissing','Select an Assistant model before copying the generation request.'),'warning');return false;}
      try{await navigator.clipboard.writeText(JSON.stringify(request,null,2));status(workflowMessage(messages,'copySuccess','Generation request copied. No network request was sent.'),'success');return true;}
      catch{status(workflowMessage(messages,'copyFailure','Clipboard is unavailable. The draft settings remain editable.'),'warning');return false;}
    }

    async function generate(){
      if(running||publishing)return false;
      const profile=readProfile(),profileError=validateProfile(profile);
      if(profileError){status(profileError,'warning');return false;}
      let request,startState;
      try{
        startState=typeof options.captureRunState==='function'?options.captureRunState({profile}):undefined;
        request=requestFor(profile,startState);
      }catch(error){status(String(error?.message||error),'error');return false;}
      if(options.requireModel!==false&&!request.model){status(workflowMessage(messages,'generateModelMissing','Select an Assistant model before generating.'),'warning');return false;}
      if(typeof runRequest!=='function'){status(workflowMessage(messages,'runtimeUnavailable','AI generation runtime helpers are unavailable on this page.'),'warning');return false;}
      controller=typeof AbortController==='function'?new AbortController():null;
      running=true;sync();stage(options.generatingStage||'draft');
      status(workflowMessage(messages,'generating','Generating a private AI draft…',{request,profile,model:activeModel()}),'working');
      try{
        const response=await runRequest(request,{signal:controller?.signal});
        const requestedMax=Number(options.maxDraftChars??50000),maxChars=Number.isInteger(requestedMax)&&requestedMax>=1000&&requestedMax<=200000?requestedMax:50000;
        if(response.reply.length>maxChars)throw new Error(workflowMessage(messages,'oversize',`Generated text exceeds the ${maxChars.toLocaleString()}-character draft limit.`,{response,request,profile}));
        const context={response,request,profile,startState};
        const draft=typeof options.createDraft==='function'?await options.createDraft(context):response.reply;
        if(typeof options.saveDraft==='function')await options.saveDraft(draft,context);
        stage(options.generatedStage||'review');
        status(workflowMessage(messages,'generated','AI draft ready for human review.',{...context,draft}),'success');
        try{if(typeof options.onGenerated==='function')await options.onGenerated({...context,draft});}catch{}
        return true;
      }catch(error){
        if(error?.name==='AbortError')status(workflowMessage(messages,'cancelled','Generation cancelled. No draft was changed.'),'warning');
        else status(workflowMessage(messages,'generateFailed','Unable to generate: '+String(error?.message||error),{error,request,profile}),'error');
        return false;
      }finally{controller=null;running=false;sync();}
    }

    function cancel(){if(controller)controller.abort();}

    async function publish(){
      if(publishing||running)return false;
      const draft=currentDraft();
      if(!draft){status(workflowMessage(messages,'publishMissingDraft','Generate or restore a draft before opening a pull request.'),'warning');sync();return false;}
      if(!publication||typeof publication.buildRequest!=='function'){status('Publication workflow is unavailable.','error');return false;}
      const confirmation=workflowMessage(messages,'publishConfirm','Send this reviewed draft to the configured publication service for a JSON-only pull request?',{draft});
      if(confirmation&&!window.confirm(confirmation))return false;
      publishing=true;sync();
      const pubStatus=publication.statusNode||null,pubLink=publication.linkNode||null;
      if(pubStatus)pubStatus.textContent=workflowMessage(messages,'publishing','Sending reviewed draft for repository validation…',{draft});
      if(pubLink){pubLink.hidden=true;pubLink.replaceChildren();}
      try{
        const ui=generationUi(),contributor=ui?.publicationContributor?.(options.root)||{display_name:'Anonymous'};
        const publicationRequest=publication.buildRequest(draft,contributor);
        const receipt=await ui?.submitPublication?.(publicationRequest);
        if(!receipt)throw new Error('Publication transport is unavailable.');
        if(pubStatus)pubStatus.textContent=ui?.publicationReceiptMessage?.(receipt)||String(receipt.message||'Publication request completed.');
        if(pubLink)ui?.appendPublicationReceiptLink?.(pubLink,receipt);
        stage(options.publishedStage||'handoff');
        status(workflowMessage(messages,'published',receipt.mode==='stub'?'Publication validated in stub mode; no GitHub write occurred.':'Draft queued for repository validation and human review.',{draft,receipt}),'success');
        try{if(typeof options.onPublished==='function')await options.onPublished({draft,receipt});}catch{}
        return true;
      }catch(error){
        if(pubStatus)pubStatus.textContent=workflowMessage(messages,'publicationStatusFailed','Publication failed; this draft remains local. '+String(error?.message||error),{draft,error});
        status(workflowMessage(messages,'publishFailed','Unable to open publication review: '+String(error?.message||error),{draft,error}),'error');
        return false;
      }finally{publishing=false;sync();}
    }

    sync();
    return{copyRequest,generate,cancel,publish,sync,request:()=>requestFor(readProfile()),isRunning:()=>running,isPublishing:()=>publishing};
  }

  function openModelPicker(button){try{const api=modelApi();return !!(api&&typeof api.openPicker==='function'&&api.openPicker(button));}catch{return false;}}
  window.AI_LEARN_TEXT_GENERATION_API={CHAT_CONTRACT,activeModel,chatEndpoint,extractReply,canonicalRequest,runRequest,run,createWorkflow,openModelPicker};
  window.dispatchEvent(new CustomEvent('ai-learn-text-generation-api-ready'));
})();
