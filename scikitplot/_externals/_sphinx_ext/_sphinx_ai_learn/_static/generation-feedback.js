/* Reviewed feedback for accepted AI Learn section generations. */
(() => {
  'use strict';
  const one=(root,selector)=>root?.querySelector?.(selector)||null;
  const all=(root,selector)=>root?.querySelectorAll?[...root.querySelectorAll(selector)]:[];
  function pageData(root){
    const page=root.closest?.('[data-learn-page]')||document.querySelector('[data-learn-page]');
    try{return JSON.parse(one(page,'.learn-page-data')?.textContent||'{}');}catch{return{};}
  }
  function feedbackId(){
    const crypto=globalThis.crypto;
    if(!crypto?.getRandomValues)return '';
    // 192 random bits, with no timestamp, device data, account data, or stable
    // browser identifier. This nonce exists only to make one feedback event
    // retry-idempotent; it is not a participant identity.
    const bytes=new Uint8Array(24);crypto.getRandomValues(bytes);
    return 'feedback-'+[...bytes].map(value=>value.toString(16).padStart(2,'0')).join('');
  }
  function normalizeComment(value){
    const text=String(value||'').trim();
    if(text.length>2000)throw new Error('Optional details must be at most 2000 characters.');
    if(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(text))throw new Error('Optional details contain unsupported control characters.');
    return text;
  }
  function validTreeRevision(value){return /^tree-[0-9a-f]{16}$/.test(String(value||'').trim());}
  /*
   * Developer helper: compact community-count scale. Keep the formatter
   * deterministic across Sphinx themes and browser locales so the quick-action
   * row keeps stable geometry.
   *
   * Abbreviation   Full number
   * 1K             1,000
   * 10K            10,000
   * 100K           100,000
   * 1M             1,000,000
   * 10M            10,000,000
   * 1B             1,000,000,000
   * 1T             1,000,000,000,000
   *
   * 0..999 stay exact. K/M/B/T use at most one decimal below 100 units,
   * trimming trailing .0 (1K, 1.5K, 2.3M, 10K). If rounding reaches 1000,
   * promote to the next suffix so 999,500 becomes 1M rather than 1000K.
   */
  function formatCompactCount(value){
    const count=Number(value);
    if(!Number.isSafeInteger(count)||count<0)return '';
    if(count<1000)return String(count);
    const units=[[1e3,'K'],[1e6,'M'],[1e9,'B'],[1e12,'T']];
    let index=units.length-1;while(index>0&&count<units[index][0])index-=1;
    let divisor=units[index][0],suffix=units[index][1],scaled=count/divisor;
    let decimals=scaled<100?1:0,factor=decimals?10:1;
    let rounded=Math.round((scaled+Number.EPSILON)*factor)/factor;
    if(rounded>=1000&&index<units.length-1){
      index+=1;divisor=units[index][0];suffix=units[index][1];scaled=count/divisor;
      decimals=scaled<100?1:0;factor=decimals?10:1;
      rounded=Math.round((scaled+Number.EPSILON)*factor)/factor;
    }
    const text=decimals?rounded.toFixed(1).replace(/\.0$/,''):String(rounded);
    return text+suffix;
  }
  function bind(root){
    const section=root.closest('.learn-section'),data=pageData(root),subject=data.subject||{};
    const generationId=String(root.dataset.generationId||section?.dataset.activeGenerationId||'').trim();
    const sectionId=String(root.dataset.feedbackSectionId||section?.dataset.section||'').trim();
    if(!generationId||!sectionId||!subject.id)return;
    const quick=all(root,'[data-learn-feedback-quick]'),expand=one(root,'[data-learn-feedback-expand]'),detail=one(root,'[data-learn-feedback-detail]'),ratings=all(root,'[data-learn-feedback-rating]'),submit=one(root,'[data-learn-feedback-submit]'),comment=one(root,'[data-learn-feedback-comment]'),contributor=one(root,'[data-learn-feedback-contributor]'),status=one(root,'[data-learn-feedback-status]'),historyOrder=one(root,'[data-learn-generation-order]'),historyList=one(root,'[data-learn-generation-history-list]');
    all(root,'[data-learn-feedback-quick-count]').forEach(node=>{
      const count=Number(node.closest('[data-feedback-count]')?.dataset.feedbackCount);
      const compact=formatCompactCount(count);if(compact)node.textContent=compact;
    });
    const totalCount=one(root,'[data-generation-feedback-count]');
    if(totalCount){
      const count=Number(totalCount.dataset.feedbackCount);
      const compact=formatCompactCount(count);
      if(compact)totalCount.textContent=compact+' rating'+(count===1?'':'s');
    }
    const pendingKey='learn-ai-feedback-pending:v1:'+String(subject.id)+':'+sectionId+':'+generationId;
    const quickKey='learn-ai-feedback-quick:v1:'+String(subject.id)+':'+sectionId+':'+generationId;
    function readPending(){
      try{
        const row=JSON.parse(sessionStorage.getItem(pendingKey)||'null'),request=row?.request;
        const id=String(request?.feedback_id||'');
        const validId=/^feedback-[0-9a-f]{48}$/.test(id);
        const revision=String(request?.base_revision||'').trim();
        // A pending feedback retry may outlive an unrelated catalog rebuild.
        // Feedback targets immutable generation identity, and the repository
        // deliberately accepts a stale base revision while that generation
        // still exists. Preserve the original envelope instead of minting a
        // second event merely because the page revision changed.
        const valid=!!row&&typeof row.fingerprint==='string'&&request&&request.contract==='learn.publication-request.v1'&&request.action==='feedback'&&validTreeRevision(revision)&&request.subject_id===String(subject.id)&&request.section_id===sectionId&&request.generation_id===generationId&&validId;
        if(valid)return row;
        sessionStorage.removeItem(pendingKey);
        return null;
      }catch{try{sessionStorage.removeItem(pendingKey);}catch{}return null;}
    }
    function writePending(value){try{if(value)sessionStorage.setItem(pendingKey,JSON.stringify(value));else sessionStorage.removeItem(pendingKey);}catch{}}
    function readQuick(){try{const value=Number(sessionStorage.getItem(quickKey));return value===-1||value===1?value:null;}catch{return null;}}
    function writeQuick(value){try{if(value===-1||value===1)sessionStorage.setItem(quickKey,String(value));else sessionStorage.removeItem(quickKey);}catch{}}
    let selected=null,busy=false,pending=readPending();
    const priorQuick=readQuick();if(priorQuick!==null)quick.forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.learnFeedbackQuick)===priorQuick)));
    const announce=value=>{if(status)status.textContent=String(value||'');};
    function setBusy(value){busy=value;quick.forEach(button=>button.disabled=value);ratings.forEach(button=>button.disabled=value);if(submit)submit.disabled=value||selected===null;if(expand)expand.disabled=value;}
    function orderHistory(mode){
      if(!historyList)return;
      const rows=[...historyList.children];
      const score=row=>Number(row.dataset.generationHistoryScore||0);
      const ratingsCount=row=>Number(row.dataset.generationHistoryRatings||0);
      const created=row=>String(row.dataset.generationHistoryCreatedAt||'');
      const id=row=>String(row.dataset.generationHistoryId||'');
      const active=row=>row.dataset.generationHistoryActive==='true';
      rows.sort((a,b)=>{
        if(mode==='newest')return created(b).localeCompare(created(a))||id(a).localeCompare(id(b));
        if(mode==='rating')return score(b)-score(a)||ratingsCount(b)-ratingsCount(a)||created(b).localeCompare(created(a))||id(a).localeCompare(id(b));
        return Number(active(b))-Number(active(a))||score(b)-score(a)||created(b).localeCompare(created(a))||id(a).localeCompare(id(b));
      });
      rows.forEach(row=>historyList.appendChild(row));
    }
    function select(value){selected=value;ratings.forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.learnFeedbackRating)===value)));if(submit)submit.disabled=busy;}
    function pendingMatches(row,fingerprint,rating,clean,displayName,mode){
      const request=row?.request;
      if(!request||row.fingerprint!==fingerprint)return false;
      const expectedComment=clean||'';
      return request.contract==='learn.publication-request.v1'&&request.action==='feedback'&&validTreeRevision(request.base_revision)&&request.subject_id===String(subject.id)&&request.section_id===sectionId&&request.generation_id===generationId&&request.rating===rating&&request.feedback_mode===mode&&String(request.comment||'')===expectedComment&&String(request.contributor?.display_name||'')===displayName&&/^feedback-[0-9a-f]{48}$/.test(String(request.feedback_id||''));
    }
    async function send(rating,textValue='',credit='',mode='detailed'){
      if(busy)return;
      if(!Number.isInteger(rating)||rating < -5||rating > 5){announce('Choose a rating from -5 to +5.');return;}
      const ui=window.AI_LEARN_GENERATION_UI;
      if(!ui?.submitPublication){announce('Reviewed feedback transport is unavailable.');return;}
      let clean='';
      try{clean=normalizeComment(textValue);}
      catch(error){announce(String(error?.message||'Invalid optional details.'));return;}
      let displayName='';
      try{if(typeof ui.normalizePublicationCredit!=='function')throw new Error('Publication credit validator is unavailable.');displayName=ui.normalizePublicationCredit(credit);}
      catch(error){announce(String(error?.message||'Invalid public credit.'));return;}
      if(displayName.length>80){announce('Contributor credit must be plain text of at most 80 characters.');return;}
      const fingerprint=JSON.stringify([rating,clean,displayName,mode]);
      const pageRevision=String(data.revision||'').trim();
      if(!validTreeRevision(pageRevision)){announce('Reviewed feedback is unavailable because this page is missing a valid canonical revision.');return;}
      // Reuse the exact envelope after an ambiguous transport failure.  The
      // repository treats feedback_id as an idempotency key, so retrying cannot
      // double-count a request whose PR dispatch succeeded but whose response
      // was lost.  Editing the rating/comment/credit intentionally creates a
      // fresh event instead.
      if(!pendingMatches(pending,fingerprint,rating,clean,displayName,mode)){
        const id=feedbackId();
        if(!id){announce('Secure feedback event nonce is unavailable in this browser.');return;}
        const request={contract:'learn.publication-request.v1',action:'feedback',base_revision:pageRevision,subject_id:String(subject.id),section_id:sectionId,generation_id:generationId,feedback_id:id,rating,feedback_mode:mode,contributor:{display_name:displayName}};
        if(clean)request.comment=clean;
        pending={fingerprint,request};writePending(pending);
      }
      setBusy(true);announce('Sending feedback for repository review…');
      try{
        const receipt=await ui.submitPublication(pending.request);
        pending=null;writePending(null);
        if(receipt?.mode==='stub'){
          announce('Feedback validated in stub mode; no repository write occurred.');
        }else{
          announce('Feedback queued for review. Community ratings update only after merge and rebuild.');
        }
        if(receipt?.mode!=='stub'&&mode==='quick'){
          quick.forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.learnFeedbackQuick)===rating)));writeQuick(rating);
        }
        if(detail&&!detail.hidden){detail.hidden=true;if(expand)expand.setAttribute('aria-expanded','false');}
        if(comment)comment.value='';if(contributor)contributor.value='';select(null);
      }catch(error){announce('Unable to send feedback: '+String(error?.message||error)+' Retry will reuse the same feedback id.');}
      finally{setBusy(false);}
    }
    quick.forEach(button=>button.addEventListener('click',()=>{
      if(button.getAttribute('aria-pressed')==='true'){announce('This quick feedback is already selected for this page session.');return;}
      send(Number(button.dataset.learnFeedbackQuick),'','','quick');
    }));
    expand?.addEventListener('click',()=>{const open=detail?.hidden!==false;if(detail)detail.hidden=!open;expand.setAttribute('aria-expanded',String(open));if(open)ratings[5]?.focus();});
    ratings.forEach(button=>button.addEventListener('click',()=>select(Number(button.dataset.learnFeedbackRating))));
    submit?.addEventListener('click',()=>{if(selected===null)return;send(selected,comment?.value||'',contributor?.value||'','detailed');});
    historyOrder?.addEventListener('change',()=>orderHistory(String(historyOrder.value||'active')));
    orderHistory(String(historyOrder?.value||'active'));
    setBusy(false);
  }
  all(document,'[data-learn-generation-feedback]').forEach(bind);
})();
