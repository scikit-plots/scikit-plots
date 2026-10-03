/* Browser-local evidence review that controls AI generation context. */
(() => {
  'use strict';
  const one=(root,selector)=>root?.querySelector?.(selector)||null;
  const all=(root,selector)=>root?.querySelectorAll?[...root.querySelectorAll(selector)]:[];
  const MAX_NOTE=1000;
  const ASSESSMENTS=new Set(['not-reviewed','supports','partial','challenges','unclear']);
  const pages=new WeakMap();
  function readJson(node){try{return JSON.parse(node?.textContent||'{}');}catch{return{};}}
  function safeState(value,ids){
    const result={};if(!value||typeof value!=='object'||Array.isArray(value))return result;
    for(const id of ids){const row=value[id];if(!row||typeof row!=='object')continue;const assessment=ASSESSMENTS.has(row.assessment)?row.assessment:'not-reviewed';const note=typeof row.note==='string'?row.note.slice(0,MAX_NOTE):'';result[id]={use:row.use!==false,assessment,note};}
    return result;
  }
  function bindPage(page,data){
    const subject=data.subject||{},sourceRows=Array.isArray(data.evidence_sources)?data.evidence_sources:[],sourceMap=new Map(sourceRows.map(row=>[row.id,row]));
    const pageState={subject,sourceMap,sections:new Map()};pages.set(page,pageState);
    all(page,'.learn-section').forEach(section=>{
      const sectionId=String(section.dataset.section||''),panel=one(section,'[data-evidence-review]'),button=one(section,'[data-verify]');if(!sectionId)return;
      const evidence=one(section,'.learn-evidence'),rows=panel?all(panel,'[data-evidence-source-id]'):[];
      if(!panel||!rows.length){button?.remove();return;}
      const ids=rows.map(row=>String(row.dataset.evidenceSourceId||'')).filter(Boolean);
      const key='learn-evidence-review:v1:'+String(data.site_id||'default')+':'+String(subject.id||'record')+':'+String(data.revision||'unknown')+':'+sectionId;
      let state={};try{const raw=localStorage.getItem(key);if(raw&&raw.length<=64000)state=safeState(JSON.parse(raw),ids);}catch{}
      const status=one(panel,'[data-evidence-review-status]'),summary=one(panel,'[data-evidence-review-summary]');
      function rowState(row){const id=String(row.dataset.evidenceSourceId||''),saved=state[id]||{use:true,assessment:'not-reviewed',note:''};return{id,...saved};}
      function apply(){
        for(const row of rows){const current=rowState(row);const use=one(row,'[data-evidence-use]'),assessment=one(row,'[data-evidence-assessment]'),note=one(row,'[data-evidence-note]');if(use)use.checked=current.use;if(assessment)assessment.value=current.assessment;if(note)note.value=current.note;row.dataset.reviewState=current.assessment;}
        const assessed=ids.filter(id=>(state[id]?.assessment||'not-reviewed')!=='not-reviewed').length,used=ids.filter(id=>state[id]?.use!==false).length;
        if(summary)summary.textContent=`${assessed}/${ids.length} assessed · ${used} in AI context`;
        if(status)status.textContent=assessed?`${assessed} of ${ids.length} references have a local assessment. This is review metadata, not verification.`:'No local evidence assessment recorded yet.';
      }
      function capture(){const next={};for(const row of rows){const id=String(row.dataset.evidenceSourceId||'');if(!id)continue;next[id]={use:!!one(row,'[data-evidence-use]')?.checked,assessment:String(one(row,'[data-evidence-assessment]')?.value||'not-reviewed'),note:String(one(row,'[data-evidence-note]')?.value||'').slice(0,MAX_NOTE)};}return next;}
      function save(){state=capture();try{localStorage.setItem(key,JSON.stringify(state));if(status)status.textContent='Evidence review state saved in this browser. Selected references will shape later AI context.';}catch{if(status)status.textContent='Browser storage is unavailable; current evidence selections remain active for this page only.';}apply();}
      button?.addEventListener('click',()=>{panel.hidden=false;evidence&&(evidence.open=true);panel.scrollIntoView({block:'nearest'});one(panel,'select,textarea,input')?.focus();});
      one(panel,'[data-evidence-review-close]')?.addEventListener('click',()=>{panel.hidden=true;button?.focus();});
      one(panel,'[data-evidence-select-all]')?.addEventListener('click',()=>{all(panel,'[data-evidence-use]').forEach(input=>input.checked=true);state=capture();apply();});
      one(panel,'[data-evidence-clear-context]')?.addEventListener('click',()=>{all(panel,'[data-evidence-use]').forEach(input=>input.checked=false);state=capture();apply();});
      one(panel,'[data-evidence-save]')?.addEventListener('click',save);
      one(panel,'[data-evidence-reset]')?.addEventListener('click',()=>{if(!window.confirm('Reset the browser-local evidence review for this section?'))return;state={};try{localStorage.removeItem(key);}catch{}apply();});
      panel.addEventListener('change',event=>{if(event.target.matches('[data-evidence-use],[data-evidence-assessment]')){state=capture();apply();}});
      pageState.sections.set(sectionId,{getState:()=>state,getContext(){const selected=[];for(const id of ids){const local=state[id]||{use:true,assessment:'not-reviewed',note:''};if(local.use===false)continue;const source=sourceMap.get(id);if(source)selected.push({source,assessment:local.assessment,note:local.note});}return selected;},open(){button?.click();}});
      apply();
    });
    return pageState;
  }
  function contextFor(page,sectionId){const state=pages.get(page),section=state?.sections.get(String(sectionId||''));return section?section.getContext():null;}
  function open(page,sectionId){const state=pages.get(page),section=state?.sections.get(String(sectionId||''));section?.open();return !!section;}
  window.AI_LEARN_EVIDENCE_API={bindPage,contextFor,open};
  all(document,'[data-learn-page]').forEach(page=>{const node=one(page,'.learn-page-data');if(!node)return;bindPage(page,readJson(node));});
})();
