/* Shared Topic / Source / URL / Prompt context chooser for AI Learn studios. */
(() => {
  'use strict';
  const TYPES=['topic','source','url','prompt'];
  const one=(root,selector)=>root?.querySelector?.(selector)||null;
  const all=(root,selector)=>root?.querySelectorAll?[...root.querySelectorAll(selector)]:[];
  const clean=value=>String(value||'').trim();

  function bind(root){
    if(!root||root._aiLearnContext)return root?._aiLearnContext||null;
    const policy=root.dataset.generationContextPolicy==='composable'?'composable':'exclusive';
    let active=TYPES.includes(root.dataset.generationContextDefault)?root.dataset.generationContextDefault:'topic';
    const tabs=all(root,'[data-generation-context-tab]');
    const panels=all(root,'[data-generation-context-panel]');
    const urlInput=one(root,'[data-generation-context-url]');
    const promptInput=one(root,'[data-generation-context-prompt]');

    function records(kind){return all(root,`[data-generation-context-record][data-context-kind="${kind}"]`);}
    function selectedIds(kind){return records(kind).filter(node=>node.checked).map(node=>node.value).filter(Boolean);}
    function value(kind){return kind==='url'?clean(urlInput?.value):kind==='prompt'?clean(promptInput?.value):selectedIds(kind)[0]||'';}
    function count(kind){if(kind==='topic'||kind==='source')return selectedIds(kind).length;return value(kind)?1:0;}
    function total(){return TYPES.reduce((sum,kind)=>sum+count(kind),0);}

    function setActive(kind,{focus=false}={}){
      if(!TYPES.includes(kind))return;
      active=kind;
      tabs.forEach(tab=>{const on=tab.dataset.generationContextTab===kind;tab.setAttribute('aria-selected',on?'true':'false');tab.tabIndex=on?0:-1;});
      panels.forEach(panel=>{panel.hidden=panel.dataset.generationContextPanel!==kind;});
      if(focus)tabs.find(tab=>tab.dataset.generationContextTab===kind)?.focus();
      render();
      root.dispatchEvent(new CustomEvent('ai-learn-context-change',{bubbles:true,detail:snapshot()}));
    }

    function summaryParts(){
      const parts=[];
      const tc=count('topic'),sc=count('source');
      if(tc)parts.push(`${tc} Topic${tc===1?'':'s'}`);
      if(sc)parts.push(`${sc} Source${sc===1?'':'s'}`);
      if(count('url'))parts.push('URL');
      if(count('prompt'))parts.push('Prompt');
      return parts;
    }
    function render(){
      for(const kind of TYPES){
        const badge=one(root,`[data-generation-context-tab-count="${kind}"]`);
        if(!badge)continue;
        const n=count(kind),show=policy==='composable'?n>0:(kind===active&&n>0);
        badge.hidden=!show;if(show)badge.textContent=kind==='url'||kind==='prompt'?'✓':String(n);
      }
      const n=policy==='composable'?total():count(active);
      const countNode=one(root,'[data-generation-context-summary-count]');
      const textNode=one(root,'[data-generation-context-summary-text]');
      if(countNode)countNode.textContent=policy==='composable'?`${n} selected`:(n?'Selected':'Not selected');
      if(textNode){
        if(policy==='composable')textNode.textContent=summaryParts().join(' · ')||'Combine Topics, Sources, URL, and Prompt only when each adds useful context.';
        else textNode.textContent=n?`${active[0].toUpperCase()+active.slice(1)} context ready.`:`Choose ${active==='url'?'a URL':active==='prompt'?'a Prompt':`a ${active[0].toUpperCase()+active.slice(1)}`}.`;
      }
      root.dataset.contextActive=active;
      root.dataset.contextCount=String(n);
    }

    function snapshot(){
      return {policy,active,topic_ids:selectedIds('topic'),source_ids:selectedIds('source'),url:clean(urlInput?.value),prompt:clean(promptInput?.value)};
    }
    function restore(state){
      if(!state||typeof state!=='object')return;
      const topicIds=new Set(Array.isArray(state.topic_ids)?state.topic_ids:state.topic_id?[state.topic_id]:[]);
      const sourceIds=new Set(Array.isArray(state.source_ids)?state.source_ids:state.source_id?[state.source_id]:[]);
      records('topic').forEach(node=>node.checked=topicIds.has(node.value));
      records('source').forEach(node=>node.checked=sourceIds.has(node.value));
      if(urlInput&&typeof state.url==='string')urlInput.value=state.url;
      else if(urlInput&&typeof state.source_url==='string')urlInput.value=state.source_url;
      if(promptInput&&typeof state.prompt==='string')promptInput.value=state.prompt;
      else if(promptInput&&typeof state.free_prompt==='string')promptInput.value=state.free_prompt;
      setActive(TYPES.includes(state.active)?state.active:TYPES.includes(state.mode)?state.mode:active);
    }
    function applyQuery(params){
      if(!params)return;
      const mode=params.get('mode'),id=params.get('id')||params.get('subject')||'';
      if(TYPES.includes(mode))setActive(mode);
      if(id&&(mode==='topic'||mode==='source'||(!mode&&policy==='composable'))){
        const candidates=mode?records(mode):[...records('topic'),...records('source')];
        const node=candidates.find(item=>item.value===id);if(node){node.checked=true;if(mode)setActive(mode);render();}
      }
      if(urlInput&&params.get('url'))urlInput.value=params.get('url');
      if(promptInput&&params.get('prompt'))promptInput.value=params.get('prompt');
      render();
    }
    function singleId(kind){return selectedIds(kind)[0]||'';}
    function selectId(kind,id){
      const node=records(kind).find(item=>item.value===String(id||''));if(!node)return false;
      if(policy==='exclusive')records(kind==='topic'?'source':'topic').forEach(item=>item.checked=false);
      node.checked=true;setActive(kind);render();return true;
    }

    tabs.forEach(tab=>{
      tab.addEventListener('click',()=>setActive(tab.dataset.generationContextTab,{focus:false}));
      tab.addEventListener('keydown',event=>{
        const index=tabs.indexOf(tab);let next=-1;
        if(event.key==='ArrowRight'||event.key==='ArrowDown')next=(index+1)%tabs.length;
        if(event.key==='ArrowLeft'||event.key==='ArrowUp')next=(index-1+tabs.length)%tabs.length;
        if(next>=0){event.preventDefault();setActive(tabs[next].dataset.generationContextTab,{focus:true});}
      });
    });
    all(root,'[data-generation-context-record]').forEach(input=>input.addEventListener('change',()=>{if(policy==='exclusive')setActive(input.dataset.contextKind);else{render();root.dispatchEvent(new CustomEvent('ai-learn-context-change',{bubbles:true,detail:snapshot()}));}}));
    [urlInput,promptInput].filter(Boolean).forEach(input=>input.addEventListener('input',()=>{render();root.dispatchEvent(new CustomEvent('ai-learn-context-change',{bubbles:true,detail:snapshot()}));}));
    all(root,'[data-generation-context-filter]').forEach(filter=>filter.addEventListener('input',()=>{
      const kind=filter.dataset.generationContextFilter,q=clean(filter.value).toLowerCase();let visible=0;
      all(root,`[data-generation-context-row][data-context-kind="${kind}"]`).forEach(row=>{row.hidden=!!q&&!String(row.dataset.search||'').includes(q);if(!row.hidden)visible++;});
      const empty=one(root,`[data-generation-context-empty="${kind}"]`);if(empty)empty.hidden=visible!==0;
    }));

    const api={policy,activeType:()=>active,setActive,selectedIds,singleId,selectId,url:()=>clean(urlInput?.value),prompt:()=>clean(promptInput?.value),snapshot,restore,applyQuery,render};
    root._aiLearnContext=api;setActive(active);return api;
  }
  function get(root){const picker=root?.matches?.('[data-generation-context-picker]')?root:one(root,'[data-generation-context-picker]');return picker?bind(picker):null;}
  all(document,'[data-generation-context-picker]').forEach(bind);
  window.AI_LEARN_CONTEXT_API={get,bind};
})();
