/* AI Learn detail pages: reading, discovery, libraries, media viewing and page-level actions. */
(() => {
  'use strict';
  const one = (root, selector) => root?.querySelector?.(selector) || null;
  const all = (root, selector) => root?.querySelectorAll ? [...root.querySelectorAll(selector)] : [];
  const text = (tag, value) => { const el = document.createElement(tag); el.textContent = value; return el; };
  // A link target taken from page data is a navigation only if it resolves to
  // http(s). Anything else - javascript:, data:, vbscript:, an unparsable
  // value - returns '' and the caller renders the label without a link.
  const safeHref = value => {
    if (typeof value !== 'string' || !value) return '';
    try {
      const url = new URL(value, document.baseURI);
      return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : '';
    } catch { return ''; }
  };
  const safeLink = (label, target) => { const a = text('a', label), href = safeHref(target); if (href) a.href = href; return a; };
  const download = (value, name, type='application/json') => {
    const blob = new Blob([typeof value === 'string' ? value : JSON.stringify(value, null, 2)], {type});
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href=url; a.download=name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const rstEscape = value => value.replace(/[\\`*_<>|]/g, '\\$&');
  const heading = (title, mark) => { const value=rstEscape(title); return value+'\n'+mark.repeat(value.length)+'\n\n'; };
  const sha = async value => [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)))].map(b=>b.toString(16).padStart(2,'0')).join('');
  all(document, '[data-enhanced]').forEach(el => el.hidden=false);

  // Reusable Topic Prompts and Skills share one bounded preference mechanism.
  // Preferences are browser-local presentation state; they never mutate canonical JSON.
  function installInteractionVisibility({toggleAttr,controlsAttr,libraryAttr,statusAttr,storagePrefix,label}) {
    const selector='['+toggleAttr+']', inputs=all(document,selector), sites=new Map();
    for (const input of inputs) {
      const host=input.closest('['+controlsAttr+']');
      if (!host) continue;
      const site=host.dataset.site || 'default';
      if (!sites.has(site)) sites.set(site, []);
      sites.get(site).push(input);
    }
    const key=site=>storagePrefix+site;
    function defaults(rows) {
      return Object.fromEntries(rows.map(input=>[input.getAttribute(toggleAttr),input.dataset.default==='true']));
    }
    function read(site,rows) {
      const fallback=defaults(rows);
      try {
        const raw=localStorage.getItem(key(site));
        if (!raw || raw.length>10000) return fallback;
        const parsed=JSON.parse(raw);
        if (!parsed || typeof parsed!=='object' || Array.isArray(parsed)) return fallback;
        for (const id of Object.keys(fallback)) if (typeof parsed[id]==='boolean') fallback[id]=parsed[id];
      } catch {}
      return fallback;
    }
    function apply(site,state) {
      const rows=sites.get(site)||[];
      for (const input of rows) input.checked=state[input.getAttribute(toggleAttr)]!==false;
      all(document,'[data-learn-page]').filter(page=>page.dataset.site===site).forEach(page=>{
        for (const [id,enabled] of Object.entries(state)) {
          const section=all(page,'.learn-section').find(item=>item.dataset.section===id);
          const container=section?.closest('section');
          if (container) container.hidden=!enabled;
          all(document,'[data-toc-section]').filter(link=>link.dataset.tocSection===id).forEach(link=>{
            const row=link.closest('li'); if(row) row.hidden=!enabled;
          });
        }
      });
    }
    for (const [site,rows] of sites) {
      const state=read(site,rows); apply(site,state);
      for (const input of rows) input.addEventListener('change',()=>{
        const id=input.getAttribute(toggleAttr); state[id]=input.checked;
        try { localStorage.setItem(key(site),JSON.stringify(state)); } catch {}
        apply(site,state);
        const status=input.closest('['+libraryAttr+']')?.querySelector('['+statusAttr+']');
        if(status) status.textContent=(input.checked?'Enabled ':'Disabled ')+id+' '+label+' for topic pages.';
      });
    }
  }
  installInteractionVisibility({
    toggleAttr:'data-prompt-toggle',controlsAttr:'data-prompt-controls',
    libraryAttr:'data-prompt-library',statusAttr:'data-prompt-status',
    storagePrefix:'learn-prompts:v1:',label:'prompt'
  });
  installInteractionVisibility({
    toggleAttr:'data-skill-toggle',controlsAttr:'data-skill-controls',
    libraryAttr:'data-skill-library',statusAttr:'data-skill-status',
    storagePrefix:'learn-skills:v1:',label:'skill'
  });


  // Bounded presentation for static card libraries (Topic Prompts, Skills, and
  // the Collections landing grid).  All cards remain semantic HTML so native
  // links/toggles and no-JS behavior stay intact; JavaScript progressively
  // limits the visible window to a safe 12-item default.
  all(document,'[data-local-display]').forEach(root=>{
    const host=one(root,'[data-display-items]'), select=one(root,'[data-display-size]');
    const button=one(root,'[data-display-load-more]'), summary=one(root,'[data-display-summary]');
    const toolbar=one(root,'[data-local-display-toolbar]'), pager=one(root,'[data-local-display-pager]');
    if(!host||!select||!button||!summary)return;
    const items=all(host,'[data-display-item]'), sizes=new Set([...select.options].map(option=>Number(option.value)).filter(Number.isFinite));
    const defaultLimit=sizes.has(12)?12:Math.min(...sizes), requested=Number(new URLSearchParams(location.search).get('limit'));
    let preset=sizes.has(requested)?requested:defaultLimit, visibleLimit=preset;
    select.value=String(preset);
    function updateUrl(){const url=new URL(location.href);if(preset===defaultLimit)url.searchParams.delete('limit');else url.searchParams.set('limit',String(preset));history.replaceState(null,'',url);}
    function render(){
      items.forEach((item,index)=>{item.hidden=index>=visibleLimit;});
      const shown=Math.min(visibleLimit,items.length), expandable=items.length>defaultLimit;
      if(toolbar)toolbar.hidden=!expandable;if(pager)pager.hidden=!expandable;
      button.hidden=shown>=items.length;button.textContent='Load 12 more';
      summary.textContent=expandable?'Showing '+shown+' of '+items.length+' items.':'';
    }
    select.addEventListener('change',()=>{const value=Number(select.value);if(!sizes.has(value))return;preset=value;visibleLimit=value;updateUrl();render();});
    button.addEventListener('click',()=>{visibleLimit+=12;render();});
    window.addEventListener('popstate',()=>{const value=Number(new URLSearchParams(location.search).get('limit'));preset=sizes.has(value)?value:defaultLimit;visibleLimit=preset;select.value=String(preset);render();});
    render();
  });
  all(document, '[data-learn-page]').forEach(page => {
    try {
    const dataNode=one(page,'.learn-page-data');
    if (!dataNode) return;
    let data;
    try { data=JSON.parse(dataNode.textContent); } catch { return; }
    const subject=data.subject, message=one(page,'.learn-message');
    const announce=value=>{message.textContent=value;};
    // Legacy revision-scoped keys remain readable only for bookmark/reading migration.
    // Section bodies are now owned by the provenance-aware AI draft controller.
    const prefix='learn-page:v1:'+data.site_id+':'+subject.id+':'+data.revision+':';
    const mediaActions=one(page,'[data-media-actions]'), mediaGeneration=mediaActions&&one(mediaActions,'[data-media-generation]');
    one(mediaActions,'[data-media-create]')?.addEventListener('click',()=>{mediaGeneration.hidden=false;one(mediaGeneration,'textarea').focus();});
    one(mediaActions,'[data-media-close]')?.addEventListener('click',()=>{mediaGeneration.hidden=true;one(mediaActions,'[data-media-create]').focus();});
    one(mediaActions,'[data-media-copy-request]')?.addEventListener('click',async()=>{
      try {
        const request={contract:'learn.media-request.v1',subject_id:subject.id,media_kind:subject.kind,base_revision:data.revision,instructions:one(mediaGeneration,'textarea').value};
        await navigator.clipboard.writeText(JSON.stringify(request,null,2));announce('Media generation request copied. No AI service was called.');
      } catch {announce('Clipboard unavailable. The media instruction remains editable above.');}
    });
    const bookmark=one(page,'[data-bookmark]'), reading=one(page,'[data-reading]');
    const userPrefix='learn-user:v1:'+data.site_id+':';
    const bookmarkKey=userPrefix+'bookmark:'+subject.id, readingKey=userPrefix+'reading:'+subject.id;
    function migratePreference(stableKey, legacyKey) {
      try {
        if(localStorage.getItem(stableKey)!==null)return;
        const legacy=localStorage.getItem(legacyKey);
        if(legacy!==null)localStorage.setItem(stableKey,legacy);
      } catch {}
    }
    migratePreference(bookmarkKey,prefix+'bookmark');
    migratePreference(readingKey,prefix+'reading');
    try {if(bookmark){const saved=localStorage.getItem(bookmarkKey)==='true';bookmark.setAttribute('aria-pressed',String(saved));bookmark.textContent=saved?'Bookmarked':'Bookmark';}if(reading){const saved=localStorage.getItem(readingKey);if([...reading.options].some(o=>o.value===saved))reading.value=saved;}}catch{}
    bookmark?.addEventListener('click',()=>{try{const saved=bookmark.getAttribute('aria-pressed')!=='true';localStorage.setItem(bookmarkKey,String(saved));bookmark.setAttribute('aria-pressed',String(saved));bookmark.textContent=saved?'Bookmarked':'Bookmark';}catch{announce('Browser storage is unavailable.');}});
    reading?.addEventListener('change',()=>{try{localStorage.setItem(readingKey,reading.value);}catch{announce('Browser storage is unavailable.');}});
    one(page,'[data-copy-url]')?.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(location.href);announce('Page URL copied.');}catch{announce('Copy the URL from your browser address bar.');}});
    function revealHash(){let id;try{id=decodeURIComponent(location.hash.slice(1));}catch{return;}const target=document.getElementById(id);if(!target)return;const section=one(target,'.learn-section')||target.closest('.learn-section')||(target.closest('section')&&one(target.closest('section'),'.learn-section'));if(section){one(section,'.learn-section-content').hidden=false;const button=one(section,'[data-collapse]');button.textContent='Hide content';button.setAttribute('aria-expanded','true');}}
    window.addEventListener('hashchange',revealHash);revealHash();
    const search=one(page,'[data-topic-search]'), results=one(search,'[data-search-results]');
    search?.addEventListener('submit',event=>{
      event.preventDefault();const query=search.elements.q.value.trim().toLowerCase();results?.replaceChildren();
      if(search.elements.mode.value!=='search'){results.append(text('p','Research generation will be available when the AI connection is configured. Search remains available now.'));return;}
      const matches=data.search.filter(s=>[s.id,s.title,s.summary,s.url,s.publisher,...s.domains].join(' ').toLowerCase().includes(query)).slice(0,10);
      for(const match of matches){results.append(safeLink(match.title,match.href));}
      if(!matches.length)results?.append(text('p','No matching topic in this snapshot. Create a custom topic from the Topics page.'));
    });
    } catch (error) {
      console.error('AI Learn: detail-page interaction initialization failed.', error);
      const message=one(page,'.learn-message');
      if(message)message.textContent='Some interactive controls could not initialize. Reload this page and check the browser console.';
    }
  });
  // Browser-local Bookmarks and Collections pages. These share stable keys
  // with topic-page controls and migrate the previous revision-scoped keys.
  all(document,'[data-user-library]').forEach(library=>{
    const dataNode=one(library,'.learn-user-library-data');
    let data; try {data=JSON.parse(dataNode.textContent);} catch {return;}
    const prefix='learn-user:v1:'+data.site_id+':';
    const status=one(library,'[data-library-status]');
    const readingValues=['Unread','Want to Read','Currently Reading','Completed'];
    const key=(type,id)=>prefix+type+':'+id;
    const gridKey=prefix+'library-grid-columns';
    const grid=one(library,'.learn-library-grid');
    const gridControl=one(library,'[data-library-grid-control]');
    const validGridColumns=new Set(['2','3','4','5']);
    const displaySelect=data.mode==='collections'?null:one(library,'[data-display-size]');
    const displayButton=data.mode==='collections'?null:one(library,'[data-display-load-more]');
    const displaySummary=data.mode==='collections'?null:one(library,'[data-display-summary]');
    const displayToolbar=data.mode==='collections'?null:one(library,'[data-local-display-toolbar]');
    const displayPager=data.mode==='collections'?null:one(library,'[data-local-display-pager]');
    const displaySizes=new Set(displaySelect?[...displaySelect.options].map(option=>Number(option.value)).filter(Number.isFinite):[12]);
    const defaultDisplayLimit=displaySizes.has(12)?12:Math.min(...displaySizes);
    const requestedDisplayLimit=Number(new URLSearchParams(location.search).get('limit'));
    let displayPreset=displaySizes.has(requestedDisplayLimit)?requestedDisplayLimit:defaultDisplayLimit;
    let displayLimit=displayPreset;
    if(displaySelect)displaySelect.value=String(displayPreset);
    function gridColumns(){
      try {const value=localStorage.getItem(gridKey);return validGridColumns.has(value)?value:'2';}
      catch {return '2';}
    }
    function applyGridColumns(value=gridColumns(),persist=false){
      const columns=validGridColumns.has(String(value))?String(value):'2';
      if(grid)grid.dataset.gridColumns=columns;
      all(gridControl,'[data-library-grid]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.libraryGrid===columns)));
      if(persist){
        try {localStorage.setItem(gridKey,columns);status.textContent='Grid preference updated.';}
        catch {status.textContent='Browser storage is unavailable.';}
      }
    }
    all(gridControl,'[data-library-grid]').forEach(button=>button.addEventListener('click',()=>applyGridColumns(button.dataset.libraryGrid,true)));
    applyGridColumns();
    function updateDisplayUrl(){
      const url=new URL(location.href);if(displayPreset===defaultDisplayLimit)url.searchParams.delete('limit');else url.searchParams.set('limit',String(displayPreset));history.replaceState(null,'',url);
    }
    function updateDisplayControls(total){
      if(!displaySelect||!displayButton||!displaySummary)return;
      const expandable=total>defaultDisplayLimit, shown=Math.min(displayLimit,total);
      if(displayToolbar)displayToolbar.hidden=!expandable;if(displayPager)displayPager.hidden=!expandable;
      displayButton.hidden=shown>=total;displayButton.textContent='Load 12 more';
      displaySummary.textContent=expandable?'Showing '+shown+' of '+total+' items.':'';
    }
    displaySelect?.addEventListener('change',()=>{const value=Number(displaySelect.value);if(!displaySizes.has(value))return;displayPreset=value;displayLimit=value;updateDisplayUrl();render();});
    displayButton?.addEventListener('click',()=>{displayLimit+=12;render();});
    function migrate(record) {
      const legacy='learn-page:v1:'+data.site_id+':'+record.id+':'+data.revision+':';
      try {
        if(localStorage.getItem(key('bookmark',record.id))===null){const value=localStorage.getItem(legacy+'bookmark');if(value!==null)localStorage.setItem(key('bookmark',record.id),value);}
        if(localStorage.getItem(key('reading',record.id))===null){const value=localStorage.getItem(legacy+'reading');if(value!==null)localStorage.setItem(key('reading',record.id),value);}
      } catch {}
    }
    data.records.forEach(migrate);
    const getState=record=>{
      try {return {bookmark:localStorage.getItem(key('bookmark',record.id))==='true',reading:localStorage.getItem(key('reading',record.id))||'Unread'};}
      catch {return {bookmark:false,reading:'Unread'};}
    };
    function makeItem(record) {
      const state=getState(record), article=document.createElement('article');article.className='learn-library-item';
      const title=safeLink(record.title,record.href);title.className='learn-library-title';article.append(title);
      if(record.summary){const summary=text('p',record.summary);summary.className='learn-library-summary';article.append(summary);}
      const meta=text('p',[record.kind,...record.domains.map(tag=>'#'+tag)].join(' · '));meta.className='learn-meta';article.append(meta);
      const actions=document.createElement('div');actions.className='learn-actions';
      const label=text('label','Reading '), select=document.createElement('select');select.setAttribute('aria-label','Reading status for '+record.title);
      for(const value of readingValues){const option=new Option(value,value);if(value===state.reading)option.selected=true;select.append(option);}label.append(select);actions.append(label);
      const bookmark=text('button',state.bookmark?'Remove bookmark':'Bookmark');bookmark.type='button';bookmark.setAttribute('aria-pressed',String(state.bookmark));actions.append(bookmark);article.append(actions);
      select.addEventListener('change',()=>{try{localStorage.setItem(key('reading',record.id),select.value);status.textContent='Reading status updated.';render();}catch{status.textContent='Browser storage is unavailable.';}});
      bookmark.addEventListener('click',()=>{try{localStorage.setItem(key('bookmark',record.id),String(!getState(record).bookmark));status.textContent='Bookmark updated.';render();}catch{status.textContent='Browser storage is unavailable.';}});
      return article;
    }
    function render() {
      if(data.mode==='collections'){
        for(const counter of all(library,'[data-collection-count-for]')){
          const name=counter.dataset.collectionCountFor;
          const matches=data.records.filter(record=>name==='Bookmarks'?getState(record).bookmark:getState(record).reading===name);
          counter.textContent=String(matches.length);
        }
        status.textContent='Collection counts reflect this browser only.';
        return;
      }
      if(data.mode==='bookmarks'){
        const host=one(library,'[data-bookmarks-list]'), empty=one(library,'[data-bookmarks-empty]');host.replaceChildren();
        const saved=data.records.filter(record=>getState(record).bookmark), visible=saved.slice(0,displayLimit);
        visible.forEach(record=>host.append(makeItem(record)));empty.hidden=saved.length>0;updateDisplayControls(saved.length);
        status.textContent='Showing '+visible.length+' of '+saved.length+' bookmark'+(saved.length===1?'':'s')+' in this browser.';
        return;
      }
      const host=one(library,'[data-collection-items]'), empty=one(library,'[data-collection-empty]');
      host.replaceChildren();
      const matches=data.records.filter(record=>getState(record).reading===data.collection_name), visible=matches.slice(0,displayLimit);
      visible.forEach(record=>host.append(makeItem(record)));empty.hidden=matches.length>0;updateDisplayControls(matches.length);
      status.textContent='Showing '+visible.length+' of '+matches.length+' record'+(matches.length===1?'':'s')+' in '+data.collection_name+'.';
    }
    window.addEventListener('storage',event=>{
      if(event.key===gridKey)applyGridColumns();
      if(event.key?.startsWith(prefix))render();
    });
    if(data.mode!=='collections')window.addEventListener('popstate',()=>{const requested=Number(new URLSearchParams(location.search).get('limit'));displayPreset=displaySizes.has(requested)?requested:defaultDisplayLimit;displayLimit=displayPreset;if(displaySelect)displaySelect.value=String(displayPreset);render();});
    render();
  });

  // Every catalog explorer shares one bounded search/disclosure/paging
  // controller.  Static RST/HTML shards remain 12 records each.  The display
  // selector can progressively fetch enough same-origin shards to satisfy one
  // bounded preset (max 150), while each explicit Load More action advances by
  // exactly one 12-record shard.  Filtering continues to describe the loaded
  // snapshot honestly rather than pretending client-side search covers records
  // that have not been fetched.
  all(document,'[data-learn-explorer]').forEach(explorer=>{
    const form=one(explorer,'[data-explorer-controls]');
    if(!form)return;
    const tbody=one(explorer,'tbody'), host=tbody||one(explorer,'.learn-items'), status=one(explorer,'[data-explorer-status]');
    if(!host||!status)return;
    const itemSelector=tbody?'.learn-topic-row':'.learn-card', nodeFor=item=>tbody?item:item.closest('.learn-entry');
    const items=tbody?all(tbody,itemSelector):all(explorer,itemSelector);
    const sortable=new Set([...form.elements.sort.options].map(option=>option.value));
    const numeric=new Set((explorer.dataset.numericSort||'').split(' ').filter(Boolean));
    const itemLabel=explorer.dataset.itemLabel||explorer.dataset.kind||'item', itemPlural=explorer.dataset.itemPlural||itemLabel+'s';
    const total=Math.max(0,Number(explorer.dataset.total)||items.length), offset=Math.max(0,Number(explorer.dataset.offset)||0);
    const chunkSize=Math.max(1,Number(explorer.dataset.pageChunkSize)||12), displaySelect=one(form,'[data-display-size]');
    const displaySizes=new Set([...displaySelect.options].map(option=>Number(option.value)).filter(Number.isFinite));
    const defaultLimit=displaySizes.has(12)?12:Math.min(...displaySizes), params=new URLSearchParams(location.search);
    const requestedLimit=Number(params.get('limit'));
    let presetLimit=displaySizes.has(requestedLimit)?requestedLimit:defaultLimit, visibleLimit=presetLimit;
    let nextLink=one(explorer,'[data-explorer-next]'), nextHref=nextLink?.href||'', loading=false;
    const loadMore=one(explorer,'[data-display-load-more]'), displaySummary=one(explorer,'[data-display-summary]');
    const fallbackLinks=one(explorer,'[data-explorer-fallback-links]'), seenPages=new Set();
    if(nextHref)seenPages.add(new URL(location.pathname,location.origin).href);
    if(displaySelect)displaySelect.value=String(presetLimit);
    if(nextLink)nextLink.hidden=true;
    const defaultDirection=key=>['title','status','publisher','format'].includes(key)?'asc':'desc';
    let sort=sortable.has(params.get('sort'))?params.get('sort'):(sortable.has('created')?'created':[...sortable][0]);
    let direction=params.get('dir')==='asc'||params.get('dir')==='desc'?params.get('dir'):defaultDirection(sort);
    const directionButton=one(form,'[data-explorer-direction]');
    const filterToggle=one(form,'[data-explorer-filter-toggle]'), filterOptions=one(form,'[data-explorer-filter-options]');
    const hasAdvancedState=p=>Boolean(
      p.get('category')||
      (p.get('timeframe')&&p.get('timeframe')!=='all')||
      (p.get('limit')&&Number(p.get('limit'))!==defaultLimit)||
      (p.get('sort')&&p.get('sort')!==(sortable.has('created')?'created':[...sortable][0]))||
      (p.get('dir')&&p.get('dir')!==defaultDirection(sortable.has(p.get('sort'))?p.get('sort'):sort))
    );
    function setFilterOptions(expanded){
      if(!filterToggle||!filterOptions)return;
      filterOptions.hidden=!expanded;
      filterToggle.setAttribute('aria-expanded',String(expanded));
      filterToggle.setAttribute('aria-label',expanded?'Hide search options':'More search options');
      filterToggle.title=expanded?'Fewer options':'More options';
    }
    setFilterOptions(hasAdvancedState(params));
    filterToggle?.addEventListener('click',()=>setFilterOptions(filterToggle.getAttribute('aria-expanded')!=='true'));
    for(const name of ['q','category','timeframe'])if(params.has(name)&&form.elements[name])form.elements[name].value=params.get(name);
    if(form.elements.sort)form.elements.sort.value=sort;
    function syncSortControls(){
      if(form.elements.sort)form.elements.sort.value=sort;
      if(directionButton){
        directionButton.textContent=direction==='asc'?'↑ Asc':'↓ Desc';
        directionButton.title=direction==='asc'?'Ascending order; activate to reverse':'Descending order; activate to reverse';
        directionButton.setAttribute('aria-label',directionButton.title);
      }
      all(explorer,'[data-sort-key]').forEach(button=>{const active=button.dataset.sortKey===sort;button.setAttribute('aria-pressed',String(active));button.dataset.direction=active?direction:'';});
    }
    function withinTimeframe(item,value){
      if(!value||value==='all')return true;
      const days=Number(value.slice(0,-1));if(!Number.isFinite(days))return true;
      const at=Date.parse(item.dataset.created);if(!Number.isFinite(at))return false;
      return Date.now()-at<=days*86400000;
    }
    function sortItems(){
      items.sort((a,b)=>{
        let av=a.dataset[sort]||'',bv=b.dataset[sort]||'';
        if(numeric.has(sort)){av=Number(av||0);bv=Number(bv||0);}
        const cmp=numeric.has(sort)?av-bv:String(av).localeCompare(String(bv));
        return direction==='asc'?cmp:-cmp;
      });
      items.forEach(item=>host.append(nodeFor(item)));
      syncSortControls();
    }
    function updateUrl(){
      const url=new URL(location.href);
      for(const name of ['q','category','timeframe']){
        const value=form.elements[name]?.value||'';
        if(value&&value!=='all')url.searchParams.set(name,value);else url.searchParams.delete(name);
      }
      if(presetLimit===defaultLimit)url.searchParams.delete('limit');else url.searchParams.set('limit',String(presetLimit));
      url.searchParams.set('sort',sort);url.searchParams.set('dir',direction);
      history.replaceState(null,'',url);
    }
    function apply(update=false){
      const q=form.elements.q.value.trim().toLowerCase(), category=form.elements.category.value, timeframe=form.elements.timeframe.value;
      sortItems();
      let matching=0, shown=0;
      for(const item of items){
        const categories=(item.dataset.categories||'').split(' ').filter(Boolean);
        const match=(!q||item.textContent.toLowerCase().includes(q))&&(!category||categories.includes(category))&&withinTimeframe(item,timeframe);
        if(match)matching++;
        const visible=match&&shown<visibleLimit;if(visible)shown++;
        nodeFor(item).hidden=!visible;
      }
      const loaded=items.length, loadedEnd=Math.min(total,offset+loaded), noun=matching===1?itemLabel:itemPlural;
      status.textContent='Showing '+shown+' of '+matching+' matching '+noun+' in '+loaded+' loaded record'+(loaded===1?'':'s')+'.';
      if(displaySummary){const start=loaded?offset+1:0;displaySummary.textContent='Catalog records '+start+'–'+loadedEnd+' of '+total+'.';}
      if(loadMore){loadMore.hidden=(!nextHref&&matching<=visibleLimit);loadMore.disabled=loading;loadMore.textContent=loading?'Loading…':'Load '+chunkSize+' more';}
      if(fallbackLinks)fallbackLinks.dataset.enhanced='true';
      if(update)updateUrl();
    }
    function addCategoryOptions(){
      const known=new Set([...form.elements.category.options].map(option=>option.value));
      for(const item of items)for(const tag of (item.dataset.categories||'').split(' '))if(tag&&!known.has(tag)){known.add(tag);form.elements.category.append(new Option(tag,tag));}
    }
    async function fetchNextChunk(){
      if(!nextHref||loading)return false;
      loading=true;apply();
      try {
        const url=new URL(nextHref,location.href);
        if(url.origin!==location.origin||seenPages.has(url.href))throw new Error('unsafe or repeated explorer page');
        const response=await fetch(url,{signal:AbortSignal.timeout(10000),credentials:'same-origin',cache:'no-store',redirect:'error'});if(!response.ok)throw new Error('explorer page unavailable');
        const reader=response.body?.getReader();if(!reader)throw new Error('stream unavailable');
        let length=0;const chunks=[];
        while(true){const {value,done}=await reader.read();if(done)break;length+=value.length;if(length>2*1024*1024){await reader.cancel();throw new Error('explorer page too large');}chunks.push(value);}
        const buffer=new Uint8Array(length);let cursor=0;for(const chunk of chunks){buffer.set(chunk,cursor);cursor+=chunk.length;}
        const doc=new DOMParser().parseFromString(new TextDecoder().decode(buffer),'text/html');
        const incoming=one(doc,'[data-learn-explorer]');if(!incoming||incoming.dataset.kind!==explorer.dataset.kind)throw new Error('explorer kind mismatch');
        const incomingHost=tbody?one(incoming,'tbody'):one(incoming,'.learn-items');if(!incomingHost)throw new Error('explorer host missing');
        const incomingItems=tbody?all(incomingHost,itemSelector):all(incoming,itemSelector);
        if(!incomingItems.length||incomingItems.length>chunkSize)throw new Error('invalid explorer shard size');
        seenPages.add(url.href);
        if(tbody){
          for(const row of incomingItems){const cloned=document.importNode(row,true);host.append(cloned);items.push(cloned);}
        }else{
          for(const item of incomingItems){const entry=item.closest('.learn-entry');if(!entry)throw new Error('card entry missing');for(const link of all(entry,'[href]'))link.setAttribute('href',new URL(link.getAttribute('href'),url).href);for(const image of all(entry,'[src]'))image.setAttribute('src',new URL(image.getAttribute('src'),url).href);const cloned=document.importNode(entry,true);host.append(cloned);const card=one(cloned,itemSelector);if(card)items.push(card);}
        }
        const after=one(incoming,'[data-explorer-next]');nextHref=after?new URL(after.getAttribute('href'),url).href:'';
        if(nextLink){if(nextHref){nextLink.href=nextHref;}else{nextLink.remove();nextLink=null;}}
        addCategoryOptions();decorateWhiteboards();return true;
      } finally {loading=false;}
    }
    async function ensureLoaded(target){
      // Automatic work is bounded by the largest allowlisted preset (150).  It
      // loads by record count, not by filter matches, avoiding unbounded scans
      // for a rare query across a very large static catalog.
      while(items.length<target&&nextHref){const before=items.length;try{if(!await fetchNextChunk())break;}catch{if(nextLink)nextLink.hidden=false;status.textContent='Could not load more items. Use the next-page link or try again.';break;}if(items.length<=before)break;}
      apply();
    }
    form.addEventListener('submit',event=>{event.preventDefault();apply(true);});
    form.addEventListener('input',event=>{if(event.target===form.elements.sort||event.target===displaySelect||event.isComposing)return;apply(true);});
    form.elements.q?.addEventListener('compositionend',()=>apply(true));
    form.elements.q?.addEventListener('search',()=>apply(true));
    form.elements.sort?.addEventListener('change',()=>{const next=form.elements.sort.value;if(!sortable.has(next))return;sort=next;direction=defaultDirection(sort);apply(true);});
    displaySelect?.addEventListener('change',async()=>{const value=Number(displaySelect.value);if(!displaySizes.has(value))return;presetLimit=value;visibleLimit=value;updateUrl();await ensureLoaded(value);setFilterOptions(true);});
    directionButton?.addEventListener('click',()=>{direction=direction==='asc'?'desc':'asc';apply(true);});
    one(form,'[data-explorer-reset]')?.addEventListener('click',()=>{form.reset();sort=sortable.has('created')?'created':[...sortable][0];direction=defaultDirection(sort);presetLimit=defaultLimit;visibleLimit=defaultLimit;if(displaySelect)displaySelect.value=String(defaultLimit);setFilterOptions(false);apply(true);});
    all(explorer,'[data-sort-key]').forEach(button=>button.addEventListener('click',()=>{const next=button.dataset.sortKey;if(!sortable.has(next))return;if(sort===next)direction=direction==='asc'?'desc':'asc';else{sort=next;direction=defaultDirection(next);}apply(true);}));
    loadMore?.addEventListener('click',async()=>{visibleLimit+=chunkSize;await ensureLoaded(visibleLimit);});
    window.addEventListener('popstate',async()=>{
      const p=new URLSearchParams(location.search);
      for(const name of ['q','category','timeframe'])if(form.elements[name])form.elements[name].value=p.get(name)||(name==='timeframe'?'all':'');
      const requested=p.get('sort');sort=sortable.has(requested)?requested:(sortable.has('created')?'created':[...sortable][0]);
      direction=p.get('dir')==='asc'||p.get('dir')==='desc'?p.get('dir'):defaultDirection(sort);
      const requestedDisplay=Number(p.get('limit'));presetLimit=displaySizes.has(requestedDisplay)?requestedDisplay:defaultLimit;visibleLimit=presetLimit;if(displaySelect)displaySelect.value=String(presetLimit);
      setFilterOptions(hasAdvancedState(p));await ensureLoaded(presetLimit);
    });
    apply();ensureLoaded(presetLimit);
  });

  // Whiteboard viewer: no external lightbox dependency. The modal is built from
  // the rendered Sphinx images so versioned documentation paths keep working.
  function whiteboardHost(image) {
    const gallery=image.closest('[data-whiteboard-gallery]');
    if(gallery)return gallery;
    const explorer=image.closest('.learn-explorer[data-kind="whiteboard"]');
    if(explorer)return one(explorer,'.learn-items')||explorer;
    return image.closest('.learn-section[data-section="whiteboard"]');
  }
  function whiteboardCaption(image) {
    return image.closest('figure')?.querySelector('figcaption')?.textContent?.trim() ||
      image.closest('.learn-entry')?.querySelector('.learn-card h3')?.textContent?.trim() ||
      image.alt || 'Whiteboard image';
  }
  function whiteboardImages(host) {
    if(!host)return [];
    return all(host,'img').filter(image=>!image.closest('[hidden]') && !image.closest('.learn-lightbox'));
  }
  const whiteboardSelector='[data-whiteboard-gallery] img, .learn-explorer[data-kind="whiteboard"] .learn-items img, .learn-section[data-section="whiteboard"] img';
  function isWhiteboardImage(image) {
    return image instanceof HTMLImageElement && image.matches(whiteboardSelector);
  }
  function decorateWhiteboard(image) {
    if(!isWhiteboardImage(image) || image.dataset.whiteboardViewer==='true')return image;
    image.dataset.whiteboardViewer='true';
    image.classList.add('learn-whiteboard-openable');
    image.tabIndex=0;
    image.setAttribute('role','button');
    image.setAttribute('aria-label','Open whiteboard image viewer: '+whiteboardCaption(image));
    image.draggable=false;
    return image;
  }
  function decorateWhiteboards() {
    all(document,whiteboardSelector).forEach(decorateWhiteboard);
  }
  function openWhiteboardViewer(opener) {
    const host=whiteboardHost(opener), images=whiteboardImages(host);
    let index=images.indexOf(opener);
    if(index<0 || !images.length)return;
    const previousOverflow=document.body.style.overflow;
    let scale=1, panX=0, panY=0, dragging=false, dragX=0, dragY=0, originX=0, originY=0, slideshow=0;

    const overlay=text('div','');overlay.textContent='';overlay.className='learn-lightbox';overlay.tabIndex=-1;
    overlay.setAttribute('role','dialog');overlay.setAttribute('aria-modal','true');overlay.setAttribute('aria-label','Whiteboard image viewer');
    const nav=text('div','');nav.className='learn-lightbox-nav';
    const toolbar=text('div','');toolbar.className='learn-lightbox-toolbar';
    const counter=text('div','');counter.className='learn-lightbox-counter';counter.setAttribute('aria-live','polite');
    const stage=text('div','');stage.className='learn-lightbox-stage';
    const image=text('img','');image.textContent='';image.className='learn-lightbox-image';image.draggable=false;image.tabIndex=0;image.setAttribute('role','button');
    const caption=text('div','');caption.className='learn-lightbox-caption';caption.setAttribute('aria-live','polite');
    const thumbs=text('div','');thumbs.className='learn-lightbox-thumbnails';thumbs.hidden=true;
    const prev=text('button','‹');prev.type='button';prev.className='learn-lightbox-slide learn-lightbox-previous';prev.title='Previous image';prev.setAttribute('aria-label','Previous image');
    const next=text('button','›');next.type='button';next.className='learn-lightbox-slide learn-lightbox-next';next.title='Next image';next.setAttribute('aria-label','Next image');

    function tool(symbol,label,handler){const button=text('button',symbol);button.type='button';button.className='learn-lightbox-tool';button.title=label;button.setAttribute('aria-label',label);button.addEventListener('click',handler);toolbar.append(button);return button;}
    const thumbsButton=tool('▦','Thumbnails',()=>{if(images.length<2)return;thumbs.hidden=!thumbs.hidden;thumbsButton.setAttribute('aria-pressed',String(!thumbs.hidden));if(!thumbs.hidden)thumbs.querySelector('[aria-current="true"]')?.scrollIntoView({block:'nearest',inline:'nearest'});});
    const zoomIn=tool('+','Zoom in',()=>zoomBy(.25));
    const zoomOut=tool('−','Zoom out',()=>zoomBy(-.25));
    const reset=tool('100%','Reset zoom',()=>setZoom(1));
    const slideshowButton=tool('▶','Turn on slideshow',()=>toggleSlideshow());
    const fullscreenButton=tool('⛶','Enter fullscreen',()=>toggleFullscreen());
    const closeButton=tool('×','Close',()=>close());
    thumbsButton.setAttribute('aria-pressed','false');slideshowButton.setAttribute('aria-pressed','false');
    if(images.length<2){thumbsButton.disabled=true;slideshowButton.disabled=true;}
    if(!overlay.requestFullscreen)fullscreenButton.disabled=true;

    nav.append(toolbar,counter);stage.append(image,prev,next);overlay.append(nav,stage,caption,thumbs);document.body.append(overlay);
    document.body.style.overflow='hidden';

    for(const [thumbIndex,source] of images.entries()){
      const button=text('button','');button.type='button';button.className='learn-lightbox-thumb';button.title='View image '+(thumbIndex+1)+': '+whiteboardCaption(source);
      const preview=text('img','');preview.textContent='';preview.src=source.currentSrc||source.src;preview.alt='';preview.draggable=false;button.append(preview);
      button.addEventListener('click',()=>show(thumbIndex));thumbs.append(button);
    }

    function clampPan(){if(scale<=1){panX=0;panY=0;return;}const maxX=Math.max(0,image.clientWidth*(scale-1)/2),maxY=Math.max(0,image.clientHeight*(scale-1)/2);panX=Math.max(-maxX,Math.min(maxX,panX));panY=Math.max(-maxY,Math.min(maxY,panY));}
    function transform(){clampPan();image.style.transform=`translate(${panX}px, ${panY}px) scale(${scale})`;image.style.cursor=scale>1?(dragging?'grabbing':'grab'):'zoom-in';image.title=scale===1?'Click once to zoom to 200%':'Click once to reset to 100%; drag to pan';image.setAttribute('aria-label',scale===1?'Zoom whiteboard image to 200%':'Reset whiteboard image to 100%; drag to pan');zoomIn.disabled=scale>=5;zoomOut.disabled=scale<=1;reset.disabled=scale===1;reset.textContent=Math.round(scale*100)+'%';}
    function setZoom(value){scale=Math.max(1,Math.min(5,Math.round(value*4)/4));if(scale===1){panX=0;panY=0;}transform();}
    function setZoomAt(value,clientX,clientY){
      const next=Math.max(1,Math.min(5,Math.round(value*4)/4));
      if(next===1){scale=1;panX=0;panY=0;transform();return;}
      const rect=image.getBoundingClientRect();
      const centerX=rect.left+rect.width/2,centerY=rect.top+rect.height/2;
      const ratio=next/scale;
      panX=(panX-(clientX-centerX)) * ratio + (clientX-centerX);
      panY=(panY-(clientY-centerY)) * ratio + (clientY-centerY);
      scale=next;transform();
    }
    function zoomBy(delta){setZoom(scale+delta);}
    function preload(at){if(at<0||at>=images.length)return;const preloadImage=new Image();preloadImage.src=images[at].currentSrc||images[at].src;}
    function show(nextIndex){
      if(nextIndex<0||nextIndex>=images.length)return;
      index=nextIndex;scale=1;panX=0;panY=0;dragging=false;
      const source=images[index], label=whiteboardCaption(source);image.src=source.currentSrc||source.src;image.alt=source.alt||label;caption.textContent=label;counter.textContent=(index+1)+' / '+images.length;
      prev.hidden=index===0;next.hidden=index===images.length-1;
      all(thumbs,'.learn-lightbox-thumb').forEach((button,i)=>{button.setAttribute('aria-current',String(i===index));if(i===index&&!thumbs.hidden)button.scrollIntoView({block:'nearest',inline:'nearest'});});
      transform();preload(index-1);preload(index+1);
    }
    function stopSlideshow(){if(slideshow){clearInterval(slideshow);slideshow=0;}slideshowButton.textContent='▶';slideshowButton.title='Turn on slideshow';slideshowButton.setAttribute('aria-label','Turn on slideshow');slideshowButton.setAttribute('aria-pressed','false');}
    function toggleSlideshow(){if(images.length<2)return;if(slideshow){stopSlideshow();return;}slideshowButton.textContent='❚❚';slideshowButton.title='Pause slideshow';slideshowButton.setAttribute('aria-label','Pause slideshow');slideshowButton.setAttribute('aria-pressed','true');slideshow=setInterval(()=>show(index+1<images.length?index+1:0),4000);}
    async function toggleFullscreen(){try{if(document.fullscreenElement===overlay){if(document.exitFullscreen)await document.exitFullscreen();}else await overlay.requestFullscreen();}catch{}}
    function updateFullscreen(){const active=document.fullscreenElement===overlay;fullscreenButton.textContent=active?'▣':'⛶';fullscreenButton.title=active?'Exit fullscreen':'Enter fullscreen';fullscreenButton.setAttribute('aria-label',fullscreenButton.title);}
    function close(){stopSlideshow();document.removeEventListener('fullscreenchange',updateFullscreen);if(document.fullscreenElement===overlay&&document.exitFullscreen){const result=document.exitFullscreen();result?.catch?.(()=>{});}overlay.remove();document.body.style.overflow=previousOverflow;opener.focus({preventScroll:true});}

    prev.addEventListener('click',()=>show(index-1));next.addEventListener('click',()=>show(index+1));
    image.addEventListener('error',()=>{caption.textContent='This whiteboard image could not be loaded.';});
    let moved=false;
    function toggleImageZoom(clientX,clientY){setZoomAt(scale===1?2:1,clientX,clientY);}
    image.addEventListener('click',event=>{
      if(moved){moved=false;return;}
      event.preventDefault();
      toggleImageZoom(event.clientX,event.clientY);
    });
    image.addEventListener('keydown',event=>{
      if(!['Enter',' '].includes(event.key))return;
      event.preventDefault();
      const rect=image.getBoundingClientRect();
      toggleImageZoom(rect.left+rect.width/2,rect.top+rect.height/2);
    });
    image.addEventListener('wheel',event=>{event.preventDefault();setZoomAt(scale+(event.deltaY<0?.25:-.25),event.clientX,event.clientY);},{passive:false});
    image.addEventListener('pointerdown',event=>{if(scale<=1)return;moved=false;dragging=true;dragX=event.clientX;dragY=event.clientY;originX=panX;originY=panY;image.setPointerCapture(event.pointerId);transform();});
    image.addEventListener('pointermove',event=>{if(!dragging)return;if(Math.abs(event.clientX-dragX)>4||Math.abs(event.clientY-dragY)>4)moved=true;panX=originX+event.clientX-dragX;panY=originY+event.clientY-dragY;transform();});
    const endDrag=()=>{dragging=false;transform();};image.addEventListener('pointerup',endDrag);image.addEventListener('pointercancel',endDrag);
    stage.addEventListener('click',event=>{if(event.target===stage)close();});
    document.addEventListener('fullscreenchange',updateFullscreen);
    overlay.addEventListener('keydown',event=>{
      if(event.key==='Escape'){event.preventDefault();close();}
      else if(event.key==='ArrowLeft'&&index>0){event.preventDefault();show(index-1);}
      else if(event.key==='ArrowRight'&&index<images.length-1){event.preventDefault();show(index+1);}
      else if(event.key==='+'||event.key==='='){event.preventDefault();zoomBy(.25);}
      else if(event.key==='-'){event.preventDefault();zoomBy(-.25);}
      else if(event.key==='0'){event.preventDefault();setZoom(1);}
      else if(event.key.toLowerCase()==='f'&&!fullscreenButton.disabled){event.preventDefault();toggleFullscreen();}
      else if(event.key.toLowerCase()==='t'&&!thumbsButton.disabled){event.preventDefault();thumbsButton.click();}
      else if(event.key===' '&&event.target===overlay&&!slideshowButton.disabled){event.preventDefault();toggleSlideshow();}
      else if(event.key==='Tab'){
        const focusable=all(overlay,'button:not([disabled]):not([hidden])').filter(button=>button.offsetParent!==null);if(!focusable.length)return;
        const first=focusable[0],last=focusable[focusable.length-1];if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
      }
    });
    show(index);closeButton.focus();
  }
  document.addEventListener('click',event=>{
    const image=event.target.closest?.('img');
    if(!isWhiteboardImage(image))return;
    decorateWhiteboard(image);
    event.preventDefault();
    event.stopPropagation();
    openWhiteboardViewer(image);
  },true);
  document.addEventListener('keydown',event=>{
    const image=event.target.closest?.('img');
    if(!isWhiteboardImage(image)||!['Enter',' '].includes(event.key))return;
    decorateWhiteboard(image);
    event.preventDefault();
    event.stopPropagation();
    openWhiteboardViewer(image);
  },true);
  decorateWhiteboards();

})();
