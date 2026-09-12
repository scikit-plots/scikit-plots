"""Local, progressive search, faceting and sorting for rendered gallery cards."""

CONTAINER_CLASS = "sk-collection"
SEARCHABLE_CLASS = "sk-collection-searchable"

ASSET_CSS = r"""/* Compact gallery controls respond to the nearest gallery's available width. */
.sk-collection-data,.sk-collection-label { display:none !important; }
.sk-collection-searchable { container:sk-gallery / inline-size; min-inline-size:0; }
.sk-collection-controls {
  display:flex; align-items:center; gap:.375rem; margin:.75rem 0 .25rem;
  padding:0; border:0; min-inline-size:0; max-inline-size:100%;
  color:var(--pst-color-text-base,CanvasText);
}
.sk-collection-control.sk-collection-search {
  display:grid; grid-template-columns:minmax(0,1fr) 3rem; align-items:stretch;
  flex:1 1 auto; inline-size:100%; max-inline-size:100%; min-inline-size:0; margin:0; gap:0;
}
.sk-collection-search input,.sk-collection-submit,.sk-collection-panel :is(input,select,textarea,button),
.sk-collection-chip {
  box-sizing:border-box; min-inline-size:0; max-inline-size:100%; min-block-size:2.75rem;
  border:1px solid var(--pst-color-border,#888); border-radius:.5rem; padding:.5rem .625rem;
  font:inherit; font-size:1rem; line-height:1.35;
  color:var(--pst-color-text-base,CanvasText); background:var(--pst-color-background,Canvas);
}
.sk-collection-search > input {
  inline-size:100%; min-inline-size:0; max-inline-size:100%; border-radius:0; padding-inline-start:1rem;
  border-start-start-radius:2rem; border-end-start-radius:2rem;
}
.sk-collection-submit {
  inline-size:3rem; min-inline-size:3rem; max-inline-size:3rem; display:grid; place-items:center; border-radius:0;
  border-start-end-radius:2rem; border-end-end-radius:2rem; border-inline-start:0;
  background:var(--pst-color-surface,Canvas); cursor:pointer;
}
.sk-collection-submit svg { inline-size:1.25rem; block-size:1.25rem; fill:currentColor; }
.sk-collection-overflow {
  flex:0 0 2.75rem; min-block-size:2.75rem; border:0; border-radius:50%; padding:0;
  font:inherit; font-size:1.5rem; color:var(--pst-color-text-base,CanvasText);
  background:var(--pst-color-surface,Canvas); cursor:pointer;
}
.sk-collection-panel {
  display:flex; flex-wrap:wrap; gap:.625rem; padding:.75rem; margin-block:.5rem;
  border:1px solid var(--pst-color-border,#888); border-radius:.875rem;
  max-block-size:32rem; max-block-size:min(32rem,65dvh); overflow:auto; overscroll-behavior:contain;
  scrollbar-gutter:stable; box-sizing:border-box;
  color:var(--pst-color-text-base,CanvasText); background:var(--pst-color-surface,Canvas);
  box-shadow:0 .5rem 1.5rem color-mix(in srgb, CanvasText 14%, transparent);
}
.sk-collection-panel[hidden],.sk-collection-panel [hidden],.sk-collection-chips[hidden],
.sk-collection-additions[hidden] { display:none !important; }
.sk-collection-control { display:flex; flex:1 1 10rem; min-inline-size:0; margin:0; flex-direction:column; gap:.2rem; }
.sk-collection-control-label { font-size:.8rem; font-weight:600; overflow-wrap:anywhere; }
.sk-collection-panel :is(select,input,textarea) { inline-size:100%; }
.sk-collection-panel select { text-overflow:ellipsis; }
.sk-collection-panel button,.sk-collection-chip { cursor:pointer; }
.sk-collection-panel button:disabled { opacity:.55; cursor:default; }
.sk-collection-reset,.sk-collection-revert { align-self:flex-end; }
.sk-collection-panel :is(details,.sk-collection-storage) { flex:1 0 100%; min-inline-size:0; }
.sk-collection-panel summary {
  cursor:pointer; padding:.5rem .25rem; min-block-size:2.75rem; box-sizing:border-box;
  font-size:.9rem; font-weight:600; overflow-wrap:anywhere;
}
.sk-collection-panel details[open] > summary { margin-block-end:.375rem; }
.sk-collection-add { display:grid; gap:.5rem; min-inline-size:0; }
.sk-collection-add-guide {
  display:grid; gap:.375rem; margin:0 0 .25rem; padding:.625rem;
  border:1px solid var(--pst-color-border,#888); border-radius:.625rem;
  background:var(--pst-color-background,Canvas); font-size:.8rem; line-height:1.45;
}
.sk-collection-add-guide p { margin:0; }
.sk-collection-capability,.sk-collection-source-help { margin:.25rem 0 0; font-size:.8rem; line-height:1.45; overflow-wrap:anywhere; }
.sk-collection-add-examples { display:flex; flex-wrap:wrap; gap:.375rem; }
.sk-collection-add-example {
  min-block-size:2rem !important; padding:.25rem .55rem !important; border-radius:999px !important;
  font-size:.78rem !important;
}
.sk-collection-source-preview {
  margin:0; padding:.4rem .5rem; border-inline-start:3px solid var(--pst-color-border,#888);
  border-radius:.25rem; font-size:.78rem; line-height:1.4; overflow-wrap:anywhere;
  background:color-mix(in srgb,var(--pst-color-background,Canvas) 82%,transparent);
}
.sk-collection-source-preview[data-state="ready"] { border-inline-start-color:var(--pst-color-success,var(--pst-color-primary,Highlight)); }
.sk-collection-source-preview[data-state="needs-site"] { border-inline-start-color:var(--pst-color-warning,var(--pst-color-border,#888)); }
.sk-collection-source-preview[data-state="invalid"] { border-inline-start-color:var(--pst-color-danger,var(--pst-color-border,#888)); }
.sk-collection-site-setup { margin-block-start:.125rem; }
.sk-collection-site-setup > summary { min-block-size:auto !important; padding:.25rem 0 !important; font-size:.8rem !important; }
.sk-collection-resolver-stub {
  max-inline-size:100%; overflow:auto; margin:.375rem 0 0; padding:.5rem;
  border:1px solid var(--pst-color-border,#888); border-radius:.375rem;
  background:var(--pst-color-background,Canvas); font-size:.72rem; line-height:1.45; white-space:pre;
}
.sk-collection-resolver-player {
  position:fixed; inline-size:1px; block-size:1px; inset:auto auto 0 0;
  overflow:hidden; clip-path:inset(50%); opacity:.01; pointer-events:none;
}
.sk-collection-add label,.sk-collection-export label { display:grid; gap:.25rem; font-size:.875rem; }
.sk-collection-source-sections { margin:0; padding:0; border:0; min-inline-size:0; }
.sk-collection-source-sections legend { margin-block-end:.25rem; font-size:.875rem; }
.sk-collection-source-section-options { display:flex; flex-wrap:wrap; gap:.375rem; }
.sk-collection-source-section-options label { position:relative; display:inline-flex; }
.sk-collection-source-section-options input { position:absolute; inline-size:1px; block-size:1px; opacity:0; pointer-events:none; }
.sk-collection-source-section-options span {
  display:inline-flex; align-items:center; min-block-size:2.25rem; padding:.35rem .65rem;
  border:1px solid var(--pst-color-border,#888); border-radius:999px; cursor:pointer; font-size:.8rem;
  color:var(--pst-color-text-base,CanvasText); background:var(--pst-color-surface,Canvas);
}
.sk-collection-source-section-options input:checked + span {
  border-color:var(--pst-color-primary,Highlight); box-shadow:inset 0 0 0 1px var(--pst-color-primary,Highlight);
}
.sk-collection-source-section-options input:focus-visible + span { outline:2px solid var(--pst-color-primary,Highlight); outline-offset:2px; }
.sk-collection-panel .sk-collection-remember { display:flex; align-items:center; gap:.5rem; margin:0; font-size:.875rem; }
.sk-collection-panel .sk-collection-remember input {
  appearance:auto; flex:0 0 1.125rem; inline-size:1.125rem; block-size:1.125rem; min-block-size:1.125rem;
  margin:0; padding:0; accent-color:var(--pst-color-primary,Highlight);
}
.sk-collection-panel .sk-collection-export-preview { resize:vertical; min-block-size:6rem; font-family:monospace; font-size:.8rem; margin-block:.5rem; }
.sk-collection-note { font-size:.8rem; line-height:1.5; margin:.25rem 0; flex-basis:100%; overflow-wrap:anywhere; }
.sk-collection-status { margin:.25rem .125rem .5rem; font-size:.8rem; line-height:1.5; overflow-wrap:anywhere; }
.sk-collection-chips { display:flex; flex-wrap:wrap; gap:.375rem; margin:0 0 .5rem; }
.sk-collection-chip { border-radius:1.5rem; font-size:.8rem; overflow-wrap:anywhere; text-align:start; }
.sk-collection-suggestions {
  display:flex; align-items:center; flex-wrap:wrap; gap:.375rem; margin:0 0 .75rem;
  min-inline-size:0; color:var(--pst-color-text-muted,var(--pst-color-text-base,CanvasText));
}
.sk-collection-suggestions[hidden] { display:none !important; }
.sk-collection-suggestions-label { font-size:.78rem; font-weight:600; margin-inline-end:.125rem; }
.sk-collection-suggestion {
  min-block-size:2rem; max-inline-size:min(100%,24rem); border:1px solid var(--pst-color-border,#888);
  border-radius:1.25rem; padding:.25rem .625rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  font:inherit; font-size:.78rem; line-height:1.25; cursor:pointer;
  color:var(--pst-color-text-base,CanvasText); background:var(--pst-color-surface,Canvas);
}
.sk-collection-suggestion-kind { opacity:.72; }
.sk-collection-view-storage { border-block-start:1px solid var(--pst-color-border,#888); padding-block-start:.625rem; }
.sk-collection-controls :is(input,button):focus-visible,.sk-collection-panel :is(input,select,textarea,button,summary):focus-visible,
.sk-collection-chip:focus-visible,.sk-collection-suggestion:focus-visible { outline:2px solid var(--pst-color-primary,Highlight); outline-offset:2px; }
@media (hover:hover) {
  .sk-collection-controls button:hover,.sk-collection-panel button:not(:disabled):hover,
  .sk-collection-chip:hover,.sk-collection-suggestion:hover { border-color:var(--pst-color-primary,Highlight); color:var(--pst-color-primary,Highlight); }
}
.sk-collection-empty { padding:.75rem; border-inline-start:3px solid var(--pst-color-border,#888); }
.sk-collection-hidden,.sk-collection-section-hidden { display:none !important; }
@container sk-gallery (max-width:22rem) {
  .sk-collection-panel { padding:.5rem; }
  .sk-collection-panel > .sk-collection-control { flex-basis:100%; }
}
@media (forced-colors:active) { .sk-collection-panel button:disabled { color:GrayText; opacity:1; } }
@media print {
  .sk-collection-searchable { container-type:normal; }
  .sk-collection-controls,.sk-collection-panel,.sk-collection-status,.sk-collection-empty,
  .sk-collection-chips,.sk-collection-suggestions { display:none !important; }
  .sk-collection-hidden,.sk-collection-section-hidden { display:revert !important; }
}
"""

ASSET_JS = r"""/* Local gallery controls with optional, explicit saved additions. */
(function () {
  'use strict';
  function text(value) { return value == null ? '' : String(value); }
  function fold(value) { return text(value).normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase(); }
  function ownData(root) {
    var carrier = Array.from(root.children).find(function (el) { return el.classList.contains('sk-collection-data'); });
    if (!carrier) return {records: {}, facets: [], sorts: ['title']};
    try { var data = JSON.parse(carrier.textContent); return data.version === 1 ? data : {records: {}}; }
    catch (_) { return {records: {}}; }
  }
  function owner(root, card) {
    var section = card.closest('section');
    if (section && section !== root && root.contains(section)) return section;
    var block = card;
    while (block.parentElement && block.parentElement !== root) block = block.parentElement;
    var previous = block.previousElementSibling;
    while (previous && !previous.matches('p.rubric')) previous = previous.previousElementSibling;
    return previous;
  }
  function values(item, field) {
    var value = item.fields[field];
    if (value == null || value === '') return [];
    return Array.from(new Set((Array.isArray(value) ? value : [value]).map(text).filter(Boolean)));
  }
  function label(field) { return field.replace(/[_.-]+/g, ' ').replace(/^./, function (c) { return c.toUpperCase(); }); }
  function element(tag, cls, content) {
    var el = document.createElement(tag); if (cls) el.className = cls;
    if (content != null) el.textContent = content; return el;
  }
  var instance=0;
  function run(root) {
    root.querySelectorAll('img:not([loading])').forEach(function (img) {
      img.loading = 'lazy'; img.decoding = 'async';
    });
    if (!root.classList.contains('sk-collection-searchable') || root.hasAttribute('data-sk-enhanced')) return;
    var config = ownData(root), caches = new Map();
    var cards = Array.from(root.querySelectorAll('.sd-card')).filter(function (card) {
      if (card.closest('.sk-collection-searchable') !== root) return false;
      var parentCard = card.parentElement.closest('.sd-card');
      return !parentCard || !root.contains(parentCard);
    }).map(function (card, index) {
      var dataRoot = card.closest('.sk-collection');
      if (!caches.has(dataRoot)) caches.set(dataRoot, ownData(dataRoot));
      var source = caches.get(dataRoot), key = Array.from(card.classList).find(function (c) { return c.startsWith('sk-collection-item-'); });
      var meta = (source.records || {})[key] || {};
      var titleNode = card.querySelector('.sd-card-title');
      var title = meta.title || (titleNode ? titleNode.textContent.trim() : card.textContent.trim());
      return {card:card, node:card.closest('.sd-col') || card, index:index,
        title:title, fields:Object.assign(Object.create(null), {title:title}, meta.fields || {}),
        search:fold(title + ' ' + (meta.search || '') + ' ' + card.textContent),
        owner:owner(root,card)};
    });
    if (!cards.length) return;
    var bar = element('div','sk-collection-controls');
    bar.setAttribute('role','group');
    var carrier = Array.from(root.children).find(function (el) { return el.classList.contains('sk-collection-label'); });
    var searchLabel = carrier ? carrier.textContent.trim() : 'Search this gallery';
    bar.setAttribute('aria-label', searchLabel + ' controls');
    function control(caption, el) {
      var wrap = element('label','sk-collection-control');
      wrap.append(element('span','sk-collection-control-label',caption),el);bar.append(wrap);return el;
    }
    var input = element('input');input.type='search';input.maxLength=1024;input.autocomplete='off';input.placeholder=searchLabel;
    input.setAttribute('aria-label',searchLabel);control('Search',input);
    var facetControls=[];
    if (config.interactive) (config.facets || []).forEach(function (field) {
      var choices = Array.from(new Set(cards.flatMap(function (c) { return values(c,field); })));
      if (!choices.length) return;
      choices.sort(function (a,b) { return a.localeCompare(b, undefined, {numeric:true, sensitivity:'base'}); });
      var select=element('select');select.setAttribute('aria-label',label(field));
      var all=element('option',null,'All '+label(field).toLowerCase());all.value='';select.append(all);
      choices.forEach(function (value) { var option=element('option',null,value);option.value=value;select.append(option); });
      control(label(field),select);facetControls.push({field:field,select:select});
    });
    var sort=null;
    var buckets=new Map();cards.forEach(function (c) { if (!buckets.has(c.node.parentElement)) buckets.set(c.node.parentElement,[]);buckets.get(c.node.parentElement).push(c); });
    var reversedRows=new Set(Array.from(buckets.keys()).filter(function(parent){return parent.classList.contains('sd-flex-row-reverse');}));
    if (config.interactive) {
      sort=element('select');var original=element('option',null,'Original order');original.value='';sort.append(original);
      (config.sorts || ['title']).filter(function(field) {
        return cards.some(function(c) { return c.fields[field]!=null && c.fields[field]!==''; });
      }).forEach(function (field) {
        [['asc','ascending'],['desc','descending']].forEach(function (direction) {
          var caption=field==='title' ? ('Title '+(direction[0]==='asc'?'A–Z':'Z–A')) : label(field)+' '+direction[1];
          var option=element('option',null,caption);option.value=field+':'+direction[0];sort.append(option);
        });
      });
      sort.setAttribute('aria-label',buckets.size>1 ? 'Sort within categories' : 'Sort');
      sort.title=buckets.size>1 ? 'Sort within each category' : 'Sort this gallery';
      control('Sort',sort);
    }
    var reset=element('button','sk-collection-reset','Reset view');reset.type='button';bar.append(reset);
    var initialCards=cards.slice(), added=[];
    var panel=element('div','sk-collection-panel');panel.hidden=true;
    do{panel.id='sk-gallery-options-'+(++instance);}while(document.getElementById(panel.id));
    var toggle=element('button','sk-collection-overflow','⋮');toggle.type='button';
    toggle.setAttribute('aria-label','Gallery options');toggle.setAttribute('aria-expanded','false');toggle.setAttribute('aria-controls',panel.id);
    var oldSearchWrap=input.parentElement;
    var searchWrap=element('form','sk-collection-control sk-collection-search');searchWrap.setAttribute('role','search');searchWrap.setAttribute('aria-label',searchLabel);
    searchWrap.append(input);oldSearchWrap.replaceWith(searchWrap);
    searchWrap.addEventListener('submit',function(event){event.preventDefault();apply();input.focus();});
    var searchButton=element('button','sk-collection-submit','');searchButton.type='submit';searchButton.setAttribute('aria-label',searchLabel);
    var svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('aria-hidden','true');
    var path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('d','M10 3a7 7 0 1 0 0 14 7 7 0 0 0 0-14Zm0 2a5 5 0 1 1 0 10 5 5 0 0 1 0-10Zm5.5 10 5.2 5.2-1.4 1.4-5.2-5.2Z');svg.append(path);searchButton.append(svg);
    searchWrap.append(searchButton);
    Array.from(bar.children).forEach(function(child){if(child!==searchWrap)panel.append(child);});
    var close=element('button','sk-collection-close','Close options');close.type='button';
    var revert=element('button','sk-collection-revert','Revert to initial gallery');revert.type='button';
    revert.title='Restore the original cards and order; clear your added cards and their saved copy';
    var note=element('p','sk-collection-note','Reset view clears search, filters and sorting. Revert also clears your additions and their saved copy.');
    panel.append(revert,note,close);bar.append(toggle);
    panel.setAttribute('role','group');panel.setAttribute('aria-label','Gallery options');
    function showPanel(open){panel.hidden=!open;toggle.setAttribute('aria-expanded',String(open));}
    toggle.addEventListener('click',function(){showPanel(panel.hidden);});
    close.addEventListener('click',function(){showPanel(false);toggle.focus();});
    panel.addEventListener('keydown',function(event){if(event.key==='Escape'){event.stopPropagation();showPanel(false);toggle.focus();}});
    document.addEventListener('pointerdown',function(event){
      if(!panel.hidden && !panel.contains(event.target) && !toggle.contains(event.target))showPanel(false);
    });
    var saveAdditions=function(){}, restoreAdditions=function(){}, forgetSaved=function(){}, refreshAdditions=function(){};
    var saveView=function(){}, restoreView=function(){}, forgetView=function(){}, rememberView=null,viewStorageDirty=false,viewForget=null;
    var storageDirty=false;
      function normalizeHandle(token){
        var text=String(token||'');
        try{text=decodeURIComponent(text);}catch(_){}
        if(text.normalize)text=text.normalize('NFC');
        if(text.startsWith('@'))text=text.slice(1);
        var chars=Array.from(text),separators='_-.·';
        if(!chars.length||chars.length>30||separators.includes(chars[0])||separators.includes(chars[chars.length-1]))return null;
        for(var i=0;i<chars.length;i++){var ch=chars[i];if(separators.includes(ch))continue;if(!/[\p{L}\p{N}\p{M}]/u.test(ch))return null;}
        return text;
      }
      function reference(raw,preferredSection){
        if(typeof raw!=='string'||raw.length>2048||/[\u0000-\u001f\u007f]/.test(raw))throw Error('Enter a video ID, channel, playlist, post, or valid YouTube URL.');
        var value=raw.trim();if(!value)throw Error('Enter a video ID, channel name, channel ID, @handle, playlist, post, or YouTube URL.');
        var sectionNames=new Set(['videos','shorts','streams','courses']);var preferred=sectionNames.has(preferredSection)?preferredSection:null;
        if(/^UC[A-Za-z0-9_-]{22}$/.test(value))return preferred
          ? {key:'channel:'+value+':'+preferred,url:'https://www.youtube.com/channel/'+value+'/'+preferred,channelId:value,section:preferred,kind:'channel'}
          : {key:'channel:'+value,url:'https://www.youtube.com/channel/'+value,channelId:value,kind:'channel'};
        if(value.startsWith('@'))value='https://www.youtube.com/'+value;
        if(/^[A-Za-z0-9_-]{11}$/.test(value))return {key:'video:'+value,url:'https://www.youtube.com/watch?v='+value,videoId:value,kind:'video'};
        if(!value.includes('://') && /^(?:(?:www|m|music)\.)?youtube\.com\/|^youtu\.be\/|^www\.youtube-nocookie\.com\//i.test(value))value='https://'+value;
        if(!value.includes('://') && !/[/?#]/.test(value) && value.length<=100){
          return preferred
            ? {key:'channel-name:'+fold(value)+':'+preferred,url:'https://www.youtube.com/results?search_query='+encodeURIComponent(value),name:value,section:preferred,kind:'channel'}
            : {key:'channel-name:'+fold(value),url:'https://www.youtube.com/results?search_query='+encodeURIComponent(value),name:value,kind:'channel'};
        }
        var u;try{u=new URL(value);}catch(_){throw Error('Enter a video ID, channel name, channel ID, @handle, playlist, post, or YouTube URL.');}
        if(u.username || u.password || u.port)throw Error('Use a standard YouTube URL without credentials or a custom port.');
        var host=u.hostname.toLowerCase();
        var youtubeHosts=['youtube.com','www.youtube.com','m.youtube.com','music.youtube.com'];
        var recognized=youtubeHosts.includes(host)||host==='youtu.be'||host==='www.youtube-nocookie.com';
        if(!recognized)throw Error('Use a YouTube URL.');
        if(u.protocol==='http:')u.protocol='https:';
        if(u.protocol!=='https:')throw Error('Use an HTTPS YouTube URL. HTTP YouTube links are upgraded automatically.');
        var parts=u.pathname.split('/').filter(Boolean),id;
        if(host==='youtu.be'&&parts.length===1)id=parts[0];
        else if(youtubeHosts.includes(host)){
          if(u.pathname==='/watch'&&u.searchParams.getAll('v').length===1)id=u.searchParams.get('v');
          else if(u.pathname==='/playlist'&&u.searchParams.getAll('list').length===1){
            var list=u.searchParams.get('list');
            if(!/^[A-Za-z0-9_-]{10,128}$/.test(list))throw Error('Use a valid YouTube playlist URL.');
            return {key:'playlist:'+list,url:'https://www.youtube.com/playlist?list='+encodeURIComponent(list),playlistId:list,section:'playlist',kind:'playlist'};
          }
          else if(parts.length===2&&['embed','shorts','live'].includes(parts[0]))id=parts[1];
          else if(parts.length===2&&parts[0]==='post'){
            var postId=parts[1];
            if(!/^[A-Za-z0-9_-]{16,200}$/.test(postId))throw Error('Use a valid YouTube post URL.');
            return {key:'post:'+postId,url:'https://www.youtube.com/post/'+postId,postId:postId,kind:'post'};
          }
          else if(parts.length===1||parts.length===2){
            var normalizedHandle=normalizeHandle(parts[0]);
            if(normalizedHandle && String(parts[0]).startsWith('@')){
              var handleToken='@'+normalizedHandle,handleUrl='https://www.youtube.com/@'+encodeURIComponent(normalizedHandle);
              var handleSection=parts.length===2?parts[1].toLowerCase():preferred;
              if(handleSection && !sectionNames.has(handleSection))throw Error('Supported channel sections are /videos, /shorts, /streams, and /courses.');
              return handleSection
                ? {key:'channel:'+fold(handleToken)+':'+handleSection,url:handleUrl+'/'+handleSection,name:normalizedHandle,handle:handleToken,section:handleSection,kind:'channel'}
                : {key:'channel:'+fold(handleToken),url:handleUrl,name:normalizedHandle,handle:handleToken,kind:'channel'};
            }
          }
          else if((parts.length===2||parts.length===3) && parts[0]==='channel' && /^UC[A-Za-z0-9_-]{22}$/.test(parts[1])){
            var idSection=parts.length===3?parts[2].toLowerCase():preferred;
            if(idSection && !sectionNames.has(idSection))throw Error('Supported channel sections are /videos, /shorts, /streams, and /courses.');
            return idSection
              ? {key:'channel:'+parts[1]+':'+idSection,url:'https://www.youtube.com/channel/'+parts[1]+'/'+idSection,channelId:parts[1],section:idSection,kind:'channel'}
              : {key:'channel:'+parts[1],url:'https://www.youtube.com/channel/'+parts[1],channelId:parts[1],kind:'channel'};
          }
        }
        if(host==='www.youtube-nocookie.com'&&parts.length===2&&parts[0]==='embed')id=parts[1];
        if(id && /^[A-Za-z0-9_-]{11}$/.test(id))return {key:'video:'+id,url:'https://www.youtube.com/watch?v='+id,videoId:id,kind:'video'};
        throw Error('Enter a valid video, channel section, playlist, post URL, channel ID, or @handle URL.');
      }
    var initialYoutubeRefs=[];
    cards.forEach(function(c){c.card.querySelectorAll('a[href],iframe[src]').forEach(function(el){if(el.closest('.sk-collection-searchable')!==root)return;try{initialYoutubeRefs.push(reference(el.href||el.src));}catch(_){}});});
    var youtubeGallery=initialYoutubeRefs.length>0;
    var channelAddingMode=youtubeGallery && initialYoutubeRefs.every(function(ref){return ref.kind==='channel';});
    var videoAddingMode=youtubeGallery && !channelAddingMode;
    if(youtubeGallery){
      var additionsSection=element('section','sk-collection-additions');additionsSection.hidden=true;
      additionsSection.append(element('h3',null,'Your additions'));
      var additionsRow=element('div',initialCards[0].node.parentElement.className);additionsRow.classList.remove('sd-flex-row-reverse');
      additionsSection.append(additionsRow);root.append(additionsSection);buckets.set(additionsRow,[]);
      var addDetails=element('details','sk-collection-add-details');addDetails.append(element('summary',null,channelAddingMode?'Add channel':'Add video, channel section, playlist, or post'));
      var addGuide=element('div','sk-collection-add-guide');
      if(channelAddingMode){
        addGuide.append(element('p',null,'Paste an @handle or channel URL. Added channel cards stay simple and open YouTube.'));
      }else{
        addGuide.append(
          element('p',null,'Works without site setup: exact video IDs and video, Short, live, or watch URLs. A watch URL with list= still plays its exact v= video.'),
          element('p',null,'Needs lookup first: latest channel sections, newest playlist items, and video attachments inside YouTube posts. Successful lookup still renders the same single-video player.')
        );
        var customResolverReady=typeof window.skCollectionResolveYouTubeLatest==='function';
        var browserKeyReady=typeof window.skCollectionYouTubeDataApiKey==='string' && /^[A-Za-z0-9_-]{20,128}$/.test(window.skCollectionYouTubeDataApiKey.trim());
        addGuide.append(element('p','sk-collection-capability',customResolverReady?'Site capability: same-origin/latest-source resolver detected.':browserKeyReady?'Site capability: browser YouTube Data API key detected for supported channel/playlist lookups; posts still need the custom resolver.':'Site capability: direct links are ready. Latest @handle/playlist/post lookup is optional and is not configured on this site.'));
        var siteSetup=element('details','sk-collection-site-setup');siteSetup.append(element('summary',null,'Site setup for latest/post sources (optional)'));
        siteSetup.append(element('p',null,'Recommended: keep provider credentials server-side and expose a same-origin resolver before the gallery script. The server should validate the source and return one exact video ID.'));
        var resolverStub=element('pre','sk-collection-resolver-stub');resolverStub.append(element('code',null,"window.skCollectionResolveYouTubeLatest = async (source) => {\n  const r = await fetch('/api/youtube/resolve', {\n    method: 'POST',\n    headers: {'Content-Type': 'application/json'},\n    body: JSON.stringify(source)\n  });\n  if (!r.ok) throw new Error('Resolver unavailable');\n  return r.json(); // { videoId, title? }\n};"));siteSetup.append(resolverStub);addGuide.append(siteSetup);
      }
      var addForm=element('form','sk-collection-add');
      var urlLabel=element('label',null,'YouTube source');
      var urlInput=element('input');urlInput.type='text';urlInput.placeholder=channelAddingMode?'@handle or channel URL':'Video/Short/live URL or ID, @handle, playlist, or post URL';urlInput.maxLength=2048;urlInput.required=true;urlInput.autocomplete='off';urlInput.spellcheck=false;urlLabel.append(urlInput);
      var sourcePreview=element('p','sk-collection-source-preview',channelAddingMode?'Paste a channel source.':'Paste a source or choose an example below.');sourcePreview.setAttribute('role','status');sourcePreview.setAttribute('aria-live','polite');sourcePreview.dataset.state='neutral';
      if(videoAddingMode){
        var examples=element('div','sk-collection-add-examples');examples.setAttribute('role','group');examples.setAttribute('aria-label','Prefill a YouTube source example');
        [
          ['Video','https://www.youtube.com/watch?v=VIDEO_ID','VIDEO_ID'],
          ['Short','https://www.youtube.com/shorts/VIDEO_ID','VIDEO_ID'],
          ['Live','https://www.youtube.com/live/VIDEO_ID','VIDEO_ID'],
          ['Watch + list','https://www.youtube.com/watch?v=VIDEO_ID&list=PLAYLIST_ID','VIDEO_ID'],
          ['Channel latest','@channel','@channel'],
          ['Playlist latest','https://www.youtube.com/playlist?list=PLAYLIST_ID','PLAYLIST_ID'],
          ['Post','https://www.youtube.com/post/POST_ID','POST_ID']
        ].forEach(function(example){
          var button=element('button','sk-collection-add-example',example[0]);button.type='button';
          button.title='Prefill '+example[0]+' example';
          button.addEventListener('click',function(){
            urlInput.value=example[1];updateSourcePreview();urlInput.focus();
            var start=urlInput.value.indexOf(example[2]);if(start>=0&&typeof urlInput.setSelectionRange==='function')urlInput.setSelectionRange(start,start+example[2].length);
          });
          examples.append(button);
        });
        addGuide.append(examples);
      }
      addDetails.append(addGuide);
      var sectionField=null;
      if(videoAddingMode){
        sectionField=element('fieldset','sk-collection-source-sections');sectionField.append(element('legend',null,'Latest channel section'));
        var sectionOptions=element('div','sk-collection-source-section-options');
        [['videos','Videos'],['shorts','Shorts'],['streams','Streams'],['courses','Courses']].forEach(function(pair,index){
          var label=element('label');var radio=element('input');radio.type='radio';radio.name='sk-source-section-'+panel.id;radio.value=pair[0];radio.checked=index===0;
          label.append(radio,element('span',null,pair[1]));sectionOptions.append(label);
        });sectionField.append(sectionOptions,element('p','sk-collection-source-help','Used only with channel handles/URLs. Direct video, Short, live, playlist and post URLs keep their own source type.'));
      }
      var titleLabel=element('label',null,'Title (optional)');var titleInput=element('input');titleInput.type='text';titleInput.maxLength=200;titleInput.autocomplete='off';titleLabel.append(titleInput);
      var addButton=element('button',null,'Add card');addButton.type='submit';
      var feedback=element('p','sk-collection-note');feedback.setAttribute('role','status');
      addForm.append(urlLabel,sourcePreview);if(sectionField)addForm.append(sectionField);addForm.append(titleLabel,addButton,feedback);addDetails.append(addForm);panel.insertBefore(addDetails,panel.firstChild);
      function selectedSection(){if(!sectionField)return null;var picked=sectionField.querySelector('input:checked');return picked?picked.value:'videos';}
      function updateSourcePreview(){
        var raw=urlInput.value.trim();if(!raw){sourcePreview.dataset.state='neutral';sourcePreview.textContent=channelAddingMode?'Paste a channel source.':'Paste a source or choose an example above.';return;}
        var willNormalize=/^http:\/\//i.test(raw)||(!raw.includes('://')&&/^(?:(?:www|m|music)\.)?youtube\.com\/|^youtu\.be\/|^www\.youtube-nocookie\.com\//i.test(raw));
        var normalizedHint=willNormalize?' · HTTPS will be used.':'';
        try{
          var ref=reference(raw,selectedSection()),resolverReady=typeof window.skCollectionResolveYouTubeLatest==='function',keyReady=!!runtimeApiKey();
          if(channelAddingMode){sourcePreview.dataset.state=ref.kind==='channel'?'ready':'invalid';sourcePreview.textContent=(ref.kind==='channel'?'Ready · adds a simple clickable channel card.':'This gallery accepts channels only.')+normalizedHint;return;}
          if(ref.kind==='video'){sourcePreview.dataset.state='ready';sourcePreview.textContent='Ready · exact video ID known · plays inline with no API key.'+normalizedHint;return;}
          if(ref.kind==='post'){sourcePreview.dataset.state=resolverReady?'ready':'needs-site';sourcePreview.textContent=(resolverReady?'Post recognized · site resolver will look for an attached video, then render the normal player.':'Post recognized · needs optional site resolver to find an attached video ID. You can also paste the video/Short URL from the post.')+normalizedHint;return;}
          if(ref.kind==='playlist'){sourcePreview.dataset.state=(resolverReady||keyReady)?'ready':'needs-site';sourcePreview.textContent=((resolverReady||keyReady)?'Playlist recognized · site lookup will choose one newest published video, then render the normal player.':'Playlist recognized · newest-item lookup needs optional site resolver or configured Data API access.')+normalizedHint;return;}
          if(ref.kind==='channel'){
            var section=ref.section||selectedSection()||'videos';
            if(section==='courses'){sourcePreview.dataset.state=resolverReady?'ready':'needs-site';sourcePreview.textContent=(resolverReady?'Courses source recognized · site resolver will return one exact video ID.':'Courses source recognized · needs the optional site resolver; there is no stable built-in Courses-tab lookup.')+normalizedHint;return;}
            if(resolverReady||keyReady){sourcePreview.dataset.state='ready';sourcePreview.textContent='Channel '+section+' source recognized · site lookup will resolve one current video, then use the normal player.'+normalizedHint;return;}
            if(ref.channelId){sourcePreview.dataset.state='ready';sourcePreview.textContent='Canonical channel ID recognized · best-effort '+section+' section lookup can run without an API key.'+normalizedHint;return;}
            sourcePreview.dataset.state='needs-site';sourcePreview.textContent='Channel '+section+' source recognized · modern @handle → latest video needs optional site resolver or configured Data API access.'+normalizedHint;return;
          }
          sourcePreview.dataset.state='invalid';sourcePreview.textContent='This source cannot become a single video card.';
        }catch(error){sourcePreview.dataset.state='invalid';sourcePreview.textContent=error.message;}
      }
      urlInput.addEventListener('input',updateSourcePreview);
      if(sectionField)sectionField.addEventListener('change',updateSourcePreview);

      // Resolution and rendering are deliberately separate. A channel section or
      // playlist never gets a cosmetic pseudo-player: it must first become one
      // exact video id, then it uses the same renderer as a direct video.
      function runtimeApiKey(){
        var value=typeof window.skCollectionYouTubeDataApiKey==='string'?window.skCollectionYouTubeDataApiKey.trim():'';
        return /^[A-Za-z0-9_-]{20,128}$/.test(value)?value:'';
      }
      function resolverInput(source){return {kind:source.kind,section:source.section||null,handle:source.handle||null,channelId:source.channelId||null,playlistId:source.playlistId||null,postId:source.postId||null,name:source.name||null,url:source.url};}
      function checkedResolution(value){
        if(!value||typeof value.videoId!=='string'||!/^[A-Za-z0-9_-]{11}$/.test(value.videoId))throw Error('Latest-video resolver did not return a valid YouTube video ID.');
        var resolvedTitle=typeof value.title==='string'?value.title.trim():'';
        if(resolvedTitle.length>200||/[\u0000-\u001f\u007f]/.test(resolvedTitle))resolvedTitle='';
        return {ref:reference(value.videoId),title:resolvedTitle};
      }
      function withTimeout(value,ms,message,onTimeout){
        return new Promise(function(resolve,reject){
          var settled=false,timer=setTimeout(function(){if(settled)return;settled=true;try{if(typeof onTimeout==='function')onTimeout();}catch(_){}reject(Error(message));},ms);
          Promise.resolve(value).then(function(result){if(settled)return;settled=true;clearTimeout(timer);resolve(result);},function(error){if(settled)return;settled=true;clearTimeout(timer);reject(error);});
        });
      }
      function apiJSON(endpoint,params,key){
        if(typeof window.fetch!=='function')return Promise.reject(Error('This browser cannot contact the YouTube Data API.'));
        var query=new URLSearchParams();Object.keys(params).forEach(function(name){if(params[name]!=null&&params[name]!=='')query.set(name,String(params[name]));});query.set('key',key);
        var controller=typeof AbortController==='function'?new AbortController():null;
        var options={method:'GET',mode:'cors',credentials:'omit',referrerPolicy:'strict-origin-when-cross-origin'};if(controller)options.signal=controller.signal;
        var request;try{request=window.fetch('https://www.googleapis.com/youtube/v3/'+endpoint+'?'+query.toString(),options);}catch(error){return Promise.reject(error);}
        return withTimeout(request,12000,'YouTube Data API request timed out.',function(){if(controller)controller.abort();})
          .then(function(response){if(!response.ok)throw Error('YouTube Data API request failed ('+response.status+').');return withTimeout(response.json(),5000,'YouTube Data API response timed out.');});
      }
      function channelIdFor(source,key){
        if(source.channelId)return Promise.resolve(source.channelId);
        if(!source.handle)return Promise.reject(Error('A plain channel name is ambiguous. Use an @handle or canonical UC… channel ID, or configure window.skCollectionResolveYouTubeLatest.'));
        if(!key)return Promise.reject(Error('Resolving a modern @handle to its latest video needs window.skCollectionResolveYouTubeLatest or a referrer-restricted window.skCollectionYouTubeDataApiKey.'));
        return apiJSON('channels',{part:'id',forHandle:source.handle,maxResults:1},key).then(function(data){
          var id=data&&data.items&&data.items[0]&&data.items[0].id;if(!/^UC[A-Za-z0-9_-]{22}$/.test(id||''))throw Error('YouTube could not resolve this channel handle.');return id;
        });
      }
      var playerApiPromise=null;
      function youtubePlayerApi(){
        if(window.YT&&typeof window.YT.Player==='function')return Promise.resolve(window.YT);
        if(playerApiPromise)return playerApiPromise;
        playerApiPromise=new Promise(function(resolve,reject){
          var done=false,started=Date.now(),timer=null;
          function finish(error){if(done)return;done=true;if(timer)clearInterval(timer);if(error){playerApiPromise=null;reject(error);}else resolve(window.YT);}
          function inspect(){if(window.YT&&typeof window.YT.Player==='function')finish();else if(Date.now()-started>12000)finish(Error('YouTube player API did not become ready.'));}
          var script=document.querySelector('script[data-sk-youtube-iframe-api]');
          if(!script){script=document.createElement('script');script.src='https://www.youtube.com/iframe_api';script.async=true;script.setAttribute('data-sk-youtube-iframe-api','');script.addEventListener('error',function(){finish(Error('Could not load the YouTube player API.'));});(document.head||document.documentElement).append(script);}
          timer=setInterval(inspect,50);inspect();
        });
        return playerApiPromise;
      }
      function firstVideoInPlayerList(listType,list){
        return youtubePlayerApi().then(function(YT){return new Promise(function(resolve,reject){
          var host=element('div','sk-collection-resolver-player');host.setAttribute('aria-hidden','true');var target=element('div');host.append(target);document.body.append(host);
          var player=null,timer=null,done=false,started=Date.now();
          function finish(error,id){if(done)return;done=true;if(timer)clearInterval(timer);try{if(player&&typeof player.destroy==='function')player.destroy();}catch(_){}host.remove();if(error)reject(error);else resolve(id);}
          try{player=new YT.Player(target,{height:'1',width:'1',playerVars:{playsinline:1},events:{
            onReady:function(event){try{event.target.cuePlaylist({listType:listType,list:list,index:0});}catch(error){finish(error);return;}
              timer=setInterval(function(){try{var ids=event.target.getPlaylist();if(Array.isArray(ids)&&/^[A-Za-z0-9_-]{11}$/.test(ids[0]||'')){finish(null,ids[0]);return;}if(Date.now()-started>10000)finish(Error('YouTube did not expose a playable item for this source.'));}catch(error){finish(error);}},100);
            },
            onError:function(){finish(Error('YouTube could not resolve a playable item for this source.'));}
          }});}catch(error){finish(error);}
        });});
      }
      function sectionPlaylistId(channelId,section){
        var suffix=channelId.slice(2),prefix={videos:'UULF',shorts:'UUSH',streams:'UULV'}[section];return prefix?prefix+suffix:null;
      }
      function videoTitleFromApi(videoId,key){
        if(!key)return Promise.resolve('');
        return apiJSON('videos',{part:'snippet',id:videoId,maxResults:1},key).then(function(data){return data&&data.items&&data.items[0]&&data.items[0].snippet?text(data.items[0].snippet.title):'';}).catch(function(){return '';});
      }
      function latestPlaylistFromApi(source,key){
        if(!key)return Promise.reject(Error('Finding the newest published video in a playlist needs window.skCollectionResolveYouTubeLatest or a referrer-restricted window.skCollectionYouTubeDataApiKey.'));
        var best=null,pages=0;
        function page(token){
          if(++pages>20)return Promise.reject(Error('This playlist is too large for bounded browser resolution; use window.skCollectionResolveYouTubeLatest.'));
          return apiJSON('playlistItems',{part:'snippet,contentDetails,status',playlistId:source.playlistId,maxResults:50,pageToken:token||''},key).then(function(data){
            (data.items||[]).forEach(function(item){
              var id=item&&item.contentDetails&&item.contentDetails.videoId;if(!/^[A-Za-z0-9_-]{11}$/.test(id||''))return;
              if(item.status&&item.status.privacyStatus==='private')return;
              var when=Date.parse(text(item.contentDetails.videoPublishedAt));if(!Number.isFinite(when))return;
              var title=text(item.snippet&&item.snippet.title).trim();if(!best||when>best.when)best={videoId:id,title:title,when:when};
            });
            return data.nextPageToken?page(data.nextPageToken):best;
          });
        }
        return page('').then(function(value){if(!value)throw Error('This playlist has no playable public videos.');return checkedResolution(value);});
      }
      function builtinLatest(source){
        var key=runtimeApiKey();
        if(source.kind==='post')return Promise.reject(Error('A YouTube post URL does not expose an attached video ID. Paste the video or Short URL from the post, or configure window.skCollectionResolveYouTubeLatest to resolve this post.'));
        if(source.kind==='playlist')return latestPlaylistFromApi(source,key);
        if(source.kind!=='channel')return Promise.reject(Error('This source cannot be resolved to a single video.'));
        var section=source.section||'videos';
        if(section==='courses')return Promise.reject(Error('YouTube does not expose a stable public Courses-tab resolver. Configure window.skCollectionResolveYouTubeLatest for /courses.'));
        return channelIdFor(source,key).then(function(channelId){
          var list=sectionPlaylistId(channelId,section);if(!list)throw Error('Unsupported channel section.');
          // UULF/UUSH/UULV are compatibility playlist IDs rather than a published
          // Data API contract. Prefer dated playlistItems when a key is configured;
          // keep the iframe-player first-item path only as a zero-key UC… fallback.
          if(key)return latestPlaylistFromApi({kind:'playlist',playlistId:list,url:source.url},key);
          return firstVideoInPlayerList('playlist',list).then(function(videoId){return checkedResolution({videoId:videoId,title:''});});
        });
      }
      function resolveLatest(source){
        if(source.kind==='video')return {ref:source,title:''};
        if(typeof window.skCollectionResolveYouTubeLatest==='function'){
          var value;try{value=window.skCollectionResolveYouTubeLatest(resolverInput(source));}catch(error){return Promise.reject(error);}
          if(value==null)return builtinLatest(source);
          if(value&&typeof value.then==='function')return withTimeout(value,15000,'Site resolver timed out.').then(function(result){return result==null?builtinLatest(source):checkedResolution(result);});
          return checkedResolution(value);
        }
        return builtinLatest(source);
      }

      var initialRefs=new Set();cards.forEach(function(c){c.card.querySelectorAll('a[href],iframe[src]').forEach(function(el){if(el.closest('.sk-collection-searchable')!==root)return;try{initialRefs.add(reference(el.href||el.src,videoAddingMode?'videos':null).key);}catch(_){}});});
      var additionGeneration=0,pendingSources=new Set();
      function insertAddition(ref,requestedTitle,sourceRef,resolvedTitle){
        sourceRef=sourceRef||ref;resolvedTitle=typeof resolvedTitle==='string'?resolvedTitle:'';
        if(initialRefs.has(ref.key)||added.some(function(c){return c.key===ref.key;}))throw Error('This item is already in the gallery.');
        if(added.length>=100)throw Error('This gallery supports up to 100 added cards.');
        if(typeof requestedTitle!=='string'||requestedTitle.length>200||/[\u0000-\u001f\u007f]/.test(requestedTitle))throw Error('Use a title of up to 200 characters on one line.');
        var suppliedTitle=requestedTitle.trim(),title;
        var isVideo=ref.kind==='video'||ref.key.startsWith('video:');
        var isChannel=ref.kind==='channel'||ref.key.startsWith('channel:')||ref.key.startsWith('channel-name:');
        if(!isVideo&&!isChannel)throw Error('Resolve this source to one video before rendering it.');
        if(isChannel)title=suppliedTitle||ref.handle||ref.name||ref.channelId||'Channel';
        else title=suppliedTitle||resolvedTitle||ref.videoId;

        var index=cards.reduce(function(n,x){return Math.max(n,x.index+1);},0);
        var node=element('div','sd-col sd-d-flex-row docutils');
        var cardClasses='sd-card sd-sphinx-override sd-w-100 sd-shadow-sm sk-collection-item-'+index+' docutils';
        if(isChannel)cardClasses+=' sd-card-hover downstream-project-links';
        var card=element('div',cardClasses);
        var body=element('div','sd-card-body docutils');
        body.append(element('div','sd-card-title sd-font-weight-bold docutils',title));
        if(isVideo){
          var wrapper=element('div','video_wrapper');
          wrapper.setAttribute('style','aspect-ratio: 16 / 9; max-width: 100%; position: relative; width: 560px');
          var frame=element('iframe');frame.setAttribute('allowfullscreen','true');frame.loading='lazy';frame.referrerPolicy='strict-origin-when-cross-origin';
          frame.src='https://www.youtube.com/embed/'+ref.videoId;frame.setAttribute('style','border: 0; height: 100%; left: 0; position: absolute; top: 0; width: 100%; max-width: 100%');frame.title='youtube video player';
          wrapper.append(frame);body.append(wrapper);
        }
        card.append(body);
        if(isChannel){var channelLink=element('a','sd-stretched-link sd-hide-link-text reference external');channelLink.href=ref.url;channelLink.append(element('span',null,title));card.append(channelLink);}
        node.append(card);additionsRow.append(node);
        var c={card:card,node:node,index:index,title:title,requestedTitle:suppliedTitle,fields:{title:title},search:fold(title+' '+sourceRef.url+' '+ref.url),owner:additionsSection,key:ref.key,url:sourceRef.url,sourceKey:sourceRef.key,mode:isVideo?'video':'channel'};
        cards.push(c);added.push(c);buckets.get(additionsRow).push(c);return c;
      }
      function finishAddition(){saveAdditions();applySort();apply();urlInput.value='';titleInput.value='';feedback.textContent='Card added under Your additions. Current filters still apply.';updateSourcePreview();urlInput.focus();}
      function resolveAndInsert(source,title,generation){
        if(pendingSources.has(source.key))return Promise.reject(Error('This source is already being resolved.'));pendingSources.add(source.key);
        var resolution;try{resolution=resolveLatest(source);}catch(error){pendingSources.delete(source.key);return Promise.reject(error);}
        if(resolution&&resolution.ref){
          try{if(generation!==additionGeneration)throw Error('Addition was cancelled.');var card=insertAddition(resolution.ref,title,source,resolution.title);pendingSources.delete(source.key);return Promise.resolve(card);}
          catch(error){pendingSources.delete(source.key);return Promise.reject(error);}
        }
        return Promise.resolve(resolution).then(function(result){if(generation!==additionGeneration)throw Error('Addition was cancelled.');return insertAddition(result.ref,title,source,result.title);}).finally(function(){pendingSources.delete(source.key);});
      }
      addForm.addEventListener('submit',function(event){event.preventDefault();var submittedRef;try{
        submittedRef=reference(urlInput.value,selectedSection());
        if(channelAddingMode && submittedRef.kind!=='channel')throw Error('Add a channel ID, @handle, or channel URL in this gallery.');
        if(channelAddingMode||submittedRef.kind==='video'){insertAddition(submittedRef,titleInput.value,submittedRef,'');finishAddition();return;}
        var generation=additionGeneration;addButton.disabled=true;feedback.textContent='Finding the latest playable video…';
        resolveAndInsert(submittedRef,titleInput.value,generation).then(finishAddition).catch(function(error){if(error.message!=='Addition was cancelled.')feedback.textContent=error.message;}).finally(function(){addButton.disabled=false;});
      }catch(error){feedback.textContent=error.message;}});
      // Persistence is available only for a unique, author-assigned per-page ID.
      var scope=config.collectionId;
      var scopeUnique=/^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(scope||'') && Array.from(document.querySelectorAll('.sk-collection-searchable')).filter(function(el){return ownData(el).collectionId===scope;}).length===1;
      var storageKey=scopeUnique?'sk-gallery-additions:v1:'+encodeURIComponent(location.pathname)+':'+scope:null;
      var remember=null,storageFeedback=null,forget=null,lastSaved=null;
      if(storageKey){
        var storageControls=element('div','sk-collection-storage');
        var rememberLabel=element('label','sk-collection-remember');remember=element('input');remember.type='checkbox';
        rememberLabel.append(remember,element('span',null,'Remember my additions in this browser'));
        storageFeedback=element('p','sk-collection-note','Off. Additions last for this visit.');storageFeedback.setAttribute('role','status');
        forget=element('button',null,'Forget saved additions');forget.type='button';forget.hidden=true;
        storageControls.append(rememberLabel,storageFeedback,forget);panel.insertBefore(storageControls,note);
        var viewStorageKey='sk-gallery-view:v1:'+encodeURIComponent(location.pathname)+':'+scope;
        var viewControls=element('div','sk-collection-storage sk-collection-view-storage');
        var viewLabel=element('label','sk-collection-remember');rememberView=element('input');rememberView.type='checkbox';
        viewLabel.append(rememberView,element('span',null,'Remember my filters & sorting'));
        var viewFeedback=element('p','sk-collection-note','Off. Search text is never remembered.');viewFeedback.setAttribute('role','status');
        viewForget=element('button',null,'Forget saved view');viewForget.type='button';viewForget.hidden=true;
        viewControls.append(viewLabel,viewFeedback,viewForget);panel.insertBefore(viewControls,note);
        var lastViewSaved=null;
        function viewFailed(message){rememberView.checked=false;viewStorageDirty=true;viewForget.hidden=false;viewFeedback.textContent=message;}
        function viewPayload(){var facets={};facetControls.forEach(function(f){if(f.select.value)facets[f.field]=f.select.value;});return {version:1,remember:true,facets:facets,sort:sort?sort.value:''};}
        function validView(raw){
          if(typeof raw!=='string'||raw.length>16384)throw Error('Saved view is too large.');var payload=JSON.parse(raw);
          if(!payload||payload.version!==1||payload.remember!==true||!payload.facets||typeof payload.facets!=='object'||typeof payload.sort!=='string')throw Error('Unsupported saved view.');
          var valuesByField={};facetControls.forEach(function(f){valuesByField[f.field]=new Set(Array.from(f.select.options).map(function(o){return o.value;}));});
          Object.keys(payload.facets).forEach(function(field){if(!valuesByField[field]||!valuesByField[field].has(payload.facets[field]))delete payload.facets[field];});
          if(sort&&!Array.from(sort.options).some(function(o){return o.value===payload.sort;}))payload.sort='';return payload;
        }
        saveView=function(){
          if(!rememberView.checked)return;try{
            if(localStorage.getItem(viewStorageKey)!==lastViewSaved){viewFailed('Saved view changed in another tab. Saving is paused; turn remembering on again to use this view.');return;}
            var raw=JSON.stringify(viewPayload());localStorage.setItem(viewStorageKey,raw);lastViewSaved=raw;viewStorageDirty=false;viewForget.hidden=false;viewFeedback.textContent='Filters and sorting are saved in this browser. Search text stays private to this visit.';
          }catch(_){viewFailed('Could not save this view. The visible gallery is unchanged.');}
        };
        forgetView=function(){rememberView.checked=false;try{localStorage.removeItem(viewStorageKey);lastViewSaved=null;viewStorageDirty=false;viewForget.hidden=true;viewFeedback.textContent='Saved view cleared. Remembering is off.';}catch(_){viewFailed('Could not clear the saved view. Use Forget saved view to retry.');}};
        rememberView.addEventListener('change',function(){if(rememberView.checked){try{lastViewSaved=localStorage.getItem(viewStorageKey);}catch(_){viewFailed('Browser storage is unavailable. The visible gallery is unchanged.');return;}saveView();}else forgetView();apply();});
        viewForget.addEventListener('click',function(){forgetView();apply();});
        restoreView=function(){try{var raw=localStorage.getItem(viewStorageKey);if(raw===null)return;lastViewSaved=raw;var state=validView(raw);facetControls.forEach(function(f){f.select.value=state.facets[f.field]||'';});if(sort)sort.value=state.sort;rememberView.checked=true;viewForget.hidden=false;viewFeedback.textContent='Saved filters and sorting restored. Search text is not stored.';}catch(_){viewFailed('Saved view could not be restored. The published gallery is still available.');}};
        window.addEventListener('storage',function(event){if(event.key!==viewStorageKey&&event.key!==null)return;rememberView.checked=false;lastViewSaved=null;viewStorageDirty=event.key!==null&&event.newValue!==null;viewForget.hidden=!viewStorageDirty;viewFeedback.textContent='Saved view changed in another tab. Current controls remain usable; saving is off.';apply();});
        function storageFailed(message){remember.checked=false;storageDirty=true;forget.hidden=false;storageFeedback.textContent=message;}
        function checkPayload(raw){
          if(typeof raw!=='string'||raw.length>131072)throw Error('Saved additions are too large.');
          var payload=JSON.parse(raw);
          if(!payload||payload.version!==1||payload.remember!==true||!Array.isArray(payload.items)||payload.items.length>100)throw Error('Unsupported saved format.');
          var seen=new Set();
          return payload.items.map(function(item){
            if(!item||typeof item.url!=='string'||typeof item.title!=='string'||item.title.length>200||/[\u0000-\u001f\u007f]/.test(item.title))throw Error('Invalid saved record.');
            var ref=reference(item.url);if(seen.has(ref.key))throw Error('Duplicate saved record.');seen.add(ref.key);
            return {ref:ref,title:item.title};
          });
        }
        saveAdditions=function(){
          if(!remember.checked)return;
          try{
            // Detect a stale tab before any overwrite; a revoked choice must stay off.
            if(localStorage.getItem(storageKey)!==lastSaved){storageFailed('Saved additions changed in another tab. Saving is paused. Export your additions or turn remembering on again to save this view.');return;}
            var raw=JSON.stringify({version:1,remember:true,items:added.map(function(c){return {url:c.url,title:c.requestedTitle||''};})});
            if(raw.length>131072)throw Error('Storage limit.');
            localStorage.setItem(storageKey,raw);lastSaved=raw;storageDirty=false;forget.hidden=false;
            storageFeedback.textContent='Additions are saved in this browser. View settings use the separate preference below.';
          }catch(_){storageFailed('Could not save additions. They remain here for this visit; export them to keep a copy.');}
        };
        forgetSaved=function(){
          remember.checked=false;
          try{localStorage.removeItem(storageKey);lastSaved=null;storageDirty=false;forget.hidden=true;storageFeedback.textContent='Saved additions cleared. Remembering is off.';}
          catch(_){storageFailed('Could not clear browser storage. Saved additions may return after reload. Use Forget saved additions to retry.');}
        };
        remember.addEventListener('change',function(){
          if(remember.checked){try{lastSaved=localStorage.getItem(storageKey);}catch(_){storageFailed('Browser storage is unavailable. Additions remain for this visit.');apply();return;}saveAdditions();}
          else forgetSaved();apply();
        });
        forget.addEventListener('click',function(){forgetSaved();apply();});
        restoreAdditions=function(){
          try{
            var raw=localStorage.getItem(storageKey);if(raw===null)return;lastSaved=raw;
            var records=checkPayload(raw),generation=additionGeneration,waiting=[];
            records.forEach(function(record){
              if(channelAddingMode){if(record.ref.kind==='channel'&&!initialRefs.has(record.ref.key))insertAddition(record.ref,record.title,record.ref,'');return;}
              if(record.ref.kind==='video'){if(!initialRefs.has(record.ref.key))insertAddition(record.ref,record.title,record.ref,'');return;}
              waiting.push(resolveAndInsert(record.ref,record.title,generation).catch(function(){return null;}));
            });
            remember.checked=true;forget.hidden=false;storageFeedback.textContent=waiting.length?'Saved direct additions restored; resolving saved latest-video sources…':'Saved additions restored in this browser.';
            if(waiting.length)Promise.all(waiting).then(function(results){var restored=results.filter(Boolean).length;storageFeedback.textContent=restored===waiting.length?'Saved additions restored in this browser.':'Saved additions restored where resolvable; some latest-video sources still need resolver access.';applySort();apply();});
          }catch(_){storageFailed('Saved additions could not be restored. The published gallery is still available. Forget saved additions to clear the saved copy.');}
        };
        window.addEventListener('storage',function(event){
          if(event.key!==storageKey && event.key!==null)return;
          remember.checked=false;lastSaved=null;storageDirty=event.key!==null&&event.newValue!==null;forget.hidden=!storageDirty;
          storageFeedback.textContent='Saving changed in another tab. Current cards remain for this visit; saving is off.';apply();
        });
      }
      var exportDetails=element('details','sk-collection-export');exportDetails.append(element('summary',null,'Export additions'));
      var exportInfo=element('p','sk-collection-note');exportInfo.setAttribute('role','status');
      var preview=element('textarea','sk-collection-export-preview');preview.readOnly=true;preview.rows=8;preview.setAttribute('aria-label','YAML export preview');preview.spellcheck=false;
      var download=element('button',null,'Download YAML');download.type='button';
      var downloadStatus=element('p','sk-collection-note');downloadStatus.setAttribute('role','status');
      exportDetails.append(exportInfo,preview,download,downloadStatus);panel.insertBefore(exportDetails,note);
      function yamlScalar(value){return JSON.stringify(String(value));}
      function exportRecords(){
        return added.reduce(function(records,c){
          if(channelAddingMode){
            if(c.mode==='channel' && /^https:\/\/www\.youtube\.com\/(?:@[^/?#]+|channel\/UC[^/?#]+)$/.test(c.url))records.push({url:c.url,title:c.title});
          }else if(c.mode==='video'){
            var id=c.key.startsWith('video:')?c.key.slice(6):'';
            if(/^[A-Za-z0-9_-]{11}$/.test(id))records.push({id:id,title:c.title});
          }
          return records;
        },[]);
      }
      function exportText(records){
        // Export the collection's real typed source model, never runtime DOM.
        // JSON quoting is valid YAML scalar syntax and safely preserves titles.
        records=records||exportRecords();
        var lines=[channelAddingMode?'channels:':'videos:'];
        if(!records.length)return lines[0]+' []\n';
        records.forEach(function(record){
          if(channelAddingMode){
            lines.push('  - url: '+yamlScalar(record.url));
            lines.push('    title: '+yamlScalar(record.title));
          }else{
            lines.push('  - id: '+yamlScalar(record.id));
            lines.push('    title: '+yamlScalar(record.title));
          }
        });
        return lines.join('\n')+'\n';
      }
      function refreshExport(){
        var records=exportRecords(),count=records.length;
        preview.value=exportText(records);download.disabled=count===0;
        exportInfo.textContent=channelAddingMode
          ? count+' channel records. The preview is directly reusable by youtube-gallery as a channels catalog.'
          : count+' video records. Resolved latest/playlist/post sources export as the exact video IDs currently rendered; restoring the downloaded catalog needs no resolver.';
      }
      download.addEventListener('click',function(){
        var url;
        try{
          url=URL.createObjectURL(new Blob([exportText()],{type:'application/yaml;charset=utf-8'}));
          var a=element('a');a.href=url;a.download=channelAddingMode?'youtube-channel-additions.yaml':'youtube-video-additions.yaml';a.hidden=true;document.body.append(a);
          try{a.click();}finally{a.remove();}
          var objectURL=url;setTimeout(function(){URL.revokeObjectURL(objectURL);},1000);
          downloadStatus.textContent='Download requested. The preview is the exact exported YAML.';
        }catch(_){if(url)URL.revokeObjectURL(url);downloadStatus.textContent='Download is unavailable here. Select and copy the YAML preview.';}
      });
      refreshAdditions=function(){refreshExport();};
    }


    var status=element('p','sk-collection-status');status.setAttribute('role','status');status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');
    var empty=element('p','sk-collection-empty','No matches. Try another search or reset the filters.');empty.hidden=true;
    var tokens=[];
    function matches(c, skip) {
      return tokens.every(function (token) { return c.search.includes(token); }) && facetControls.every(function (facet) {
        return facet.field===skip || !facet.select.value || values(c,facet.field).includes(facet.select.value);
      });
    }
    function applySort() {
      var field=sort && sort.value ? sort.value.split(':')[0] : null;
      var direction=sort && sort.value.endsWith(':desc') ? -1 : 1;
      var candidates=cards.map(function(c){return field ? c.fields[field] : null;}).filter(function(v){return v!=null && v!=='';});
      var numeric=candidates.length && candidates.every(function(v){return (typeof v==='number' || (typeof v==='string' && /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(v))) && Number.isFinite(Number(v));});
      var dates=!numeric && candidates.length && candidates.every(function(v){return typeof v==='string' && /^\d{4}-\d{2}-\d{2}(?:T|$)/.test(v) && Number.isFinite(Date.parse(v));});
      buckets.forEach(function (bucket,parent) {
        // Explicit live sorting uses normal visual order; Reset restores source reversal.
        parent.classList.toggle('sd-flex-row-reverse', !field && reversedRows.has(parent));
        var ordered=bucket.slice().sort(function(a,b) {
          if (!field) return a.index-b.index;
          var left=a.fields[field],right=b.fields[field];
          var missingA=left==null || left==='',missingB=right==null || right==='';
          if (missingA || missingB) return missingA===missingB ? a.index-b.index : missingA ? 1 : -1;
          var cmp=numeric ? Number(left)-Number(right) : dates ? Date.parse(left)-Date.parse(right) : text(left).localeCompare(text(right),undefined,{numeric:true,sensitivity:'base'});
          return cmp*direction || a.index-b.index;
        });
        // Reorder only on a sort change. moveBefore preserves iframe state where available.
        var position=parent.firstElementChild;
        ordered.forEach(function (c) {
          if (c.node===position) { position=position.nextElementSibling;return; }
          if (typeof parent.moveBefore==='function') parent.moveBefore(c.node,position);
          else parent.insertBefore(c.node,position);
        });
      });
    }
    function apply() {
      tokens=fold(input.value).trim().split(/\s+/).filter(Boolean);
      var shown=0,alive=new Map();
      cards.forEach(function(c){var hit=matches(c);c.node.classList.toggle('sk-collection-hidden',!hit);if(hit)shown++;if(c.owner)alive.set(c.owner,(alive.get(c.owner)||false)||hit);});
      alive.forEach(function(hit,heading){heading.classList.toggle('sk-collection-section-hidden',!hit);});
      facetControls.forEach(function(facet){
        var counts=new Map();
        cards.forEach(function(c){if(matches(c,facet.field)) values(c,facet.field).forEach(function(value){counts.set(value,(counts.get(value)||0)+1);});});
        Array.from(facet.select.options).forEach(function(option,index){if(!index)return;
          option.textContent=option.value+' ('+(counts.get(option.value)||0)+')';
        });
      });
      status.textContent=shown+' of '+cards.length+' cards';empty.hidden=shown!==0;
      reset.disabled=!input.value && !facetControls.some(function(f){return f.select.value;}) && !(sort && sort.value);
      revert.disabled=reset.disabled && !added.length && !storageDirty && !viewStorageDirty && !(youtubeGallery && remember && remember.checked) && !(rememberView&&rememberView.checked);
      if(youtubeGallery)additionsSection.hidden=!added.length;
      renderChips();renderSuggestions();refreshAdditions();
    }
    var chips=element('div','sk-collection-chips');chips.setAttribute('role','group');chips.setAttribute('aria-label','Active filters and sorting');
    var suggestions=element('div','sk-collection-suggestions');suggestions.setAttribute('role','group');suggestions.setAttribute('aria-label','Suggested searches, filters and sorting');
    var suggestionLabel=element('span','sk-collection-suggestions-label','Try');suggestions.append(suggestionLabel);
    var chipCache=new Map();
    function renderChips(){
      var active=[];
      if(input.value)active.push({key:'search',caption:'Search: '+input.value,clear:function(){input.value='';}});
      facetControls.forEach(function(f){if(f.select.value)active.push({key:'facet:'+f.field,caption:label(f.field)+': '+f.select.value,clear:function(){f.select.value='';}});});
      if(sort && sort.value)active.push({key:'sort',caption:sort.options[sort.selectedIndex].textContent,clear:function(){sort.value='';applySort();}});
      var live=new Set(active.map(function(c){return c.key;}));
      chipCache.forEach(function(button,key){if(!live.has(key)){button.remove();chipCache.delete(key);}});
      active.forEach(function(entry){
        var button=chipCache.get(entry.key);
        if(!button){button=element('button','sk-collection-chip');button.type='button';chips.append(button);chipCache.set(entry.key,button);}
        button.textContent=entry.caption+' ×';button.title='Clear '+entry.caption;button.setAttribute('aria-label','Clear '+entry.caption);
        button.onclick=function(){entry.clear();apply();var next=chips.querySelector('button');if(next)next.focus();else input.focus();};
      });
      chips.hidden=!active.length;
      toggle.title=active.length?'Gallery options ('+active.length+' active settings)':'Gallery options';
    }
    function suggestionButton(kind,caption,action){
      var button=element('button','sk-collection-suggestion');button.type='button';button.title=kind+': '+caption;button.setAttribute('aria-label',kind+': '+caption);
      var prefix=element('span','sk-collection-suggestion-kind',kind+' · ');button.append(prefix,document.createTextNode(caption));button.addEventListener('click',action);suggestions.append(button);
    }
    function renderSuggestions(){
      suggestions.replaceChildren(suggestionLabel);var made=0,query=fold(input.value).trim();
      if(query.length>=2){
        var seenTitles=new Set();cards.filter(function(c){return fold(c.title).includes(query)&&fold(c.title)!==query;}).slice(0,3).forEach(function(c){
          if(seenTitles.has(c.title)||made>=3)return;seenTitles.add(c.title);suggestionButton('Search',c.title,function(){input.value=c.title;apply();input.focus();});made++;
        });
      }
      var facetCandidates=[];facetControls.forEach(function(f){if(f.select.value)return;var counts=new Map();cards.forEach(function(c){if(matches(c,f.field))values(c,f.field).forEach(function(v){counts.set(v,(counts.get(v)||0)+1);});});counts.forEach(function(count,value){facetCandidates.push({facet:f,value:value,count:count});});});
      facetCandidates.sort(function(a,b){return b.count-a.count||a.value.localeCompare(b.value,undefined,{numeric:true,sensitivity:'base'});});
      facetCandidates.slice(0,Math.max(0,5-made)).forEach(function(entry){if(made>=5)return;suggestionButton(label(entry.facet.field),entry.value,function(){entry.facet.select.value=entry.value;saveView();apply();entry.facet.select.focus();});made++;});
      if(sort&&!sort.value&&made<6){
        var preferred=Array.from(sort.options).find(function(o){return o.value==='published:desc';})||Array.from(sort.options).find(function(o){return o.value==='title:asc';});
        if(preferred&&preferred.value){var caption=preferred.value==='published:desc'?'Newest first':preferred.textContent;suggestionButton('Sort',caption,function(){sort.value=preferred.value;applySort();saveView();apply();sort.focus();});made++;}
      }
      suggestions.hidden=made===0;
    }
    input.addEventListener('input',function(event){if(!event.isComposing)apply();});
    input.addEventListener('compositionend',apply);
    input.addEventListener('keydown',function(event){if(event.key==='Escape'){input.value='';apply();}});
    facetControls.forEach(function(f){f.select.addEventListener('change',apply);});
    if(sort)sort.addEventListener('change',function(){applySort();apply();});
    function resetView(){input.value='';facetControls.forEach(function(f){f.select.value='';});if(sort)sort.value='';applySort();apply();input.focus();}
    reset.addEventListener('click',resetView);
    revert.addEventListener('click',function(){
      if(youtubeGallery){additionGeneration++;pendingSources.clear();}
      added.forEach(function(c){c.node.remove();});added=[];cards=initialCards.slice();
      buckets.forEach(function(bucket,parent){buckets.set(parent,bucket.filter(function(c){return initialCards.includes(c);}));});
      if(youtubeGallery){urlInput.value='';titleInput.value='';feedback.textContent='Initial gallery restored.';updateSourcePreview();forgetSaved();forgetView();addDetails.open=false;exportDetails.open=false;}
      resetView();showPanel(storageDirty||viewStorageDirty);if(storageDirty && forget)forget.focus();else if(viewStorageDirty&&viewForget)viewForget.focus();
    });
    input.addEventListener('keydown',function(event){if(event.key==='Enter' && event.isComposing)event.preventDefault();});
    root.insertBefore(bar,root.firstChild);bar.after(panel);panel.after(status);status.after(chips);chips.after(suggestions);root.append(empty);
    root.setAttribute('data-sk-enhanced','true');restoreAdditions();restoreView();applySort();apply();
  }
  function start(){document.querySelectorAll('.sk-collection').forEach(function(root){try{run(root);}catch(error){console.warn('Gallery controls unavailable',error);}});}
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start,{once:true});else start();
})();
"""  # ruff: ignore[ambiguous-unicode-character-string]
