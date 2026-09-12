/* Behavioral checks with controlled browser storage; no resource fetching. */
const {JSDOM}=require('jsdom');
const fs=require('fs'),path=require('path'),assert=require('node:assert/strict');
const dir=process.argv[2],html=fs.readFileSync(path.join(dir,'index.html'),'utf8'),js=fs.readFileSync(path.join(dir,'_static/sk-collection.js'),'utf8');
const key='sk-gallery-additions:v1:%2Flearn%2Findex.html:youtube-videos';
let checks=0;
function load(saved={},configure=()=>{},page='/learn/index.html'){
 const dom=new JSDOM(html,{runScripts:'outside-only',url:'https://docs.example'+page});
 for(const [k,v] of Object.entries(saved))dom.window.localStorage.setItem(k,v);
 configure(dom.window);
 dom.window.eval(js);dom.window.document.dispatchEvent(new dom.window.Event('DOMContentLoaded'));
 return dom;
}
function scope(dom){return [...dom.window.document.querySelectorAll('.sk-collection-searchable')][1];}
function channelScope(dom){return [...dom.window.document.querySelectorAll('.sk-collection-searchable')][0];}
function count(root){return root.querySelectorAll('.sd-card').length;}
function add(dom,raw,title='Saved test'){
 const root=scope(dom),form=root.querySelector('.sk-collection-add'),fields=form.querySelectorAll('input[type=text]');
 fields[0].value=raw;fields[1].value=title;form.dispatchEvent(new dom.window.Event('submit',{bubbles:true,cancelable:true}));
}
function remember(dom){const el=scope(dom).querySelector('.sk-collection-remember input');el.checked=true;el.dispatchEvent(new dom.window.Event('change',{bubbles:true}));return el;}
function snapshot(dom){return Object.fromEntries(Object.keys(dom.window.localStorage).map(k=>[k,dom.window.localStorage.getItem(k)]));}
function search(dom,value){const el=scope(dom).querySelector('input[type=search]');el.value=value;el.dispatchEvent(new dom.window.Event('input',{bubbles:true}));}
function test(name,fn){fn();checks++;console.log('PASS '+name);}

test('No writes until visitor opts in; search/filter text excluded',()=>{
 const d=load();add(d,'abcdefghijk');search(d,'PRIVATE SEARCH');assert.equal(d.window.localStorage.length,0);
 remember(d);const raw=d.window.localStorage.getItem(key);assert(raw);assert(!raw.includes('PRIVATE SEARCH'));assert.equal(JSON.parse(raw).items.length,1);
 d.window.close();
});
test('Reload restores cards and consent; Reset retains saved additions',()=>{
 const d=load();add(d,'abcdefghijk');remember(d);const saved=snapshot(d);d.window.close();
 const restored=load(saved),r=scope(restored);assert.equal(count(r),14);assert(r.querySelector('.sk-collection-remember input').checked);
 search(restored,'saved');r.querySelector('.sk-collection-reset').click();assert.equal(count(r),14);assert(restored.window.localStorage.getItem(key));restored.window.close();
});
test('Revert clears only this gallery and restores original DOM identity',()=>{
 const d=load(),r=scope(d),original=[...r.querySelectorAll('.sd-card')];
 d.window.localStorage.setItem('unrelated','keep');
 add(d,'abcdefghijk');remember(d);r.querySelector('.sk-collection-revert').click();
 assert.deepEqual([...r.querySelectorAll('.sd-card')],original);assert.equal(d.window.localStorage.getItem(key),null);assert.equal(d.window.localStorage.getItem('unrelated'),'keep');
 assert(!r.querySelector('.sk-collection-remember input').checked);add(d,'abcdefghijl');assert.equal(d.window.localStorage.getItem(key),null);d.window.close();
});
test('Turning remembering off clears saved copy but retains visible cards',()=>{
 const d=load(),r=scope(d);add(d,'abcdefghijk');const box=remember(d);box.checked=false;box.dispatchEvent(new d.window.Event('change'));
 assert.equal(count(r),14);assert.equal(d.window.localStorage.getItem(key),null);d.window.close();
});
test('Runtime addition cards contain no extra removal chrome',()=>{
 const d=load(),r=scope(d);add(d,'abcdefghijk');assert.equal(r.querySelector('.sk-collection-additions .sk-collection-remove'),null);d.window.close();
});
test('Blocked read/write leaves page and visit additions usable',()=>{
 for(const operation of ['getItem','setItem']){
  const d=load({},w=>{w.Storage.prototype[operation]=()=>{throw Error('denied');};}),r=scope(d);
  add(d,'abcdefghijk');remember(d);assert.equal(count(r),14);assert(!r.querySelector('.sk-collection-remember input').checked);
  assert.match(r.querySelector('.sk-collection-storage').textContent,/unavailable|Could not/);d.window.close();
 }
});
test('Failed storage deletion keeps Revert retry visible and restores cards',()=>{
 const d=load(),r=scope(d);add(d,'abcdefghijk');remember(d);
 const remove=d.window.Storage.prototype.removeItem;d.window.Storage.prototype.removeItem=()=>{throw Error('denied');};
 r.querySelector('.sk-collection-revert').click();assert.equal(count(r),13);assert(!r.querySelector('.sk-collection-panel').hidden);
 const forget=[...r.querySelectorAll('button')].find(el=>el.textContent==='Forget saved additions');assert(!forget.hidden);assert.equal(d.window.document.activeElement,forget);
 d.window.Storage.prototype.removeItem=remove;forget.click();assert.equal(d.window.localStorage.getItem(key),null);assert(r.querySelector('.sk-collection-revert').disabled);d.window.close();
});
test('Malformed/oversized/duplicate saved records restore nothing',()=>{
 const record={url:'https://youtu.be/abcdefghijk',title:'Valid'};
 for(const raw of ['{','x'.repeat(131073),JSON.stringify({version:1,remember:false,items:[record]}),JSON.stringify({version:1,remember:true,items:[record,{url:'javascript:alert(1)',title:'bad'}]}),JSON.stringify({version:1,remember:true,items:[record,record]}),JSON.stringify({version:1,remember:true,items:Array(101).fill(record)})]){
  const d=load({[key]:raw}),r=scope(d);assert.equal(count(r),13);assert(!r.querySelector('.sk-collection-remember input').checked);d.window.close();
 }
});
test('Published duplicate is skipped after docs rebuild',()=>{
 const d=load({[key]:JSON.stringify({version:1,remember:true,items:[{url:'https://youtu.be/UNzCG3lw6O0',title:'Already published'},{url:'https://youtu.be/abcdefghijk',title:'New'}]})});
 assert.equal(count(scope(d)),14);d.window.close();
});
test('Different pages and duplicate collection IDs cannot share saved state',()=>{
 const payload=JSON.stringify({version:1,remember:true,items:[{url:'https://youtu.be/abcdefghijk',title:'New'}]});
 const d=load({[key]:payload},()=>{},'/other.html');assert.equal(count(scope(d)),13);d.window.close();
 const duplicate=load({[key]:payload},w=>{for(const root of w.document.querySelectorAll('.sk-collection-searchable')){const node=[...root.children].find(el=>el.classList.contains('sk-collection-data'));const data=JSON.parse(node.textContent);data.collectionId='youtube-videos';node.textContent=JSON.stringify(data);}});
 assert.equal(count(scope(duplicate)),13);assert(!duplicate.window.document.querySelector('.sk-collection-remember'));duplicate.window.close();
});
test('Revocation in another tab pauses saving and does not resurrect consent',()=>{
 const d=load(),r=scope(d);add(d,'abcdefghijk');remember(d);d.window.localStorage.removeItem(key);
 d.window.dispatchEvent(new d.window.StorageEvent('storage',{key,newValue:null}));add(d,'abcdefghijl');assert.equal(d.window.localStorage.getItem(key),null);assert(!r.querySelector('.sk-collection-remember input').checked);d.window.close();
});
test('Stale write is detected even before a storage event',()=>{
 const d=load(),r=scope(d);add(d,'abcdefghijk');remember(d);d.window.localStorage.setItem(key,'newer-value');add(d,'abcdefghijl');
 assert.equal(d.window.localStorage.getItem(key),'newer-value');assert(!r.querySelector('.sk-collection-remember input').checked);d.window.close();
});
test('Filter chips remove only their own settings and retain focus',()=>{
 const d=load(),r=scope(d);search(d,'agent');const select=r.querySelector('select[aria-label=Category]');select.value='Agent Skills & Design Patterns';select.dispatchEvent(new d.window.Event('change'));
 const chips=r.querySelector('.sk-collection-chips');assert.equal(chips.querySelectorAll('button').length,2);
 chips.querySelector('button').click();assert.equal(r.querySelector('input[type=search]').value,'');assert(select.value);assert(chips.contains(d.window.document.activeElement));
 chips.querySelector('button').click();assert(chips.hidden);assert.equal(select.value,'');assert.equal(d.window.document.activeElement,r.querySelector('input[type=search]'));d.window.close();
});
test('Unicode composition does not prematurely apply search; search form is local',()=>{
 const d=load(),r=scope(d),input=r.querySelector('input[type=search]');input.value='absent';input.dispatchEvent(new d.window.InputEvent('input',{isComposing:true}));assert.equal(r.querySelector('.sk-collection-status').textContent,'13 of 13 cards');
 input.dispatchEvent(new d.window.CompositionEvent('compositionend'));assert.equal(r.querySelector('.sk-collection-status').textContent,'0 of 13 cards');
 const event=new d.window.Event('submit',{cancelable:true});r.querySelector('form[role=search]').dispatchEvent(event);assert(event.defaultPrevented);assert.equal(d.window.location.pathname,'/learn/index.html');d.window.close();
});
test('Typed YAML export round-trips video and channel additions without flattening card semantics',()=>{
 const d=load({},w=>{w.skCollectionResolveYouTubeLatest=()=>({videoId:'abcdefghijl',title:'Resolved latest'});}),r=scope(d),cr=channelScope(d);
 add(d,'abcdefghijk','[click](https://evil.example) <script> *x* `code`');add(d,'@example','Latest from channel');
 const videos=r.querySelector('.sk-collection-export-preview').value;
 assert.match(videos,/^videos:\n/);assert.match(videos,/id: \"abcdefghijk\"/);assert.match(videos,/id: \"abcdefghijl\"/);assert(!videos.includes('link:'));
 const cform=cr.querySelector('.sk-collection-add'),ctext=cform.querySelectorAll('input[type=text]');ctext[0].value='@export-channel';ctext[1].value='Export channel';cform.dispatchEvent(new d.window.Event('submit',{bubbles:true,cancelable:true}));
 const channels=cr.querySelector('.sk-collection-export-preview').value;
 assert.match(channels,/^channels:\n/);assert.match(channels,/url: \"https:\/\/www\.youtube\.com\/@export-channel\"/);assert(!channels.includes('id:'));
 assert.equal(r.querySelector('.sk-collection-export select'),null);assert.equal(cr.querySelector('.sk-collection-export select'),null);
 fs.writeFileSync(path.join(dir,'test-export-videos.yaml'),videos);fs.writeFileSync(path.join(dir,'test-export-channels.yaml'),channels);
 // URL APIs are absent here: export preview remains a usable fallback.
 r.querySelector('.sk-collection-export button').click();assert.match(r.querySelector('.sk-collection-export').textContent,/Select and copy/);d.window.close();
});
test('Download cleanup, canonical reference validation and programmatic bounds',()=>{
 let created=0,revoked=0;
 const d=load({},w=>{w.URL.createObjectURL=()=>{created++;return 'blob:https://docs.example/test';};w.URL.revokeObjectURL=()=>revoked++;w.setTimeout=fn=>{fn();return 1;};w.HTMLAnchorElement.prototype.click=function(){};}),r=scope(d);
 for(const value of ['https://youtube.com.evil.example/watch?v=abcdefghijk','https://youtu.be/abcdefghijk/extra','https://youtube.com/watch?v=abcdefghijk&v=abcdefghijl','https://youtube.com/watch?v=abcdefghijk\u0000'])add(d,value);
 assert.equal(count(r),13);add(d,'abcdefghijk','x'.repeat(201));assert.equal(count(r),13);add(d,'abcdefghijk');r.querySelector('.sk-collection-export button').click();assert.equal(created,1);assert.equal(revoked,1);assert(!d.window.document.querySelector('a[download]'));d.window.close();
});
test('No collection ID means no browser-storage access',()=>{
 let reads=0,writes=0;
 const d=load({},w=>{
  for(const node of w.document.querySelectorAll('.sk-collection-data')){const data=JSON.parse(node.textContent);delete data.collectionId;node.textContent=JSON.stringify(data);}
  w.Storage.prototype.getItem=()=>{reads++;throw Error('unexpected read');};w.Storage.prototype.setItem=()=>{writes++;throw Error('unexpected write');};
 });
 add(d,'abcdefghijk');scope(d).querySelector('.sk-collection-revert').click();assert.equal(reads,0);assert.equal(writes,0);assert(!scope(d).querySelector('.sk-collection-remember'));d.window.close();
});
test('Reinitialization with a new gallery preserves unique disclosure IDs',()=>{
 const d=load(),source=new JSDOM(html);const fresh=source.window.document.querySelector('.sk-collection-searchable');
 const node=[...fresh.children].find(el=>el.classList.contains('sk-collection-data'));const data=JSON.parse(node.textContent);data.collectionId='new-gallery';node.textContent=JSON.stringify(data);
 d.window.document.body.append(d.window.document.importNode(fresh,true));d.window.eval(js);
 d.window.document.dispatchEvent(new d.window.Event('DOMContentLoaded'));
 const panels=[...d.window.document.querySelectorAll('.sk-collection-panel')];assert.equal(panels.length,3);assert.equal(new Set(panels.map(el=>el.id)).size,3);
 for(const toggle of d.window.document.querySelectorAll('.sk-collection-overflow'))assert(d.window.document.getElementById(toggle.getAttribute('aria-controls')));
 source.window.close();d.window.close();
});

test('Direct video and channel-gallery additions keep distinct published card shapes',()=>{
 const d=load(),r=scope(d),cr=channelScope(d);add(d,'abcdefghijk','Playable video');
 const channelForm=cr.querySelector('.sk-collection-add'),channelTexts=channelForm.querySelectorAll('input[type=text]');channelTexts[0].value='@example';channelTexts[1].value='';channelForm.dispatchEvent(new d.window.Event('submit',{bubbles:true,cancelable:true}));
 const video=r.querySelector('.sk-collection-additions .sd-card'),videoBody=video.querySelector('.sd-card-body');assert.equal(videoBody.children.length,2);assert.equal(video.querySelector('.sd-card-title').textContent,'Playable video');
 assert(video.querySelector('.video_wrapper'));assert.equal(video.querySelector('iframe').src,'https://www.youtube.com/embed/abcdefghijk');assert.equal(video.querySelector('a'),null);assert.equal(video.querySelector('button'),null);assert.equal(video.querySelector('.sk-collection-note'),null);
 const channel=cr.querySelector('.sk-collection-additions .sd-card'),channelBody=channel.querySelector('.sd-card-body');assert.equal(channelBody.children.length,1);assert.equal(channel.querySelector('.sd-card-title').textContent,'@example');
 assert.equal(channel.querySelector('iframe'),null);const link=channel.querySelector('a.sd-stretched-link');assert(link);assert.equal(link.href,'https://www.youtube.com/@example');assert.equal(channel.querySelector('button'),null);assert.equal(channel.querySelector('.sk-collection-note'),null);d.window.close();
});

test('Video section radios resolve source first, then render the normal video card',()=>{
 const d=load({},w=>{w.skCollectionResolveYouTubeLatest=req=>({videoId:'abcdefghijl',title:'Resolved short'});w.__latestRequest=null;const fn=w.skCollectionResolveYouTubeLatest;w.skCollectionResolveYouTubeLatest=req=>{w.__latestRequest=req;return fn(req);};}),r=scope(d),form=r.querySelector('.sk-collection-add'),texts=form.querySelectorAll('input[type=text]');
 const radios=[...form.querySelectorAll('.sk-collection-source-sections input[type=radio]')];assert.deepEqual(radios.map(x=>x.value),['videos','shorts','streams','courses']);
 radios.find(x=>x.value==='shorts').click();texts[0].value='@example';texts[1].value='';form.dispatchEvent(new d.window.Event('submit',{bubbles:true,cancelable:true}));
 const card=[...r.querySelectorAll('.sk-collection-additions .sd-card')].at(-1);assert.equal(d.window.__latestRequest.section,'shorts');assert.equal(card.querySelector('.sd-card-title').textContent,'Resolved short');assert.equal(card.querySelector('.sd-card-body').children.length,2);assert.equal(card.querySelector('iframe').src,'https://www.youtube.com/embed/abcdefghijl');assert.equal(card.querySelector('a'),null);
 remember(d);const saved=JSON.parse(d.window.localStorage.getItem(key));assert.equal(saved.items.at(-1).url,'https://www.youtube.com/@example/shorts');assert.equal(saved.items.at(-1).title,'');d.window.close();
});

test('Resolved latest titles refresh with the source instead of being frozen in saved state',()=>{
 const oldId='ZZZZZZZZZZ1',newId='ZZZZZZZZZZ2';
 const d=load({},w=>{w.skCollectionResolveYouTubeLatest=()=>({videoId:oldId,title:'Old resolver title'});}),r=scope(d),form=r.querySelector('.sk-collection-add'),fields=form.querySelectorAll('input[type=text]');
 fields[0].value='@example';fields[1].value='';form.dispatchEvent(new d.window.Event('submit',{bubbles:true,cancelable:true}));remember(d);
 const saved=snapshot(d),payload=JSON.parse(saved[key]);assert.equal(payload.items.at(-1).url,'https://www.youtube.com/@example/videos');assert.equal(payload.items.at(-1).title,'');d.window.close();
 const restored=load(saved,w=>{w.skCollectionResolveYouTubeLatest=()=>({videoId:newId,title:'New resolver title'});}),rr=scope(restored),card=[...rr.querySelectorAll('.sk-collection-additions .sd-card')].at(-1);
 assert.equal(card.querySelector('.sd-card-title').textContent,'New resolver title');assert.equal(card.querySelector('iframe').src,'https://www.youtube.com/embed/'+newId);restored.window.close();
});

test('Options close on outside pointer interaction without stealing destination focus',()=>{
 const d=load(),r=scope(d),toggle=r.querySelector('.sk-collection-overflow'),panel=r.querySelector('.sk-collection-panel');
 toggle.click();assert(!panel.hidden);const destination=d.window.document.createElement('button');destination.textContent='Outside';d.window.document.body.append(destination);
 destination.focus();destination.dispatchEvent(new d.window.Event('pointerdown',{bubbles:true}));assert(panel.hidden);assert.equal(d.window.document.activeElement,destination);d.window.close();
});
test('Remembered view stores facets and sort but never search text',()=>{
 const d=load(),r=scope(d),view=r.querySelector('.sk-collection-view-storage input');assert(view);
 const category=r.querySelector('select[aria-label="Category"]'),sort=r.querySelector('select[aria-label="Sort within categories"]');
 category.value='Agent Skills & Design Patterns';category.dispatchEvent(new d.window.Event('change',{bubbles:true}));sort.value='title:desc';sort.dispatchEvent(new d.window.Event('change',{bubbles:true}));search(d,'PRIVATE VIEW SEARCH');
 view.checked=true;view.dispatchEvent(new d.window.Event('change',{bubbles:true}));const viewKey='sk-gallery-view:v1:%2Flearn%2Findex.html:youtube-videos',raw=d.window.localStorage.getItem(viewKey);assert(raw);assert(!raw.includes('PRIVATE VIEW SEARCH'));
 const saved=snapshot(d);d.window.close();const restored=load(saved),rr=scope(restored);assert.equal(rr.querySelector('select[aria-label="Category"]').value,'Agent Skills & Design Patterns');assert.equal(rr.querySelector('select[aria-label="Sort within categories"]').value,'title:desc');assert.equal(rr.querySelector('input[type=search]').value,'');restored.window.close();
});
test('Metadata-driven suggestions expose filter and sort shortcuts',()=>{
 const d=load(),r=scope(d),suggestions=r.querySelector('.sk-collection-suggestions');assert(!suggestions.hidden);assert(suggestions.querySelectorAll('button').length>0);
 search(d,'Principal');assert([...suggestions.querySelectorAll('button')].some(b=>b.textContent.includes('Search ·')));d.window.close();
});

console.log(checks+' saved-addition/export/chip checks passed');
