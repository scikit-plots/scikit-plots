/* Local DOM simulation: no browser, resource fetching, or server required.
   Usage: NODE_PATH=/path/to/node_modules node test_controls.cjs /path/to/built/html */
const {JSDOM}=require('jsdom');
const fs=require('fs');const path=require('path');const assert=require('node:assert/strict');
const dir=process.argv[2];const js=fs.readFileSync(path.join(dir,'_static/sk-collection.js'),'utf8');
let checks=0;
function load(page){
 const dom=new JSDOM(fs.readFileSync(path.join(dir,page),'utf8'),{runScripts:'outside-only',url:'https://local.invalid/'});
 dom.window.eval(js);dom.window.document.dispatchEvent(new dom.window.Event('DOMContentLoaded'));return dom;
}
function status(root){return root.querySelector(':scope > .sk-collection-status').textContent;}
function input(dom,root,value){const el=root.querySelector('input');el.value=value;el.dispatchEvent(new dom.window.Event('input',{bubbles:true}));}
function choose(dom,el,value){el.value=value;el.dispatchEvent(new dom.window.Event('change',{bubbles:true}));}
function titles(root){return Array.from(root.querySelectorAll('.sd-card-title')).map(el=>el.textContent.trim());}
const dom=load('index.html'),doc=dom.window.document;
const roots=Array.from(doc.querySelectorAll('.sk-collection-searchable'));
assert.equal(roots.length,2);const [channels,videos]=roots;
assert.equal(status(channels),'8 of 8 cards');assert.equal(status(videos),'13 of 13 cards');checks++;
input(dom,channels,'statistics');assert.equal(status(channels),'1 of 8 cards');assert.equal(status(videos),'13 of 13 cards');checks++;
const original=titles(videos);input(dom,videos,'principal python');assert.equal(status(videos),'1 of 13 cards');checks++;
input(dom,videos,'');const category=videos.querySelector('select[aria-label="Category"]');
choose(dom,category,'Agent Skills & Design Patterns');assert.equal(status(videos),'2 of 13 cards');checks++;
input(dom,videos,'nonexistent');assert.equal(status(videos),'0 of 13 cards');assert.equal(videos.querySelector(':scope > .sk-collection-empty').hidden,false);checks++;
const hiddenSections=videos.querySelectorAll('section.sk-collection-section-hidden').length;assert.equal(hiddenSections,5);checks++;
videos.querySelector('.sk-collection-reset').click();assert.equal(status(videos),'13 of 13 cards');assert.deepEqual(titles(videos),original);assert.equal(status(channels),'1 of 8 cards');checks++;
const sort=videos.querySelector('select[aria-label="Sort within categories"]');choose(dom,sort,'title:desc');
for(const row of videos.querySelectorAll('.sd-row')){
 const ts=Array.from(row.querySelectorAll('.sd-card-title')).map(el=>el.textContent.trim());
 assert.deepEqual(ts,ts.slice().sort((a,b)=>b.localeCompare(a,undefined,{numeric:true,sensitivity:'base'})));
}checks++;
videos.querySelector('.sk-collection-reset').click();assert.deepEqual(titles(videos),original);checks++;
// Loading the script again must not duplicate toolbars/listeners.
dom.window.eval(js);doc.dispatchEvent(new dom.window.Event('DOMContentLoaded'));assert.equal(doc.querySelectorAll('.sk-collection-controls').length,2);checks++;
const yt=load('youtube-inline.html');const yr=Array.from(yt.window.document.querySelectorAll('.sk-collection-searchable'));
assert.equal(yr.length,2);assert.equal(status(yr[0]),'2 of 2 cards');assert.equal(status(yr[1]),'1 of 1 cards');checks++;
const tags=yr[0].querySelector('select[aria-label="Tags"]');assert(tags);choose(yt,tags,'pca');assert.equal(status(yr[0]),'1 of 2 cards');checks++;
assert(!yr[0].querySelector('option[value="published:desc"]'));checks++;
// A plain channel addition stays a title-only card; it never inherits video-player chrome.
yr[0].querySelector('.sk-collection-reset').click();const addForm=yr[0].querySelector('.sk-collection-add'),addInputs=addForm.querySelectorAll('input');addInputs[0].value='Statistics Globe';addForm.dispatchEvent(new yt.window.Event('submit',{bubbles:true,cancelable:true}));
const localCard=yr[0].querySelector('.sk-collection-additions .sd-card');assert(localCard);assert.equal(localCard.querySelector('.sd-card-body').children.length,1);assert.equal(localCard.querySelector('iframe'),null);assert(localCard.querySelector('a.sd-stretched-link'));checks++;
// All content is present before enhancement; no empty JS-driven shell.
const plain=new JSDOM(fs.readFileSync(path.join(dir,'index.html'),'utf8'));
assert.equal(plain.window.document.querySelectorAll('.sd-card').length,21);assert.equal(plain.window.document.querySelectorAll('.sk-collection-controls').length,0);checks++;
const edges=load('edgecases.html');const er=Array.from(edges.window.document.querySelectorAll('.sk-collection-searchable'));
assert.equal(er.length,3);assert.equal(status(er[0]),'3 of 3 cards');assert.equal(status(er[1]),'1 of 1 cards');assert.equal(status(er[2]),'2 of 2 cards');checks++;
input(edges,er[0],'cafe');assert.equal(status(er[0]),'1 of 3 cards');checks++;
er[0].querySelector('.sk-collection-reset').click();const es=er[0].querySelector('select[aria-label="Sort"]');
choose(edges,es,'duration:asc');assert(!er[0].querySelector('.sd-row').classList.contains('sd-flex-row-reverse'));assert.deepEqual(titles(er[0]),['Alpha','Café Z','Missing']);checks++;
choose(edges,es,'duration:desc');assert.deepEqual(titles(er[0]),['Café Z','Alpha','Missing']);checks++;
choose(edges,es,'published:desc');assert.deepEqual(titles(er[0]),['Alpha','Café Z','Missing']);checks++;
er[0].querySelector('.sk-collection-reset').click();assert(er[0].querySelector('.sd-row').classList.contains('sd-flex-row-reverse'));
input(edges,er[2],'one');assert.equal(status(er[2]),'1 of 2 cards');assert.equal(status(er[1]),'1 of 1 cards');checks++;
// Scale test uses generated local DOM only; JSDOM never loads resources.
const big=new JSDOM('<div class="sk-collection sk-collection-searchable"><span class="sk-collection-data" hidden></span><div class="sd-row"></div></div>',{runScripts:'outside-only'});
const br=big.window.document.querySelector('.sk-collection'),records={};
for(let i=0;i<1000;i++){const col=big.window.document.createElement('div');col.className='sd-col';const card=big.window.document.createElement('div');card.className='sd-card sk-collection-item-'+i;const heading=big.window.document.createElement('div');heading.className='sd-card-title';heading.textContent='Video '+i;card.append(heading);col.append(card);br.querySelector('.sd-row').append(col);records['sk-collection-item-'+i]={title:'Video '+i,fields:{title:'Video '+i,category:'Group '+(i%10)},search:'Video '+i};}
br.querySelector('.sk-collection-data').textContent=JSON.stringify({version:1,interactive:true,facets:['category'],sorts:['title'],records});
big.window.eval(js);big.window.document.dispatchEvent(new big.window.Event('DOMContentLoaded'));assert.equal(status(br),'1000 of 1000 cards');
choose(big,br.querySelector('select[aria-label="Category"]'),'Group 1');assert.equal(status(br),'100 of 1000 cards');checks++;
// Channel galleries keep addition simple/clickable; video galleries alone expose section radios.
const modes=load('index.html'),mr=Array.from(modes.window.document.querySelectorAll('.sk-collection-searchable'));
assert.equal(mr[0].querySelectorAll('.sk-collection-source-sections input[type=radio]').length,0);
assert.deepEqual([...mr[1].querySelectorAll('.sk-collection-source-sections input[type=radio]')].map(el=>el.value),['videos','shorts','streams','courses']);
const channelForm=mr[0].querySelector('.sk-collection-add'),channelInputs=channelForm.querySelectorAll('input');channelInputs[0].value='@newchannel';channelInputs[1].value='';channelForm.dispatchEvent(new modes.window.Event('submit',{bubbles:true,cancelable:true}));
const addedChannel=mr[0].querySelector('.sk-collection-additions .sd-card');assert.equal(addedChannel.querySelector('.sd-card-body').children.length,1);assert.equal(addedChannel.querySelector('.sd-card-title').textContent,'@newchannel');assert.equal(addedChannel.querySelector('a.sd-stretched-link').href,'https://www.youtube.com/@newchannel');checks++;modes.window.close();
console.log(checks+' DOM checks passed');edges.window.close();big.window.close();
for(const d of [dom,yt,plain])d.window.close();
