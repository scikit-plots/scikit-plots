// Run 105 regression — execute slash DOM row rendering, not only parser/registry.
// Guards the real Run 104 browser failure: `_esc is not defined` threw after
// discovery recognized `/`, leaving the hint visible while the menu stayed closed.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
let passed=0,failed=0; function ok(c,n){if(c)passed++;else{failed++;console.error('FAIL '+n);}}
function extract(name){const i=src.indexOf('function '+name+'(');if(i<0)throw new Error('missing '+name);let d=0,start=false;for(let j=i;j<src.length;j++){if(src[j]==='{'){d++;start=true;}else if(src[j]==='}'){d--;if(start&&d===0)return src.slice(i,j+1);}}throw new Error('unbalanced '+name);}

ok(!src.includes('_esc('),'slash renderer has no undefined _esc helper dependency');
ok(src.includes('function _buildSlashCommandPaletteRows(commands)'),'row construction has a dedicated executable helper');
ok(src.includes("document.createDocumentFragment()"),'rows are built off-DOM before palette commit');
ok(src.includes("commandName.textContent = String(item.command || '').replace(/^\\//, '')"),'visible row renders exactly one slash via icon + slashless label');
ok(src.includes("commandDescription.textContent = String(item.description || item.title || '')"),'description uses textContent rather than HTML interpolation');
ok(src.includes("commandArg.textContent = String(item.argumentHint || item.badge || '')"),'badge/argument uses textContent');
ok(src.includes("try {\n                built = _buildSlashCommandPaletteRows(commands);"),'palette build is failure-contained before opening');
ok(src.includes("catch (renderError) {\n                _closeSlashCommandPalette(false);"),'render failure atomically closes the palette');
ok(src.includes("slashPaletteList.replaceChildren(built.fragment);"),'built fragment is committed only after successful construction');
ok(src.includes("try { slashPaletteList.replaceChildren(); } catch (_e) {}"),'closing palette removes stale command DOM');
ok(src.includes("slashPaletteStatus.textContent = '';"),'closing palette clears stale live-region status');

class FakeNode {
  constructor(tag, fragment=false){this.tagName=tag;this.fragment=fragment;this.children=[];this.attrs={};this.listeners={};this.hidden=false;this.className='';this.textContent='';this.id='';this.type='';}
  setAttribute(k,v){this.attrs[k]=String(v);}
  getAttribute(k){return Object.prototype.hasOwnProperty.call(this.attrs,k)?this.attrs[k]:null;}
  removeAttribute(k){delete this.attrs[k];}
  appendChild(n){if(n&&n.fragment){this.children.push(...n.children);n.children=[];}else this.children.push(n);return n;}
  replaceChildren(...nodes){this.children=[];for(const n of nodes){if(n&&n.fragment)this.children.push(...n.children);else if(n)this.children.push(n);}}
  addEventListener(k,fn){this.listeners[k]=fn;}
  scrollIntoView(){}
}
const document={createElement:t=>new FakeNode(t),createDocumentFragment:()=>new FakeNode('#fragment',true)};
let highlighted=[];let executed=[];
const buildSrc=extract('_buildSlashCommandPaletteRows');
const build=new Function('document','_highlightSlashCommand','_executeSlashCommand',buildSrc+'\nreturn _buildSlashCommandPaletteRows;')(document,i=>highlighted.push(i),i=>executed.push(i));
const commands=[
 {id:'add-current-page-context',command:'/Add current page context',description:'Include <page> & context.',badge:'Added'},
 {id:'pin-current-page',command:'/Pin current page',description:'Keep a snapshot.',badge:'Keep'},
 {id:'add-files',command:'/Add files or photos',description:'Stage local files.',badge:'Local'},
 {id:'skill-creator',command:'/skill-creator',description:'Create a skill.',argumentHint:'optional goal'}
];
let built;
try { built=build(commands); ok(true,'real row builder executes without unresolved helper errors'); }
catch(e){ console.error(e); ok(false,'real row builder executes without unresolved helper errors'); }
ok(built && built.rows.length===4,'bare-slash command set builds four menu rows');
ok(built && built.fragment.children.length===4,'all four rows remain in detached fragment before commit');
const first=built.rows[0];
ok(first.getAttribute('data-command')==='/Add current page context','canonical slash command is preserved in data-command');
ok(first.children[0].textContent==='/' && first.children[1].children[0].textContent==='Add current page context','visual command contains exactly one slash');
ok(first.children[1].children[1].textContent==='Include <page> & context.','markup-like description remains inert text');
ok(first.children[2].textContent==='Added','state badge is rendered as text');
first.listeners.pointermove(); first.listeners.click();
ok(highlighted[0]===0,'detached row retains pointer highlight handler');
ok(executed[0]===commands[0],'detached row retains canonical execution item');
const empty=build([]);
ok(empty.rows.length===0 && empty.fragment.children.length===1,'true no-match result gets one empty-state node');
ok(empty.fragment.children[0].textContent==='No matching local commands','empty-state copy remains explicit');

// Execute the actual palette commit function with a browser-like fake textarea.
const renderSrc=extract('_renderSlashCommandPalette');
const input={value:'/',selectionEnd:1,hidden:false,attrs:{},setAttribute(k,v){this.attrs[k]=String(v);},removeAttribute(k){delete this.attrs[k];}};
const list=new FakeNode('div');
const palette=new FakeNode('div'); palette.hidden=true; palette.attrs['data-open']='false';
const hint={hidden:true}; const status={textContent:''};
let items=[],index=0,discovery=null,dismissed=null,attachClosed=0,positioned=0,highlightedCommit=-1,closeCalls=0;
const normalize=v=>String(v??'').replace(/\s+/g,' ').trim().toLowerCase();
const active={start:0,end:1,raw:'',query:''};

const committedSrc=extract('_slashCommandCommittedItem');
const committed=new Function('_normalizeSlashCommandText',committedSrc+'\nreturn _slashCommandCommittedItem;')(normalize);
ok(committed('skill-creator ',commands)?.id==='skill-creator','exact skill command + trailing space commits discovery');
ok(committed('skill-creator build an API skill',commands)?.id==='skill-creator','skill arguments cannot reopen autocomplete');
ok(committed('Add current page context ',commands)?.id==='add-current-page-context','multi-word utility + trailing space commits discovery');
ok(committed('Add current page context later text',commands)?.id==='add-current-page-context','utility trailing prose cannot reopen autocomplete');
ok(committed('Add current page',commands)===null,'partial multi-word command remains discoverable');
ok(committed('',commands)===null,'bare slash is never treated as committed');
const runtime=new Function(
 'input','slashPalette','slashPaletteList','slashInlineHint','slashPaletteStatus','document','commands','_buildSlashCommandPaletteRows',
 '_composerSlashDiscovery','_slashCommandResults','_availableLocalSlashCommands','_normalizeSlashCommandText','_slashCommandCommittedItem','_closeAttachMenu','_highlightSlashCommand','_positionSlashCommandPalette','_closeSlashCommandPalette','_log','state',
 `${renderSrc}\nreturn ()=>{ slashPaletteItems=state.items; slashPaletteIndex=state.index; slashPaletteDiscovery=state.discovery; slashPaletteDismissedValue=state.dismissed; _renderSlashCommandPalette(false); state.items=slashPaletteItems; state.index=slashPaletteIndex; state.discovery=slashPaletteDiscovery; state.dismissed=slashPaletteDismissedValue; };`
)(input,palette,list,hint,status,document,commands,build,
 ()=>active,()=>commands,()=>commands,normalize,committed,()=>{attachClosed++;},i=>{highlightedCommit=i;},()=>{positioned++;},()=>{closeCalls++;palette.hidden=true;palette.attrs['data-open']='false';input.attrs['aria-expanded']='false';hint.hidden=true;list.replaceChildren();},()=>{},
 {items,index,discovery,dismissed});
try { runtime(); ok(true,'actual render commit function executes for bare slash'); }
catch(e){console.error(e);ok(false,'actual render commit function executes for bare slash');}
ok(palette.hidden===false && palette.attrs['data-open']==='true','bare slash opens the palette shell');
ok(input.attrs['aria-expanded']==='true','bare slash exposes expanded state on textarea');
ok(list.children.length===4,'bare slash commits four actual menu rows');
ok(hint.hidden===false,'leading bare slash exposes Type to filter hint only after successful row build');
ok(status.textContent.startsWith('4 commands available.'),'live region reports four available commands');
ok(attachClosed===1,'opening slash palette closes attachment menu once');
ok(positioned===1 && highlightedCommit===0,'opened palette positions and highlights first command');
ok(closeCalls===0,'successful bare slash render never enters failure close path');

console.log(`${passed} passed, ${failed} failed`); if(failed)process.exit(1);
