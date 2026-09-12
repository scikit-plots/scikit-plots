// Run 142 — reader-owned binary ZIP replacements remain separate from AI media input capability.
import fs from 'node:fs';
const src=fs.readFileSync(process.argv[2],'utf8');
const css=fs.readFileSync(process.argv[3],'utf8');
let pass=0,fail=0;
function ok(c,n){if(c)pass++;else{fail++;console.error('FAIL '+n)}}
function extract(name){
  for(const pre of ['async function ','function ']){
    const i=src.indexOf(pre+name+'('); if(i<0)continue;
    let d=0,st=false,q=null,esc=false,line=false,block=false;
    for(let j=i;j<src.length;j++){
      const c=src[j],n=src[j+1]||'';
      if(line){if(c==='\n')line=false;continue}
      if(block){if(c==='*'&&n==='/'){block=false;j++}continue}
      if(q){if(esc)esc=false;else if(c==='\\')esc=true;else if(c===q)q=null;continue}
      if(c==='/'&&n==='/'){line=true;j++;continue}
      if(c==='/'&&n==='*'){block=true;j++;continue}
      if(c==='"'||c==="'"||c==='`'){q=c;continue}
      if(c==='{'){d++;st=true}else if(c==='}'&&st&&--d===0)return src.slice(i,j+1)
    }
  }
  throw new Error('missing '+name);
}

ok(src.includes('var _ZIP_EDIT_MAX_IMAGE_REPLACEMENT_BYTES = 16 * 1024 * 1024'),'image replacement has independent 16 MiB ceiling');
ok(src.includes('var _ZIP_EDIT_MAX_AUDIO_REPLACEMENT_BYTES = 32 * 1024 * 1024'),'audio replacement has independent 32 MiB ceiling');
ok(src.includes('var _ZIP_EDIT_MAX_VIDEO_REPLACEMENT_BYTES = 64 * 1024 * 1024'),'video replacement has independent 64 MiB ceiling');
ok(src.includes('var _ZIP_EDIT_MAX_IMAGE_DIMENSION = 16384'),'image dimension ceiling is explicit');
ok(src.includes('var _ZIP_EDIT_MAX_IMAGE_PIXELS = 40 * 1000 * 1000'),'image pixel ceiling is explicit');
ok(src.includes('var _ZIP_EDIT_MAX_AUDIO_DURATION_SECONDS = 60 * 60'),'audio duration ceiling is explicit');
ok(src.includes('var _ZIP_EDIT_MAX_VIDEO_DURATION_SECONDS = 30 * 60'),'video duration ceiling is explicit');
ok(/async function _zipEditStageLocalReplacement[\s\S]*?_extractZipEntry\(st\.job,entry,shouldCancel\)[\s\S]*?_zipEditValidateBinaryBlob\(entry,original[\s\S]*?_zipEditValidateBinaryBlob\(entry,file/.test(src),'local replacement validates both original archive bytes and reader-supplied bytes');
ok(/async function _zipEditValidateBinaryBlob[\s\S]*?_zipEditBinaryMagic\(probeBytes\)[\s\S]*?token!==spec\.token/.test(src),'replacement signature is derived from bytes rather than filename/MIME declaration');
ok(/async function _zipEditValidateBinaryBlob[\s\S]*?size>spec\.maxBytes[\s\S]*?size>serverMax/.test(src),'local replacement enforces both client lane and server entry ceilings');
ok(/async function _zipEditStageLocalReplacement[\s\S]*?maxReplacements[\s\S]*?maxAuthorizedPaths[\s\S]*?maxReplacementTotalBytes/.test(src),'staging obeys server-advertised count and aggregate replacement ceilings');
ok(/function _zipEditProbeMediaBlob[\s\S]*?Number\.isFinite\(duration\)[\s\S]*?duration>maximum/.test(src),'audio/video metadata probe enforces finite bounded duration');
ok(/function _zipEditProbeMediaBlob[\s\S]*?_zipEditValidateImageBounds\(out\)/.test(src),'video metadata probe also bounds decoded frame dimensions');
ok(/function _zipEditProbeImageBlob[\s\S]*?naturalWidth[\s\S]*?expectedDimensions/.test(src),'image header dimensions are cross-checked against browser decode dimensions');
ok(/jpegOrientationSwap=token==='jpeg'/.test(src),'JPEG EXIF orientation may swap decoded width/height without falsely failing the header/decode cross-check');
ok(src.includes('Replacement metadata is preserved as supplied; original EXIF/media metadata is not copied or merged.'),'UI states whole-file metadata replacement policy');
ok(src.includes('Media mN ids never enter server authorization'),'read-only AI media capability remains separate from ZIP write authority');
ok(src.includes("source:'reader-binary', lane:'binary-local'"),'reader binary replacement has explicit non-model provenance/lane');
ok(/async function _zipEditApply[\s\S]*?row\.source === 'reader-binary'[\s\S]*?authorized\.push\(row\.path\)/.test(src),'binary path enters server authorization only from accepted reader-owned replacement state');
ok(/function _zipEditAcceptedProposals[\s\S]*?st\.proposals\.concat\(st\.localReplacements\)/.test(src),'apply review unifies model text proposals and reader binary replacements without conflating provenance');
ok(/function _zipEditInvalidateProposal[\s\S]*?st\.proposals = \[\][\s\S]*?st\.localReplacements/.test(src),'model prompt invalidation does not silently discard reader-owned binary replacement intent');
ok(/function _closeZipEditLayer[\s\S]*?_zipEditClearLocalReplacements\(\)/.test(src),'closing workflow revokes local binary preview capabilities/state');
ok(/function _zipEditPickLocalReplacement[\s\S]*?addEventListener\('cancel',cleanup[\s\S]*?window\.addEventListener\('focus',focusHandler/.test(src),'ephemeral local file picker cleans itself on cancel including older-browser focus fallback');
ok(/function _zipEditProbeMediaBlob[\s\S]*?removeEventListener\('loadedmetadata'[\s\S]*?removeEventListener\('error'/.test(src),'media metadata probe removes event handlers when it completes or times out');
ok(src.includes('Replace locally…'),'inventory provides explicit reader-owned replacement action');
ok(src.includes('Input / local-replace eligible'),'inventory filter exposes local replacement independently from model-input support');
ok(src.includes('Local replacements'),'inventory can isolate staged local binary writes');
ok(src.includes('I reviewed the accepted file diffs and, for binary replacements'),'final apply gate covers binary preview review as well as text diffs');
ok(/proposal\.accepted = checkbox\.checked;[\s\S]*?if \(st\.review\) st\.review\.checked = false/.test(src),'any accepted-set change invalidates the aggregate review attestation');
ok(css.includes('.ai-assistant-panel-zip-edit-row[data-lane="binary-local"]'),'binary-local inventory lane has explicit UI treatment');
ok(css.includes('.ai-assistant-panel-zip-edit-binary-pair'),'binary review uses paired original/replacement surface');
ok(css.includes('.ai-assistant-panel-zip-edit-proposal[data-lane="binary-local"]'),'binary proposals have distinct review styling');

const factory=new Function(`
 var _ZIP_EDIT_MAX_IMAGE_REPLACEMENT_BYTES=16*1024*1024,_ZIP_EDIT_MAX_AUDIO_REPLACEMENT_BYTES=32*1024*1024,_ZIP_EDIT_MAX_VIDEO_REPLACEMENT_BYTES=64*1024*1024;
 var _ZIP_EDIT_MAX_IMAGE_DIMENSION=16384,_ZIP_EDIT_MAX_IMAGE_PIXELS=40*1000*1000;
 ${extract('_zipEditBinaryReplacementSpec')}
 ${extract('_zipEditBinaryMagic')}
 ${extract('_zipEditImageDimensions')}
 ${extract('_zipEditValidateImageBounds')}
 return {spec:_zipEditBinaryReplacementSpec,magic:_zipEditBinaryMagic,dim:_zipEditImageDimensions,bounds:_zipEditValidateImageBounds};
`);
const f=factory();
function bytes(...xs){return new Uint8Array(xs)}
let spec=f.spec({kind:'image',relativePath:'assets/plot.png'});
ok(spec&&spec.token==='png'&&spec.kind==='image','PNG path gets reader image replacement spec');
spec=f.spec({kind:'image',relativePath:'assets/photo.jpeg'});
ok(spec&&spec.token==='jpeg','JPEG path gets reader image replacement spec');
spec=f.spec({kind:'audio',relativePath:'audio/theme.flac'});
ok(spec&&spec.token==='flac'&&spec.kind==='audio','FLAC path gets reader audio replacement spec');
spec=f.spec({kind:'video',relativePath:'media/demo.webm'});
ok(spec&&spec.token==='webm'&&spec.kind==='video','WebM path gets reader video replacement spec');
ok(f.spec({kind:'image',relativePath:'assets/a.avif'})===null,'AVIF remains fail-closed until a validated replacement parser/lane exists');
ok(f.spec({kind:'file',relativePath:'bin/model.bin'})===null,'arbitrary binary files do not inherit media replacement authority');

ok(f.magic(bytes(0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a))==='png','PNG magic is recognized');
ok(f.magic(bytes(0xff,0xd8,0xff,0xe0))==='jpeg','JPEG magic is recognized');
ok(f.magic(new TextEncoder().encode('GIF89a'))==='gif','GIF magic is recognized');
let webp=new Uint8Array(16); webp.set(new TextEncoder().encode('RIFF'),0); webp.set(new TextEncoder().encode('WEBP'),8);
ok(f.magic(webp)==='webp','WebP RIFF signature is recognized');
let wav=new Uint8Array(16); wav.set(new TextEncoder().encode('RIFF'),0); wav.set(new TextEncoder().encode('WAVE'),8);
ok(f.magic(wav)==='wav','WAV RIFF signature is distinguished from WebP');
let avi=new Uint8Array(16); avi.set(new TextEncoder().encode('RIFF'),0); avi.set(new TextEncoder().encode('AVI '),8);
ok(f.magic(avi)==='avi','AVI RIFF signature is distinguished from WAV/WebP');
ok(f.magic(new TextEncoder().encode('fLaC'))==='flac','FLAC magic is recognized');
ok(f.magic(new TextEncoder().encode('OggS'))==='ogg','Ogg container magic is recognized');
ok(f.magic(new Uint8Array([0x49,0x44,0x33]))==='mp3','ID3 MP3 magic is recognized');
let bmff=new Uint8Array(16); bmff.set(new TextEncoder().encode('ftyp'),4);
ok(f.magic(bmff)==='iso-bmff','ISO-BMFF family is recognized without assuming input capability is output capability');
ok(f.magic(new Uint8Array([0x1a,0x45,0xdf,0xa3]))==='webm','WebM EBML signature is recognized');
ok(f.magic(new Uint8Array([0,0,1,0xba]))==='mpeg','MPEG program stream signature is recognized');
ok(f.magic(new TextEncoder().encode('not-media'))==='','unknown bytes fail closed');

let png=new Uint8Array(24); png.set(bytes(0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a),0); png[18]=0x04; png[19]=0x00; png[22]=0x03; png[23]=0x00;
let d=f.dim(png,'png');
ok(d&&d.width===1024&&d.height===768,'PNG dimensions parse from IHDR');
let gif=new Uint8Array(10); gif.set(new TextEncoder().encode('GIF89a'),0); gif[6]=0x80; gif[7]=0x02; gif[8]=0xe0; gif[9]=0x01;
d=f.dim(gif,'gif'); ok(d&&d.width===640&&d.height===480,'GIF logical screen dimensions parse');
let bounded=f.bounds({width:4000,height:3000}); ok(bounded.width===4000&&bounded.height===3000,'reasonable image dimensions pass');
let huge=false;try{f.bounds({width:16384,height:16384})}catch(e){huge=e.message==='ZIP_EDIT_BINARY_IMAGE_DIMENSIONS'}
ok(huge,'excessive pixel count fails closed even when each dimension is individually bounded');

console.log(`${pass} passed, ${fail} failed`); if(fail)process.exit(1);
