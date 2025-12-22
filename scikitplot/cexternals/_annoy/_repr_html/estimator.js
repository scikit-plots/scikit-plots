// Annoy HTML repr — minimal copy-to-clipboard behavior.
// Scoped by __ANNOY_REPR_ID__ to avoid cross-output collisions.
// JS must reference document.getElementById('__ANNOY_REPR_ID__')

(function(){
  var root=document.getElementById('__ANNOY_REPR_ID__');
  if(!root)return;
  var btns=root.querySelectorAll('button.annoy-copy');
  for(var i=0;i<btns.length;i++){
    btns[i].addEventListener('click',function(e){
      e.preventDefault();
      var tr=this.closest('tr'); if(!tr) return;
      var val=tr.querySelector('.annoy-value'); if(!val) return;
      var txt=val.textContent || ''; if(!txt) return;

      function done(btn){
        var old=btn.textContent;
        btn.textContent='✔︎ Copied';
        setTimeout(function(){btn.textContent=old;},800);
      }

      if(navigator.clipboard && navigator.clipboard.writeText){
        navigator.clipboard.writeText(txt).then(done.bind(null,this),function(){done(this);}.bind(this));
      } else {
        var ta=document.createElement('textarea');
        ta.value=txt;
        ta.style.position='fixed';
        ta.style.left='-9999px';
        document.body.appendChild(ta);
        ta.select();
        try{document.execCommand('copy');}catch(_e){}
        document.body.removeChild(ta);
        done(this);
      }
    });
  }
})();
