<script>
(function(){
  var btns=[].slice.call(document.querySelectorAll('.fbtn'));
  var cards=[].slice.call(document.querySelectorAll('.card'));
  var grps=[].slice.call(document.querySelectorAll('.grp'));
  function apply(f){
    cards.forEach(function(c){
      var show = f==='all' || (f==='win'&&c.dataset.win==='win') || (f==='reg'&&c.dataset.win==='reg') || (f==='wrong'&&c.dataset.tl==='wrong');
      c.classList.toggle('hide', !show);
    });
    grps.forEach(function(g){
      var any=[].slice.call(g.querySelectorAll('.card')).some(function(c){return !c.classList.contains('hide');});
      g.style.display = any ? '' : 'none';
    });
  }
  btns.forEach(function(b){
    b.addEventListener('click', function(){
      btns.forEach(function(x){x.setAttribute('aria-pressed', x===b ? 'true':'false');});
      apply(b.dataset.f);
    });
  });
})();
</script>
