'use strict';
const bundle = JSON.parse(document.getElementById('task-data').textContent);
const $ = id => document.getElementById(id);
let rows, annotator, cursor = 0, revision = null, storageKey, storageAvailable = true, conflict = false, needsBackup = false;
$('total-intro').textContent = `${bundle.tasks.length} responses. You can do a few at a time and return later.`;
function warning(text) { $('storage-warning').hidden=false; $('storage-warning').textContent=text; }
function snapshot() { return {dataset_id:bundle.dataset_id,annotator,cursor,revision,rows}; }
function persist() {
  if (conflict) return false;
  if (!storageAvailable) { $('saved').textContent='Browser saving unavailable. Download your labels.'; return true; }
  try {
    const old = localStorage.getItem(storageKey);
    if ((old ? JSON.parse(old).revision : null) !== revision) {
      conflict=true;
      $('saved').textContent='Saving stopped: another tab changed this session.';
      warning('Another tab changed these labels. Download this tab’s copy if needed, then reload before continuing.');
      document.querySelectorAll('input[type=radio],textarea,#save-next,#import').forEach(el=>el.disabled=true);
      return false;
    }
    revision = `${Date.now()}-${Math.random()}`;
    localStorage.setItem(storageKey,JSON.stringify(snapshot()));
    $('saved').textContent='Saved in this browser. Download a backup when you finish.';
  } catch (error) {
    storageAvailable=false;
    warning('This browser could not save progress. Download your labels before closing, then import them next time.');
    $('saved').textContent='Not saved in browser. Download your labels.';
  }
  return true;
}
function render() {
  const row=rows[cursor], counts=LabelingCore.counts(rows);
  $('position').textContent=`Response ${cursor+1} of ${rows.length}`;
  $('progress-text').textContent=`${counts.reviewed} of ${counts.total} reviewed · ${annotator}`;
  $('uncertain-count').textContent=`${counts.uncertain} marked uncertain`;
  $('progress-bar').max=counts.total; $('progress-bar').value=counts.reviewed;
  $('prompt').textContent=row.prompt === null ? 'The original user request was not saved. Use “Cannot tell” for question 2.' : row.prompt || '(Empty user request)';
  $('completion').textContent=row.completion || '(Empty response)';
  $('instructions-text').textContent=row.instructions;
  $('instruction-version').textContent=`Instructions: ${row.instruction_version}`;
  $('system-box').hidden=row.system_prompt===null;
  $('missing-system').hidden=row.system_prompt!==null;
  $('system').textContent=row.system_prompt || '(Empty system instructions)';
  $('item-status').textContent=row.review_status;
  $('notes').value=row.notes;
  for (const group of ['response','request']) document.querySelectorAll(`input[name=${group}]`).forEach(el=>el.checked=el.value===row[`${group}_label`]);
  $('previous').disabled=cursor===0; $('next').disabled=cursor===rows.length-1;
}
function jumpUnreviewed() {
  for(let step=1;step<=rows.length;step++) {
    const index=(cursor+step)%rows.length;
    if(!LabelingCore.reviewed(rows[index])) { cursor=index; return true; }
  }
  return false;
}
$('start').onclick=()=> {
  annotator=$('annotator').value.trim();
  if(!annotator) { $('welcome-error').textContent='Enter a name or ID to keep your labels separate.'; return; }
  storageKey=`compass-rest-labels-v1:${bundle.dataset_id}:${encodeURIComponent(annotator)}`;
  rows=bundle.tasks.map(task=>LabelingCore.blank(task,annotator)); cursor=0; revision=null; conflict=false; storageAvailable=true; needsBackup=false;
  try {
    let saved;
    try { saved=localStorage.getItem(storageKey); } catch(error) { storageAvailable=false; }
    if(saved) {
      const state=JSON.parse(saved);
      if(state.dataset_id!==bundle.dataset_id || state.annotator!==annotator || typeof state.revision!=='string') throw new Error('Saved session does not match these tasks.');
      rows=LabelingCore.validateRows(bundle.tasks,state.rows,annotator);
      revision=state.revision;
      cursor=Number.isInteger(state.cursor) && state.cursor>=0 && state.cursor<rows.length ? state.cursor : 0;
    }
  } catch(error) { $('welcome-error').textContent=`Could not load saved labels: ${error.message} The saved copy has not been changed.`; return; }
  $('storage-warning').hidden=true;
  document.querySelectorAll('input[type=radio],textarea,#save-next,#import').forEach(el=>el.disabled=false);
  $('welcome').hidden=true; $('workspace').hidden=false; $('session-tools').hidden=false;
  $('message').textContent='';
  if(!storageAvailable) warning('Browser storage is unavailable. Download your labels before closing.');
  render(); persist();
};
function change(patch) { needsBackup=true; rows[cursor]=LabelingCore.edit(rows[cursor],patch); persist(); $('item-status').textContent='draft'; $('message').textContent='Draft. Save when both answers are ready.'; const count=LabelingCore.counts(rows); $('progress-text').textContent=`${count.reviewed} of ${count.total} reviewed · ${annotator}`; $('progress-bar').value=count.reviewed; $('uncertain-count').textContent=`${count.uncertain} marked uncertain`; }
for (const group of ['response','request']) document.querySelectorAll(`input[name=${group}]`).forEach(el=>el.onchange=()=>change({[`${group}_label`]:el.value}));
$('notes').oninput=()=>change({notes:$('notes').value});
$('save-next').onclick=()=> {
  try { rows[cursor]=LabelingCore.commit(rows[cursor]); needsBackup=true; }
  catch(error) { $('message').textContent=error.message; return; }
  if(!persist()) return;
  const found=jumpUnreviewed(); render(); persist();
  $('message').textContent=found ? 'Saved. Next unreviewed response.' : 'All responses reviewed. Download your labels to keep a backup.';
  window.scrollTo({top:0,behavior:'smooth'});
};
$('unreviewed').onclick=()=> { const found=jumpUnreviewed(); render(); persist(); $('message').textContent=found ? '' : 'All responses are reviewed.'; };
for(const [id,step] of [['previous',-1],['next',1]]) $(id).onclick=()=> { cursor=Math.max(0,Math.min(rows.length-1,cursor+step)); render(); persist(); $('message').textContent=''; window.scrollTo({top:0,behavior:'smooth'}); };
$('download').onclick=()=> {
  const blob=new Blob([rows.map(row=>JSON.stringify(row)).join('\n')+'\n'],{type:'application/x-ndjson'});
  const url=URL.createObjectURL(blob), link=document.createElement('a');
  link.href=url; link.download=`rest-labels-${annotator.replace(/[^a-z0-9_-]/gi,'_')}.jsonl`; link.click(); needsBackup=false;
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$('import').onchange=async()=> {
  const file=$('import').files[0]; if(!file) return;
  try {
    const incoming=(await file.text()).split(/\r?\n/).filter(line=>line.trim()).map(line=>JSON.parse(line));
    const validated=LabelingCore.validateRows(bundle.tasks,incoming,annotator);
    if(rows.some(row=>row.review_status!=='unreviewed') && !window.confirm('Replace this session with the imported labels? Download your current copy first if you need to keep it.')) return;
    rows=validated; needsBackup=true; cursor=0; if(LabelingCore.reviewed(rows[0])) jumpUnreviewed();
    if(!persist()) return;
    render(); $('message').textContent='Labels imported.';
  } catch(error) { $('message').textContent=`Import failed: ${error.message} Your current labels were not replaced.`; }
  finally { $('import').value=''; }
};
$('switch').onclick=()=> { if(needsBackup && (!storageAvailable || conflict) && !window.confirm('These labels are not saved in this browser. Download them before switching. Switch anyway and discard this tab’s unsaved copy?')) return; $('workspace').hidden=true; $('session-tools').hidden=true; $('welcome').hidden=false; $('welcome-error').textContent=''; $('annotator').value=''; $('annotator').focus(); };

window.addEventListener('beforeunload',event=> {
  if(needsBackup && (!storageAvailable || conflict)) { event.preventDefault(); event.returnValue=''; }
});
