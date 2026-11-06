'use strict';
const bundle = JSON.parse(document.getElementById('task-data').textContent);
const $ = id => document.getElementById(id);
let rows, annotator, cursor = 0, revision = null, storageKey, storageAvailable = true, conflict = false, needsBackup = false;
const editable='input[type=radio],textarea,#save-next,#import,#save-request,#edit-request,#cancel-request';
$('total-intro').textContent = `${bundle.tasks.length} responses. You can do a few at a time and return later.`;
function warning(text) { $('storage-warning').hidden=false; $('storage-warning').textContent=text; }
function stopForConflict() {
  conflict=true;
  $('saved').textContent='Saving stopped: another tab changed this session.';
  warning('Another tab changed these labels. Download this tab’s copy if needed, then reload before continuing. Your latest change was not applied.');
  document.querySelectorAll(editable).forEach(el=>el.disabled=true);
}
function persist(nextRows=rows, nextCursor=cursor) {
  if (conflict) return false;
  let nextRevision=revision;
  if (storageAvailable) {
    try {
      const old=localStorage.getItem(storageKey);
      if ((old ? JSON.parse(old).revision : null) !== revision) { stopForConflict(); return false; }
    } catch (error) {
      storageAvailable=false;
      warning('This browser could not read saved progress. Download your labels before closing.');
    }
    if (storageAvailable) {
      nextRevision=`${Date.now()}-${Math.random()}`;
      try {
        localStorage.setItem(storageKey,JSON.stringify({dataset_id:bundle.dataset_id,annotator,
          cursor:nextCursor,revision:nextRevision,rows:nextRows}));
      } catch (error) {
        storageAvailable=false;
        nextRevision=revision;
        warning('This browser could not save progress. Download your labels before closing, then import them next time.');
      }
    }
  }
  rows=nextRows; cursor=nextCursor; revision=nextRevision;
  $('saved').textContent=storageAvailable ? 'Saved in this browser. Download a backup when you finish.' : 'Not saved in browser. Download your labels.';
  return true;
}
function render() {
  const row=rows[cursor], counts=LabelingCore.counts(rows), request=LabelingCore.requestState(rows,row.id);
  $('position').textContent=`Response ${cursor+1} of ${rows.length}`;
  $('progress-text').textContent=`${counts.reviewed} of ${counts.total} reviewed · ${annotator}`;
  $('uncertain-count').textContent=`${counts.uncertain} marked uncertain`;
  $('progress-bar').max=counts.total; $('progress-bar').value=counts.reviewed;
  $('prompt').textContent=typeof row.prompt !== 'string' ? 'The original user request was not saved. Use “Cannot tell” for question 2.' : row.prompt || '(Empty user request)';
  $('completion').textContent=row.completion || '(Empty response)';
  $('instructions-text').textContent=row.instructions;
  $('instruction-version').textContent=`Instructions: ${row.instruction_version}`;
  $('system-box').hidden=row.system_prompt===null;
  $('missing-system').hidden=row.system_prompt!==null;
  $('system').textContent=row.system_prompt || '(Empty system instructions)';
  $('item-status').textContent=row.request_draft ? `${row.review_status} · request edit pending` : row.review_status;
  if ($('notes').value!==row.notes) $('notes').value=row.notes;
  document.querySelectorAll('input[name=response]').forEach(el=>el.checked=el.value===row.response_label);
  const draft=request.draft || {label:request.label,notes:request.notes};
  document.querySelectorAll('input[name=request]').forEach(el=>el.checked=el.value===draft.label);
  if ($('request-notes').value!==draft.notes) $('request-notes').value=draft.notes;
  $('request-summary').hidden=!request.saved;
  $('request-saved-answer').textContent=request.saved ? `Saved answer: ${request.label==='unclear' ? 'Cannot tell' : request.label==='yes' ? 'Yes' : 'No'}.` : '';
  $('request-saved-notes').textContent=request.notes;
  $('request-saved-notes').hidden=!request.notes;
  $('request-editor').hidden=request.saved && !request.draft;
  $('edit-request').hidden=!request.saved || !!request.draft;
  $('cancel-request').hidden=!request.draft;
  $('request-scope').textContent=request.count===1 ? 'This answer applies to this request.' : `Answer once for this user request. It applies to ${request.count} responses with the same prompt and labeling instructions.`;
  $('request-pending').hidden=!request.draft;
  const quality=row.generation_quality || {};
  $('quality-warning').hidden=!(quality.hit_token_cap || quality.is_fragment || quality.quality_flagged);
  $('quality-warning').textContent=quality.hit_token_cap ? 'This response may have been cut off. Label the text shown; choose Cannot tell if you cannot decide.' : 'This response was flagged as possibly incomplete. Label the text shown; choose Cannot tell if you cannot decide.';
  document.querySelectorAll(editable).forEach(el=>el.disabled=conflict);
  $('previous').disabled=cursor===0; $('next').disabled=cursor===rows.length-1;
}
function nextUnreviewed(start=cursor, candidates=rows) {
  for(let step=1;step<=candidates.length;step++) {
    const index=(start+step)%candidates.length;
    if(!LabelingCore.reviewed(candidates[index])) return index;
  }
  return null;
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
  $('welcome').hidden=true; $('workspace').hidden=false; $('session-tools').hidden=false;
  $('message').textContent='';
  if(!storageAvailable) warning('Browser storage is unavailable. Download your labels before closing.');
  render(); persist();
};
function apply(nextRows, message) {
  if (!persist(nextRows)) { render(); return false; }
  needsBackup=true; render(); $('message').textContent=message; return true;
}
function changeResponse(patch) {
  const changed=rows.map((row,index)=>index===cursor ? LabelingCore.edit(row,patch) : row);
  apply(changed,'Response draft. Save when your answer is ready.');
}
document.querySelectorAll('input[name=response]').forEach(el=>el.onchange=()=>changeResponse({response_label:el.value}));
$('notes').oninput=()=>changeResponse({notes:$('notes').value});
function changeRequest(patch={}) {
  apply(LabelingCore.editRequest(rows,rows[cursor].id,patch),'Request edit saved as a draft. Save the request answer to apply it to matching responses.');
}
document.querySelectorAll('input[name=request]').forEach(el=>el.onchange=()=>changeRequest({label:el.value}));
$('request-notes').oninput=()=>changeRequest({notes:$('request-notes').value});
$('edit-request').onclick=()=>changeRequest();
$('cancel-request').onclick=()=>apply(LabelingCore.cancelRequest(rows,rows[cursor].id),'Request edit canceled. The saved answer is unchanged.');
$('save-request').onclick=()=> {
  try { apply(LabelingCore.saveRequest(rows,rows[cursor].id),'Request answer saved for every matching response. Read and label each response separately.'); }
  catch(error) { $('message').textContent=error.message; }
};
$('save-next').onclick=()=> {
  let candidate;
  try { candidate=rows.map((row,index)=>index===cursor ? LabelingCore.commit(row) : row); }
  catch(error) { $('message').textContent=error.message; return; }
  const next=nextUnreviewed(cursor,candidate);
  if(!persist(candidate,next ?? cursor)) { render(); return; }
  needsBackup=true; render();
  $('message').textContent=next!==null ? 'Saved. Next unreviewed response.' : 'All responses reviewed. Download your labels to keep a backup.';
  window.scrollTo({top:0,behavior:'smooth'});
};
$('unreviewed').onclick=()=> {
  const next=nextUnreviewed();
  if (!persist(rows,next ?? cursor)) return;
  render(); $('message').textContent=next!==null ? '' : 'All responses are reviewed.';
};
for(const [id,step] of [['previous',-1],['next',1]]) $(id).onclick=()=> {
  if (!persist(rows,Math.max(0,Math.min(rows.length-1,cursor+step)))) return;
  render(); $('message').textContent=''; window.scrollTo({top:0,behavior:'smooth'});
};
$('download').onclick=()=> {
  const blob=new Blob([rows.map(row=>JSON.stringify(row)).join('\n')+'\n'],{type:'application/x-ndjson'});
  const url=URL.createObjectURL(blob), link=document.createElement('a');
  link.href=url; link.download=`rest-labels-${annotator.replace(/[^a-z0-9_-]/gi,'_')}.jsonl`; link.click(); needsBackup=false;
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$('import').onchange=async()=> {
  const file=$('import').files[0]; if(!file) return;
  const targetKey=storageKey, targetRevision=revision, targetRows=rows, targetAnnotator=annotator;
  try {
    const incoming=(await file.text()).split(/\r?\n/).filter(line=>line.trim()).map(line=>JSON.parse(line));
    if (storageKey!==targetKey || revision!==targetRevision || rows!==targetRows || annotator!==targetAnnotator || conflict)
      throw new Error('This session changed while the file was being read. Choose the file again.');
    const validated=LabelingCore.validateRows(bundle.tasks,incoming,annotator);
    if(rows.some(row=>row.review_status!=='unreviewed' || row.request_reviewed || row.request_draft) && !window.confirm('Replace this session with the imported labels? Download your current copy first if you need to keep it.')) return;
    const next=LabelingCore.reviewed(validated[0]) ? nextUnreviewed(0,validated) ?? 0 : 0;
    if(!persist(validated,next)) { $('message').textContent='Import was not applied because another tab changed this session.'; return; }
    needsBackup=true; render(); $('message').textContent='Labels imported. Matching requests share their saved answer.';
  } catch(error) { $('message').textContent=`Import failed: ${error.message} Your current labels were not replaced.`; }
  finally { $('import').value=''; }
};
$('switch').onclick=()=> {
  if(needsBackup && (!storageAvailable || conflict) && !window.confirm('These labels are not saved in this browser. Download them before switching. Switch anyway and discard this tab’s unsaved copy?')) return;
  $('workspace').hidden=true; $('session-tools').hidden=true; $('welcome').hidden=false; $('welcome-error').textContent=''; $('annotator').value=''; $('annotator').focus();
};
window.addEventListener('storage',event=> {
  if (!$('workspace').hidden && (event.key===storageKey || event.key===null)) {
    try { if ((event.newValue ? JSON.parse(event.newValue).revision : null)!==revision) stopForConflict(); }
    catch(error) { stopForConflict(); }
  }
});
window.addEventListener('beforeunload',event=> {
  if(needsBackup && (!storageAvailable || conflict)) { event.preventDefault(); event.returnValue=''; }
});
