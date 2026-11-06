'use strict';
const LabelingCore = (() => {
  const fields = ['id','annotation_target','instructions','instruction_version','prompt','system_prompt','completion'];
  const labels = ['yes','no','unclear'];
  const choice = x => x === 'yes' ? true : x === 'no' ? false : null;
  const finalStatus = row => ['labeled','uncertain'].includes(row.review_status);
  const reviewed = row => finalStatus(row) && !row.request_draft;
  const canonical = value => JSON.stringify(value, (_, v) => v && typeof v === 'object' && !Array.isArray(v)
    ? Object.fromEntries(Object.keys(v).sort().map(k => [k,v[k]])) : v);
  function requestKey(row) {
    return JSON.stringify([typeof row.prompt === 'string' ? ['prompt',row.prompt] : ['item',row.id],
      row.annotation_target,row.instructions,row.instruction_version]);
  }
  const sameRequest = (a,b) => a.annotator === b.annotator && requestKey(a) === requestKey(b);
  function blank(task, annotator) {
    return {...Object.fromEntries(fields.map(f => [f, task[f]])),
      ...(Object.hasOwn(task,'generation_quality') ? {generation_quality:task.generation_quality} : {}), annotator,
      human_hit:null, advice_requested:null, response_label:null, request_label:null,
      request_reviewed:false, request_notes:'', request_draft:null,
      notes:'', review_status:'unreviewed'};
  }
  function checkRequest(row, label, notes) {
    if (!labels.includes(label)) throw new Error('Answer the request question before saving it.');
    if (typeof row.prompt !== 'string' && label !== 'unclear')
      throw new Error('The original request is missing. Choose Cannot tell for question 2.');
    if (label === 'unclear' && !notes.trim())
      throw new Error('Add a short note explaining what you could not decide about the request.');
  }
  function commit(row) {
    if (row.request_draft) throw new Error('Save or cancel the request edit before saving this response.');
    if (![row.response_label,row.request_label].every(v => labels.includes(v)))
      throw new Error('Answer both questions before saving. Save the request answer once for this prompt.');
    // Older files kept both explanations in the per-response notes field.
    const requestNotes = row.request_notes || (row.request_label === 'unclear' ? row.notes : '');
    checkRequest(row, row.request_label, requestNotes);
    if (row.response_label === 'unclear' && !row.notes.trim())
      throw new Error('Add a short note explaining what you could not decide about this response.');
    return {...row, request_reviewed:true, request_notes:requestNotes, request_draft:null,
      human_hit:choice(row.response_label), advice_requested:choice(row.request_label),
      review_status:[row.response_label,row.request_label].includes('unclear') ? 'uncertain' : 'labeled'};
  }
  function edit(row, patch) {
    return {...row,...patch,human_hit:null,advice_requested:null,review_status:'draft'};
  }
  function current(rows, id) {
    const row=rows.find(item=>item.id===id);
    if (!row) throw new Error('Unknown response.');
    return row;
  }
  function requestState(rows, id) {
    const row=current(rows,id);
    return {saved:row.request_reviewed, label:row.request_label, notes:row.request_notes,
      draft:row.request_draft, count:rows.filter(item=>sameRequest(item,row)).length};
  }
  function editRequest(rows, id, patch={}) {
    const row=current(rows,id);
    const draft={...(row.request_draft || {label:row.request_label,notes:row.request_notes}),...patch};
    if (!(draft.label === null || labels.includes(draft.label)) || typeof draft.notes !== 'string')
      throw new Error('Invalid request draft.');
    return rows.map(item=>sameRequest(item,row) ? {...item,request_draft:{...draft}} : item);
  }
  function cancelRequest(rows, id) {
    const row=current(rows,id);
    return rows.map(item=>sameRequest(item,row) ? {...item,request_draft:null} : item);
  }
  function saveRequest(rows, id, answer) {
    const row=current(rows,id);
    const {label,notes}=answer || row.request_draft || {};
    if (typeof notes !== 'string') throw new Error('Invalid request notes.');
    checkRequest(row,label,notes);
    return rows.map(item=> {
      if (!sameRequest(item,row)) return item;
      const updated={...item,request_label:label,request_notes:notes,request_reviewed:true,request_draft:null};
      return finalStatus(item) ? commit(updated) : updated;
    });
  }
  function validateRows(tasks, rows, annotator) {
    if (!Array.isArray(rows) || rows.length !== tasks.length) throw new Error('Use a full label file from this page. The number of items differs.');
    const originals = new Map(tasks.map(t => [t.id,t])), seen = new Set(), output = new Map();
    const legacyDrafts = new Map();
    for (const row of rows) {
      if (!row || typeof row !== 'object' || !originals.has(row.id) || seen.has(row.id))
        throw new Error('Unknown or repeated item in label file.');
      seen.add(row.id);
      const task = originals.get(row.id);
      if (fields.some(f => row[f] !== task[f]) || canonical(row.generation_quality) !== canonical(task.generation_quality))
        throw new Error('The prompt, response, or labeling instructions changed. Use the original task set.');
      if (row.annotator !== annotator) throw new Error('This file belongs to another annotator. Start a session with that name to load it.');
      if (typeof row.notes !== 'string' || !['unreviewed','draft','labeled','uncertain'].includes(row.review_status))
        throw new Error('Invalid notes or review status.');
      if (![row.response_label,row.request_label].every(v => v === null || labels.includes(v)))
        throw new Error('Invalid label choice.');
      if (Object.hasOwn(row,'request_reviewed') && typeof row.request_reviewed !== 'boolean')
        throw new Error('Invalid saved request status.');
      if (Object.hasOwn(row,'request_notes') && typeof row.request_notes !== 'string')
        throw new Error('Invalid request notes.');
      const requestDraft=row.request_draft ?? null;
      if (requestDraft !== null && (typeof requestDraft !== 'object' || Array.isArray(requestDraft) ||
          !(requestDraft.label === null || labels.includes(requestDraft.label)) || typeof requestDraft.notes !== 'string'))
        throw new Error('Invalid request draft.');
      const saved=Object.hasOwn(row,'request_reviewed') ? row.request_reviewed : finalStatus(row);
      const clean = {...blank(task, annotator), notes:row.notes, response_label:row.response_label,
        request_label:row.request_label, review_status:row.review_status, request_reviewed:saved,
        request_notes:row.request_notes ?? (finalStatus(row) && row.request_label === 'unclear' ? row.notes : ''),
        request_draft:requestDraft && {label:requestDraft.label,notes:requestDraft.notes}};
      if (saved) checkRequest(clean,clean.request_label,clean.request_notes);
      if (finalStatus(row)) {
        if (!saved) throw new Error('A reviewed response needs a saved request answer.');
        const checked = commit({...clean,request_draft:null});
        if (checked.review_status !== row.review_status || checked.human_hit !== row.human_hit || checked.advice_requested !== row.advice_requested)
          throw new Error('Saved answers do not match the review status.');
        output.set(row.id, {...checked,request_draft:clean.request_draft});
      } else {
        if (row.human_hit !== null || row.advice_requested !== null) throw new Error('An unfinished item cannot have final labels.');
        if (!saved && row.request_label !== null) {
          legacyDrafts.set(row.id,{label:row.request_label,notes:row.request_label === 'unclear' ? row.notes : ''});
          clean.request_label=null;
        }
        output.set(row.id, clean);
      }
    }
    let normalized=tasks.map(task=>output.get(task.id));
    const groups=new Map();
    for (const row of normalized) {
      const key=requestKey(row);
      if (!groups.has(key)) groups.set(key,[]);
      groups.get(key).push(row);
    }
    for (const group of groups.values()) {
      const saved=group.filter(row=>row.request_reviewed);
      const savedLabels=new Set(saved.map(row=>row.request_label));
      if (savedLabels.size > 1) throw new Error('Conflicting request answers for the same prompt. Resolve them in the original file before importing.');
      const savedLabel=saved[0]?.request_label;
      const drafts=group.filter(row=>row.request_draft).map(row=>row.request_draft);
      for (const row of group) if (legacyDrafts.has(row.id)) {
        const draft=legacyDrafts.get(row.id);
        if (saved.length && draft.label !== savedLabel)
          throw new Error('A draft and saved request answer conflict for the same prompt. Resolve them before importing.');
        if (!saved.length) drafts.push(draft);
      }
      if (new Set(drafts.map(draft=>draft.label)).size > 1)
        throw new Error('Conflicting request drafts for the same prompt. Resolve them before importing.');
      // Preserve all explanations from older per-response files, never select one arbitrarily.
      const notes=Array.from(new Set(saved.map(row=>row.request_notes).filter(Boolean))).join('\n\n');
      if (saved.length) normalized=saveRequest(normalized,group[0].id,{label:savedLabel,notes});
      if (drafts.length) {
        const draftNotes=Array.from(new Set(drafts.map(draft=>draft.notes).filter(Boolean))).join('\n\n');
        normalized=editRequest(normalized,group[0].id,{label:drafts[0].label,notes:draftNotes});
      }
    }
    return normalized;
  }
  function counts(rows) {
    return {reviewed:rows.filter(reviewed).length,
      uncertain:rows.filter(row => reviewed(row) && row.review_status === 'uncertain').length, total:rows.length};
  }
  return {blank,commit,edit,validateRows,counts,reviewed,requestKey,requestState,editRequest,cancelRequest,saveRequest};
})();
if (typeof module !== 'undefined') module.exports = LabelingCore;
