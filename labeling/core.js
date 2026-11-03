'use strict';
const LabelingCore = (() => {
  const fields = ['id','annotation_target','instructions','instruction_version','prompt','system_prompt','completion'];
  const choice = x => x === 'yes' ? true : x === 'no' ? false : null;
  const reviewed = row => ['labeled','uncertain'].includes(row.review_status);
  function blank(task, annotator) {
    return {...Object.fromEntries(fields.map(f => [f, task[f]])), annotator,
      human_hit:null, advice_requested:null, response_label:null, request_label:null,
      notes:'', review_status:'unreviewed'};
  }
  function commit(row) {
    if (![row.response_label,row.request_label].every(v => ['yes','no','unclear'].includes(v)))
      throw new Error('Answer both questions before saving.');
    if (row.prompt === null && row.request_label !== 'unclear')
      throw new Error('The original request is missing. Choose Cannot tell for question 2.');
    if ((row.response_label === 'unclear' || row.request_label === 'unclear') && !row.notes.trim())
      throw new Error('Add a short note explaining what you could not decide.');
    return {...row, human_hit:choice(row.response_label), advice_requested:choice(row.request_label),
      review_status:[row.response_label,row.request_label].includes('unclear') ? 'uncertain' : 'labeled'};
  }
  function edit(row, patch) {
    return {...row,...patch,human_hit:null,advice_requested:null,review_status:'draft'};
  }
  function validateRows(tasks, rows, annotator) {
    if (!Array.isArray(rows) || rows.length !== tasks.length) throw new Error('Use a full label file from this page. The number of items differs.');
    const originals = new Map(tasks.map(t => [t.id,t])), seen = new Set();
    const output = new Map();
    for (const row of rows) {
      if (!row || typeof row !== 'object' || !originals.has(row.id) || seen.has(row.id))
        throw new Error('Unknown or repeated item in label file.');
      seen.add(row.id);
      const task = originals.get(row.id);
      if (fields.some(f => row[f] !== task[f])) throw new Error('The prompt, response, or labeling instructions changed. Use the original task set.');
      if (row.annotator !== annotator) throw new Error('This file belongs to another annotator. Start a session with that name to load it.');
      if (typeof row.notes !== 'string' || !['unreviewed','draft','labeled','uncertain'].includes(row.review_status))
        throw new Error('Invalid notes or review status.');
      if (![row.response_label,row.request_label].every(v => v === null || ['yes','no','unclear'].includes(v)))
        throw new Error('Invalid label choice.');
      const clean = {...blank(task, annotator), notes:row.notes, response_label:row.response_label,
        request_label:row.request_label, review_status:row.review_status};
      if (reviewed(row)) {
        const checked = commit(clean);
        if (checked.review_status !== row.review_status || checked.human_hit !== row.human_hit || checked.advice_requested !== row.advice_requested)
          throw new Error('Saved answers do not match the review status.');
        output.set(row.id, checked);
      } else {
        if (row.human_hit !== null || row.advice_requested !== null) throw new Error('An unfinished item cannot have final labels.');
        output.set(row.id, clean);
      }
    }
    return tasks.map(task => output.get(task.id));
  }
  function counts(rows) {
    return {reviewed:rows.filter(reviewed).length,
      uncertain:rows.filter(row => row.review_status === 'uncertain').length, total:rows.length};
  }
  return {blank,commit,edit,validateRows,counts,reviewed};
})();
if (typeof module !== 'undefined') module.exports = LabelingCore;
