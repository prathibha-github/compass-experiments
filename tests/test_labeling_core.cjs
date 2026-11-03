const test = require('node:test');
const assert = require('node:assert/strict');
const core = require('../labeling/core.js');
const task = {id:'one',annotation_target:'rest recommendation to the user',instructions:'Label the advice.',instruction_version:'v1',prompt:'Help.',system_prompt:null,completion:'Take a break.'};

test('all label combinations survive a JSONL round trip',()=>{
  for(const a of ['yes','no','unclear']) for(const b of ['yes','no','unclear']) {
    const row=core.commit(core.edit(core.blank(task,'Alice'),{response_label:a,request_label:b,notes:'Context is ambiguous.'}));
    const imported=core.validateRows([task],JSON.parse(JSON.stringify([row])),'Alice');
    assert.deepEqual(imported,[row]);
    assert.equal(row.human_hit,a==='yes'?true:a==='no'?false:null);
    assert.equal(row.advice_requested,b==='yes'?true:b==='no'?false:null);
    assert.equal(core.counts(imported).reviewed,1);
  }
});
test('editing completed labels makes them drafts, not final observations',()=>{
  const saved=core.commit({...core.blank(task,'Alice'),response_label:'yes',request_label:'no'});
  const edited=core.edit(saved,{response_label:'no'});
  assert.equal(edited.human_hit,null); assert.equal(edited.review_status,'draft');
  assert.equal(core.counts([edited]).reviewed,0);
});
test('uncertain needs notes and missing request cannot be inferred',()=>{
  assert.throws(()=>core.commit({...core.blank(task,'Alice'),response_label:'unclear',request_label:'no'}),/short note/);
  assert.throws(()=>core.commit({...core.blank({...task,prompt:null},'Alice'),response_label:'no',request_label:'no'}),/missing/);
});
test('imports reject changes atomically, including mixed authors and final draft scores',()=>{
  const second={...task,id:'two'}, originals=[task,second];
  const valid=originals.map(t=>core.blank(t,'Alice'));
  for(const patch of [{prompt:'changed'},{completion:'changed'},{instructions:'changed'},
    {annotator:'Bob'},{human_hit:false},{id:'one'},{response_label:'maybe'}]) {
    const bad=[valid[0],{...valid[1],...patch}];
    assert.throws(()=>core.validateRows(originals,bad,'Alice'));
    assert.equal(valid[1].review_status,'unreviewed');
  }
  assert.throws(()=>core.validateRows(originals,valid.slice(0,1),'Alice'),/number/);
});
