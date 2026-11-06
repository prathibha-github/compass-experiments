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

const pairedTasks=()=>[task,{...task,id:'two',system_prompt:'Be concise.',completion:'Keep working.'}];
const blankRows=(tasks=pairedTasks(),author='Alice')=>tasks.map(item=>core.blank(item,author));
const roundTrip=(tasks,rows)=>core.validateRows(tasks,JSON.parse(JSON.stringify(rows)),'Alice');
const legacy=row=>Object.fromEntries(Object.entries(row).filter(([key])=>!['request_reviewed','request_notes','request_draft'].includes(key)));

test('request identity uses exact prompt and instructions, excludes generation context, and isolates authors',()=>{
  const base=core.blank(task,'Alice');
  assert.equal(core.requestKey(base),core.requestKey({...base,id:'other',system_prompt:'Other condition',completion:'Other response',model:'other'}));
  for(const patch of [{prompt:'Help. '},{instructions:'New instructions'},{instruction_version:'v2'},{annotation_target:'Other target'}])
    assert.notEqual(core.requestKey(base),core.requestKey({...base,...patch}));
  assert.notEqual(core.requestKey({...base,prompt:null}),core.requestKey({...base,prompt:null,id:'other'}));
  const rows=core.saveRequest([base,core.blank({...task,id:'bob'},'Bob')],'one',{label:'no',notes:''});
  assert.equal(rows[0].request_reviewed,true);
  assert.equal(rows[1].request_reviewed,false);
});

test('saving one request answer never reviews an unread response',()=>{
  const tasks=pairedTasks();
  const rows=core.saveRequest(blankRows(tasks),'one',{label:'no',notes:'No request for rest.'});
  for(const row of rows) {
    assert.equal(row.request_label,'no'); assert.equal(row.request_reviewed,true);
    assert.equal(row.response_label,null); assert.equal(row.human_hit,null);
    assert.equal(row.advice_requested,null); assert.equal(row.review_status,'unreviewed');
  }
  rows[0]=core.commit(core.edit(rows[0],{response_label:'yes'}));
  assert.equal(core.counts(rows).reviewed,1);
  assert.deepEqual(roundTrip(tasks,rows),rows);
});

test('shared request uncertainty needs one explanation while response uncertainty needs its own',()=>{
  let rows=core.saveRequest(blankRows(),'one',{label:'unclear',notes:'The request is ambiguous.'});
  rows=rows.map(row=>core.commit(core.edit(row,{response_label:'no'})));
  assert.equal(core.counts(rows).uncertain,2);
  assert.equal(rows[1].notes,'');
  assert.throws(()=>core.commit(core.edit(rows[1],{response_label:'unclear'})),/this response/);
  assert.throws(()=>core.saveRequest(rows,'one',{label:'unclear',notes:'  '}),/short note/);
});

test('pending shared edits round trip, cancel cleanly, and save consistently without losing response drafts',()=>{
  const tasks=[...pairedTasks(),{...task,id:'three'}];
  let rows=core.saveRequest(blankRows(tasks),'one',{label:'no',notes:''});
  rows[0]=core.commit(core.edit(rows[0],{response_label:'yes',notes:'The response recommends a break.'}));
  rows[1]=core.edit(rows[1],{response_label:'no',notes:'Still checking this response.'});
  const saved=structuredClone(rows);
  rows=core.editRequest(rows,'two',{label:'unclear',notes:'Unclear wording in the request.'});
  assert.equal(rows[0].request_label,'no');
  assert.equal(rows[0].advice_requested,false);
  assert.equal(core.counts(rows).reviewed,0);
  assert.throws(()=>core.commit(rows[0]),/Save or cancel/);
  assert.deepEqual(roundTrip(tasks,rows),rows);
  assert.deepEqual(core.cancelRequest(rows,'three'),saved);
  rows=core.saveRequest(rows,'two');
  assert.equal(rows[0].review_status,'uncertain'); assert.equal(rows[0].human_hit,true);
  assert.equal(rows[0].advice_requested,null); assert.equal(rows[0].notes,saved[0].notes);
  assert.equal(rows[1].review_status,'draft'); assert.equal(rows[1].response_label,'no');
  assert.equal(rows[1].human_hit,null); assert.equal(rows[1].notes,saved[1].notes);
  assert.equal(rows[2].review_status,'unreviewed');
  rows=core.saveRequest(rows,'one',{label:'yes',notes:''});
  assert.equal(rows[0].review_status,'labeled'); assert.equal(rows[0].advice_requested,true);
  assert.equal(rows[1].advice_requested,null);
  assert.deepEqual(roundTrip(tasks,rows),rows);
});

test('legacy reviewed answers migrate and preserve all uncertainty explanations',()=>{
  const tasks=pairedTasks();
  const old=tasks.map((item,index)=>legacy(core.commit({...core.blank(item,'Alice'),
    response_label:'yes',request_label:'unclear',notes:`Ambiguity ${index+1}.`})));
  const imported=roundTrip(tasks,old);
  for(let index=0;index<2;index++) {
    assert.equal(imported[index].request_notes,'Ambiguity 1.\n\nAmbiguity 2.');
    assert.equal(imported[index].notes,old[index].notes);
  }
  assert.deepEqual(roundTrip(tasks,imported),imported);
});

test('legacy response drafts remain unfinished when a matching request answer is reused',()=>{
  const tasks=pairedTasks();
  const rows=blankRows(tasks);
  rows[0]=legacy(core.commit({...rows[0],response_label:'yes',request_label:'no'}));
  rows[1]=legacy(core.edit(rows[1],{response_label:'no',request_label:'no',notes:'Not finished.'}));
  const imported=roundTrip(tasks,rows);
  assert.equal(imported[1].review_status,'draft');
  assert.equal(imported[1].request_reviewed,true); assert.equal(imported[1].request_draft,null);
  assert.equal(imported[1].human_hit,null);
  const onlyDrafts=blankRows(tasks).map(row=>legacy(core.edit(row,{request_label:'yes'})));
  const drafts=roundTrip(tasks,onlyDrafts);
  assert.equal(drafts[0].request_reviewed,false);
  assert.deepEqual(drafts[1].request_draft,{label:'yes',notes:''});
  assert.equal(core.counts(drafts).reviewed,0);
});

test('conflicting saved or draft request imports fail atomically',()=>{
  const tasks=pairedTasks();
  const saved=tasks.map((item,index)=>core.commit({...core.blank(item,'Alice'),response_label:'no',request_label:index?'yes':'no'}));
  const original=structuredClone(saved);
  assert.throws(()=>roundTrip(tasks,saved),/Conflicting request answers/);
  assert.deepEqual(saved,original);
  const drafts=blankRows(tasks).map((row,index)=>({...row,request_draft:{label:index?'yes':'no',notes:''}}));
  assert.throws(()=>roundTrip(tasks,drafts),/Conflicting request drafts/);
  const oldMixed=[legacy(saved[0]),legacy(core.edit(blankRows(tasks)[1],{request_label:'yes'}))];
  assert.throws(()=>roundTrip(tasks,oldMixed),/draft and saved request answer conflict/);
});

test('missing requests remain separate and require explicit uncertainty',()=>{
  const tasks=pairedTasks().map(row=>({...row,prompt:null}));
  let rows=blankRows(tasks);
  assert.throws(()=>core.saveRequest(rows,'one',{label:'no',notes:''}),/missing/);
  rows=core.saveRequest(rows,'one',{label:'unclear',notes:'Original request unavailable.'});
  assert.equal(rows[1].request_reviewed,false);
  assert.equal(core.requestState(rows,'one').count,1);
  assert.deepEqual(roundTrip(tasks,rows),rows);
});

test('optional generation quality is preserved and cannot change on import',()=>{
  const tasks=[{...task,generation_quality:{hit_token_cap:true,is_fragment:null,quality_flagged:true}}];
  const rows=blankRows(tasks);
  assert.deepEqual(roundTrip(tasks,rows),rows);
  const changed=structuredClone(rows); changed[0].generation_quality.hit_token_cap=false;
  assert.throws(()=>roundTrip(tasks,changed),/changed/);
});

// Run the real app handlers against a minimal DOM and shared browser storage.
// Rendering and layout are checked separately in browser QA.
function appSession(storage=new Map()) {
  const fs=require('node:fs'), vm=require('node:vm');
  const elements=new Map(), listeners=new Map();
  const element=()=>({value:'',textContent:'',hidden:false,disabled:false,checked:false,files:[],focus(){}});
  const html=fs.readFileSync(require.resolve('../labeling/page.html'),'utf8');
  for(const [,id] of html.matchAll(/id="([^"]+)"/g)) elements.set(id,element());
  elements.get('task-data').textContent=JSON.stringify({dataset_id:'synthetic',tasks:pairedTasks()});
  const radios=Object.fromEntries(['response','request'].map(name=>[name,['yes','no','unclear'].map(value=>({...element(),value}))]));
  const document={getElementById:id=>{
    assert.ok(elements.has(id),`Unknown element ${id}`); return elements.get(id);
  },querySelectorAll:selector=>selector==='input[name=response]' ? radios.response :
    selector==='input[name=request]' ? radios.request : [...radios.response,...radios.request,...elements.values()]};
  const context=vm.createContext({document,localStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value)},
    window:{addEventListener:(name,handler)=>listeners.set(name,handler),confirm:()=>true,scrollTo(){}},console});
  vm.runInContext(fs.readFileSync(require.resolve('../labeling/core.js'),'utf8'),context);
  vm.runInContext(fs.readFileSync(require.resolve('../labeling/app.js'),'utf8'),context);
  const read=expression=>JSON.parse(vm.runInContext(`JSON.stringify(${expression})`,context));
  const start=name=>{elements.get('annotator').value=name; elements.get('start').onclick();};
  start('Alice');
  return {elements,radios,listeners,storage,read,start};
}

test('another tab saving first blocks a shared edit without changing either copy',()=>{
  const first=appSession();
  first.radios.request[1].onchange(); first.elements.get('save-request').onclick();
  const second=appSession(first.storage);
  second.radios.response[0].onchange();
  const saved=JSON.stringify([...first.storage]), before=first.read('rows');
  first.elements.get('edit-request').onclick();
  assert.equal(first.read('conflict'),true);
  assert.deepEqual(first.read('rows'),before);
  assert.equal(JSON.stringify([...first.storage]),saved);
  assert.equal(first.elements.get('save-request').disabled,true);
});

test('import checks storage revision before replacing the current in-memory labels',async()=>{
  const app=appSession(), before=app.read('rows');
  const incoming=core.saveRequest(before,'one',{label:'no',notes:''});
  let resolveFile;
  app.elements.get('import').files=[{text:()=>new Promise(resolve=>{resolveFile=resolve;})}];
  const pending=app.elements.get('import').onchange();
  const key=app.read('storageKey');
  const external=JSON.parse(app.storage.get(key)); external.revision='another-tab';
  app.storage.set(key,JSON.stringify(external));
  resolveFile(incoming.map(JSON.stringify).join('\n'));
  await pending;
  assert.deepEqual(app.read('rows'),before);
  assert.equal(app.storage.get(key),JSON.stringify(external));
  assert.match(app.elements.get('message').textContent,/Import was not applied/);
});

test('an asynchronous import cannot replace labels after switching annotators',async()=>{
  const app=appSession(), incoming=app.read('rows');
  let resolveFile;
  app.elements.get('import').files=[{text:()=>new Promise(resolve=>{resolveFile=resolve;})}];
  const pending=app.elements.get('import').onchange();
  app.start('Bob');
  const current=app.read('rows'), saved=JSON.stringify([...app.storage]);
  resolveFile(incoming.map(JSON.stringify).join('\n'));
  await pending;
  assert.deepEqual(app.read('rows'),current);
  assert.equal(app.read('annotator'),'Bob');
  assert.equal(JSON.stringify([...app.storage]),saved);
});

test('conflicting file import leaves both displayed labels and saved progress untouched',async()=>{
  const app=appSession(), before=app.read('rows'), saved=JSON.stringify([...app.storage]);
  const incoming=pairedTasks().map((item,index)=>core.commit({...core.blank(item,'Alice'),response_label:'no',request_label:index?'yes':'no'}));
  app.elements.get('import').files=[{text:async()=>incoming.map(JSON.stringify).join('\n')}];
  await app.elements.get('import').onchange();
  assert.match(app.elements.get('message').textContent,/Conflicting request answers/);
  assert.deepEqual(app.read('rows'),before);
  assert.equal(JSON.stringify([...app.storage]),saved);
});
