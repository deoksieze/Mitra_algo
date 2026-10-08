// Minimal DOM/canvas smoke test; browser appearance still requires visual inspection.
'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const handlers={};
const context=new Proxy({},{get:(_,name)=>name==='setTransform'?()=>{}:(...args)=>{
  for(const arg of args)if(typeof arg==='number')assert(Number.isFinite(arg),'non-finite canvas coordinate');
},set:()=>true});
const elements={};
for(const id of ['canvas','cluster','display','faces','transformed','size','reset','stats','empty','file']){
  elements[id]={value:id==='display'?'patches':id==='size'?'3':'0',checked:id==='faces',clientWidth:900,clientHeight:700,
    textContent:'',options:[],getContext:()=>context,
    replaceChildren(){this.options=[];},append(option){this.options.push(option);},
    addEventListener(name,fn){handlers[id+':'+name]=fn;},setPointerCapture(){}};
}
const sandbox={console,window:{devicePixelRatio:1,addEventListener(){}},document:{getElementById:id=>elements[id],createElement:()=>({})}};
vm.createContext(sandbox);vm.runInContext(script,sandbox);
assert(elements.cluster.options.length>0);
assert(elements.stats.textContent.includes('Coverage:'));
for(const mode of ['patches','sample','pruned','pairs']){elements.display.value=mode;handlers['display:input']();}
elements.transformed.checked=true;handlers['transformed:input']();
handlers['canvas:pointerdown']({clientX:10,clientY:20,pointerId:1});
handlers['canvas:pointermove']({clientX:35,clientY:40,shiftKey:false});
handlers['canvas:pointerup']();
handlers['canvas:wheel']({preventDefault(){},deltaY:-100});
elements.reset.onclick();
vm.runInContext('load({...data,clusters:[]})',sandbox);
assert(elements.empty.textContent.includes('Кластеров нет'));
console.log('Viewer smoke passed: rendering modes, controls, empty result');
