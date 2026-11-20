import assert from "node:assert/strict";
import { trialPrompt, trialSpan, windowBox } from "../web/h3_window.js";
const frames = Array.from({length:22},(_,i)=>7+i);
const extent = {box:[140,60,259,100],frames};
assert.deepEqual(windowBox(373,278,extent,{}),[71,0,327,256]);
assert.deepEqual(windowBox(373,278,extent,{mode:"full"}),[0,0,373,278]);
assert.deepEqual(trialSpan(frames,24),[17,29]);
const prompt = {output:{
    "7":{class_type:"THSUniversalH3Window",inputs:{settings:"{}"}},
    "8":{class_type:"THSUniversalRepairCompare",inputs:{work_window:["7",3]}},
    "9":{class_type:"SaveVideo",inputs:{video:["7",0]}},
}};
const {target,scoped} = trialPrompt(prompt,"7",24,[17,29],36);
assert.equal(target,"8"); assert.deepEqual(Object.keys(scoped.output),["7","8"]);
assert.equal(scoped.output["7"].inputs.probe_frame,24);
assert.equal(prompt.output["7"].inputs.probe_frame,undefined);
