import assert from "node:assert/strict";

class FakeElement {
    constructor(tagName) { this.tagName = String(tagName).toUpperCase(); this.children = []; this.style = {}; this.listeners = new Map(); this.attributes = {}; this.dataset = {}; this.hidden = false; this.textContent = ""; this.value = ""; }
    append(...children) { this.children.push(...children.filter(Boolean)); }
    appendChild(child) { this.append(child); return child; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    addEventListener(type, handler) { const list = this.listeners.get(type) || []; list.push(handler); this.listeners.set(type, list); }
    dispatchEvent(event) { event.target ||= this; event.currentTarget = this; for (const handler of this.listeners.get(event.type) || []) handler(event); return !event.defaultPrevented; }
    click() { this.dispatchEvent({ type: "click" }); }
    focus() {}
    getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 360 }; }
    setPointerCapture() {}
    releasePointerCapture() {}
}

let hitBits = 0;
class FakeCanvas extends FakeElement {
    constructor() { super("canvas"); this.width = 640; this.height = 360; }
    getContext() {
        const noop = () => {};
        const getImageData = () => this.width === 1 && this.height === 1 ? {data: [hitBits, 0, 0, 255]} : undefined;
        return new Proxy({ getImageData }, { get: (target, key) => target[key] || noop, set: () => true });
    }
}

globalThis.document = {
    activeElement: null,
    createElement(tagName) { return String(tagName).toLowerCase() === "canvas" ? new FakeCanvas() : new FakeElement(tagName); },
};
globalThis.window = { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {}, sessionStorage: null };

const { createUniversalMaskEditor } = await import("../web/universal_mask_editor.js");

function find(root, predicate) {
    if (!root) return null;
    if (predicate(root)) return root;
    for (const child of root.children || []) { const match = find(child, predicate); if (match) return match; }
    return null;
}
function button(root, label) { return find(root, (item) => item.tagName === "BUTTON" && item.attributes["aria-label"] === label); }
function visibleButton(root, label) { return find(root, (item) => item.tagName === "BUTTON" && item.attributes["aria-label"] === label && item.style.display !== "none"); }
function control(root, label) { return find(root, (item) => item.attributes["aria-label"] === label); }
const source = { frames: 3, width: 640, height: 360, fingerprint: "sam3-clip" };
const widgets = [
    { name: "mask_data", value: "{}", options: {} },
    { name: "tracking_mode", value: "sam3" },
    { name: "grow", value: 8, options: {} }, { name: "feather", value: 4, options: {} },
    { name: "start_frame", value: 0 }, { name: "end_frame", value: -1 }, { name: "fps", value: 24 },
];
const submitted = [];
let promptCounter = 0;
const app = {
    graph: {},
    async graphToPrompt() { return { output: { "5": { class_type: "LoadVideo", inputs: {} }, "7": { class_type: "THSUniversalMaskEditor", inputs: { images: ["5", 0] } }, "99": { class_type: "KSampler", inputs: { mask: ["7", 0] } } } }; },
    queuePrompt(_number, prompt) { submitted.push(prompt); return Promise.resolve({ prompt_id: `sam3-${++promptCounter}` }); },
    async cancelPrompt() {},
};
let interruptCalls = 0;
const api = { queuePrompt(_number, prompt) { submitted.push(prompt); return Promise.resolve({ prompt_id: `sam3-${++promptCounter}` }); }, async interrupt() { interruptCalls += 1; }, addEventListener() {}, removeEventListener() {} };
const node = { id: 7, widgets, graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const editor = createUniversalMaskEditor(node, { app, api });
editor.state.source = source; editor.state.frameCount = source.frames; editor.document.source = source;
function sourceFrame(frame = 0) {
    const item = { sourceImage: { naturalWidth: 640, naturalHeight: 360 }, edit_base: { data_url: "base" } };
    editor.state.frameItems.set(frame, item); editor.state.previews.set(frame, item); editor.setFrame(frame, { load: false });
}
sourceFrame();
const tick = () => new Promise(resolve => setImmediate(resolve));
function click(x, y, extra = {}) {
    editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 1, clientX: x, clientY: y, preventDefault() {}, ...extra });
    editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 1, clientX: x, clientY: y, preventDefault() {}, ...extra });
}
async function ready(token, track = false, objectMap = null) {
    const id = editor.state.pendingPreview.promptId;
    const patch = objectMap ? { universal_mask_frame_patch: [{ frame: editor.document.sam3.frame, object_map: objectMap }] } : {};
    await editor.applyExecuted({ ...patch, prompt_id: id, universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: track, frame: editor.document.sam3.frame, seed_token: token, direction: "both", computed_frames: track ? [0,1,2] : [editor.document.sam3.frame] } }] });
    sourceFrame(editor.state.frame);
}
assert.equal(editor.state.mode, "sam_point");
assert.equal(button(editor.root, "Выбрать объект").attributes["aria-pressed"], "true");
assert.equal(submitted.length, 0, "loading an editor must not run inference");
button(editor.root, "Исключить из объекта").click();click(100, 100);await tick();
assert.equal(submitted.length, 0, "negative-only prompts must not run");
button(editor.root, "Выбрать объект").click();
click(160,90);await tick();
assert.deepEqual(editor.document.sam3.prompt, {frame:0,points:[[.25,.25]],labels:[1]});
assert.deepEqual(JSON.parse(widgets[0].value).sam3.prompt, editor.document.sam3.prompt, "point survives serialization");
assert.equal(submitted.length,1);assert.equal(submitted[0].output["99"],undefined,"click never queues H3");
assert.equal(JSON.parse(submitted[0].output["7"].inputs.mask_data).editor_request,"sam3_frame_v1");
assert.equal(JSON.parse(widgets[0].value).editor_request,undefined,"the scoped preview flag must never enter the workflow");
assert.equal(editor.state.pendingPreview.sam3Action,"frame");
assert.equal(button(editor.root,"Отследить видео").disabled,true);
const before=widgets[0].value;click(300,100);await tick();
assert.equal(submitted.length,1,"rapid clicks do not flood the queue");assert.equal(widgets[0].value,before);
editor.state.frameItems.set(1, { source: "existing-next-frame.jpg", mask: "old-track.png", edit_base: "old-base.png" });
const patch = { frame: 0, mask: "data:image/png;base64,mask", edit_base: "data:image/png;base64,base" };
const patched = await editor.applyExecuted({
    prompt_id: editor.state.pendingPreview.promptId,
    universal_mask_frame_patch: [patch],
    universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: false, frame: 0, seed_token: "point-seed-1", computed_frames: [0] } }],
});
assert.equal(patched.hasMask,true);
assert.equal(editor.state.frameItems.get(0).mask,patch.mask);
assert.deepEqual(editor.state.frameItems.get(1), { source: "existing-next-frame.jpg", mask: null, edit_base: null });
assert.equal((await editor.applyExecuted({ universal_mask_frame_patch: [patch] })).stale,true,"a patch without an active request is ignored");
sourceFrame();
assert.equal(button(editor.root,"Отследить видео").disabled,false);
assert.equal(button(editor.root,"Запустить H3").disabled,true);
assert.equal(submitted.length,1,"preview result never starts tracking automatically");
button(editor.root,"Добавить к объекту").click();click(200,100);await tick();
assert.deepEqual(editor.document.sam3.prompt.labels,[1,1]);
await ready("point-seed-2");
button(editor.root,"Исключить из объекта").click();click(280,140);await tick();
assert.deepEqual(editor.document.sam3.prompt.labels,[1,1,0]);
await ready("point-seed-3");
const pointDocument=JSON.parse(widgets[0].value);
button(editor.root,"Отследить видео").click();await tick();
assert.equal(editor.state.pendingPreview.sam3Action,"track");
assert.deepEqual(JSON.parse(widgets[0].value).sam3.prompt,pointDocument.sam3.prompt);
assert.equal(submitted.at(-1).output["99"],undefined);
assert.equal(JSON.parse(submitted.at(-1).output["7"].inputs.mask_data || "{}").editor_request,undefined);
await ready("point-seed-3",true);
assert.equal(button(editor.root,"Запустить H3").disabled,false);
sourceFrame(1);
assert.equal(button(editor.root,"Скопировать предыдущий ключ").disabled,true,"a point seed alone is not a copyable manual edit");
button(editor.root,"Кисть").click();click(220,120);await tick();
assert.equal(editor.document.frames[1].inherit,true,"brush inherits the point-seeded mask");
const stroke=JSON.stringify(editor.document.frames[1]);
assert.equal(editor.state.sam3.frameReady,true);
button(editor.root,"Отменить").click();button(editor.root,"Повторить").click();
assert.equal(JSON.stringify(editor.document.frames[1]),stroke);
assert.equal(editor.state.sam3.frameReady,true,"brush history keeps verified point seed");
sourceFrame(2);button(editor.root,"Скопировать предыдущий ключ").click();
assert.equal(JSON.stringify(editor.document.frames[2]),stroke);
assert.deepEqual(editor.document.sam3.prompt,pointDocument.sam3.prompt,"copying brush edits retains the point seed on its original frame");
assert.equal(editor.state.sam3.frameReady,true);
sourceFrame(0);button(editor.root,"Выбрать объект").click();click(400,180);await tick();
assert.equal(editor.document.sam3.prompt,undefined);
const group = editor.document.sam3.objects;
assert.equal(group.length,2,"normal click adds a second object");
assert.deepEqual(group[0],pointDocument.sam3.prompt);
assert.deepEqual(group[1].points,[[.625,.5]]);
assert.equal(control(editor.root,"Активный объект SAM3").value,"1");
assert.equal(JSON.stringify(editor.document.frames[1]),stroke,"reselection retains manual corrections");
assert.equal(button(editor.root,"Запустить H3").disabled,true);
const pendingId=editor.state.pendingPreview.promptId;
const stale=await editor.applyExecuted({
    prompt_id:"old-point-request", universal_mask_frame_patch:[patch],
    universal_mask_meta:[{source:{...source,fingerprint:"old-source"},has_mask:true}],
});
assert.equal(stale.stale,true);assert.equal(editor.state.pendingPreview.promptId,pendingId);
assert.equal(editor.state.mismatchBlocked,false);
await ready("point-seed-4");
const objectSelect = control(editor.root,"Активный объект SAM3");
objectSelect.value="0"; objectSelect.dispatchEvent({type:"change"});
button(editor.root,"Исключить из объекта").click();click(128,90);await tick();
assert.deepEqual(editor.document.sam3.objects[0].labels,[1,1,0,0]);
assert.deepEqual(editor.document.sam3.objects[1],group[1],"negative point refines only the active object");
await ready("point-seed-5");
const refinedGroup = JSON.parse(widgets[0].value).sam3.objects;
assert.equal(button(editor.root,"Удалить объект"),null);
function rightClick(x, y) {
    let prevented = false;
    editor.canvas.dispatchEvent({type:"contextmenu",button:2,clientX:x,clientY:y,preventDefault(){ prevented=true; }});
    assert.equal(prevented,true);
}
rightClick(128,90);await tick();
assert.equal(editor.document.sam3.objects.length,2);
assert.deepEqual(editor.document.sam3.objects[0].labels,[1,1,0]);
await ready("point-seed-marker-removed", false, "data:image/png;base64,objectMap");
const afterMarker = JSON.parse(widgets[0].value).sam3.objects;
globalThis.Image = class {
    naturalWidth = 640;
    naturalHeight = 360;
    set src(_value) { queueMicrotask(() => this.onload?.()); }
};
objectSelect.value="1"; objectSelect.dispatchEvent({type:"change"});
hitBits = 3; // Both masks cover the pixel. The active object is removed.
rightClick(450,220);await tick();
assert.deepEqual(editor.document.sam3.prompt,afterMarker[0]);
assert.equal(editor.document.sam3.objects,undefined);
await ready("point-seed-6");
delete globalThis.Image;
button(editor.root,"Отменить").click();
assert.deepEqual(editor.document.sam3.objects,afterMarker);
button(editor.root,"Отменить").click();
assert.deepEqual(editor.document.sam3.objects,refinedGroup);
button(editor.root,"Очистить кадр").click();
assert.equal(editor.document.sam3.prompt,undefined);
assert.equal(editor.document.sam3.objects,undefined);
assert.equal(editor.state.sam3.frameReady,false);
button(editor.root,"Отменить").click();
assert.deepEqual(editor.document.sam3.objects,refinedGroup);
assert.equal(editor.state.sam3.frameReady,false,"undo of a changed selection requires fresh proof");
// Dragging, middle mouse and cancelled pointers never select an object.
const count=submitted.length;
click(160,90,{button:1});await tick();assert.equal(submitted.length,count);
editor.canvas.dispatchEvent({type:"pointerdown",button:0,pointerId:3,clientX:160,clientY:90,preventDefault(){}});
editor.canvas.dispatchEvent({type:"pointermove",pointerId:3,clientX:200,clientY:90});
editor.canvas.dispatchEvent({type:"pointerup",pointerId:3,clientX:200,clientY:90});
await tick();assert.equal(submitted.length,count);
editor.canvas.dispatchEvent({type:"pointerdown",button:0,pointerId:3,clientX:160,clientY:90,preventDefault(){}});
editor.canvas.dispatchEvent({type:"pointercancel",pointerId:3});await tick();assert.equal(submitted.length,count);
editor.commit();const saved=widgets[0].value;editor.destroy();
const restored=createUniversalMaskEditor(node,{app,api});
assert.deepEqual(restored.document.sam3.objects,refinedGroup);
assert.equal(restored.state.sam3.frameReady,false,"saved seed does not invent live readiness");
restored.destroy();console.log("universal_mask_editor_sam3_points: PASS");
