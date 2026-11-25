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

class FakeCanvas extends FakeElement {
    constructor() { super("canvas"); this.width = 640; this.height = 360; }
    getContext() { const noop = () => {}; return new Proxy({}, { get: () => noop, set: () => true }); }
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
const apiEvents = new Map();
const api = {
    queuePrompt(_number, prompt) { submitted.push(prompt); return Promise.resolve({ prompt_id: `sam3-${++promptCounter}` }); },
    async interrupt() { interruptCalls += 1; },
    addEventListener(type, listener) { apiEvents.set(type, listener); },
    removeEventListener(type) { apiEvents.delete(type); },
};
const node = { id: 7, widgets, graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const editor = createUniversalMaskEditor(node, { app, api });
editor.state.source = source; editor.state.frameCount = source.frames; editor.document.source = source;
editor.state.frameItems.set(0, { sourceImage: { naturalWidth: source.width, naturalHeight: source.height } });
editor.state.previews.set(0, editor.state.frameItems.get(0)); editor.setFrame(0, { load: false });

assert.equal(control(editor.root, "Трекинг").value, "sam3");
assert.equal(control(editor.root, "Направление").value, "both");
assert.match(control(editor.root, "Grow").title, /Отследить видео/);
assert.doesNotMatch(control(editor.root, "Grow").title, /Рассчитать маску/);
assert.equal(button(editor.root, "Отследить видео").disabled, true, "tracking waits for frame result");
assert.equal(button(editor.root, "Запустить граф").disabled, true, "graph queue waits for full track");

// Seed boxes are local; drawing never queues a request.
button(editor.root, "Прямоугольник").click();
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 1, clientX: 120, clientY: 80, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointermove", pointerId: 1, clientX: 300, clientY: 250 });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 1, clientX: 300, clientY: 250 });
assert.equal(submitted.length, 0);
assert.equal(editor.document.frames["0"].events.at(-1).refine_method, "sam3");
const seedToken = editor.state.sam3.seedToken;
const frameButton = button(editor.root, "Рассчитать кадр");
frameButton.click();
await new Promise((resolve) => setImmediate(resolve));
assert.equal(submitted.length, 1);
const framePrompt = submitted.at(-1);
assert.equal(framePrompt.output["99"], undefined, "frame preview has no downstream descendants");
assert.equal(JSON.parse(widgets[0].value).sam3.action, "frame");
assert.equal(JSON.parse(widgets[0].value).sam3.seed_token, seedToken);
const framePromptId = editor.state.pendingPreview.promptId;
const backendToken = "sam3v1-authoritative-token";
await editor.applyExecuted({ prompt_id: framePromptId, universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: false, frame: 0, seed_token: backendToken, direction: "forward", computed_frames: [0] } }], universal_mask_frames: [{ frame: 0, source: { data_url: "data:image/png;base64,source" }, mask: { data_url: "data:image/png;base64,mask" }, edit_base: { data_url: "data:image/png;base64,base" } }] });
assert.equal(editor.state.sam3.frameReady, true);
assert.equal(editor.state.sam3.trackReady, false);
assert.equal(editor.state.sam3.seedToken, backendToken, "backend token replaces the client placeholder");
assert.equal(button(editor.root, "Отследить видео").disabled, false);
assert.equal(button(editor.root, "Запустить граф").disabled, true);

// A targeted cancellation or unrelated track error keeps the approved frame
// proof/token usable for a retry. It must not queue another request or invoke a
// global interrupt on its own.
const approvedFrameSignature = editor.state.sam3.frameSignature;
button(editor.root, "Отследить видео").click();
await new Promise((resolve) => setImmediate(resolve));
const cancelledTrackPromptId = editor.state.pendingPreview.promptId;
const progress = find(editor.root, item => item.tagName === "PROGRESS");
const overlay = find(editor.root, item => item.className === "ths-mask-progress");
const caption = find(editor.root, item => item.className === "ths-mask-progress-caption");
const time = find(editor.root, item => item.className === "ths-mask-progress-time");
assert.equal(overlay.style.display, "grid");
assert.match(caption.textContent, /Очереди|очереди/);
apiEvents.get("execution_start")({ detail: { prompt_id: cancelledTrackPromptId } });
assert.match(caption.textContent, /Подготовка/);
const report = (value, promptId = cancelledTrackPromptId, nodeId = "7") => {
    apiEvents.get("universal_inpaint_mask_progress")({ detail: {
        node: nodeId, prompt_id: promptId, phase: "sam3_track", value, max: 56,
    } });
};
report(1, "another-job");
report(1, cancelledTrackPromptId, "99");
assert.equal(progress.value, "");
const realNow = Date.now;
try {
    let now = realNow();
    Date.now = () => now;
    report(1);
    assert.equal(progress.max, 56);
    assert.equal(progress.value, 1);
    now += 2500;
    report(6);
    assert.match(caption.textContent, /6 из 56 кадров/);
    assert.match(time.textContent, /Осталось ≈ 0:25/);
    report(3);
    assert.equal(progress.value, 6, "late frame samples cannot move progress backwards");
} finally {
    Date.now = realNow;
}
const submittedBeforeCancellation = submitted.length;
assert.equal(editor.handleRuntimeEvent({ prompt_id: cancelledTrackPromptId, exception_message: "user cancelled track" }, "execution_interrupted"), true);
assert.equal(submitted.length, submittedBeforeCancellation, "cancellation does not auto-queue another request");
assert.equal(interruptCalls, 0, "targeted runtime cancellation does not invoke a global interrupt");
assert.equal(editor.state.sam3.frameReady, true);
assert.equal(editor.state.sam3.frameSignature, approvedFrameSignature);
assert.equal(editor.state.sam3.trackReady, false);
assert.equal(overlay.style.display, "none", "interruption clears the overlay");
assert.equal(button(editor.root, "Отследить видео").disabled, false, "track stays enabled after targeted cancellation");
assert.equal(button(editor.root, "Запустить граф").disabled, true, "queue stays disabled until a full track succeeds");

button(editor.root, "Отследить видео").click();
await new Promise((resolve) => setImmediate(resolve));
const failedTrackPromptId = editor.state.pendingPreview.promptId;
const submittedBeforeTrackError = submitted.length;
assert.equal(editor.handleRuntimeEvent({ prompt_id: failedTrackPromptId, exception_message: "transient SAM3 runtime failure" }, "execution_error"), true);
assert.equal(submitted.length, submittedBeforeTrackError, "track error does not auto-queue another request");
assert.equal(interruptCalls, 0, "track error does not invoke a global interrupt");
assert.equal(editor.state.sam3.frameReady, true);
assert.equal(editor.state.sam3.frameSignature, approvedFrameSignature);
assert.equal(editor.state.sam3.trackReady, false);
assert.equal(button(editor.root, "Отследить видео").disabled, false, "track stays enabled after targeted error");
assert.equal(button(editor.root, "Запустить граф").disabled, true);

const direction = control(editor.root, "Направление"); direction.value = "forward"; direction.dispatchEvent({ type: "change" });
assert.equal(editor.state.sam3.frameReady, true, "direction does not invalidate the seed frame");
assert.equal(editor.state.sam3.trackReady, false, "direction invalidates a completed track");
direction.value = "both"; direction.dispatchEvent({ type: "change" });
button(editor.root, "Отследить видео").click();
await new Promise((resolve) => setImmediate(resolve));
assert.equal(submitted.length, 4);
const trackPrompt = submitted.at(-1);
assert.equal(trackPrompt.output["99"], undefined, "track preview has no downstream descendants");
assert.equal(JSON.parse(widgets[0].value).sam3.action, "track");
const trackPromptId = editor.state.pendingPreview.promptId;
await editor.applyExecuted({ prompt_id: trackPromptId, universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: true, frame: 0, seed_token: backendToken, direction: "both", computed_frames: [0, 1, 2] } }], universal_mask_frames: [{ frame: 0, source: { data_url: "data:image/png;base64,source" }, mask: { data_url: "data:image/png;base64,mask" }, edit_base: { data_url: "data:image/png;base64,base" } }] });
assert.equal(editor.state.sam3.trackReady, true);
assert.equal(button(editor.root, "Запустить граф").disabled, false);

// A local correction is retained in mask_data but invalidates only the full track.
editor.state.frameItems.set(1, { sourceImage: { naturalWidth: source.width, naturalHeight: source.height }, edit_base: { data_url: "base" } });
editor.state.previews.set(1, editor.state.frameItems.get(1)); editor.setFrame(1, { load: false });
button(editor.root, "Кисть").click();
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 2, clientX: 180, clientY: 140, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 2, clientX: 180, clientY: 140 });
assert.ok(editor.document.frames["1"].events.some((event) => event.type === "stroke"));
assert.equal(editor.state.sam3.frameReady, true);
assert.equal(editor.state.sam3.trackReady, false);
assert.equal(button(editor.root, "Запустить граф").disabled, true);
// Undo/redo of a correction must keep the approved object seed usable.
// The full track still needs recalculation before the downstream graph can run.
const correctedFrames = JSON.stringify(editor.document.frames);
button(editor.root, "Отменить").click();
assert.equal(editor.state.sam3.frameReady, true, "undoing a stroke keeps the approved SAM3 seed");
assert.equal(button(editor.root, "Отследить видео").disabled, false);
assert.equal(button(editor.root, "Запустить граф").disabled, true);
button(editor.root, "Повторить").click();
assert.equal(JSON.stringify(editor.document.frames), correctedFrames);
assert.equal(editor.state.sam3.frameReady, true, "redoing a stroke keeps the approved SAM3 seed");
assert.equal(editor.state.sam3.seedToken, backendToken);
assert.equal(button(editor.root, "Отследить видео").disabled, false);
const correctedHistoryIndex = editor.state.historyIndex;
while (editor.state.historyIndex > 1) button(editor.root, "Отменить").click();
assert.equal(editor.state.sam3.frameReady, true, "history from before frame calculation can reuse the identical live verified seed");
assert.equal(editor.document.sam3.seed_token, backendToken, "a historical client token never replaces the verified backend token");
while (editor.state.historyIndex < correctedHistoryIndex) button(editor.root, "Повторить").click();
assert.equal(JSON.stringify(editor.document.frames), correctedFrames);
// Changing the actual object prompt revokes the proof. Undo must not invent
// readiness from an old history token after that revocation.
editor.setFrame(0, { load: false });
button(editor.root, "Очистить кадр").click();
assert.equal(editor.state.sam3.frameReady, false);
button(editor.root, "Отменить").click();
assert.equal(editor.state.sam3.frameReady, false, "undo cannot resurrect a revoked seed proof");
assert.equal(button(editor.root, "Отследить видео").disabled, true);
button(editor.root, "Рассчитать кадр").click();
await new Promise((resolve) => setImmediate(resolve));
await editor.applyExecuted({ prompt_id: editor.state.pendingPreview.promptId, universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: false, frame: 0, seed_token: backendToken, direction: "both", computed_frames: [0] } }] });
editor.setFrame(1, { load: false });
button(editor.root, "Отследить видео").click();
await new Promise((resolve) => setImmediate(resolve));
const staleTrackPromptId = editor.state.pendingPreview.promptId;
await editor.applyExecuted({ prompt_id: staleTrackPromptId, universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: true, frame: 0, seed_token: "stale-token", direction: "both", computed_frames: [0, 1, 2] } }] });
assert.equal(editor.state.sam3.trackReady, false, "track output with the wrong token is rejected");
assert.equal(button(editor.root, "Запустить граф").disabled, true);

// Source mismatch clears stale readiness and blocks downstream work.
await editor.applyExecuted({ universal_mask_meta: [{ source_mismatch: true, source_mismatch_reason: "fingerprint", previous_source: source, source: { ...source, fingerprint: "other" }, has_mask: false }] });
assert.equal(editor.state.sam3.frameReady, false);
assert.equal(editor.state.sam3.trackReady, false);
assert.equal(button(editor.root, "Запустить граф").disabled, true);
editor.destroy();

const freshNode = { id: 9, widgets: [{ name: "mask_data", value: "{}" }], graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const freshEditor = createUniversalMaskEditor(freshNode, { app: null, api: null });
assert.equal(control(freshEditor.root, "Трекинг").value, "sam3", "empty nodes default to SAM3");
assert.ok(visibleButton(freshEditor.root, "Загрузить кадры"), "empty SAM3 editor tells the user to load frames first");
assert.equal(visibleButton(freshEditor.root, "Рассчитать кадр"), null, "empty SAM3 editor does not claim a frame is loaded");
freshEditor.destroy();
const legacyNode = { id: 10, widgets: [{ name: "mask_data", value: JSON.stringify({ version: 1, frames: { "0": { events: [{ type: "rect", x: 0.1, y: 0.1, w: 0.2, h: 0.2 }] } } }) }], graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const legacyEditor = createUniversalMaskEditor(legacyNode, { app: null, api: null });
assert.equal(control(legacyEditor.root, "Трекинг").value, "optical_flow", "content-bearing legacy documents retain Optical Flow");
legacyEditor.destroy();
console.log("universal_mask_editor_sam3_video: PASS");
