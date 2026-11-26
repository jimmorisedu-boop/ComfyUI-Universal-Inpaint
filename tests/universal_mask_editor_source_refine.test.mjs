import assert from "node:assert/strict";

class FakeElement {
    constructor(tagName) {
        this.tagName = String(tagName).toUpperCase();
        this.children = [];
        this.style = {};
        this.listeners = new Map();
        this.attributes = {};
        this.textContent = "";
        this.dataset = {};
        this.className = "";
        this.parentNode = null;
    }
    append(...children) { for (const child of children.filter(Boolean)) { if (typeof child === "object") child.parentNode = this; this.children.push(child); } }
    appendChild(child) { this.append(child); return child; }
    after(...children) { this.parentNode?.append(...children); }
    remove() { this.removed = true; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    addEventListener(type, handler) { const list = this.listeners.get(type) || []; list.push(handler); this.listeners.set(type, list); }
    removeEventListener(type, handler) { const list = this.listeners.get(type) || []; this.listeners.set(type, list.filter((item) => item !== handler)); }
    dispatchEvent(event) { event.target ||= this; event.currentTarget = this; for (const handler of this.listeners.get(event.type) || []) handler(event); return !event.defaultPrevented; }
    click() { this.dispatchEvent({ type: "click" }); }
    focus() {}
    getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 360 }; }
    setPointerCapture() {}
    releasePointerCapture() {}
    getContext() { return new Proxy({}, { get: () => () => {}, set: () => true }); }
}

class FakeCanvas extends FakeElement {
    constructor() { super("canvas"); this.width = 640; this.height = 360; }
}

const apiListeners = new Map();
const submitted = [];
const api = {
    apiURL: (path) => `http://comfy.test${path}`,
    queuePrompt: async (_number, prompt) => { submitted.push(prompt); return { prompt_id: `p-${submitted.length}` }; },
    addEventListener(type, handler) { const list = apiListeners.get(type) || []; list.push(handler); apiListeners.set(type, list); },
    removeEventListener(type, handler) { apiListeners.set(type, (apiListeners.get(type) || []).filter((item) => item !== handler)); },
    emit(type, detail) { for (const handler of apiListeners.get(type) || []) handler({ detail }); },
};
const app = {
    graph: {},
    graphToPrompt: async () => ({ output: {
        "1": { class_type: "LoadVideo", inputs: {} },
        "2": { class_type: "GetVideoComponents", inputs: { video: ["1", 0] } },
        "3": { class_type: "THSUniversalMaskEditor", inputs: { images: ["2", 0] } },
        "8": { class_type: "KSampler", inputs: { mask: ["3", 0] } },
    } }),
};

globalThis.document = {
    activeElement: null,
    createElement(tagName) { return String(tagName).toLowerCase() === "canvas" ? new FakeCanvas() : new FakeElement(tagName); },
};
globalThis.window = { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {} };

const { createUniversalMaskEditor } = await import("../web/universal_mask_editor.js");

function find(root, predicate) {
    if (!root) return null;
    if (predicate(root)) return root;
    for (const child of root.children || []) { const found = find(child, predicate); if (found) return found; }
    return null;
}
function findButton(root, text) { return find(root, (item) => item.tagName === "BUTTON" && (item.textContent === text || item.attributes["aria-label"] === text)); }

const oldDocument = {
    version: 1,
    source: { frames: 2, width: 160, height: 128, fingerprint: "old-source" },
    frames: { "1": { events: [{ type: "rect", mode: "add", x: 0.1, y: 0.2, w: 0.3, h: 0.4 }] } },
    anchors: [1],
    tracking_mode: "keyframes",
};
const widgets = [
    { name: "mask_data", value: JSON.stringify(oldDocument), options: {} },
    { name: "tracking_mode", value: "keyframes" },
    { name: "grow", value: 17 }, { name: "feather", value: 9 },
    { name: "start_frame", value: 1 }, { name: "end_frame", value: 1 }, { name: "fps", value: 24 },
];
const node = { id: 3, widgets, properties: {}, graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const editor = createUniversalMaskEditor(node, { app, api });
const beforeMismatch = JSON.stringify(editor.document);
const mismatchOutput = {
    universal_mask_meta: [{
        source_mismatch: true,
        source_mismatch_reason: "frames",
        previous_source: oldDocument.source,
        source: { frames: 3, width: 320, height: 256, fingerprint: "new-source" },
        has_mask: false,
        anchors: [],
    }],
    universal_mask_frames: [{ frame: 0, source: { data_url: "data:image/png;base64,new" } }],
};
const deferredImages = [];
globalThis.Image = class {
    constructor() { this.naturalWidth = 320; this.naturalHeight = 256; }
    set src(value) { deferredImages.push(this); }
};
const mismatch = await editor.applyExecuted(mismatchOutput);
for (const image of deferredImages.splice(0)) image.onload();
await new Promise((resolve) => setTimeout(resolve, 0));
delete globalThis.Image;
assert.equal(mismatch.sourceMismatch, true);
assert.equal(editor.state.mismatchBlocked, true);
assert.equal(JSON.stringify(editor.document), beforeMismatch, "persisted old document is retained on mismatch");
assert.equal(node.properties.__thsPreviousMask.grow, 17);
assert.equal(node.properties.__thsPreviousMask.feather, 9);
assert.match(editor.status.textContent, /2 кадр.*160×128.*3 кадр.*320×256/);
assert.equal(editor.status.dataset.state, "warn", "asynchronous source loading must retain the mismatch warning");
assert.ok(editor.debugReport().events.some((entry) => entry.event === "frame.ready"));
assert.ok(editor.debugReport().events.some((entry) => entry.event === "preview.received" && entry.details.mismatch));
assert.ok(!JSON.stringify(editor.debugReport()).includes("data:image"), "diagnostics never include image payloads");
assert.equal(findButton(editor.root, "Начать маску для нового клипа").style.display, "inline-flex");
assert.equal(findButton(editor.root, "Запустить граф").disabled, true);

findButton(editor.root, "Начать маску для нового клипа").click();
assert.deepEqual(editor.document.frames, {});
assert.deepEqual(editor.document.source, mismatchOutput.universal_mask_meta[0].source);
assert.equal(widgets.find((item) => item.name === "start_frame").value, 0);
assert.equal(widgets.find((item) => item.name === "end_frame").value, -1);
assert.equal(editor.state.frame, 0);
assert.equal(editor.state.zoom, 1);
assert.equal(node.properties.__thsPreviousMask.mask_data, JSON.stringify(oldDocument));
const editorAlertBanner = find(editor.root, (item) => String(item.className || "").includes("ths-mask-alert-overlay"));
assert.equal(editorAlertBanner.style.display, "flex", "recovery banner remains visible while accepted clip is blank");
assert.equal(editorAlertBanner.dataset.state, "info", "neutral status banner does not show warning border");
assert.equal(findButton(editor.root, "Вернуть прежнюю маску").style.display, "inline-flex");

findButton(editor.root, "Вернуть прежнюю маску").click();
assert.deepEqual(editor.document.frames, oldDocument.frames);
assert.equal(widgets.find((item) => item.name === "grow").value, 17);
assert.equal(widgets.find((item) => item.name === "feather").value, 9);
assert.equal(widgets.find((item) => item.name === "start_frame").value, 1);
assert.equal(widgets.find((item) => item.name === "end_frame").value, 1);
assert.equal(editor.state.mismatchBlocked, true);
assert.match(editor.status.textContent, /Выберите прежний источник/);
findButton(editor.root, "Вернуть прежнюю маску").click();
assert.deepEqual(editor.document.frames, {}, "restore swaps the accepted empty B document into the backup slot");
findButton(editor.root, "Вернуть прежнюю маску").click();
assert.deepEqual(editor.document.frames, oldDocument.frames, "a second restore swaps back to A");
findButton(editor.root, "Рассчитать маску")?.click() || findButton(editor.root, "Загрузить кадры")?.click();
await new Promise((resolve) => setTimeout(resolve, 0));
assert.ok(editor.state.pendingPreview, "manual preview remains available while mismatch is blocked");
await editor.applyExecuted({
    universal_mask_meta: [{ source: oldDocument.source, has_mask: false, anchors: [] }],
    universal_mask_frames: [{ frame: 0, source: { data_url: "data:image/png;base64,old" } }],
});
assert.equal(editor.state.mismatchBlocked, false, "matching previous source recovers the preserved document");
// A later B→C transition must replace the one-slot backup with B, rather than
// silently retaining the older A document.
const sourceB = { frames: 2, width: 200, height: 140, fingerprint: "source-b" };
editor.document.frames = {};
editor.document.source = sourceB;
editor.state.source = sourceB;
editor.state.frameCount = sourceB.frames;
editor.state.localEdits = false;
editor.commit();
editor.state.frameItems.set(0, { sourceImage: { naturalWidth: 200, naturalHeight: 140 } });
editor.state.previews.set(0, editor.state.frameItems.get(0));
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 22, clientX: 180, clientY: 120, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 22, clientX: 180, clientY: 120 });
assert.equal(editor.state.localEdits, true);
await editor.applyExecuted({ universal_mask_meta: [{ source_mismatch: true, source_mismatch_reason: "fingerprint", previous_source: sourceB, source: { frames: 3, width: 200, height: 140, fingerprint: "source-c" }, has_mask: false, anchors: [] }] });
assert.equal(node.properties.__thsPreviousMask.source.fingerprint, "source-b");
assert.ok(JSON.parse(node.properties.__thsPreviousMask.mask_data).frames["0"], "B document is the new recoverable backup");

// Re-create a clean editor for refinement/queue tests.
const refineWidgets = [
    { name: "mask_data", value: "{}", options: {} }, { name: "tracking_mode", value: "optical_flow" },
    { name: "grow", value: 8 }, { name: "feather", value: 4 }, { name: "start_frame", value: 0 }, { name: "end_frame", value: -1 },
];
const refineNode = { id: 3, widgets: refineWidgets, graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const refineEditor = createUniversalMaskEditor(refineNode, { app, api });
refineEditor.state.source = { frames: 1, width: 160, height: 128, fingerprint: "new-source" };
refineEditor.state.frameCount = 1;
refineEditor.state.frameItems.set(0, { sourceImage: { naturalWidth: 160, naturalHeight: 128 }, source: { data_url: "data:image/png;base64,source" } });
refineEditor.state.previews.set(0, refineEditor.state.frameItems.get(0));
const refinementPrepare = findButton(refineEditor.root, "Загрузить кадры");
const requestsBeforeDraft = submitted.length;
findButton(refineEditor.root, "Выделять объект").click();
findButton(refineEditor.root, "Прямоугольник").click();
refineEditor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 1, clientX: 150, clientY: 60, preventDefault() {} });
refineEditor.canvas.dispatchEvent({ type: "pointermove", pointerId: 1, clientX: 400, clientY: 300 });
refineEditor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 1, clientX: 400, clientY: 300 });
await new Promise((resolve) => setTimeout(resolve, 0));
assert.equal(refineEditor.state.pendingPreview, null, "drawing GrabCut does not start calculation");
assert.equal(submitted.length, requestsBeforeDraft);
assert.match(refineEditor.status.textContent, /Рамка сохранена.*Рассчитать маску/);
findButton(refineEditor.root, "Отменить").click();
assert.equal(refineEditor.document.frames["0"]?.events?.length || 0, 0, "an accidental box can be undone before any request");
findButton(refineEditor.root, "Повторить").click();
assert.equal(submitted.length, requestsBeforeDraft, "undo/redo remain local");
refinementPrepare.click();
await new Promise((resolve) => setTimeout(resolve, 0));
assert.equal(submitted.length, requestsBeforeDraft + 1, "the calculate button submits exactly once");
assert.ok(refineEditor.state.pendingPreview);
assert.equal(submitted.at(-1).output["8"], undefined, "manual refinement submits ancestors only");
assert.deepEqual(Object.keys(submitted.at(-1).output).sort(), ["1", "2", "3"]);
const pendingPrompt = refineEditor.state.pendingPreview.promptId;
api.emit("execution_error", { prompt_id: pendingPrompt, node_id: "2", exception_message: "upstream failed" });
assert.equal(refineEditor.state.pendingPreview, null);
assert.equal(findButton(refineEditor.root, "Рассчитать маску").disabled, false);
assert.match(refineEditor.status.textContent, /Ошибка выполнения/);
assert.equal(refineEditor.document.frames["0"].events[0].refine_method, "grabcut");
const methodSelect = find(refineEditor.root, (item) => item.attributes?.["aria-label"] === "Метод выделения");
methodSelect.value = "sam3";
methodSelect.dispatchEvent({ type: "change" });
assert.equal(refineNode.properties.__thsRefineMethod, "sam3");
assert.equal(refineEditor.state.grabcut, true);
assert.equal(refineEditor.state.mode, "rect");
refineEditor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 2, clientX: 150, clientY: 60, preventDefault() {} });
refineEditor.canvas.dispatchEvent({ type: "pointermove", pointerId: 2, clientX: 400, clientY: 300 });
refineEditor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 2, clientX: 400, clientY: 300 });
await new Promise((resolve) => setTimeout(resolve, 0));
assert.equal(refineEditor.document.frames["0"].events.at(-1).refine_method, "sam3");
assert.equal(refineEditor.document.frames["0"].events[0].refine_method, "grabcut", "switching method preserves earlier rectangles");
assert.equal(JSON.parse(refineWidgets[0].value).frames["0"].events.at(-1).refine_method, "sam3");
assert.equal(submitted.length, requestsBeforeDraft + 1, "SAM3 selection and rectangle stay local until Calculate");
assert.equal(methodSelect.disabled, false, "a draft leaves the method selector available");
for (const [tool, pointerId] of [["Кисть", 3], ["Ластик", 4]]) {
    findButton(refineEditor.root, tool).click();
    refineEditor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId, clientX: 240, clientY: 160, preventDefault() {} });
    refineEditor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId, clientX: 240, clientY: 160 });
}
await new Promise((resolve) => setTimeout(resolve, 0));
assert.equal(submitted.length, requestsBeforeDraft + 1, "brush and eraser on a refined frame do not auto-submit");
findButton(refineEditor.root, "Отменить").click();
findButton(refineEditor.root, "Отменить").click();
assert.equal(refineEditor.document.frames["0"].events.at(-1).refine_method, "sam3");
refinementPrepare.click();
refinementPrepare.click();
await new Promise((resolve) => setTimeout(resolve, 0));
assert.equal(submitted.length, requestsBeforeDraft + 2, "repeated Calculate during a pending request does not duplicate it");
assert.equal(methodSelect.disabled, true, "method cannot change during calculation");
await refineEditor.applyExecuted({ universal_mask_meta: [{ source: refineEditor.state.source, has_mask: false, refinement_error: "SAM3: local checkpoint is missing" }] });
assert.equal(methodSelect.disabled, false);
assert.match(refineEditor.status.textContent, /SAM3: local checkpoint is missing/, "SAM3 remedy is not replaced with a generic warning");
refineEditor.syncFromConfiguredWidget(true);
assert.equal(methodSelect.value, "sam3", "configured node restores method preference");
refineEditor.destroy();
const listenerCount = (apiListeners.get("execution_error") || []).length;
assert.equal(listenerCount, 1, "destroy detaches only the refined editor listener");
editor.destroy();
assert.equal((apiListeners.get("execution_error") || []).length, 0);

// Alert banner auto-hides once a new mask is created or computed, and dismisses manually when idle.
const bannerWidgets = [
    { name: "mask_data", value: "{}", options: {} }, { name: "tracking_mode", value: "static" },
    { name: "grow", value: 8 }, { name: "feather", value: 4 }, { name: "start_frame", value: 0 }, { name: "end_frame", value: -1 },
];
const bannerSourceOld = { frames: 2, width: 160, height: 128, fingerprint: "old-fp" };
const bannerSourceNew = { frames: 3, width: 320, height: 256, fingerprint: "new-fp" };
const bannerDocOld = { version: 1, source: bannerSourceOld, frames: { "0": { events: [{ type: "rect", mode: "add", x: 0.2, y: 0.2, w: 0.3, h: 0.3 }] } } };
bannerWidgets[0].value = JSON.stringify(bannerDocOld);
const bannerNode = { id: 4, widgets: bannerWidgets, properties: {}, graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const bannerEditor = createUniversalMaskEditor(bannerNode, { app, api });
await bannerEditor.applyExecuted({
    universal_mask_meta: [{ source_mismatch: true, source_mismatch_reason: "frames", previous_source: bannerSourceOld, source: bannerSourceNew, has_mask: false, anchors: [] }],
    universal_mask_frames: [{ frame: 0, source: { data_url: "data:image/png;base64,new" } }],
});
findButton(bannerEditor.root, "Начать маску для нового клипа").click();
const bannerOverlay = find(bannerEditor.root, (item) => String(item.className || "").includes("ths-mask-alert-overlay"));
const dismissButton = find(bannerEditor.root, (item) => item.attributes?.["aria-label"] === "Скрыть уведомление");
assert.equal(bannerOverlay.style.display, "flex");
assert.equal(dismissButton.style.display, "inline-flex");

// Computing a mask hides the recovery banner instead of leaving "Маска рассчитана..." stuck in an orange box.
await bannerEditor.applyExecuted({
    universal_mask_meta: [{ source: bannerSourceNew, has_mask: true, anchors: [0] }],
    universal_mask_frames: [{ frame: 0, source: { data_url: "data:image/png;base64,new" }, mask: { data_url: "data:image/png;base64,mask" } }],
});
assert.equal(bannerOverlay.style.display, "none", "computed mask hides the old-clip recovery banner");

// Dismissing manually on an accepted blank clip clears the backup banner and persisted backup.
await bannerEditor.applyExecuted({
    universal_mask_meta: [{ source_mismatch: true, source_mismatch_reason: "fingerprint", previous_source: bannerSourceNew, source: bannerSourceOld, has_mask: false, anchors: [] }],
    universal_mask_frames: [{ frame: 0, source: { data_url: "data:image/png;base64,old" } }],
});
findButton(bannerEditor.root, "Начать маску для нового клипа").click();
assert.ok(bannerNode.properties.__thsPreviousMask);
dismissButton.click();
assert.equal(bannerOverlay.style.display, "none", "dismiss button hides recovery banner");
assert.equal(bannerNode.properties.__thsPreviousMask, undefined, "dismiss clears persisted backup property");
bannerEditor.destroy();

// "Новое видео" resets the old mask/source binding cleanly and triggers a partial preview request for the new clip.
const newVideoDoc = {
    version: 1,
    source: { frames: 2, width: 160, height: 128, fingerprint: "prev-clip" },
    crop: { mode: "manual", margin: 32, multiple: 16, box: [10, 10, 80, 80], size: [160, 128] },
    frames: { "0": { events: [{ type: "rect", mode: "add", x: 0.2, y: 0.2, w: 0.3, h: 0.3 }] } },
    sam3: { action: "track", frame: 0, direction: "both", seed_token: "old-token" },
};
const newVideoWidgets = [
    { name: "mask_data", value: JSON.stringify(newVideoDoc), options: {} },
    { name: "tracking_mode", value: "sam3" },
    { name: "grow", value: 8 },
    { name: "feather", value: 4 },
    { name: "start_frame", value: 1 },
    { name: "end_frame", value: 1 },
];
const newVideoNode = { id: 3, widgets: newVideoWidgets, properties: {}, graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const newVideoApp = {
    graph: {},
    graphToPrompt: async () => ({
        output: {
            "1": { class_type: "LoadVideo", inputs: {} },
            "2": { class_type: "GetVideoComponents", inputs: { video: ["1", 0] } },
            "3": { class_type: "THSUniversalMaskEditor", inputs: { images: ["2", 0], mask_data: newVideoWidgets[0].value } },
            "8": { class_type: "KSampler", inputs: { mask: ["3", 0] } },
        },
    }),
};
const newVideoEditor = createUniversalMaskEditor(newVideoNode, { app: newVideoApp, api });
const submittedBeforeNewVideo = submitted.length;
const newVideoButton = findButton(newVideoEditor.root, "Новое видео");
assert.ok(newVideoButton, "header renders 'Новое видео' button");
newVideoButton.click();
await new Promise((resolve) => setTimeout(resolve, 0));

assert.deepEqual(newVideoEditor.document.frames, {}, "old frames cleared");
assert.equal(newVideoEditor.document.source, undefined, "old source fingerprint cleared");
assert.equal(newVideoWidgets.find((item) => item.name === "start_frame").value, 0, "start_frame reset to 0");
assert.equal(newVideoWidgets.find((item) => item.name === "end_frame").value, -1, "end_frame reset to -1");
assert.deepEqual(newVideoEditor.document.crop, { mode: "auto", margin: 32, multiple: 16 }, "stale manual box switches to auto while keeping margin & multiple");
assert.equal(submitted.length, submittedBeforeNewVideo + 1, "submits partial preview immediately");
assert.equal(submitted.at(-1).output["8"], undefined, "partial preview excludes downstream nodes");
const submittedMaskPayload = JSON.parse(submitted.at(-1).output["3"].inputs.mask_data);
assert.equal(submittedMaskPayload.source, undefined, "queued preview has no stale source");
assert.deepEqual(submittedMaskPayload.frames, {});

// Receiving frames for the newly loaded video succeeds without sourceMismatch or stuck alert banner.
const nextClipResponse = await newVideoEditor.applyExecuted({
    universal_mask_meta: [{ source: { frames: 5, width: 320, height: 240, fingerprint: "next-clip" }, has_mask: false, anchors: [] }],
    universal_mask_frames: [{ frame: 0, source: { data_url: "data:image/png;base64,next" } }],
});
assert.equal(Boolean(nextClipResponse.sourceMismatch), false, "new clip loads cleanly without mismatch warning");
const newVideoBanner = find(newVideoEditor.root, (item) => String(item.className || "").includes("ths-mask-alert-overlay"));
assert.equal(newVideoBanner.style.display, "none", "alert overlay stays hidden on clean new video");
newVideoEditor.destroy();

console.log("universal_mask_editor_source_refine: PASS");
