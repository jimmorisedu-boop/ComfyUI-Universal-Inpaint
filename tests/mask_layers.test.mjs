import assert from "node:assert/strict";

import {
    blendLabel,
    blendSymbol,
    ensureDocumentLayers,
    layerHasContent,
    nextBlendMode,
    nextLayerId,
    nextLayerName,
    projectLayerToDocument,
    shouldSerializeLayers,
    syncActiveLayerToList,
} from "../web/mask_layers.js";

// 1. Pure unit tests for mask_layers.js helpers.
assert.equal(nextLayerId([]), "layer-1");
assert.equal(nextLayerId([{ id: "layer-1" }, { id: "layer-2" }]), "layer-3");
assert.equal(nextLayerName([], "sam3"), "SAM3");
assert.equal(nextLayerName([{ name: "SAM3" }], "sam3"), "SAM3 #2");
assert.equal(nextBlendMode("add"), "subtract");
assert.equal(nextBlendMode("subtract"), "intersect");
assert.equal(nextBlendMode("intersect"), "add");
assert.equal(blendSymbol("add"), "+");
assert.equal(blendSymbol("subtract"), "−");
assert.equal(blendSymbol("intersect"), "∩");
assert.match(blendLabel("subtract"), /Вычесть/);

assert.equal(layerHasContent({ type: "static", frames: {} }), false);
assert.equal(layerHasContent({ type: "static", frames: { "0": { events: [{ type: "rect" }] } } }), true);
assert.equal(layerHasContent({ type: "sam3", frames: {}, sam3: { prompt: { frame: 0, points: [[0.5, 0.5]], labels: [1] } } }), true);

assert.equal(
    shouldSerializeLayers([{ id: "layer-1", type: "sam3", blend: "add", enabled: true, start_frame: 0, end_frame: -1 }], { startFrame: 0, endFrame: -1 }),
    false,
    "single default layer preserves legacy v1 JSON",
);
assert.equal(
    shouldSerializeLayers([
        { id: "layer-1", type: "sam3", blend: "add", enabled: true, start_frame: 0, end_frame: -1 },
        { id: "layer-2", type: "static", blend: "subtract", enabled: true, start_frame: 0, end_frame: -1 },
    ], { startFrame: 0, endFrame: -1 }),
    true,
);

const migratedDoc = {
    version: 1,
    tracking_mode: "optical_flow",
    frames: { "0": { events: [{ type: "rect", mode: "add", x: 0.1, y: 0.1, w: 0.2, h: 0.2 }] } },
    anchors: [0],
};
ensureDocumentLayers(migratedDoc, { mode: "optical_flow", startFrame: 2, endFrame: 10 });
assert.equal(migratedDoc.layers.length, 1);
assert.equal(migratedDoc.layers[0].type, "optical_flow");
assert.equal(migratedDoc.layers[0].start_frame, 2);
syncActiveLayerToList(migratedDoc, { mode: "keyframes", startFrame: 3, endFrame: 9 });
assert.equal(migratedDoc.layers[0].type, "keyframes");
assert.equal(migratedDoc.layers[0].end_frame, 9);

const targetSamLayer = {
    id: "layer-sam",
    name: "SAM3",
    type: "sam3",
    blend: "add",
    enabled: true,
    start_frame: 0,
    end_frame: -1,
    frames: {},
    sam3: { action: "track", frame: 0, direction: "both", seed_token: "tok-42", prompt: { frame: 0, points: [[0.5, 0.5]], labels: [1] } },
};
projectLayerToDocument(migratedDoc, targetSamLayer);
assert.equal(migratedDoc.tracking_mode, "sam3");
assert.equal(migratedDoc.sam3.seed_token, "tok-42");

// 2. Full DOM & lifecycle integration tests with createUniversalMaskEditor.
class FakeElement {
    constructor(tagName) {
        this.tagName = String(tagName || "DIV").toUpperCase();
        this.children = [];
        this.attributes = {};
        this.dataset = {};
        this.style = {};
        this.listeners = new Map();
        this.parentNode = null;
        this._text = "";
        this.value = "";
        this.disabled = false;
        this.className = "";
        this.classList = {
            add: (...names) => {
                const set = new Set(String(this.className || "").split(/\s+/).filter(Boolean));
                for (const name of names) set.add(name);
                this.className = [...set].join(" ");
            },
            remove: (...names) => {
                const drop = new Set(names);
                this.className = String(this.className || "").split(/\s+/).filter((item) => item && !drop.has(item)).join(" ");
            },
            contains: (name) => String(this.className || "").split(/\s+/).includes(name),
        };
    }
    set textContent(value) { this._text = String(value ?? ""); }
    get textContent() {
        return this._text + this.children.map((child) => child.textContent).join("");
    }
    append(...nodes) {
        for (const node of nodes) {
            if (!node) continue;
            if (typeof node === "string") this._text += node;
            else { node.parentNode = this; this.children.push(node); }
        }
    }
    after(node) { this.parentNode?.append(node); }
    replaceChildren(...nodes) { this.children = []; this._text = ""; this.append(...nodes); }
    setAttribute(name, value) {
        this.attributes[name] = String(value);
        if (name.startsWith("data-")) {
            const camel = name.slice(5).replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());
            this.dataset[camel] = String(value);
        }
    }
    getAttribute(name) { return this.attributes[name] ?? null; }
    removeAttribute(name) { delete this.attributes[name]; }
    addEventListener(type, fn) {
        const list = this.listeners.get(type) || [];
        list.push(fn);
        this.listeners.set(type, list);
    }
    removeEventListener(type, fn) {
        this.listeners.set(type, (this.listeners.get(type) || []).filter((item) => item !== fn));
    }
    dispatchEvent(event) {
        const list = [...(this.listeners.get(event.type) || [])];
        for (const fn of list) fn({ target: this, currentTarget: this, ...event });
    }
    click() { this.dispatchEvent({ type: "click" }); }
    remove() {}
    focus() {}
    getBoundingClientRect() { return { left: 0, top: 0, width: 400, height: 300 }; }
    setPointerCapture() {}
    releasePointerCapture() {}
}

class FakeCanvas extends FakeElement {
    constructor() {
        super("canvas");
        this.width = 400;
        this.height = 300;
    }
    getContext() {
        return {
            setTransform() {}, clearRect() {}, fillRect() {}, drawImage() {},
            save() {}, restore() {}, beginPath() {}, rect() {}, arc() {},
            moveTo() {}, lineTo() {}, closePath() {}, fill() {}, stroke() {},
            clip() {}, strokeRect() {}, setLineDash() {}, fillText() {},
            getImageData: (_x, _y, w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
            putImageData() {},
        };
    }
    getBoundingClientRect() { return { left: 0, top: 0, width: 400, height: 300 }; }
}

const submitted = [];
const apiListeners = new Map();
const api = {
    queuePrompt: async (_n, prompt) => { submitted.push(prompt); return { prompt_id: `p-${submitted.length}` }; },
    addEventListener(type, handler) { const list = apiListeners.get(type) || []; list.push(handler); apiListeners.set(type, list); },
    removeEventListener(type, handler) { apiListeners.set(type, (apiListeners.get(type) || []).filter((h) => h !== handler)); },
};
const app = {
    graph: {},
    queuePrompt: async () => ({ prompt_id: "queue-1" }),
    graphToPrompt: async () => ({
        output: {
            "1": { class_type: "LoadVideo", inputs: {} },
            "2": { class_type: "THSUniversalMaskEditor", inputs: { images: ["1", 0] } },
        },
    }),
};

globalThis.document = {
    activeElement: null,
    createElement(tag) { return String(tag).toLowerCase() === "canvas" ? new FakeCanvas() : new FakeElement(tag); },
};
globalThis.window = { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {} };

const { createUniversalMaskEditor } = await import("../web/universal_mask_editor.js");

function find(root, predicate) {
    if (!root) return null;
    if (predicate(root)) return root;
    for (const child of root.children || []) { const hit = find(child, predicate); if (hit) return hit; }
    return null;
}
function findAll(root, predicate, acc = []) {
    if (!root) return acc;
    if (predicate(root)) acc.push(root);
    for (const child of root.children || []) findAll(child, predicate, acc);
    return acc;
}
function findButton(root, text) {
    return find(root, (el) => el.tagName === "BUTTON" && (el._text === text || el.textContent === text || el.attributes["aria-label"] === text));
}

const widgets = [
    { name: "mask_data", value: "{}", options: {} },
    { name: "tracking_mode", value: "sam3" },
    { name: "grow", value: 8 },
    { name: "feather", value: 4 },
    { name: "start_frame", value: 0 },
    { name: "end_frame", value: -1 },
];
const node = { id: 2, widgets, properties: {}, graph: { setDirtyCanvas() {} }, addDOMWidget(_n, _t, root) { this.root = root; } };
const editor = createUniversalMaskEditor(node, { app, api });

// Single default SAM3 layer must NOT serialize "layers" in v1 mask_data.
editor.commit();
assert.equal(JSON.parse(widgets[0].value).layers, undefined);

// Simulate SAM3 frame calculation & tracking on layer-1.
const clipSource = { frames: 4, width: 320, height: 240, fingerprint: "clip-1" };
editor.state.source = clipSource;
editor.state.frameCount = 4;
editor.document.source = clipSource;
const frameItem0 = { sourceImage: { naturalWidth: 320, naturalHeight: 240 }, edit_base: { data_url: "base" } };
editor.state.frameItems.set(0, frameItem0);
editor.state.previews.set(0, frameItem0);
editor.setFrame(0, { load: false });
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 1, clientX: 100, clientY: 100, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 1, clientX: 100, clientY: 100 });
await new Promise((r) => setTimeout(r, 0));
assert.ok(editor.document.sam3.prompt?.points?.length === 1);

await editor.applyExecuted({
    prompt_id: editor.state.pendingPreview?.promptId,
    universal_mask_meta: [{
        source: clipSource,
        has_mask: true,
        sam3: { frame_ready: true, track_ready: false, frame: 0, direction: "both", seed_token: "tok-s1", computed_frames: [0] },
    }],
    universal_mask_frame_patch: [{ frame: 0, mask: "data:image/png;base64,m", edit_base: "data:image/png;base64,b" }],
});

// Smart switch via tracking dropdown ("Режим") preserves SAM3 points/tokens in Layer 1 and adds Layer 2 (Optical Flow)!
const modeSelect = find(editor.root, (el) => el.tagName === "SELECT" && el.attributes["aria-label"] === "Трекинг");
modeSelect.value = "optical_flow";
modeSelect.dispatchEvent({ type: "change" });
assert.equal(editor.document.layers.length, 2, "switching mode on populated SAM3 creates a new layer instead of wiping SAM3");
assert.equal(editor.document.layers[0].type, "sam3");
assert.equal(editor.document.layers[0].sam3.prompt.points.length, 1, "SAM3 points preserved inside Layer 1");
assert.equal(editor.document.layers[0].sam3.seed_token, "tok-s1");
assert.equal(editor.document.layers[1].type, "optical_flow");
assert.equal(editor.document.active_layer_id, editor.document.layers[1].id);
assert.equal(editor.state.layerSam3.get(editor.document.layers[0].id).seedToken, "tok-s1");

// Draw brush stroke in Layer 2 (Optical Flow).
findButton(editor.root, "Кисть").click();
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 2, clientX: 200, clientY: 150, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 2, clientX: 210, clientY: 160 });
assert.equal(editor.document.frames["0"].events.length, 1);

// Add Layer 3 (+ Статика), cycle blend -> subtract, toggle visibility, reorder up, switch back to Layer 1.
findButton(editor.root, "+ Статика").click();
assert.equal(editor.document.layers.length, 3);
assert.equal(editor.document.layers[2].type, "static");

const rowItems = findAll(editor.root, (el) => String(el.className || "").includes("ths-layer-item"));
assert.equal(rowItems.length, 3);
const thirdBlend = findButton(rowItems[2], "Режим смешивания слоя");
thirdBlend.click();
assert.equal(editor.document.layers[2].blend, "subtract");
const thirdVis = findButton(rowItems[2], "Видимость слоя");
thirdVis.click();
assert.equal(editor.document.layers[2].enabled, false);
thirdVis.click();
assert.equal(editor.document.layers[2].enabled, true);

// Reorder layer 3 up -> moves to index 1.
const thirdUp = findButton(findAll(editor.root, (el) => String(el.className || "").includes("ths-layer-item"))[2], "Слой выше");
thirdUp.click();
assert.equal(editor.document.layers[1].type, "static");

// Select Layer 1 (SAM3) again -> restores SAM3 points and saved readiness!
const rowsAfterReorder = findAll(editor.root, (el) => String(el.className || "").includes("ths-layer-item"));
find(rowsAfterReorder[0], (el) => el.tagName === "BUTTON" && String(el.className || "").includes("ths-layer-main")).click();
assert.equal(editor.document.active_layer_id, editor.document.layers[0].id);
assert.equal(editor.document.sam3.prompt.points.length, 1);
assert.equal(editor.state.sam3.seedToken, "tok-s1");

// Deleting Static layer shrinks layers to 2; Undo brings it back.
const rowsForDelete = findAll(editor.root, (el) => String(el.className || "").includes("ths-layer-item"));
findButton(rowsForDelete[1], "Удалить слой").click();
assert.equal(editor.document.layers.length, 2);
findButton(editor.root, "Отменить").click();
assert.equal(editor.document.layers.length, 3, "Undo restores deleted layer");

// Multi-layer serialization preserves all 3 layers and reloads cleanly in a fresh editor instance.
const serialized = JSON.parse(widgets[0].value);
assert.equal(serialized.layers.length, 3);
assert.equal(serialized.active_layer_id, editor.document.active_layer_id);

const reloadWidgets = [
    { name: "mask_data", value: widgets[0].value, options: {} },
    { name: "tracking_mode", value: "sam3" },
    { name: "grow", value: 8 },
    { name: "feather", value: 4 },
    { name: "start_frame", value: 0 },
    { name: "end_frame", value: -1 },
];
const reloadNode = { id: 9, widgets: reloadWidgets, properties: {}, graph: { setDirtyCanvas() {} }, addDOMWidget(_n, _t, root) { this.root = root; } };
const reloadEditor = createUniversalMaskEditor(reloadNode, { app, api });
assert.equal(reloadEditor.document.layers.length, 3, "reloaded node restores all layers");

// Empty SAM3 layer must NOT block queue button when another Static layer has content!
const mixedQueueDoc = {
    layers: [
        { id: "l-empty-sam3", name: "SAM3", type: "sam3", blend: "add", enabled: true, start_frame: 0, end_frame: -1, frames: {} },
        { id: "l-static-ok", name: "Статика", type: "static", blend: "add", enabled: true, start_frame: 0, end_frame: -1, frames: { "0": { events: [{ type: "rect", mode: "add", x: 0.2, y: 0.2, w: 0.3, h: 0.3 }] } } },
    ],
    active_layer_id: "l-empty-sam3",
};
reloadWidgets[0].value = JSON.stringify(mixedQueueDoc);
reloadWidgets[1].value = "sam3";
reloadEditor.syncFromConfiguredWidget(true);
const queueBtn = findButton(reloadEditor.root, "Запустить граф");
assert.equal(queueBtn.disabled, false, "empty SAM3 layer does not block Queue when Static layer is populated");

reloadEditor.destroy();
editor.destroy();
console.log("mask_layers: PASS");
