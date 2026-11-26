import assert from "node:assert/strict";

class FakeElement {
    constructor(tagName) {
        this.tagName = String(tagName).toUpperCase();
        this.children = [];
        this.style = {};
        this.listeners = new Map();
        this.attributes = {};
        this.textContent = "";
        this.hidden = false;
        this.dataset = {};
        this.className = "";
        this.disabled = false;
        this.value = "";
    }
    append(...children) { this.children.push(...children.filter(Boolean)); }
    appendChild(child) { this.append(child); return child; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    addEventListener(type, handler) { const list = this.listeners.get(type) || []; list.push(handler); this.listeners.set(type, list); }
    dispatchEvent(event) { event.target ||= this; event.currentTarget = this; for (const handler of this.listeners.get(event.type) || []) handler(event); return !event.defaultPrevented; }
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

const { cropWindowBox } = await import("../web/crop_window.js");
const { createUniversalMaskEditor } = await import("../web/universal_mask_editor.js");

function find(root, predicate) {
    if (!root) return null;
    if (predicate(root)) return root;
    for (const child of root.children || []) {
        const hit = find(child, predicate);
        if (hit) return hit;
    }
    return null;
}

// 1. Exact box math parity with Python plan_crop_window
assert.deepEqual(cropWindowBox(960, 720, { box: [200, 150, 360, 520], frames: [0, 1] }, { mode: "full", margin: 64 }), [0, 0, 960, 720]);
assert.deepEqual(cropWindowBox(960, 720, { box: [200, 150, 360, 520], frames: [0, 1] }, { mode: "auto", margin: 64 }), [136, 86, 424, 584]);
assert.deepEqual(cropWindowBox(960, 720, { box: [200, 150, 360, 520], frames: [0, 1] }, { mode: "manual", margin: 64, box: [100, 80, 500, 620], size: [960, 720] }), [100, 80, 500, 620]);

// 1b. Side-multiple alignment across auto, manual, 90% fallback, and multiple > limit
for (const mult of [16, 24, 32, 64]) {
    const autoBox = cropWindowBox(960, 720, { box: [200, 150, 360, 520], frames: [0, 1] }, { mode: "auto", margin: 64, multiple: mult });
    assert.equal((autoBox[2] - autoBox[0]) % mult, 0, `auto width divisible by ${mult}`);
    assert.equal((autoBox[3] - autoBox[1]) % mult, 0, `auto height divisible by ${mult}`);
    assert.ok(autoBox[0] <= 200 && autoBox[2] >= 360 && autoBox[1] <= 150 && autoBox[3] >= 520);

    const manualBox = cropWindowBox(960, 720, { box: [200, 150, 360, 520], frames: [0, 1] }, { mode: "manual", margin: 64, multiple: mult, box: [150, 120, 410, 560], size: [960, 720] });
    assert.equal((manualBox[2] - manualBox[0]) % mult, 0, `manual width divisible by ${mult}`);
    assert.equal((manualBox[3] - manualBox[1]) % mult, 0, `manual height divisible by ${mult}`);
    assert.ok(manualBox[0] <= 150 && manualBox[2] >= 410 && manualBox[1] <= 120 && manualBox[3] >= 560);
}
// Non-divisible 1920x1080 with multiple=64 & >=90% fallback must snap height to 1024 (64*16), never 1080
const fullFallback64 = cropWindowBox(1920, 1080, { box: [20, 20, 1900, 1060], frames: [0] }, { mode: "auto", margin: 64, multiple: 64 });
assert.equal((fullFallback64[2] - fullFallback64[0]) % 64, 0);
assert.equal((fullFallback64[3] - fullFallback64[1]) % 64, 0);
assert.equal(fullFallback64[3] - fullFallback64[1], 1024);
// multiple > limit safely returns full axis
assert.deepEqual(cropWindowBox(300, 200, { box: [50, 40, 200, 160], frames: [0] }, { mode: "auto", margin: 20, multiple: 512 }), [0, 0, 300, 200]);

// 2. Editor integration: hover-only tooltips, zero layout shift on mode changes, preserved SAM3 track readiness
const widgets = [
    { name: "mask_data", value: "{}", options: {} },
    { name: "tracking_mode", value: "sam3" },
    { name: "grow", value: 8, options: {} },
    { name: "feather", value: 4, options: {} },
    { name: "start_frame", value: 0 },
    { name: "end_frame", value: -1 },
    { name: "fps", value: 24 },
];
const node = { id: 9, widgets, graph: { setDirtyCanvas() {} }, addDOMWidget(_n, _t, root) { this.root = root; } };
const editor = createUniversalMaskEditor(node, { app: null, api: null });
editor.state.source = { frames: 10, width: 960, height: 720, fingerprint: "crop-clip" };
editor.state.frameCount = 10;
editor.state.metadata = { crop_extent: { box: [180, 140, 340, 540], frames: [0, 1, 2] }, sam3: { frame_ready: true, track_ready: true } };
editor.state.sam3.frameReady = true;
editor.state.sam3.trackReady = true;
editor.state.previewStale = false;

const cropMode = find(editor.root, (el) => el.tagName === "SELECT" && el.attributes["aria-label"] === "Режим автокропа");
const cropMargin = find(editor.root, (el) => el.tagName === "INPUT" && el.attributes["aria-label"] === "Запас автокропа");
const cropMultiple = find(editor.root, (el) => el.tagName === "INPUT" && el.attributes["aria-label"] === "Кратность сторон кропа");
const presetChips = [1, 8, 16, 32, 64].map((n) => find(editor.root, (el) => el.tagName === "BUTTON" && el.attributes["aria-label"] === `Кратность ${n}`));
const cropInfo = find(editor.root, (el) => String(el.className || "").includes("ths-crop-readout"));
const nwKnob = find(editor.root, (el) => el.tagName === "BUTTON" && String(el.className || "").includes("ths-crop-nw"));
const seKnob = find(editor.root, (el) => el.tagName === "BUTTON" && String(el.className || "").includes("ths-crop-se"));
assert.ok(cropMode && cropMargin && cropMultiple && cropInfo && nwKnob && seKnob);
assert.ok(presetChips.every(Boolean), "all 5 multiple preset chips exist");
assert.equal(cropMargin.disabled, true, "margin disabled when mode is full without hiding");
assert.equal(cropMultiple.disabled, true, "multiple disabled when mode is full without hiding");
assert.ok(presetChips.every((chip) => chip.disabled === true), "preset chips disabled in full mode");
assert.notEqual(cropMargin.style.display, "none", "margin stays mounted for zero layout jump");
assert.notEqual(cropMultiple.style.display, "none", "multiple stays mounted for zero layout jump");
assert.ok(cropMode.dataset.tooltip && cropMode.title, "tooltip metadata populated for Top-Layer hover controller");

cropMode.value = "auto";
cropMode.dispatchEvent({ type: "change" });
assert.equal(cropMargin.disabled, false);
assert.equal(cropMultiple.disabled, false);
assert.ok(presetChips.every((chip) => chip.disabled === false), "preset chips enabled in auto mode");
assert.equal(presetChips[0].attributes["aria-pressed"], "true");
assert.equal(editor.state.sam3.trackReady, true, "toggling autocrop must preserve SAM3 readiness");
assert.equal(editor.state.previewStale, false, "toggling autocrop must not stale computed masks");
assert.match(cropInfo.textContent, /Кроп 288×528 px/);
assert.equal(JSON.parse(widgets[0].value).crop.mode, "auto");

// 1-click preset chip (Кратность 16)
presetChips[2].dispatchEvent({ type: "click" });
assert.equal(cropMultiple.value, "16");
assert.equal(presetChips[2].attributes["aria-pressed"], "true");
assert.equal(presetChips[0].attributes["aria-pressed"], "false");
assert.equal(JSON.parse(widgets[0].value).crop.multiple, 16);
assert.match(cropInfo.textContent, /×16/);
assert.equal(editor.state.sam3.trackReady, true);
assert.equal(editor.state.previewStale, false);

// Reset to x1 via chip before checking raw margin size
presetChips[0].dispatchEvent({ type: "click" });
assert.equal(JSON.parse(widgets[0].value).crop.multiple, 1);

cropMargin.value = "96";
cropMargin.dispatchEvent({ type: "change" });
assert.match(cropInfo.textContent, /Кроп 352×592 px/);
assert.equal(JSON.parse(widgets[0].value).crop.margin, 96);

// Live typing + commit for side multiple (e.g., 64 px)
cropMultiple.value = "64";
cropMultiple.dispatchEvent({ type: "input" });
assert.match(cropInfo.textContent, /×64/, "live input immediately updates crop size badge");
assert.equal(presetChips[4].attributes["aria-pressed"], "true", "64 chip highlights during live input");
cropMultiple.dispatchEvent({ type: "change" });
assert.equal(JSON.parse(widgets[0].value).crop.multiple, 64);
assert.equal(editor.state.sam3.trackReady, true);
assert.equal(editor.state.previewStale, false);
const savedCrop64 = cropWindowBox(960, 720, editor.state.metadata.crop_extent, JSON.parse(widgets[0].value).crop);
assert.equal((savedCrop64[2] - savedCrop64[0]) % 64, 0);
assert.equal((savedCrop64[3] - savedCrop64[1]) % 64, 0);

// Manual mode keeps multiple & preset chips enabled & steps cleanly by multiple via arrow keys
cropMode.value = "manual";
cropMode.dispatchEvent({ type: "change" });
assert.equal(cropMargin.disabled, true);
assert.equal(cropMultiple.disabled, false);
assert.ok(presetChips.every((chip) => chip.disabled === false), "preset chips enabled in manual mode");
assert.notEqual(cropMargin.style.display, "none");
assert.equal(JSON.parse(widgets[0].value).crop.mode, "manual");
cropMultiple.value = "32";
cropMultiple.dispatchEvent({ type: "change" });
const boxBeforeArrow = JSON.parse(widgets[0].value).crop.box;
seKnob.dispatchEvent({ type: "keydown", key: "ArrowRight", stopPropagation() {}, preventDefault() {} });
const boxAfterArrow = JSON.parse(widgets[0].value).crop.box;
assert.equal((boxAfterArrow[2] - boxAfterArrow[0]) % 32, 0);
assert.ok((boxAfterArrow[2] - boxAfterArrow[0]) >= (boxBeforeArrow[2] - boxBeforeArrow[0]));

// Changing crop must not drop the clean computed mask baseline on seed frames!
const dirtyMask = editor.canvas ? editor.document : null;
assert.ok(dirtyMask);
editor.document.frames["0"] = { inherit: false, events: [{ type: "rect", mode: "add", refine: true, refine_method: "sam3", x: 0.2, y: 0.2, w: 0.2, h: 0.2 }] };
const computedMask = { naturalWidth: 960, naturalHeight: 720, label: "computed-mask" };
const sourceImage = { naturalWidth: 960, naturalHeight: 720, label: "source" };
editor.state.frameItems.set(0, { sourceImage, maskImage: computedMask });
editor.state.previews.set(0, { sourceImage, maskImage: computedMask });
editor.state.previewStale = false;
const { crop: _dropCrop, ...maskOnly } = editor.document;
editor.state.previewSignature = JSON.stringify(maskOnly);
cropMode.value = "auto";
cropMode.dispatchEvent({ type: "change" });
assert.equal(editor.state.previewStale, false);
assert.equal(editor.state.dirty, false);
console.log("crop_window: PASS");
