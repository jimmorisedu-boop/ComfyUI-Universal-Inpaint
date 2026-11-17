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
        this.isContentEditable = false;
    }

    append(...children) { this.children.push(...children.filter((child) => child !== undefined && child !== null)); }
    appendChild(child) { this.append(child); return child; }
    after(...children) { this.parentNode?.append(...children); }
    remove() { this.removed = true; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    addEventListener(type, handler) { const handlers = this.listeners.get(type) || []; handlers.push(handler); this.listeners.set(type, handlers); }
    click() { this.dispatchEvent({ type: "click" }); }
    dispatchEvent(event) { event.target ||= this; event.currentTarget = this; for (const handler of this.listeners.get(event.type) || []) handler(event); return !event.defaultPrevented; }
    getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 360 }; }
    focus() { this.focused = true; }
    setPointerCapture() {}
    releasePointerCapture() {}
    getContext() { const noop = () => {}; return new Proxy({}, { get: () => noop, set: () => true }); }
    showModal() { this.opened = true; }
    close() { this.opened = false; }
}

class DeferredImage {
    static pending = [];
    constructor() { DeferredImage.pending.push(this); }
    set src(value) { this.url = value; }
    resolve(width = 640, height = 360) { this.naturalWidth = width; this.naturalHeight = height; this.onload?.(); }
    reject() { this.onerror?.(); }
}

globalThis.Image = DeferredImage;
globalThis.document = { createElement(tagName) { return new FakeElement(tagName); } };
globalThis.window = { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {} };

const { createUniversalMaskEditor, imageUrl } = await import("../web/universal_mask_editor.js");

const widgets = [
    { name: "mask_data", value: "{}", options: {} },
    { name: "tracking_mode", value: "optical_flow" },
    { name: "start_frame", value: 0 },
    { name: "end_frame", value: -1 },
    { name: "fps", value: 24 },
];
const node = {
    id: 31,
    widgets,
    graph: { setDirtyCanvas() {} },
    addDOMWidget(_name, _type, root) { this.root = root; },
};
const editor = createUniversalMaskEditor(node, { app: null, api: { apiURL: (path) => `http://comfy.test${path}` } });
function findNode(root, predicate) {
    if (!root) return null;
    if (predicate(root)) return root;
    for (const child of root.children || []) { const found = findNode(child, predicate); if (found) return found; }
    return null;
}

// Exact frame descriptors are retained; no neighboring-frame snapping.
const output = {
    universal_mask_meta: [{ source: { frames: 120, width: 640, height: 360, fingerprint: "clip-a" }, fps: 30, has_mask: true, tracking_available: true }],
    universal_mask_frames: [0, 1, 2, 5, 119].map((frame) => ({ frame, filename: `frame-${frame}.png`, subfolder: "temp", type: "temp", mask: { filename: `mask-${frame}.png`, subfolder: "temp", type: "temp" } })),
};
await editor.applyExecuted(output);
assert.equal(editor.state.frameCount, 120);
editor.setFrame(119, { load: false });
assert.equal(editor.state.frame, 119);
editor.root.dispatchEvent({ type: "keydown", key: "ArrowLeft", preventDefault() {} });
assert.equal(editor.state.frame, 118);
editor.root.dispatchEvent({ type: "keydown", key: "ArrowRight", shiftKey: true, preventDefault() {} });
assert.equal(editor.state.frame, 119);
const frameInput = findNode(editor.root, (element) => element.tagName === "INPUT" && element.attributes["aria-label"] === "Номер кадра");
frameInput.value = "73";
let directStopped = false;
frameInput.dispatchEvent({ type: "keydown", key: "Enter", preventDefault() {}, stopPropagation() { directStopped = true; } });
assert.equal(editor.state.frame, 72);
assert.equal(directStopped, true);

// URL descriptors use Comfy's /view contract and URLSearchParams escaping.
assert.equal(imageUrl({ filename: "a frame.png", subfolder: "tmp/x", type: "temp" }, { apiURL: (path) => `https://x${path}` }), "https://x/view?filename=a+frame.png&subfolder=tmp%2Fx&type=temp");

// A stale source response cannot replace the selected source after a new run.
const firstLoad = DeferredImage.pending.length;
await editor.applyExecuted({ universal_mask_meta: [{ source: { frames: 120, width: 640, height: 360, fingerprint: "clip-b" }, fps: 24 }], universal_mask_frames: [{ frame: 0, filename: "new.png", type: "temp" }] });
for (const image of DeferredImage.pending.slice(firstLoad)) image.resolve();
assert.equal(editor.state.source.fingerprint, "clip-b");

// The first independent stroke must not inherit merely because the clip has
// tracking metadata.  A later correction inherits only when an independent
// anchor and a raw edit_base descriptor are both available.
editor.state.previews.set(0, { sourceImage: { naturalWidth: 640, naturalHeight: 360 }, maskImage: null });
editor.setFrame(0, { load: false });
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 2, clientX: 150, clientY: 130, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 2, clientX: 150, clientY: 130 });
assert.equal(editor.document.frames["0"].inherit, false);
assert.equal(editor.document.frames["0"].events.length, 1);
editor.state.previews.set(5, {
    sourceImage: { naturalWidth: 640, naturalHeight: 360 },
    maskImage: null,
    editBaseImage: { naturalWidth: 640, naturalHeight: 360 },
    edit_base: { data_url: "data:image/png;base64,raw-baseline" },
});
editor.setFrame(5, { load: false });
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 3, clientX: 180, clientY: 130, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 3, clientX: 180, clientY: 130 });
assert.equal(editor.document.frames["5"].inherit, true);
editor.root.dispatchEvent({ type: "keydown", key: "z", ctrlKey: true, preventDefault() {} });
assert.equal(editor.document.frames["5"], undefined);
editor.root.dispatchEvent({ type: "keydown", key: "y", ctrlKey: true, preventDefault() {} });
assert.equal(editor.document.frames["5"].inherit, true);

// Clearing a frame is an explicit independent empty anchor and must survive
// serialization without inventing a backend-unsupported `cleared` field.
editor.setFrame(5, { load: false });
function findButton(root, label) {
    if (!root) return null;
    if (root.tagName === "BUTTON" && root.textContent === label) return root;
    for (const child of root.children || []) { const found = findButton(child, label); if (found) return found; }
    return null;
}
findButton(editor.root, "Очистить кадр")?.click();
assert.deepEqual(editor.document.frames["5"], { events: [], inherit: false });
editor.commit();
const serializedAfterClear = JSON.parse(widgets.find((widget) => widget.name === "mask_data").value);
assert.deepEqual(serializedAfterClear.frames["5"], { events: [], inherit: false });
assert.equal(serializedAfterClear.frames["5"].cleared, undefined);

// Fullscreen wheel zoom stays anchored at the cursor; navigation never edits a mask.
const maskBeforeNavigation = JSON.stringify(editor.document);
const widgetBeforeNavigation = widgets[0].value;
const frameBeforeNavigation = editor.state.frame;
editor.state.zoom = 1; editor.state.panX = 0; editor.state.panY = 0;
editor.state.dialogOpen = true;
let prevented = false, stopped = false;
const wheel = { type: "wheel", deltaY: -100, clientX: 160, clientY: 90,
    preventDefault() { prevented = true; }, stopPropagation() { stopped = true; } };
editor.canvas.dispatchEvent(wheel);
assert.equal(editor.state.zoom, 1.12);
assert.ok(prevented && stopped);
assert.ok(Math.abs((640 - 640 * editor.state.zoom) / 2 + editor.state.panX + .25 * 640 * editor.state.zoom - 160) < 1e-8);
assert.ok(Math.abs((360 - 360 * editor.state.zoom) / 2 + editor.state.panY + .25 * 360 * editor.state.zoom - 90) < 1e-8);
for (const modifiers of [{ button: 1 }, { button: 0, altKey: true }, { button: 0, shiftKey: true }]) {
    const before = { x: editor.state.panX, y: editor.state.panY };
    editor.canvas.dispatchEvent({ type: "pointerdown", pointerId: 99, clientX: 160, clientY: 90, ...modifiers, preventDefault() {}, stopPropagation() {} });
    assert.equal(editor.state.active, null);
    assert.equal(editor.canvas.style.cursor, "grabbing");
    editor.canvas.dispatchEvent({ type: "pointermove", pointerId: 99, clientX: 200, clientY: 120 });
    assert.ok(Math.abs(editor.state.panX - before.x - 40) < 1e-8);
    assert.ok(Math.abs(editor.state.panY - before.y - 30) < 1e-8);
    editor.canvas.dispatchEvent({ type: "pointerup", pointerId: 99, ...modifiers });
    assert.equal(editor.state.panning, null);
    assert.equal(editor.canvas.style.cursor, "crosshair");
}
editor.canvas.dispatchEvent({ type: "pointerdown", button: 1, pointerId: 100, clientX: 160, clientY: 90 });
editor.canvas.dispatchEvent({ type: "pointercancel", pointerId: 100 });
assert.equal(editor.state.panning, null);
assert.equal(editor.canvas.style.cursor, "crosshair");
editor.state.dialogOpen = false;
const zoomBeforeInlineWheel = editor.state.zoom;
prevented = false; stopped = false;
editor.canvas.dispatchEvent(wheel);
assert.equal(editor.state.zoom, zoomBeforeInlineWheel);
assert.ok(!prevented && !stopped);
editor.canvas.dispatchEvent({ ...wheel, ctrlKey: true });
assert.ok(editor.state.zoom > zoomBeforeInlineWheel);
findNode(editor.root, (element) => element.attributes["aria-label"] === "Вписать")?.click();
assert.equal(editor.state.zoom, 1);
assert.equal(editor.state.panX, 0);
assert.equal(editor.state.panY, 0);
assert.equal(editor.state.frame, frameBeforeNavigation);
assert.equal(JSON.stringify(editor.document), maskBeforeNavigation);
assert.equal(widgets[0].value, widgetBeforeNavigation);

// Fullscreen uses unscaled CSS; Comfy's inline widget inherits graph scale.
// Exercise a complete round-trip with zoomed-in/out graphs and HiDPI.
const cursor = findNode(editor.root, (e) => e.style?.transform === "translate(-50%, -50%)" && e.style?.borderRadius === "50%");
const expandButton = findNode(editor.root, (e) => e.attributes?.["aria-label"] === "Развернуть");
const closeButton = findNode(editor.root, (e) => e.attributes?.["aria-label"] === "Закрыть");
const fitButton = findNode(editor.root, (e) => e.attributes?.["aria-label"] === "Вписать");
let graphScale = .5;
Object.defineProperties(editor.canvas, {
    clientWidth: { get: () => editor.state.dialogOpen ? 960 : 640 },
    clientHeight: { get: () => editor.state.dialogOpen ? 540 : 360 },
});
editor.canvas.getBoundingClientRect = () => ({ left: 100, top: 50,
    width: editor.canvas.clientWidth * (editor.state.dialogOpen ? 1 : graphScale),
    height: editor.canvas.clientHeight * (editor.state.dialogOpen ? 1 : graphScale) });
for (const dpr of [1, 2]) for (const scale of [.5, 1.5]) {
    window.devicePixelRatio = dpr; graphScale = scale;
    expandButton.click();
    editor.canvas.dispatchEvent({ type: "pointermove", pointerId: 221, clientX: 340, clientY: 185 });
    assert.equal(parseFloat(cursor.style.left), 240);
    closeButton.click();
    assert.equal(cursor.style.display, "none", "hide a pointer from the former layout");
    fitButton.click();
    const event = { pointerId: 222, button: 0, clientX: 100 + 160 * scale, clientY: 50 + 90 * scale, preventDefault() {}, stopPropagation() {} };
    editor.canvas.dispatchEvent({ ...event, type: "pointermove" });
    assert.equal(parseFloat(cursor.style.left), 160, `local cursor x at graph scale ${scale}`);
    assert.equal(parseFloat(cursor.style.top), 90, `local cursor y at graph scale ${scale}`);
    assert.ok(Math.abs(parseFloat(cursor.style.width) - .035 * 360 * 2) < 1e-6);
    editor.canvas.dispatchEvent({ ...event, type: "pointerdown" });
    editor.canvas.dispatchEvent({ ...event, type: "pointerup" });
    assert.deepEqual(editor.document.frames["5"].events.at(-1).points[0], [.25, .25]);
    assert.equal(editor.canvas.width, 640 * dpr, "bitmap applies DPR to local dimensions once");
    editor.canvas.dispatchEvent({ ...event, type: "wheel", deltaY: -100, ctrlKey: true });
    assert.ok(Math.abs((640 - 640 * editor.state.zoom) / 2 + editor.state.panX + .25 * 640 * editor.state.zoom - 160) < 1e-8);
    editor.canvas.dispatchEvent({ ...event, type: "pointerdown", button: 1 });
    const beforePan = editor.state.panX;
    editor.canvas.dispatchEvent({ ...event, type: "pointermove", clientX: event.clientX + 40 * scale });
    assert.ok(Math.abs(editor.state.panX - beforePan - 40) < 1e-8, "pan uses local delta");
    editor.canvas.dispatchEvent({ ...event, type: "pointerup", button: 1 });
    fitButton.click();
}

// Removal cleans playback/timers and invalidates future image loads.
editor.state.playing = true;
node.onRemoved?.();
assert.equal(editor.state.destroyed, true);
assert.equal(editor.state.playing, false);

console.log("universal_mask_editor_navigation: PASS");
