import assert from "node:assert/strict";

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
        this._html = "";
        this.value = "";
        this.disabled = false;
        this.className = "";
        this.isContentEditable = false;
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
    set innerHTML(value) { this._html = String(value ?? ""); }
    get innerHTML() { return this._html; }
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
        const payload = { target: this, currentTarget: this, ...event };
        for (const fn of [...(this.listeners.get(event.type) || [])]) fn(payload);
        return !payload.defaultPrevented;
    }
    click() { this.dispatchEvent({ type: "click" }); }
    remove() {}
    focus() {}
    getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 360 }; }
    setPointerCapture() {}
    releasePointerCapture() {}
}

class FakeContext {
    constructor() { this.draws = []; this.circles = []; }
    save() {}
    restore() {}
    setTransform() {}
    clearRect() {}
    fillRect() {}
    strokeRect() {}
    setLineDash() {}
    fillText() {}
    beginPath() {}
    moveTo() {}
    lineTo() {}
    arc(x, y, r) { this.circles.push({ x, y, r }); }
    closePath() {}
    fill() {}
    stroke() {}
    drawImage(...args) { this.draws.push(args[0]); }
    getImageData(_x, _y, width, height) { return { width, height, data: new Uint8ClampedArray(width * height * 4) }; }
    putImageData() {}
}

class FakeCanvas extends FakeElement {
    constructor() { super("canvas"); this.width = 640; this.height = 360; this.context = new FakeContext(); }
    getContext() { return this.context; }
}

const canvases = [];
globalThis.document = {
    activeElement: null,
    createElement(tag) {
        if (String(tag).toLowerCase() === "canvas") {
            const canvas = new FakeCanvas();
            canvases.push(canvas);
            return canvas;
        }
        return new FakeElement(tag);
    },
};
globalThis.window = { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {} };

const { createUniversalMaskEditor } = await import("../web/universal_mask_editor.js");

function find(root, predicate) {
    if (!root) return null;
    if (predicate(root)) return root;
    for (const child of root.children || []) { const hit = find(child, predicate); if (hit) return hit; }
    return null;
}
function findButton(root, text) {
    return find(root, (el) => el.tagName === "BUTTON" && (el._text === text || el.textContent === text || el.attributes["aria-label"] === text));
}

const submitted = [];
const api = {
    queuePrompt: async (_n, prompt) => { submitted.push(prompt); return { prompt_id: `p-${submitted.length}` }; },
    addEventListener() {},
    removeEventListener() {},
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
const widgets = [
    { name: "mask_data", value: "{}", options: {} },
    { name: "tracking_mode", value: "sam3" },
    { name: "grow", value: 0 },
    { name: "feather", value: 0 },
    { name: "start_frame", value: 0 },
    { name: "end_frame", value: -1 },
    { name: "fps", value: 24 },
];
const node = { id: 2, widgets, properties: {}, graph: { setDirtyCanvas() {} }, addDOMWidget(_n, _t, root) { this.root = root; } };
const editor = createUniversalMaskEditor(node, { app, api });
const mainContext = canvases[0].context;

// 1. Alert close button uses centered SVG close artwork instead of unaligned text glyph.
const dismissBtn = findButton(editor.root, "Скрыть уведомление");
assert.ok(dismissBtn, "dismiss button exists");
assert.ok(String(dismissBtn.className || "").includes("ths-mask-alert-dismiss"));
assert.equal(dismissBtn._text, "");
const svgChild = find(dismissBtn, (el) => String(el.innerHTML || "").includes("<svg"));
assert.ok(svgChild, "dismiss button uses vector SVG close icon");
const styleEl = find(editor.root, (el) => el.tagName === "STYLE");
assert.match(styleEl._text, /\.ths-mask-alert-head\s*\{[^}]*align-items:\s*center/);
assert.match(styleEl._text, /\.ths-mask-alert-dismiss\s*\{[^}]*align-items:\s*center;\s*justify-content:\s*center/);

// Setup source images on frames 0, 2, 3.
const source = { frames: 5, width: 640, height: 360, fingerprint: "clip-mf" };
const img = { naturalWidth: 640, naturalHeight: 360 };
editor.state.source = source;
editor.state.frameCount = 5;
editor.document.source = source;
for (const f of [0, 2, 3]) {
    const maskImg = { naturalWidth: 640, naturalHeight: 360, tag: `m-${f}` };
    const baseImg = { naturalWidth: 640, naturalHeight: 360, tag: `b-${f}` };
    editor.state.frameItems.set(f, { sourceImage: img, mask: { data_url: `m-${f}` }, edit_base: { data_url: `b-${f}` } });
    editor.state.previews.set(f, { sourceImage: img, maskImage: maskImg, editBaseImage: baseImg, edit_base: { data_url: `b-${f}` } });
}

function click(x, y, pointerId = 1) {
    editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId, clientX: x, clientY: y, preventDefault() {} });
    editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId, clientX: x, clientY: y });
}

// 2. Object 1 on frame 0 -> scrub to frame 2 -> add Object 2 on frame 2 without single-frame lock warning!
editor.setFrame(0, { load: false });
click(160, 90, 1);
await new Promise((r) => setTimeout(r, 0));
assert.equal(editor.document.sam3.prompt.frame, 0);
await editor.applyExecuted({
    prompt_id: editor.state.pendingPreview?.promptId,
    universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: false, frame: 0, direction: "both", seed_token: "tok-f0", computed_frames: [0] } }],
    universal_mask_frame_patch: [{ frame: 0, mask: "data:image/png;base64,m0", edit_base: "data:image/png;base64,b0" }],
});

editor.setFrame(2, { load: false });
findButton(editor.root, "Выбрать объект").click();
click(400, 180, 2);
await new Promise((r) => setTimeout(r, 0));
assert.equal(editor.document.sam3.objects.length, 2, "object added on frame 2 without blocking alert");
assert.equal(editor.document.sam3.objects[0].frame, 0);
assert.equal(editor.document.sam3.objects[1].frame, 2);
assert.ok(!/Добавляйте и уточняйте объекты на опорном кадре/.test(editor.status.textContent));

await editor.applyExecuted({
    prompt_id: editor.state.pendingPreview?.promptId,
    universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: false, frame: 2, direction: "both", seed_token: "tok-f2", computed_frames: [0, 2] } }],
    universal_mask_frame_patch: [{ frame: 2, mask: "data:image/png;base64,m2", edit_base: "data:image/png;base64,b2" }],
});
const objSelect = find(editor.root, (el) => el.tagName === "SELECT" && el.attributes["aria-label"] === "Активный объект SAM3");
assert.match(objSelect.children[0].textContent, /к\.1/);
assert.match(objSelect.children[1].textContent, /к\.3/);

// 3. Select Object 1, scrub to frame 3, add negative cross-frame refinement -> preserves frame 3 mask!
objSelect.value = "0";
objSelect.dispatchEvent({ type: "change" });
editor.setFrame(3, { load: false });
const existingMask3 = { data_url: "existing-m3" };
editor.state.frameItems.get(3).mask = existingMask3;
findButton(editor.root, "Исключить из объекта").click();
click(220, 140, 3);
await new Promise((r) => setTimeout(r, 0));

assert.equal(editor.state.resetBlank, false, "negative cross-frame refinement preserves existing mask preview");
assert.equal(editor.document.sam3.objects[0].refinements.length, 1);
assert.deepEqual(editor.document.sam3.objects[0].refinements[0], { frame: 3, points: [[0.34375, 0.388889]], labels: [0] });
assert.equal(submitted.at(-1).output["2"].inputs.mask_data.includes("sam3_frame_v1"), true);

mainContext.circles.length = 0;
await editor.applyExecuted({
    prompt_id: editor.state.pendingPreview?.promptId,
    universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: false, frame: 3, direction: "both", seed_token: "tok-f3", computed_frames: [0, 2, 3], keep_existing_mask: true } }],
    universal_mask_frame_patch: [{ frame: 3, mask: "data:image/png;base64,m3", edit_base: "data:image/png;base64,b3", keep_existing_mask: true }],
});
assert.equal(editor.state.frameItems.get(3).mask, existingMask3, "keep_existing_mask keeps tracked frame mask intact");
assert.equal(editor.state.sam3.frameReady, true);
assert.equal(findButton(editor.root, "Отследить видео").disabled, false);

// 4. Right-click deletes only the refinement point on frame 3, leaving Object 1 on frame 0 intact!
let prevented = false;
editor.canvas.dispatchEvent({ type: "contextmenu", button: 2, clientX: 220, clientY: 140, preventDefault() { prevented = true; } });
await new Promise((r) => setTimeout(r, 0));
assert.equal(prevented, true);
assert.equal(editor.document.sam3.objects.length, 2, "primary objects remain after deleting refinement point");
assert.equal(editor.document.sam3.objects[0].refinements, undefined);

// 5. Clear frame 2 removes only Object 2, leaving Object 1 on frame 0!
editor.setFrame(2, { load: false });
findButton(editor.root, "Очистить кадр").click();
assert.equal(editor.document.sam3.prompt.frame, 0, "clearing frame 2 keeps Object 1 on frame 0");
assert.equal(editor.document.sam3.objects, undefined);

editor.destroy();
console.log("sam3_multiframe: PASS");
