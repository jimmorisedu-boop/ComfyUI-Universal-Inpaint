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
    constructor() {
        this.draws = [];
        this.ops = [];
        this.globalAlpha = 1;
        this.globalCompositeOperation = "source-over";
    }
    save() {}
    restore() { this.globalCompositeOperation = "source-over"; }
    setTransform() {}
    clearRect() {}
    fillRect() { this.ops.push({ op: "fillRect", gco: this.globalCompositeOperation }); }
    strokeRect() { this.ops.push({ op: "strokeRect", gco: this.globalCompositeOperation }); }
    setLineDash() {}
    fillText() {}
    beginPath() {}
    rect() {}
    moveTo() {}
    lineTo() {}
    arc() { this.ops.push({ op: "arc", gco: this.globalCompositeOperation }); }
    closePath() {}
    fill() { this.ops.push({ op: "fill", gco: this.globalCompositeOperation }); }
    stroke() { this.ops.push({ op: "stroke", gco: this.globalCompositeOperation }); }
    clip() {}
    drawImage(...args) { this.draws.push(args[0]); }
    getImageData(_x, _y, width, height) { return { width, height, data: new Uint8ClampedArray(width * height * 4) }; }
    putImageData() {}
}

class FakeCanvas extends FakeElement {
    constructor() {
        super("canvas");
        this.width = 640;
        this.height = 360;
        this.context = new FakeContext();
    }
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
function createEditor(mode = "optical_flow") {
    const startIdx = canvases.length;
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
        { name: "tracking_mode", value: mode },
        { name: "grow", value: 0 },
        { name: "feather", value: 0 },
        { name: "start_frame", value: 0 },
        { name: "end_frame", value: -1 },
        { name: "fps", value: 24 },
    ];
    const node = { id: 2, widgets, properties: {}, graph: { setDirtyCanvas() {} }, addDOMWidget(_n, _t, root) { this.root = root; } };
    const editor = createUniversalMaskEditor(node, { app, api });
    const maskContext = canvases[startIdx + 1].context;
    return { editor, node, widgets, maskContext };
}

function clickCanvas(editor, x = 520, y = 280, pointerId = 10) {
    editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId, clientX: x, clientY: y, preventDefault() {} });
    editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId, clientX: x, clientY: y });
}

// 1. ingestDescriptors merging universal_mask_frames + universal_mask_preview preserves edit_base & raw_mask.
{
    const { editor, maskContext } = createEditor("optical_flow");
    const source = { frames: 10, width: 640, height: 360, fingerprint: "clip-preview-merge" };
    editor.state.frameCount = 10;
    editor.state.source = source;
    editor.document.source = source;
    editor.document.frames["0"] = { events: [{ type: "rect", mode: "add", x: 0.1, y: 0.1, w: 0.2, h: 0.2 }], inherit: false };
    editor.commit();

    await editor.applyExecuted({
        universal_mask_meta: [{ source, has_mask: true, anchors: [0] }],
        universal_mask_frames: [
            {
                frame: 5,
                source: { filename: "frame_000005.jpg", type: "temp" },
                mask: { filename: "mask_000005.png", type: "temp" },
                edit_base: { filename: "edit_base_000005.png", type: "temp" },
                raw_mask: { filename: "raw_mask_000005.png", type: "temp" },
            },
        ],
        universal_mask_preview: [
            { frame: 5, width: 640, height: 360, data_url: "data:image/jpeg;base64,s5", mask_url: "data:image/png;base64,m5" },
        ],
    });
    const item5 = editor.state.frameItems.get(5);
    assert.deepEqual(item5.edit_base, { filename: "edit_base_000005.png", type: "temp" }, "sampled preview must not overwrite edit_base");
    assert.deepEqual(item5.raw_mask, { filename: "raw_mask_000005.png", type: "temp" }, "sampled preview must not overwrite raw_mask");

    const src5 = { naturalWidth: 640, naturalHeight: 360, tag: "s5" };
    const m5 = { naturalWidth: 640, naturalHeight: 360, tag: "m5" };
    const base5 = { naturalWidth: 640, naturalHeight: 360, tag: "edit-base-5" };
    editor.state.previews.set(5, { sourceImage: src5, maskImage: m5, editBaseImage: base5, edit_base: item5.edit_base });
    editor.setFrame(5, { load: false });
    findButton(editor.root, "Ластик").click();
    maskContext.draws.length = 0;
    clickCanvas(editor, 520, 280, 11);
    assert.equal(editor.document.frames["5"].inherit, true);
    assert.ok(maskContext.draws.includes(base5), "eraser click outside mask on sampled frame retains editBaseImage");
    editor.destroy();
}

// 2. SAM3 rectangle seed (inherit: false) keeps SAM3 baseline visible on eraser click and flips inherit -> true.
{
    const { editor, maskContext } = createEditor("sam3");
    const source = { frames: 5, width: 640, height: 360, fingerprint: "clip-sam3-rect" };
    const src0 = { naturalWidth: 640, naturalHeight: 360, tag: "s0" };
    const m0 = { naturalWidth: 640, naturalHeight: 360, tag: "sam3-m0" };
    const base0 = { naturalWidth: 640, naturalHeight: 360, tag: "sam3-base0" };
    editor.state.source = source;
    editor.state.frameCount = 5;
    editor.document.source = source;
    editor.state.frameItems.set(0, { sourceImage: src0, edit_base: { data_url: "b0" } });
    editor.state.previews.set(0, { sourceImage: src0, editBaseImage: base0 });
    editor.setFrame(0, { load: false });

    findButton(editor.root, "Прямоугольник").click();
    editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 20, clientX: 80, clientY: 60, preventDefault() {} });
    editor.canvas.dispatchEvent({ type: "pointermove", pointerId: 20, clientX: 220, clientY: 200 });
    editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 20, clientX: 220, clientY: 200 });
    assert.equal(editor.document.frames["0"].inherit, false);

    findButton(editor.root, "Рассчитать кадр").click();
    await new Promise((r) => setTimeout(r, 0));
    await editor.applyExecuted({
        prompt_id: editor.state.pendingPreview?.promptId,
        universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: false, frame: 0, direction: "both", seed_token: "tok-rect-1", computed_frames: [0] } }],
        universal_mask_frame_patch: [{ frame: 0, mask: "data:image/png;base64,m", edit_base: "data:image/png;base64,b" }],
    });
    editor.state.previews.set(0, { sourceImage: src0, maskImage: m0, editBaseImage: base0, edit_base: { data_url: "b" } });
    editor.setFrame(0, { load: false });

    findButton(editor.root, "Ластик").click();
    maskContext.draws.length = 0;
    clickCanvas(editor, 560, 320, 21);
    assert.equal(editor.document.frames["0"].inherit, true, "SAM3 frame edit switches seed frame bucket to inherit:true");
    assert.ok(maskContext.draws.includes(base0), "eraser click on SAM3 rect frame retains SAM3 baseline");
    editor.destroy();
}

// 3. SAM3 point mode: eraser click then Undo (Ctrl+Z) does not set resetBlank or drop SAM3 mask.
{
    const { editor, maskContext } = createEditor("sam3");
    const source = { frames: 5, width: 640, height: 360, fingerprint: "clip-sam3-undo" };
    const src0 = { naturalWidth: 640, naturalHeight: 360, tag: "s0" };
    const m0 = { naturalWidth: 640, naturalHeight: 360, tag: "sam3-m0" };
    const base0 = { naturalWidth: 640, naturalHeight: 360, tag: "sam3-base0" };
    editor.state.source = source;
    editor.state.frameCount = 5;
    editor.document.source = source;
    editor.state.frameItems.set(0, { sourceImage: src0, edit_base: { data_url: "b0" } });
    editor.state.previews.set(0, { sourceImage: src0, editBaseImage: base0 });
    editor.setFrame(0, { load: false });

    clickCanvas(editor, 160, 120, 30);
    await new Promise((r) => setTimeout(r, 0));
    await editor.applyExecuted({
        prompt_id: editor.state.pendingPreview?.promptId,
        universal_mask_meta: [{ source, has_mask: true, sam3: { frame_ready: true, track_ready: false, frame: 0, direction: "both", seed_token: "tok-pt-1", computed_frames: [0] } }],
        universal_mask_frame_patch: [{ frame: 0, mask: "data:image/png;base64,m", edit_base: "data:image/png;base64,b" }],
    });
    editor.state.previews.set(0, { sourceImage: src0, maskImage: m0, editBaseImage: base0, edit_base: { data_url: "b" } });
    editor.setFrame(0, { load: false });

    findButton(editor.root, "Ластик").click();
    clickCanvas(editor, 520, 300, 31);
    maskContext.draws.length = 0;
    editor.root.dispatchEvent({ type: "keydown", key: "z", ctrlKey: true, preventDefault() {} });
    assert.equal(editor.state.resetBlank, false, "Undo in SAM3 point mode must not trigger resetBlank");
    assert.ok(maskContext.draws.includes(base0), "Undo in SAM3 point mode keeps SAM3 baseline visible");
    editor.destroy();
}

// 4. Static mode: paint on frame 0 -> scrub to frame 3 -> click eraser replays frame 0 strokes.
{
    const { editor, maskContext } = createEditor("static");
    const source = { frames: 5, width: 640, height: 360, fingerprint: "clip-static" };
    const src = { naturalWidth: 640, naturalHeight: 360, tag: "s" };
    editor.state.source = source;
    editor.state.frameCount = 5;
    editor.document.source = source;
    for (const f of [0, 3]) {
        editor.state.frameItems.set(f, { sourceImage: src });
        editor.state.previews.set(f, { sourceImage: src });
    }
    editor.setFrame(0, { load: false });
    findButton(editor.root, "Кисть").click();
    clickCanvas(editor, 160, 120, 40);

    editor.setFrame(3, { load: false });
    findButton(editor.root, "Ластик").click();
    maskContext.ops.length = 0;
    clickCanvas(editor, 540, 300, 41);
    assert.ok(maskContext.ops.some((item) => item.gco === "source-over"), "Static frame 3 replays frame 0 brush paint");
    assert.ok(maskContext.ops.some((item) => item.gco === "destination-out"), "Static frame 3 applies eraser cut");
    editor.destroy();
}

// 5. Optical flow GrabCut refined rect on anchor frame 0 retains rawMaskImage on eraser click.
{
    const { editor, maskContext } = createEditor("optical_flow");
    const source = { frames: 4, width: 640, height: 360, fingerprint: "clip-grabcut" };
    const src0 = { naturalWidth: 640, naturalHeight: 360, tag: "s0" };
    const raw0 = { naturalWidth: 640, naturalHeight: 360, tag: "gc-raw-0" };
    const m0 = { naturalWidth: 640, naturalHeight: 360, tag: "gc-mask-0" };
    editor.state.source = source;
    editor.state.frameCount = 4;
    editor.document.source = source;
    editor.state.frameItems.set(0, { sourceImage: src0 });
    editor.state.previews.set(0, { sourceImage: src0 });
    editor.setFrame(0, { load: false });

    findButton(editor.root, "Выделять объект").click();
    editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 50, clientX: 80, clientY: 60, preventDefault() {} });
    editor.canvas.dispatchEvent({ type: "pointermove", pointerId: 50, clientX: 240, clientY: 220 });
    editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 50, clientX: 240, clientY: 220 });
    await editor.applyExecuted({
        universal_mask_meta: [{ source, has_mask: true, anchors: [0] }],
        universal_mask_frames: [{ frame: 0, mask: { data_url: "m0" }, raw_mask: { data_url: "r0" }, edit_base: { data_url: "b0" } }],
    });
    editor.state.previews.set(0, { sourceImage: src0, maskImage: m0, rawMaskImage: raw0 });
    editor.setFrame(0, { load: false });

    findButton(editor.root, "Ластик").click();
    maskContext.draws.length = 0;
    clickCanvas(editor, 520, 300, 51);
    assert.ok(maskContext.draws.includes(raw0), "GrabCut anchor frame keeps computed rawMaskImage baseline when erasing");
    editor.destroy();
}

// 6. Multi-layer: Layer 1 calculated as active -> add Layer 2 -> eraser on frame 0 prefers rawMaskImage over Layer 1's zero editBaseImage.
{
    const { editor, maskContext } = createEditor("optical_flow");
    const source = { frames: 4, width: 640, height: 360, fingerprint: "clip-multilayer" };
    const src0 = { naturalWidth: 640, naturalHeight: 360, tag: "s0" };
    const base0Zero = { naturalWidth: 640, naturalHeight: 360, tag: "l1-zero-edit-base" };
    const raw0Full = { naturalWidth: 640, naturalHeight: 360, tag: "l1-full-raw" };
    const m0Full = { naturalWidth: 640, naturalHeight: 360, tag: "l1-full-mask" };
    editor.state.source = source;
    editor.state.frameCount = 4;
    editor.document.source = source;
    editor.state.frameItems.set(0, { sourceImage: src0 });
    editor.state.previews.set(0, { sourceImage: src0 });
    editor.setFrame(0, { load: false });

    clickCanvas(editor, 160, 120, 60);
    await editor.applyExecuted({
        universal_mask_meta: [{
            source,
            has_mask: true,
            active_layer_id: "layer-1",
            layers: [{ id: "layer-1", type: "optical_flow", blend: "add", enabled: true, has_mask: true }],
        }],
        universal_mask_frames: [{ frame: 0, mask: { data_url: "m0" }, raw_mask: { data_url: "r0" }, edit_base: { data_url: "z0" } }],
    });
    editor.state.previews.set(0, { sourceImage: src0, maskImage: m0Full, rawMaskImage: raw0Full, editBaseImage: base0Zero, edit_base: { data_url: "z0" } });
    editor.setFrame(0, { load: false });

    findButton(editor.root, "+ Статика").click();
    findButton(editor.root, "Ластик").click();
    maskContext.draws.length = 0;
    clickCanvas(editor, 520, 300, 61);
    assert.ok(maskContext.draws.includes(raw0Full), "other layer uses rawMaskImage instead of stale zero editBaseImage");
    assert.ok(!maskContext.draws.includes(base0Zero));
    editor.destroy();
}

// 7. Fallback preview without edit_base (only maskImage): eraser click on tracked frame still inherits instead of blanking.
{
    const { editor, maskContext } = createEditor("optical_flow");
    const source = { frames: 4, width: 640, height: 360, fingerprint: "clip-fallback" };
    const src2 = { naturalWidth: 640, naturalHeight: 360, tag: "s2" };
    const m2 = { naturalWidth: 640, naturalHeight: 360, tag: "m2-only" };
    editor.state.source = source;
    editor.state.frameCount = 4;
    editor.document.source = source;
    editor.document.frames["0"] = { events: [{ type: "rect", mode: "add", x: 0.1, y: 0.1, w: 0.2, h: 0.2 }], inherit: false };
    editor.commit();
    editor.state.frameItems.set(2, { sourceImage: src2, mask: { data_url: "m2" } });
    editor.state.previews.set(2, { sourceImage: src2, maskImage: m2 });
    editor.setFrame(2, { load: false });

    findButton(editor.root, "Ластик").click();
    maskContext.draws.length = 0;
    clickCanvas(editor, 520, 300, 70);
    assert.equal(editor.document.frames["2"].inherit, true, "eraser fallback inherits visible mask when edit_base is missing");
    assert.ok(maskContext.draws.includes(m2), "eraser fallback draws maskImage instead of blanking");
    editor.destroy();
}

console.log("eraser_preserve_mask: PASS");
