import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";

import {
    applyFeatherBlur,
    applyGrowFromDistanceField,
    computeSquaredDistanceField,
    effectiveCropExtent,
    postprocessMaskAlpha,
} from "../web/mask_postprocess.js";

// 1. Distance field & circular Grow symmetry + empty mask fast-exit
{
    const empty = new Uint8Array(32 * 32);
    assert.equal(computeSquaredDistanceField(empty, 32, 32), null, "empty mask returns null");

    const width = 260;
    const height = 260;
    const cx = 130;
    const cy = 130;
    const seed = new Uint8Array(width * height);
    seed[cy * width + cx] = 255;

    const field = computeSquaredDistanceField(seed, width, height);
    assert.ok(field);
    assert.equal(field.fillVal, 255);

    const out0 = applyGrowFromDistanceField(seed, field, 0);
    assert.equal(out0[cy * width + cx], 255);
    assert.equal(out0[cy * width + cx + 1], 0);

    for (const radius of [1, 8, 50, 100]) {
        const grown = applyGrowFromDistanceField(seed, field, radius);
        assert.equal(grown[cy * width + (cx + radius)], 255, `axis pixel at radius=${radius} is inside`);
        assert.equal(grown[(cy + radius) * width + cx], 255, `vertical axis pixel at radius=${radius} is inside`);
        const diagDist = Math.hypot(radius, radius);
        if (diagDist > radius + 1) {
            assert.equal(grown[(cy + radius) * width + (cx + radius)], 0, `diagonal (${radius}, ${radius}) is outside circle`);
        }
        assert.equal(grown[cy * width + Math.min(width - 1, cx + radius + 2)], 0, `pixel outside radius=${radius}+2 is 0`);
    }
}

// 2. Feather blur smoothness, monotonicity & edge clamping (no border vignette)
{
    const w = 120;
    const h = 80;
    const stepMask = new Uint8Array(w * h);
    for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < 40; x += 1) {
            stepMask[y * w + x] = 255;
        }
    }
    const f0 = applyFeatherBlur(stepMask, w, h, 0);
    assert.equal(f0[40 * w + 0], 255);
    assert.equal(f0[40 * w + 40], 0);

    for (const feather of [4, 16, 64, 256]) {
        const blurred = applyFeatherBlur(stepMask, w, h, feather);
        if (feather <= 16) {
            assert.equal(blurred[0], 255, `top-left corner touching edge keeps 255 for feather=${feather}`);
            assert.equal(blurred[(h - 1) * w], 255, `bottom-left edge keeps 255 for feather=${feather}`);
            assert.equal(blurred[40 * w], 255, `left edge keeps 255 for feather=${feather}`);
        } else {
            assert.ok(blurred[40 * w] > blurred[40 * w + 39], `left edge stays brighter than interior step for feather=${feather}`);
        }
        let prev = 255;
        for (let x = 0; x < w; x += 1) {
            const val = blurred[40 * w + x];
            assert.ok(val <= prev, `monotonically non-increasing across step at x=${x}, feather=${feather}`);
            prev = val;
        }
        if (feather <= 64) {
            assert.ok(blurred[40 * w + 42] > 0 && blurred[40 * w + 42] < 255, `smooth falloff near edge for feather=${feather}`);
        }
    }

    const fullMask = new Uint8Array(w * h).fill(255);
    const blurredFull = applyFeatherBlur(fullMask, w, h, 256);
    assert.equal(blurredFull[0], 255);
    assert.equal(blurredFull[Math.floor(h / 2) * w + Math.floor(w / 2)], 255);
}

// 3. 640x480 performance benchmark (initial pass + cached Grow sweep)
{
    const w = 640;
    const h = 480;
    const mask = new Uint8Array(w * h);
    for (let y = 160; y < 320; y += 1) {
        for (let x = 220; x < 420; x += 1) {
            mask[y * w + x] = 255;
        }
    }
    const t0 = performance.now();
    const distField = computeSquaredDistanceField(mask, w, h);
    const out = postprocessMaskAlpha(mask, w, h, { growPx: 67, featherPx: 27, distField });
    const firstMs = performance.now() - t0;
    assert.ok(out[160 * w + 220] > 0);
    assert.ok(firstMs < 150, `initial 640x480 pass should be fast, got ${firstMs.toFixed(2)}ms`);

    const reuseBuf = new Uint8Array(w * h);
    // Warm up JIT once before measuring cached sweep
    applyGrowFromDistanceField(mask, distField, 66, reuseBuf);
    const t1 = performance.now();
    applyGrowFromDistanceField(mask, distField, 67, reuseBuf);
    const cachedGrowMs = performance.now() - t1;
    assert.ok(cachedGrowMs < 15, `cached Grow sweep must stay < 15ms, got ${cachedGrowMs.toFixed(2)}ms`);
}

// 4. effectiveCropExtent parity and clamped symmetric expansion
{
    const source = { width: 400, height: 300 };
    const meta = {
        crop_extent: { box: [80, 60, 160, 140], frames: [0, 1] },
        raw_crop_extent: { box: [88, 64, 152, 136], frames: [0, 1] },
        mask_postprocess: { grow: 8, feather: 4 },
    };
    assert.deepEqual(effectiveCropExtent(meta, 8, 4, source), meta.crop_extent);
    assert.deepEqual(effectiveCropExtent(meta, 40, 20, source), {
        frames: [0, 1],
        box: [28, 4, 212, 196],
    });
    assert.deepEqual(effectiveCropExtent(meta, 200, 100, source), {
        frames: [0, 1],
        box: [0, 0, 400, 300],
    });
}

// 5. Full editor live Grow/Feather slider + pixel integration test
{
    class FakeElement {
        constructor(tagName) {
            this.tagName = String(tagName).toUpperCase();
            this.children = [];
            this.style = { setProperty(k, v) { this[k] = v; } };
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
        click() { this.dispatchEvent({ type: "click" }); }
        focus() {}
        getBoundingClientRect() { return { left: 0, top: 0, width: 120, height: 120 }; }
        setPointerCapture() {}
        releasePointerCapture() {}
    }

    class PixelContext {
        constructor(canvas) {
            this.canvas = canvas;
            this.rgba = new Uint8ClampedArray(canvas.width * canvas.height * 4);
            this.globalAlpha = 1;
        }
        _ensureBuffer() {
            const needed = this.canvas.width * this.canvas.height * 4;
            if (this.rgba.length !== needed) {
                this.rgba = new Uint8ClampedArray(needed);
            }
        }
        save() {}
        restore() {}
        setTransform() {}
        fillRect() { this._ensureBuffer(); }
        clearRect() {
            this._ensureBuffer();
            this.rgba.fill(0);
        }
        beginPath() {}
        moveTo() {}
        lineTo() {}
        arc() {}
        closePath() {}
        fill() {}
        stroke() {}
        strokeRect() {}
        drawImage(img) {
            this._ensureBuffer();
            if (img?.rgba) {
                this.rgba.set(img.rgba);
            }
        }
        getImageData() {
            this._ensureBuffer();
            return {
                width: this.canvas.width,
                height: this.canvas.height,
                data: new Uint8ClampedArray(this.rgba),
            };
        }
        putImageData(imageData) {
            this._ensureBuffer();
            this.rgba.set(imageData.data);
        }
    }

    const canvases = [];
    class FakeCanvas extends FakeElement {
        constructor() {
            super("canvas");
            this.width = 120;
            this.height = 120;
            this.context = new PixelContext(this);
            canvases.push(this);
        }
        getContext() { return this.context; }
    }

    globalThis.Image = class FakeImage {};
    globalThis.document = {
        activeElement: null,
        createElement(tagName) { return String(tagName).toLowerCase() === "canvas" ? new FakeCanvas() : new FakeElement(tagName); },
    };
    globalThis.window = { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {}, sessionStorage: null };

    const { createUniversalMaskEditor } = await import("../web/universal_mask_editor.js");

    const find = (root, predicate) => {
        if (!root) return null;
        if (predicate(root)) return root;
        for (const child of root.children || []) {
            const hit = find(child, predicate);
            if (hit) return hit;
        }
        return null;
    };

    const makeMaskBuffer = (w, h, x0, y0, x1, y1) => {
        const rgba = new Uint8ClampedArray(w * h * 4);
        for (let y = y0; y < y1; y += 1) {
            for (let x = x0; x < x1; x += 1) {
                const p = (y * w + x) * 4;
                rgba[p] = 255;
                rgba[p + 1] = 255;
                rgba[p + 2] = 255;
                rgba[p + 3] = 255;
            }
        }
        return { naturalWidth: w, naturalHeight: h, rgba };
    };

    const W = 120;
    const H = 120;
    const widgets = [
        { name: "mask_data", value: JSON.stringify({ crop: { mode: "auto", margin: 10 }, tracking_mode: "sam3", sam3: { action: "track", frame: 0, direction: "both", seed_token: "tok-live", prompt: { frame: 0, points: [[0.5, 0.5]], labels: [1] } }, frames: {} }), options: {} },
        { name: "tracking_mode", value: "sam3" },
        { name: "grow", value: 8, options: {} },
        { name: "feather", value: 0, options: {} },
        { name: "start_frame", value: 0 },
        { name: "end_frame", value: -1 },
        { name: "fps", value: 24 },
    ];
    const node = { id: 77, widgets, graph: { setDirtyCanvas() {} }, addDOMWidget(_n, _t, root) { this.root = root; } };
    const editor = createUniversalMaskEditor(node, { app: null, api: null });

    const sourceImage = { naturalWidth: W, naturalHeight: H, rgba: new Uint8ClampedArray(W * H * 4) };
    const rawMaskImage = makeMaskBuffer(W, H, 50, 50, 70, 70);
    const serverMaskImage = makeMaskBuffer(W, H, 42, 42, 78, 78);

    editor.state.source = { frames: 3, width: W, height: H, fingerprint: "live-edge-fp" };
    editor.state.frameCount = 3;
    editor.state.metadata = {
        crop_extent: { box: [42, 42, 78, 78], frames: [0, 1, 2] },
        raw_crop_extent: { box: [50, 50, 70, 70], frames: [0, 1, 2] },
        mask_postprocess: { grow: 8, feather: 0 },
        sam3: { frame_ready: true, track_ready: true, frame: 0, seed_token: "tok-live", direction: "both" },
    };
    editor.state.serverMaskSettings = { grow: 8, feather: 0 };
    editor.state.sam3.frameReady = true;
    editor.state.sam3.trackReady = true;
    editor.state.sam3.frame = 0;
    editor.state.sam3.seedToken = "tok-live";
    editor.state.sam3.frameSignature = editor.state.sam3.frameSignature || "";
    editor.state.previewStale = false;
    editor.state.frameItems.set(0, { sourceImage, maskImage: serverMaskImage, rawMaskImage });
    editor.state.previews.set(0, { sourceImage, maskImage: serverMaskImage, rawMaskImage, lastUsed: 1 });
    editor.state.previewSignature = JSON.stringify((({ crop: _c, ...rest }) => rest)(editor.document));

    const growRange = find(editor.root, (el) => el.tagName === "INPUT" && el.attributes["aria-label"] === "Ползунок Grow");
    const growInput = find(editor.root, (el) => el.tagName === "INPUT" && el.attributes["aria-label"] === "Grow");
    const featherRange = find(editor.root, (el) => el.tagName === "INPUT" && el.attributes["aria-label"] === "Ползунок Feather");
    const featherInput = find(editor.root, (el) => el.tagName === "INPUT" && el.attributes["aria-label"] === "Feather");
    const cropInfo = find(editor.root, (el) => String(el.className || "").includes("ths-crop-readout"));
    const maskViewBtn = find(editor.root, (el) => el.tagName === "BUTTON" && el.textContent === "Маска");
    assert.ok(growRange && growInput && featherRange && featherInput && cropInfo && maskViewBtn);

    // Switch to "Маска" view so RGBA channels directly show 0..255 mask value
    maskViewBtn.click();
    const maskCanvas = canvases[1];
    const alphaAt = (x, y) => maskCanvas.context.rgba[(y * W + x) * 4];

    // Initial server state (grow=8): (45, 60) is inside serverMaskImage [42..78), (35, 60) is 0
    assert.equal(alphaAt(45, 60), 255);
    assert.equal(alphaAt(35, 60), 0);

    // Live drag Grow via input event to 40 (without pushing history)
    const historyLenBeforeDrag = editor.state.history.length;
    growRange.value = "25";
    growRange.dispatchEvent({ type: "input" });
    growRange.value = "40";
    growRange.dispatchEvent({ type: "input" });
    assert.equal(growInput.value, "40");
    assert.equal(widgets[2].value, 40);
    assert.equal(editor.state.history.length, historyLenBeforeDrag, "live drag does not spam history");
    assert.equal(editor.state.previewStale, false, "live Grow does not stale computed mask");
    assert.ok(alphaAt(15, 60) > 200, "Grow=40 expands mask out to x=15 (from x0=50)");
    assert.match(cropInfo.textContent, /Кроп 120×120 px|Кроп 100×100 px/);

    // Release slider -> single history commit, SAM3 readiness preserved
    growRange.dispatchEvent({ type: "change" });
    assert.equal(editor.state.history.length, historyLenBeforeDrag + 1);
    assert.equal(editor.state.previewStale, false);

    // Shrink Grow from 40 -> 0 (must shrink back to rawMaskImage [50..70), proving no double dilation!)
    growRange.value = "0";
    growRange.dispatchEvent({ type: "input" });
    growRange.dispatchEvent({ type: "change" });
    assert.equal(alphaAt(45, 60), 0, "Grow=0 shrinks back inside rawMaskImage (45 < 50)");
    assert.equal(alphaAt(55, 60), 255, "core raw mask remains 255");

    // Live Feather 0 -> 20 produces soft intermediate alpha around edge x=50
    featherRange.value = "20";
    featherRange.dispatchEvent({ type: "input" });
    featherRange.dispatchEvent({ type: "change" });
    assert.equal(featherInput.value, "20");
    const edgeVal = alphaAt(46, 60);
    assert.ok(edgeVal > 0 && edgeVal < 255, `Feather=20 creates soft alpha at x=46, got ${edgeVal}`);

    // Undo restores Feather=0 cleanly
    find(editor.root, (el) => el.tagName === "BUTTON" && el.attributes["aria-label"] === "Отменить").click();
    assert.equal(featherInput.value, "0");
    assert.equal(featherRange.value, "0");
    assert.equal(alphaAt(46, 60), 0);
    assert.equal(editor.state.previewStale, false);
}

console.log("mask_postprocess: PASS");
