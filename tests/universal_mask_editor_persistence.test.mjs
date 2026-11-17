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
    }

    append(...children) {
        this.children.push(...children.filter(Boolean));
    }

    appendChild(child) {
        this.append(child);
        return child;
    }

    setAttribute(name, value) {
        this.attributes[name] = String(value);
    }

    addEventListener(type, handler) {
        const handlers = this.listeners.get(type) || [];
        handlers.push(handler);
        this.listeners.set(type, handlers);
    }

    dispatchEvent(event) {
        event.target ||= this;
        event.currentTarget = this;
        for (const handler of this.listeners.get(event.type) || []) handler(event);
        return !event.defaultPrevented;
    }

    getBoundingClientRect() {
        return { left: 0, top: 0, width: 640, height: 330 };
    }

    focus() {}
    setPointerCapture() {}
    releasePointerCapture() {}
}

class FakeCanvas extends FakeElement {
    constructor() {
        super("canvas");
        this.width = 640;
        this.height = 330;
    }

    getContext() {
        const noop = () => {};
        return new Proxy({}, { get: () => noop, set: () => true });
    }
}

globalThis.document = {
    createElement(tagName) {
        return String(tagName).toLowerCase() === "canvas" ? new FakeCanvas() : new FakeElement(tagName);
    },
};
globalThis.window = { devicePixelRatio: 1, addEventListener() {} };

const { createUniversalMaskEditor } = await import("../web/universal_mask_editor.js");

const saved = {
    version: 1,
    source: { frames: 12, width: 640, height: 360, fingerprint: "saved-clip" },
    anchors: [2],
    tracking_mode: "keyframes",
    frames: {
        "2": { events: [{ type: "rect", mode: "add", x: 0.2, y: 0.3, w: 0.2, h: 0.15 }] },
    },
};
const widgets = [
    { name: "mask_data", value: "{}", options: { multiline: true } },
    { name: "tracking_mode", value: "optical_flow" },
    { name: "start_frame", value: 0 },
    { name: "end_frame", value: -1 },
];
let configureCalls = 0;
const fakeNode = {
    id: 7,
    widgets,
    graph: { setDirtyCanvas() {} },
    addDOMWidget(_name, _type, root) { this.root = root; },
    onConfigure(config) {
        configureCalls += 1;
        this.widgets[0].value = config.widgets_values[0];
        this.widgets[1].value = config.widgets_values[1];
    },
};

const editor = createUniversalMaskEditor(fakeNode, { app: null, api: null });
assert.equal(widgets[0].hidden, true, "mask_data remains in widgets but is hidden from normal UI");
assert.equal(widgets[0].options.hidden, true, "current Comfy widget API hidden option is set");

// Reproduce Comfy's order: nodeCreated created the editor with "{}", then
// workflow configuration fills widgets_values and invokes onConfigure.
fakeNode.onConfigure({ widgets_values: [JSON.stringify(saved), "keyframes"] });
assert.equal(configureCalls, 1, "the previous onConfigure handler still runs");
assert.equal(editor.state.frameCount, 12);
assert.deepEqual(editor.state.source, saved.source);
assert.deepEqual(editor.state.anchors, [2]);
assert.deepEqual(editor.document.frames, saved.frames);

// The first queue/prepare commit must serialize the configured document,
// including source and explicit anchors, rather than the initial empty state.
editor.commit();
let serialized = JSON.parse(widgets[0].value);
assert.deepEqual(serialized.frames, saved.frames);
assert.deepEqual(serialized.source, saved.source);
assert.deepEqual(serialized.anchors, saved.anchors);
assert.equal(serialized.tracking_mode, "keyframes");

// A real canvas gesture marks the editor locally dirty.  A later configure
// callback must not discard that unsaved paint, even if it supplies the old
// workflow value again.
editor.state.previews.set(2, { sourceImage: { naturalWidth: 640, naturalHeight: 360 }, maskImage: null });
editor.setFrame(2);
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 1, clientX: 120, clientY: 120, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 1, clientX: 120, clientY: 120 });
assert.equal(editor.state.localEdits, true);
const localFrame = editor.document.frames["2"];
assert.ok(localFrame.events.some((event) => event.type === "stroke"), "gesture is in the live document");

fakeNode.onConfigure({ widgets_values: [JSON.stringify(saved), "keyframes"] });
assert.equal(editor.document.frames["2"], localFrame, "unsaved local document survives configure");
editor.commit();
serialized = JSON.parse(widgets[0].value);
assert.ok(serialized.frames["2"].events.some((event) => event.type === "stroke"));
assert.deepEqual(serialized.source, saved.source);
assert.deepEqual(serialized.anchors, saved.anchors);

console.log("universal_mask_editor_persistence: PASS");
