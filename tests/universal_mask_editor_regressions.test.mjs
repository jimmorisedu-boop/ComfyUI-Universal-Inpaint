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
    append(...children) { this.children.push(...children.filter(Boolean)); }
    appendChild(child) { this.append(child); return child; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    addEventListener(type, handler) { const list = this.listeners.get(type) || []; list.push(handler); this.listeners.set(type, list); }
    dispatchEvent(event) { event.target ||= this; event.currentTarget = this; for (const handler of this.listeners.get(event.type) || []) handler(event); return !event.defaultPrevented; }
    click() { this.dispatchEvent({ type: "click" }); }
    focus() { this.focused = true; }
    getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 360 }; }
    setPointerCapture() {}
    releasePointerCapture() {}
}

class FakeContext {
    constructor() { this.draws = []; this.globalAlpha = 1; }
    save() {}
    restore() {}
    setTransform() {}
    clearRect() {}
    fillRect() {}
    strokeRect() {}
    beginPath() {}
    moveTo() {}
    lineTo() {}
    arc() {}
    closePath() {}
    fill() {}
    stroke() {}
    drawImage(...args) { this.draws.push(args[0]); }
    getImageData(_x, _y, width, height) { return { data: new Uint8ClampedArray(width * height * 4) }; }
    putImageData() {}
}

class FakeCanvas extends FakeElement {
    constructor() { super("canvas"); this.width = 640; this.height = 360; this.context = new FakeContext(); }
    getContext() { return this.context; }
}

class DeferredImage {
    static pending = [];
    constructor() { DeferredImage.pending.push(this); }
    set src(value) { this.url = value; }
    resolve(width = 640, height = 360) { this.naturalWidth = width; this.naturalHeight = height; this.onload?.(); }
}

const canvases = [];
globalThis.Image = DeferredImage;
globalThis.document = {
    activeElement: null,
    createElement(tagName) {
        if (String(tagName).toLowerCase() === "canvas") { const canvas = new FakeCanvas(); canvases.push(canvas); return canvas; }
        return new FakeElement(tagName);
    },
};
globalThis.window = { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {} };

const { createUniversalMaskEditor } = await import("../web/universal_mask_editor.js");

function find(root, predicate) {
    if (!root) return null;
    if (predicate(root)) return root;
    for (const child of root.children || []) { const found = find(child, predicate); if (found) return found; }
    return null;
}
function button(root, label) { return find(root, (node) => node.tagName === "BUTTON" && node.attributes["aria-label"] === label); }

const widgets = [
    { name: "mask_data", value: "{}", options: {} },
    { name: "tracking_mode", value: "optical_flow" },
    { name: "grow", value: 8, options: {} }, { name: "feather", value: 4, options: {} },
    { name: "start_frame", value: 0 }, { name: "end_frame", value: -1 }, { name: "fps", value: 24 },
];
const node = { id: 44, widgets, graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const editor = createUniversalMaskEditor(node, { app: null, api: null });

// Modal moves the full workbench, including navigation and actions.
button(editor.root, "Развернуть").click();
const dialog = find(editor.root, (node) => node.tagName === "DIALOG");
assert.ok(button(dialog, "Воспроизвести"));
assert.ok(find(dialog, (node) => node.tagName === "BUTTON" && node.textContent === "Очистить кадр"));
button(dialog, "Закрыть").click();
assert.equal(editor.state.dialogOpen, false);

const sourceImage = { naturalWidth: 640, naturalHeight: 360, label: "source" };
const computedMask = { naturalWidth: 640, naturalHeight: 360, label: "computed-mask" };
const editBaseImage = { naturalWidth: 640, naturalHeight: 360, label: "edit-base" };
editor.state.frameCount = 2;
editor.state.frameItems.set(0, { sourceImage, maskImage: computedMask, editBaseImage, edit_base: { data_url: "raw" } });
editor.state.previews.set(0, { sourceImage, maskImage: computedMask, editBaseImage, edit_base: { data_url: "raw" } });
editor.state.previewStale = false;
editor.state.previewSignature = JSON.stringify(editor.document);
editor.setFrame(0, { load: false });
const maskContext = canvases[1].context;
assert.ok(maskContext.draws.includes(computedMask), "clean frame keeps computed preview");

// The first independent stroke starts from zero, not a cached old mask.
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 1, clientX: 180, clientY: 140, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 1, clientX: 180, clientY: 140 });
maskContext.draws.length = 0;
editor.setFrame(0, { load: false });
assert.ok(!maskContext.draws.includes(editBaseImage), "independent key starts from zero");
assert.ok(!maskContext.draws.includes(computedMask), "dirty frame does not replay over computed mask");

// A later tracked correction uses the raw baseline and retains it after undo.
editor.state.frameItems.set(1, { sourceImage, maskImage: computedMask, editBaseImage, edit_base: { data_url: "raw" } });
editor.state.previews.set(1, { sourceImage, maskImage: computedMask, editBaseImage, edit_base: { data_url: "raw" } });
editor.setFrame(1, { load: false });
button(editor.root, "Ластик").click();
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 5, clientX: 180, clientY: 140, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 5, clientX: 180, clientY: 140 });
assert.equal(editor.document.frames["1"].inherit, true);
maskContext.draws.length = 0;
editor.setFrame(1, { load: false });
assert.ok(maskContext.draws.includes(editBaseImage), "correction starts from raw edit_base");
button(editor.root, "Очистить кадр").click();
maskContext.draws.length = 0;
editor.setFrame(1, { load: false });
assert.ok(!maskContext.draws.includes(editBaseImage), "clear key is empty");
button(editor.root, "Отменить").click();
assert.equal(editor.document.frames["1"].inherit, true);

// Reset hides stale computed pixels immediately while retaining immutable bases.
button(editor.root, "Сбросить маску").click();
maskContext.draws.length = 0;
editor.setFrame(0, { load: false });
assert.ok(!maskContext.draws.includes(computedMask), "reset shows an empty draft");
assert.ok(!maskContext.draws.includes(editBaseImage), "reset does not retain old raw base");
button(editor.root, "Отменить").click();
assert.equal(editor.state.resetBlank, false);
editor.setFrame(1, { load: false });
assert.ok(maskContext.draws.includes(editBaseImage), "undo reset restores inherited basis");
editor.setFrame(0, { load: false });

// Enter completes a straight Pen path and only consumed keys stop propagation.
button(editor.root, "Перо").click();
assert.equal(button(editor.root, "Полигон"), null, "Полигон button replaced by Перо");
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 2, clientX: 120, clientY: 100, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 3, clientX: 240, clientY: 100, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 4, clientX: 180, clientY: 220, preventDefault() {} });
let stopped = false;
editor.root.dispatchEvent({ type: "keydown", key: "Enter", preventDefault() {}, stopPropagation() { stopped = true; } });
assert.equal(editor.document.frames["0"].events.at(-1).type, "polygon");
assert.equal(editor.document.frames["0"].events.at(-1).points.length, 3, "straight pen clicks keep exact corner vertices");
assert.equal(stopped, true);

// Dragging with the Pen pulls Bezier handles and produces a curved contour.
const eventsBeforeBezier = editor.document.frames["0"].events.length;
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 10, clientX: 140, clientY: 160, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointermove", button: 0, pointerId: 10, clientX: 140, clientY: 90, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 10, clientX: 140, clientY: 90 });
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 11, clientX: 380, clientY: 160, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointermove", button: 0, pointerId: 11, clientX: 380, clientY: 230, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 11, clientX: 380, clientY: 230 });
editor.root.dispatchEvent({ type: "keydown", key: "Enter", preventDefault() {}, stopPropagation() {} });
assert.equal(editor.document.frames["0"].events.length, eventsBeforeBezier + 1);
const bezierEvent = editor.document.frames["0"].events.at(-1);
assert.equal(bezierEvent.type, "polygon");
assert.ok(bezierEvent.points.length > 6, "curved Bezier segments subdivide smoothly");
assert.ok(bezierEvent.points.every(([x, y]) => x >= 0 && x <= 1 && y >= 0 && y <= 1), "flattened Bezier coordinates stay normalized");

// Clicking near the first Pen anchor closes the path automatically; Backspace and Escape edit or abort open paths.
const eventsBeforeClose = editor.document.frames["0"].events.length;
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 12, clientX: 160, clientY: 120, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 12, clientX: 160, clientY: 120 });
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 13, clientX: 300, clientY: 120, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 13, clientX: 300, clientY: 120 });
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 14, clientX: 230, clientY: 220, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 14, clientX: 230, clientY: 220 });
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 15, clientX: 163, clientY: 123, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 15, clientX: 163, clientY: 123 });
assert.equal(editor.document.frames["0"].events.length, eventsBeforeClose + 1, "clicking near first anchor closes path");
assert.equal(editor.state.penPath.length, 0);

editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 16, clientX: 180, clientY: 140, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 16, clientX: 180, clientY: 140 });
editor.canvas.dispatchEvent({ type: "pointerdown", button: 0, pointerId: 17, clientX: 240, clientY: 180, preventDefault() {} });
editor.canvas.dispatchEvent({ type: "pointerup", button: 0, pointerId: 17, clientX: 240, clientY: 180 });
assert.equal(editor.state.penPath.length, 2);
editor.root.dispatchEvent({ type: "keydown", key: "Backspace", preventDefault() {}, stopPropagation() {} });
assert.equal(editor.state.penPath.length, 1, "Backspace removes last Pen anchor");
editor.root.dispatchEvent({ type: "keydown", key: "Escape", preventDefault() {}, stopPropagation() {} });
assert.equal(editor.state.penPath.length, 0, "Escape cancels open Pen path");
assert.equal(editor.document.frames["0"].events.length, eventsBeforeClose + 1, "Escape does not commit open Pen path");

// A visible tracking selector must undo/redo the actual serialized mode too.
const trackingSelect = find(editor.root, (node) => node.tagName === "SELECT" && node.attributes["aria-label"] === "Трекинг");
trackingSelect.value = "keyframes";
trackingSelect.dispatchEvent({ type: "change" });
button(editor.root, "Отменить").click();
assert.equal(trackingSelect.value, "optical_flow");
assert.equal(widgets.find((widget) => widget.name === "tracking_mode").value, "optical_flow");
assert.equal(JSON.parse(widgets[0].value).tracking_mode, "optical_flow");
button(editor.root, "Повторить").click();
assert.equal(trackingSelect.value, "keyframes");
assert.equal(widgets.find((widget) => widget.name === "tracking_mode").value, "keyframes");
assert.equal(JSON.parse(widgets[0].value).tracking_mode, "keyframes");

// Grow/Feather remain native serialized inputs while their editor controls
// support history, limits, Enter, and restoration from a configured workflow.
const growInput = find(editor.root, (node) => node.tagName === "INPUT" && node.attributes["aria-label"] === "Grow");
const featherInput = find(editor.root, (node) => node.tagName === "INPUT" && node.attributes["aria-label"] === "Feather");
const growWidget = widgets.find((widget) => widget.name === "grow");
const featherWidget = widgets.find((widget) => widget.name === "feather");
assert.equal(growInput.value, "8"); assert.equal(featherInput.value, "4");
assert.equal(growWidget.hidden, true); assert.equal(featherWidget.hidden, true);
growInput.value = "18";
growInput.dispatchEvent({ type: "keydown", key: "Enter", preventDefault() {}, stopPropagation() {} });
assert.equal(growWidget.value, 18);
featherInput.value = "9"; featherInput.dispatchEvent({ type: "change" });
assert.equal(featherWidget.value, 9);
button(editor.root, "Отменить").click();
assert.equal(featherInput.value, "4"); assert.equal(featherWidget.value, 4);
assert.equal(growWidget.value, 18);
button(editor.root, "Отменить").click();
assert.equal(growInput.value, "8"); assert.equal(growWidget.value, 8);
button(editor.root, "Повторить").click();
assert.equal(growWidget.value, 18);
growInput.value = "999"; growInput.dispatchEvent({ type: "change" });
assert.equal(growWidget.value, 256);
featherInput.value = "-2"; featherInput.dispatchEvent({ type: "change" });
assert.equal(featherWidget.value, 0);
growInput.value = ""; growInput.dispatchEvent({ type: "blur" });
assert.equal(growWidget.value, 256); assert.equal(growInput.value, "256");
editor.state.localEdits = false;
growWidget.value = 12; featherWidget.value = 6;
node.onConfigure({});
assert.equal(growInput.value, "12"); assert.equal(featherInput.value, "6");
assert.equal(JSON.parse(widgets[0].value).grow, undefined, "native grow input is not duplicated in mask_data");
assert.equal(JSON.parse(widgets[0].value).feather, undefined);

// A configured workflow can replace the initial empty SAM3 defaults after the
// editor has already mounted.  The imported legacy mode must rebind the active
// tool/status without rewriting its saved document, while an explicit refined
// rectangle remains opt-in for Optical Flow.
const configuredWidgets = [
    { name: "mask_data", value: "{}", options: {} }, { name: "tracking_mode", value: "sam3" },
    { name: "grow", value: 8 }, { name: "feather", value: 4 }, { name: "start_frame", value: 0 }, { name: "end_frame", value: -1 },
];
const configuredNode = { id: 45, widgets: configuredWidgets, graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const configuredEditor = createUniversalMaskEditor(configuredNode, { app: null, api: null });
assert.equal(configuredEditor.state.mode, "sam_point");
assert.equal(configuredEditor.state.grabcut, true);
const legacyDocument = {
    version: 1,
    source: { frames: 2, width: 320, height: 180, fingerprint: "configured-optical-flow" },
    frames: { "0": { events: [{ type: "stroke", mode: "add", radius: 0.03, points: [[0.2, 0.2]] }] } },
    tracking_mode: "optical_flow",
};
const legacyMaskData = JSON.stringify(legacyDocument);
configuredWidgets[0].value = legacyMaskData;
configuredWidgets[1].value = "optical_flow";
configuredNode.onConfigure({});
assert.equal(configuredEditor.state.mode, "brush", "configured legacy mode selects ordinary painting");
assert.equal(configuredEditor.state.grabcut, false, "configured legacy mode leaves optional refinement off");
assert.equal(configuredEditor.state.refineMethod, "grabcut");
assert.doesNotMatch(configuredEditor.status.textContent, /SAM3/);
assert.equal(configuredWidgets[0].value, legacyMaskData, "configure sync does not rewrite saved mask_data");
assert.deepEqual(configuredEditor.document.frames, legacyDocument.frames, "configure sync preserves saved events");

const refinedLegacyDocument = {
    ...legacyDocument,
    frames: { "0": { events: [{ type: "rect", mode: "add", x: 0.1, y: 0.1, w: 0.3, h: 0.4, refine: true, refine_method: "grabcut" }] } },
};
configuredWidgets[0].value = JSON.stringify(refinedLegacyDocument);
configuredWidgets[1].value = "optical_flow";
configuredNode.onConfigure({});
assert.equal(configuredEditor.state.mode, "rect", "persisted refinement intent opts into the rectangle tool");
assert.equal(configuredEditor.state.grabcut, true);
assert.match(configuredEditor.status.textContent, /GrabCut/);

// Switching refinement off is a saved preference, even if earlier events used
// GrabCut. Reopening must preserve those masks without enabling new refinement.
const configuredRefine = find(configuredEditor.root, (item) => item.tagName === "BUTTON" && item.textContent === "Выделять объект");
configuredRefine.click();
assert.equal(configuredNode.properties.__thsRefineEnabled, false);
const refinedBeforeReload = configuredWidgets[0].value;
const reopenedNode = { id: 46, widgets: structuredClone(configuredWidgets), properties: structuredClone(configuredNode.properties), graph: { setDirtyCanvas() {} }, addDOMWidget(_name, _type, root) { this.root = root; } };
const reopenedEditor = createUniversalMaskEditor(reopenedNode, { app: null, api: null });
assert.equal(reopenedEditor.state.grabcut, false, "disabled refinement survives reload with refined events");
assert.equal(reopenedEditor.state.mode, "brush");
assert.equal(reopenedNode.widgets[0].value, refinedBeforeReload, "refinement preference never rewrites old events");
reopenedEditor.destroy();

const sam3Document = {
    version: 1,
    source: { frames: 2, width: 320, height: 180, fingerprint: "configured-sam3" },
    frames: {}, tracking_mode: "sam3", sam3: { action: "frame", frame: 0, direction: "both", seed_token: "" },
};
configuredWidgets[0].value = JSON.stringify(sam3Document);
configuredWidgets[1].value = "sam3";
configuredNode.onConfigure({});
assert.equal(configuredEditor.state.mode, "sam_point", "configured SAM3 mode selects click preview");
assert.equal(configuredEditor.state.grabcut, true);
assert.equal(configuredEditor.state.refineMethod, "sam3");
assert.match(configuredEditor.status.textContent, /общей маске/);

// Direct mode switches use the same visible status/tool contract.
const configuredTracking = find(configuredEditor.root, (item) => item.tagName === "SELECT" && item.attributes["aria-label"] === "Трекинг");
configuredTracking.value = "optical_flow"; configuredTracking.dispatchEvent({ type: "change" });
assert.equal(configuredEditor.state.mode, "brush");
assert.equal(configuredEditor.state.grabcut, false);
assert.doesNotMatch(configuredEditor.status.textContent, /SAM3/);
configuredTracking.value = "sam3"; configuredTracking.dispatchEvent({ type: "change" });
assert.equal(configuredEditor.state.mode, "sam_point");
assert.match(configuredEditor.status.textContent, /общей маске/);

// Modern-web-guidance & zero-H3 UI checks.
const allButtons = [];
(function walkUi(item) {
    if (!item) return;
    if (item.tagName === "BUTTON") allButtons.push(item);
    for (const child of item.children || []) walkUi(child);
})(configuredEditor.root);
assert.equal(allButtons.some((item) => /h3/i.test(`${item.textContent} ${item.attributes?.["aria-label"] || ""} ${item.title || ""}`)), false, "no button exposes H3 labels");
assert.ok(button(configuredEditor.root, "Запустить граф"), "queue button uses generic graph label");

const toolToolbar = find(configuredEditor.root, (item) => item.className === "ths-mask-tools");
assert.equal(toolToolbar.attributes.role, "toolbar");
assert.equal(toolToolbar.attributes["aria-orientation"], "vertical");
assert.equal(toolToolbar.attributes.focusgroup, "toolbar wrap block");
assert.ok(find(toolToolbar, (item) => item.attributes?.role === "separator"));
assert.equal(button(toolToolbar, "Отменить").attributes["aria-pressed"], undefined, "action buttons omit aria-pressed");
assert.equal(button(toolToolbar, "Кисть").attributes["aria-pressed"], "false");
assert.equal(button(toolToolbar, "Перо").attributes["aria-pressed"], "false");
assert.equal(button(toolToolbar, "Полигон"), null);
const verticalButtons = [button(toolToolbar, "Выбрать объект"), button(toolToolbar, "Добавить к объекту"), button(toolToolbar, "Исключить из объекта"), button(toolToolbar, "Кисть")];
assert.equal(verticalButtons.filter((item) => item.tabIndex === 0).length, 1, "roving tabindex keeps one stop in tab order");
let arrowDownFocused = false;
let arrowUpFocused = false;
verticalButtons[1].focus = () => { arrowDownFocused = true; };
verticalButtons[0].focus = () => { arrowUpFocused = true; };
toolToolbar.dispatchEvent({ type: "keydown", key: "ArrowDown", target: verticalButtons[0], preventDefault() {}, stopPropagation() {} });
assert.equal(arrowDownFocused, true);
assert.equal(verticalButtons[1].tabIndex, 0);
assert.equal(verticalButtons[0].tabIndex, -1);
toolToolbar.dispatchEvent({ type: "keydown", key: "ArrowUp", target: verticalButtons[1], preventDefault() {}, stopPropagation() {} });
assert.equal(arrowUpFocused, true);
assert.equal(verticalButtons[0].tabIndex, 0);
assert.equal(verticalButtons[1].tabIndex, -1);

// IME composing guards prevent premature Enter submission on numeric inputs.
const directFrameInput = find(configuredEditor.root, (item) => item.tagName === "INPUT" && item.attributes?.["aria-label"] === "Номер кадра");
configuredEditor.state.frameCount = 5;
configuredEditor.setFrame(0, { load: false });
directFrameInput.value = "4";
directFrameInput.dispatchEvent({ type: "keydown", key: "Enter", isComposing: true, preventDefault() {}, stopPropagation() {} });
assert.equal(configuredEditor.state.frame, 0, "isComposing ignores Enter");
directFrameInput.dispatchEvent({ type: "keydown", key: "Enter", keyCode: 229, isComposing: false, preventDefault() {}, stopPropagation() {} });
assert.equal(configuredEditor.state.frame, 0, "keyCode 229 fallback ignores Enter");
directFrameInput.dispatchEvent({ type: "keydown", key: "Enter", keyCode: 13, isComposing: false, preventDefault() {}, stopPropagation() {} });
assert.equal(configuredEditor.state.frame, 3, "non-composing Enter commits frame");

const sam3Dialog = find(configuredEditor.root, (item) => item.tagName === "DIALOG");
const sam3DialogTitle = find(sam3Dialog, (item) => item.tagName === "STRONG");
assert.equal(sam3Dialog.attributes["aria-labelledby"], sam3DialogTitle.id);
assert.equal(sam3Dialog.attributes.closedby, "closerequest");
assert.ok(configuredEditor.root.attributes["data-instance"]);
const frameRange = find(configuredEditor.root, (item) => item.tagName === "INPUT" && item.attributes?.type === "range" || item.type === "range");
assert.match(frameRange.attributes["aria-valuetext"], /Кадр|%|px/);

// Hover-only tooltips & layout declutter checks: no permanent inline hint blocks.
assert.equal(find(configuredEditor.root, (item) => item.tagName === "DETAILS"), null, "permanent details accordion removed");
const visibleModeCopy = find(configuredEditor.root, (item) => String(item.className || "").includes("ths-mask-section-copy"));
assert.ok(String(visibleModeCopy.className).includes("ths-sr-only"), "mode hint is visually hidden outside layout flow");
const hiddenQueueHint = find(configuredEditor.root, (item) => String(item.className || "").includes("ths-mask-queue-hint"));
assert.ok(String(hiddenQueueHint.className).includes("ths-sr-only"), "queue hint is visually hidden outside layout flow");
assert.ok(String(configuredEditor.status.className).includes("ths-sr-only"), "status live region preserves textContent outside layout flow");
assert.ok(button(configuredEditor.root, "Управление и горячие клавиши"), "header help button exposes keyboard tips");
assert.ok(button(configuredEditor.root, "О режиме выделения"), "selection info button exposes mode hint");

const sharedTooltip = find(configuredEditor.root, (item) => item.attributes?.role === "tooltip" && item.className === "ths-mask-tooltip");
assert.ok(sharedTooltip, "shared hover tooltip element exists");
assert.ok(["hint", "auto"].includes(sharedTooltip.attributes.popover));
const tooltipBody = find(sharedTooltip, (item) => item.className === "ths-mask-tooltip-body");

const penBtn = button(toolToolbar, "Перо");
configuredEditor.root.dispatchEvent({ type: "focusin", target: penBtn });
assert.equal(sharedTooltip.attributes["data-open"], "true", "focusing trigger opens Top-Layer tooltip");
assert.match(tooltipBody.textContent, /кривая Безье/);
assert.equal(penBtn.attributes["aria-describedby"], sharedTooltip.id);
assert.equal(penBtn.title, "", "native title suppressed while custom tooltip is active");
configuredEditor.root.dispatchEvent({ type: "keydown", key: "Escape", preventDefault() {}, stopPropagation() {} });
assert.equal(sharedTooltip.attributes["data-open"], undefined, "Escape closes open tooltip first");
assert.match(penBtn.title, /кривая Безье/, "native title restored when tooltip closes");

const queueBtn = button(configuredEditor.root, "Запустить граф");
const queueWrap = find(configuredEditor.root, (item) => item.className === "ths-tip-wrap" && item.children.includes(queueBtn));
assert.ok(queueWrap, "queue action button wrapped so disabled hover still shows tooltip");
configuredEditor.root.dispatchEvent({ type: "pointerover", target: queueWrap });
await new Promise((resolve) => setTimeout(resolve, 250));
assert.equal(sharedTooltip.attributes["data-open"], undefined, "tooltip waits past brief pointer pass before opening");
await new Promise((resolve) => setTimeout(resolve, 460));
assert.equal(sharedTooltip.attributes["data-open"], "true");
assert.match(tooltipBody.textContent, /Сначала рассчитайте кадр SAM3/);

configuredEditor.destroy();

console.log("universal_mask_editor_regressions: PASS");
