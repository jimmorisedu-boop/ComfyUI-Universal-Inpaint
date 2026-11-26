/* global document, window, Image, HTMLElement, IntersectionObserver */
import { createCropControls } from "./crop_window.js";
import { createMaskPostprocessor, effectiveCropExtent } from "./mask_postprocess.js";
import { previewPrompt } from "./preview_prompt.js";
import {
    MAX_MASK_LAYERS,
    activeLayerOf,
    blendLabel,
    blendSymbol,
    ensureDocumentLayers,
    isDefaultLayerName,
    layerHasContent,
    layerShortLabel,
    layerTypeLabel,
    nextBlendMode,
    nextLayerId,
    nextLayerName,
    projectLayerToDocument,
    shouldSerializeLayers,
    syncActiveLayerToList,
} from "./mask_layers.js";

// Universal mask editor for ComfyUI.  This file intentionally has no framework
// dependency: the same module is loaded by ComfyUI and by the standalone
// harness.  The UI is a small VFX workbench rather than a dashboard: frame
// navigation and the image stay primary, while paint settings stay compact.

let comfyApp = null;
let comfyApi = null;
try {
    ({ app: comfyApp } = await import("../../scripts/app.js"));
    ({ api: comfyApi } = await import("../../scripts/api.js"));
} catch (_) {
    // Standalone harness / unit DOM: Comfy is intentionally absent.
}

const NODE_CLASS = "THSUniversalMaskEditor";
const MAX_EVENTS = 4096;
const MAX_POINTS_PER_STROKE = 4096;
const MAX_TOTAL_POINTS = 65536;
const MAX_PREVIEWS = 48;
const MAX_HISTORY = 80;
const EDITOR_BUILD = "20261126-mask-editor-live-edges";
const TOOLTIP_SHOW_DELAY_MS = 650;
const TOOLTIP_WARM_DELAY_MS = 180;
const SAM3_FRAME_PREVIEW = "sam3_frame_v1";
const MAX_SAM3_OBJECTS = 8;
let nextEditorInstance = 0;

// A bounded, local flight recorder. Callers pass only explicit diagnostic
// fields: never the prompt, mask document, image URLs or image buffers.
function diagnosticText(value) {
    return String(value ?? "").replace(/data:[^\s"']+/gi, "[media omitted]")
        .replace(/(?:https?:\/\/|file:\/\/)[^\s"']+/gi, "[url omitted]")
        .replace(/[A-Za-z]:[\\/][^\r\n"']+/g, "[path omitted]").slice(0, 600);
}

export function createMaskDiagnostics({ nodeId, snapshot, storage = null, limit = 400 } = {}) {
    const key = `ths-mask-debug-v1:${nodeId}`;
    const session = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    let entries = [], timer = null, disposed = false;
    try {
        const saved = JSON.parse(storage?.getItem(key) || "null");
        if (saved?.schema === 1 && Array.isArray(saved.events)) entries = saved.events.slice(-limit);
    } catch (_) { /* Private mode / full storage must not affect editing. */ }
    function report() {
        return { schema: 1, build: EDITOR_BUILD, node_id: nodeId, session, exported_at: new Date().toISOString(),
            privacy: "Local diagnostics only; no frames, mask strokes, prompts or video filenames.",
            events: clone(entries) };
    }
    function flush() {
        if (timer) clearTimeout(timer); timer = null;
        try { storage?.setItem(key, JSON.stringify(report())); } catch (_) { /* Best effort. */ }
    }
    function record(event, details = {}) {
        if (disposed) return;
        try {
            entries.push({ at: new Date().toISOString(), session, event, details, state: snapshot?.() || {} });
            if (entries.length > limit) entries.splice(0, entries.length - limit);
            if (storage && !timer) timer = setTimeout(flush, 500);
        } catch (_) { /* Diagnostics must never break editor input. */ }
    }
    function dispose() { flush(); disposed = true; }
    return { record, report, flush, dispose };
}

const COLORS = {
    // Follow the active ComfyUI palette, including user-defined and light themes.
    page: "var(--bg-color, #202020)",
    panel: "var(--comfy-menu-bg, #171718)",
    panel2: "var(--comfy-menu-secondary-bg, #303030)",
    input: "var(--comfy-input-bg, #222)",
    border: "var(--border-color, #4e4e4e)",
    borderSoft: "var(--border-color, #4e4e4e)",
    text: "var(--input-text, #ddd)",
    muted: "var(--descrip-text, #999)",
    dim: "var(--descrip-text, #999)",
    accent: "var(--primary-background, var(--content-bg, #4e4e4e))",
    accentStrong: "var(--primary-background, var(--border-color, #6e6e6e))",
    accentText: "var(--primary-foreground, var(--content-fg, #fff))",
    selected: "var(--content-bg, #4e4e4e)",
    hover: "var(--content-hover-bg, #222)",
    mask: "#4fe38b",
    maskSoft: "rgba(79,227,139,.34)",
    warning: "var(--warning-text, #d69b48)",
    danger: "var(--error-text, #ff4444)",
};

function findWidget(node, name) {
    return (node?.widgets || []).find((widget) => widget.name === name);
}

function widgetValue(node, name, fallback) {
    const value = findWidget(node, name)?.value;
    return value === undefined ? fallback : value;
}

function setWidget(node, name, value) {
    const widget = findWidget(node, name);
    if (!widget) return false;
    widget.value = value;
    widget.callback?.(value);
    node.graph?.setDirtyCanvas?.(true, true);
    return true;
}

function hideTechnicalWidget(node, name) {
    const widget = findWidget(node, name);
    if (!widget) return false;
    widget.hidden = true;
    if (!widget.options || typeof widget.options !== "object") widget.options = {};
    widget.options.hidden = true;
    return true;
}

function number(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp(value, low = 0, high = 1) {
    return Math.max(low, Math.min(high, value));
}

function roundCoord(value) {
    return Number(clamp(value, 0, 1).toFixed(6));
}

function roundOffset(value) {
    return Number(clamp(value, -1, 1).toFixed(6));
}

function penAnchorHasHandles(anchor) {
    return Boolean(anchor && (anchor.inX || anchor.inY || anchor.outX || anchor.outY));
}

function penPathHasHandles(anchors) {
    return Array.isArray(anchors) && anchors.some(penAnchorHasHandles);
}

function canClosePenPath(anchors) {
    return Array.isArray(anchors) && (anchors.length >= 3 || (anchors.length >= 2 && penPathHasHandles(anchors)));
}

export function flattenPenPath(anchors, { closed = true, maxPoints = MAX_POINTS_PER_STROKE } = {}) {
    if (!Array.isArray(anchors) || !anchors.length || maxPoints < 1) return [];
    if (anchors.length === 1) return [[roundCoord(anchors[0].x), roundCoord(anchors[0].y)]];
    const segmentCount = closed ? anchors.length : anchors.length - 1;
    let curvedCount = 0;
    let straightCount = 0;
    for (let index = 0; index < segmentCount; index += 1) {
        const a = anchors[index];
        const b = anchors[(index + 1) % anchors.length];
        if (a.outX || a.outY || b.inX || b.inY) curvedCount += 1;
        else straightCount += 1;
    }
    const perCurveCap = curvedCount
        ? Math.max(4, Math.floor(Math.max(4, maxPoints - straightCount - (closed ? 0 : 1)) / curvedCount))
        : 0;
    const points = [];
    for (let index = 0; index < segmentCount; index += 1) {
        const a = anchors[index];
        const b = anchors[(index + 1) % anchors.length];
        if (!(a.outX || a.outY || b.inX || b.inY)) {
            points.push([roundCoord(a.x), roundCoord(a.y)]);
            continue;
        }
        const p0x = number(a.x, 0), p0y = number(a.y, 0);
        const p1x = p0x + number(a.outX, 0), p1y = p0y + number(a.outY, 0);
        const p3x = number(b.x, 0), p3y = number(b.y, 0);
        const p2x = p3x + number(b.inX, 0), p2y = p3y + number(b.inY, 0);
        const controlLen = Math.hypot(p1x - p0x, p1y - p0y)
            + Math.hypot(p2x - p1x, p2y - p1y)
            + Math.hypot(p3x - p2x, p3y - p2y);
        const steps = clamp(Math.ceil(controlLen * 48), 8, Math.min(32, perCurveCap));
        for (let step = 0; step < steps; step += 1) {
            const t = step / steps;
            const u = 1 - t;
            const x = u * u * u * p0x + 3 * u * u * t * p1x + 3 * u * t * t * p2x + t * t * t * p3x;
            const y = u * u * u * p0y + 3 * u * u * t * p1y + 3 * u * t * t * p2y + t * t * t * p3y;
            points.push([roundCoord(x), roundCoord(y)]);
        }
    }
    if (!closed) {
        const tail = anchors.at(-1);
        points.push([roundCoord(tail.x), roundCoord(tail.y)]);
    }
    return points.slice(0, maxPoints);
}

function fallbackPlainClone(value, seen = new WeakMap()) {
    if (value === null || typeof value !== "object") return value;
    if (seen.has(value)) return seen.get(value);
    if (Array.isArray(value)) {
        const copy = [];
        seen.set(value, copy);
        for (let index = 0; index < value.length; index += 1) copy[index] = fallbackPlainClone(value[index], seen);
        return copy;
    }
    const copy = {};
    seen.set(value, copy);
    for (const [key, item] of Object.entries(value)) {
        if (item !== undefined && typeof item !== "function") copy[key] = fallbackPlainClone(item, seen);
    }
    return copy;
}

function clone(value) {
    if (value === null || typeof value !== "object") return value;
    if (typeof structuredClone === "function") {
        try { return structuredClone(value); } catch (_) { /* Fallback for host proxies or non-cloneable fields. */ }
    }
    return fallbackPlainClone(value);
}

function isImeComposing(event) {
    return Boolean(event?.isComposing || event?.keyCode === 229);
}

function replaceChildrenSafe(element, ...nodes) {
    if (!element) return;
    if (typeof element.replaceChildren === "function") {
        element.replaceChildren(...nodes);
        return;
    }
    element.textContent = "";
    if (Array.isArray(element.children)) element.children.length = 0;
    if (nodes.length) element.append?.(...nodes);
}

function syncSliderMetrics(input, valueText = "") {
    if (!input) return;
    input.style?.setProperty?.("--attr-min", String(input.min ?? 0));
    input.style?.setProperty?.("--attr-max", String(input.max ?? 100));
    input.style?.setProperty?.("--control-value", String(input.value ?? 0));
    if (valueText) input.setAttribute?.("aria-valuetext", String(valueText));
}

function collectToolbarButtons(root) {
    const items = [];
    const walk = (node) => {
        if (!node || typeof node !== "object") return;
        if (node.tagName === "BUTTON") { items.push(node); return; }
        if (node.style?.display === "none") return;
        for (const child of node.children || []) walk(child);
    };
    walk(root);
    return items.filter((item) => item.style?.display !== "none" && !item.disabled);
}

function installToolbarRovingTabindex(toolbar, orientation = "horizontal") {
    if (!toolbar) return { refresh() {} };
    const nativeFocusGroup = typeof HTMLElement !== "undefined" && "focusGroup" in HTMLElement.prototype;
    if (nativeFocusGroup) return { refresh() {} };

    const sync = (preferred = null) => {
        const items = collectToolbarButtons(toolbar);
        if (!items.length) return;
        const target = (preferred && items.includes(preferred))
            ? preferred
            : items.find((item) => item.attributes?.["aria-pressed"] === "true" || item.getAttribute?.("aria-pressed") === "true")
            || items.find((item) => item.tabIndex === 0)
            || items[0];
        for (const item of items) item.tabIndex = item === target ? 0 : -1;
    };

    toolbar.addEventListener("focusin", (event) => {
        const target = event?.target;
        if (target?.tagName !== "BUTTON") return;
        for (const item of collectToolbarButtons(toolbar)) item.tabIndex = item === target ? 0 : -1;
    });

    toolbar.addEventListener("keydown", (event) => {
        if (isImeComposing(event)) return;
        const items = collectToolbarButtons(toolbar);
        if (!items.length) return;
        const current = items.indexOf(event.target);
        if (current < 0) return;
        const rtl = typeof getComputedStyle === "function" && typeof Element !== "undefined" && toolbar instanceof Element && getComputedStyle(toolbar).direction === "rtl";
        const nextKey = orientation === "vertical" ? "ArrowDown" : (rtl ? "ArrowLeft" : "ArrowRight");
        const prevKey = orientation === "vertical" ? "ArrowUp" : (rtl ? "ArrowRight" : "ArrowLeft");
        let nextIndex = -1;
        if (event.key === nextKey) nextIndex = (current + 1) % items.length;
        else if (event.key === prevKey) nextIndex = (current - 1 + items.length) % items.length;
        else if (event.key === "Home") nextIndex = 0;
        else if (event.key === "End") nextIndex = items.length - 1;
        if (nextIndex < 0) return;
        event.preventDefault?.();
        event.stopPropagation?.();
        const nextItem = items[nextIndex];
        for (const item of items) item.tabIndex = item === nextItem ? 0 : -1;
        nextItem.focus?.();
    });

    sync();
    return { refresh: sync };
}

function widgetSnapshot(node) {
    const value = widgetValue(node, "mask_data", "{}");
    if (value && typeof value === "object") {
        try { return JSON.stringify(value); } catch (_) { return ""; }
    }
    return String(value ?? "");
}

function explicitAnchors(documentState) {
    const values = Array.isArray(documentState?.anchors)
        ? documentState.anchors
        : Array.isArray(documentState?.keyframes)
            ? documentState.keyframes
            : Array.isArray(documentState?.keys) ? documentState.keys : [];
    return [...new Set(values.map((item) => {
        const frame = item && typeof item === "object" ? item.frame : item;
        const parsed = Number(frame);
        return Number.isFinite(parsed) ? Math.max(0, Math.round(parsed)) : null;
    }).filter((frame) => frame !== null))].sort((a, b) => a - b);
}

function replaceDocument(target, next) {
    for (const key of Object.keys(target)) delete target[key];
    const value = next && typeof next === "object" && !Array.isArray(next)
        ? clone(next) : { version: 1, frames: {} };
    Object.assign(target, value);
    if (!target.frames || typeof target.frames !== "object" || Array.isArray(target.frames)) target.frames = {};
}

function readDocument(node) {
    const raw = widgetValue(node, "mask_data", "{}");
    if (raw && typeof raw === "object") return clone(raw);
    if (typeof raw !== "string" || !raw.trim()) return { version: 1, frames: {} };
    try {
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" && !Array.isArray(parsed)
            ? parsed : { version: 1, frames: {} };
    } catch (_) {
        return { version: 1, frames: {} };
    }
}

function parseEvents(bucket) {
    if (!bucket) return [];
    if (Array.isArray(bucket)) return bucket.filter((event) => event && typeof event === "object");
    if (Array.isArray(bucket.events)) return bucket.events.filter((event) => event && typeof event === "object");
    const result = [];
    for (const [key, type] of [["strokes", "stroke"], ["rects", "rect"], ["rectangles", "rect"], ["polygons", "polygon"]]) {
        for (const event of (Array.isArray(bucket[key]) ? bucket[key] : [])) result.push({ ...event, type });
    }
    return result;
}

function sourceInfo(meta) {
    const source = meta?.source;
    if (!source || typeof source !== "object") return null;
    const frames = Math.round(number(source.frames ?? source.frame_count, 0));
    const width = Math.round(number(source.width, 0));
    const height = Math.round(number(source.height, 0));
    return frames > 0 && width > 0 && height > 0
        ? { frames, width, height, fingerprint: String(source.fingerprint || "") }
        : null;
}

function sourceSummary(source) {
    const value = sourceInfo({ source });
    if (!value) return "неизвестный источник";
    return `${value.frames} кадр. · ${value.width}×${value.height}`;
}

function sourceMismatchReason(previous, current, explicit = "") {
    const reason = String(explicit || "").toLowerCase();
    if (["frames", "width", "height", "fingerprint"].includes(reason)) return reason;
    if (!previous || !current) return "fingerprint";
    if (previous.frames !== current.frames) return "frames";
    if (previous.width !== current.width) return "width";
    if (previous.height !== current.height) return "height";
    if (previous.fingerprint && current.fingerprint && previous.fingerprint !== current.fingerprint) return "fingerprint";
    return "fingerprint";
}

function sourceMismatchLabel(reason) {
    return ({ frames: "число кадров", width: "ширина", height: "высота", fingerprint: "содержимое клипа" })[reason] || "источник";
}

function refinementMessage(error) {
    const raw = String(error || "").replace(/^\s*\d+\s*:\s*/, "").trim();
    if (/SAM3/i.test(raw)) return raw;
    if (/unavailable/i.test(raw)) return "Уточнение недоступно для этого источника";
    if (/no separable foreground|separable foreground|grab.?cut/i.test(raw)) return "Не удалось отделить объект от фона";
    return raw || "Не удалось уточнить объект";
}

function safeMode(node, documentState = null) {
    const widget = findWidget(node, "tracking_mode");
    const configured = widget?.value;
    const persisted = documentState?.tracking_mode;
    // Explicit widget/document values preserve every legacy workflow.  Only
    // a genuinely empty document with no configured mode gets the new SAM3
    // default; an old document containing frames/anchors keeps Optical Flow.
    const mode = configured !== undefined && configured !== null && String(configured).trim()
        ? String(configured)
        : persisted !== undefined && persisted !== null && String(persisted).trim()
            ? String(persisted)
            : (documentState && (Object.keys(documentState.frames || {}).length || explicitAnchors(documentState).length)
                ? "optical_flow" : "sam3");
    return ["sam3", "optical_flow", "static", "keyframes"].includes(mode) ? mode : "sam3";
}

function makeSeedToken() {
    // Tokens are opaque to the UI/backend.  Keep generation available in
    // offline harnesses that do not expose crypto.randomUUID().
    try { if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID(); } catch (_) { /* best effort */ }
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

function sam3Direction(value) {
    return ["forward", "backward", "both"].includes(String(value)) ? String(value) : "both";
}

function imageUrl(value, api = null) {
    if (typeof value === "string") return value;
    if (!value || typeof value !== "object") return "";
    if (typeof value.data_url === "string") return value.data_url;
    if (typeof value.url === "string") return value.url;
    if (!value.filename) return "";
    const params = new URLSearchParams();
    params.set("filename", String(value.filename));
    if (value.subfolder) params.set("subfolder", String(value.subfolder));
    if (value.type) params.set("type", String(value.type));
    const path = `/view?${params.toString()}`;
    if (typeof api?.apiURL === "function") return api.apiURL(path);
    if (typeof api?.apiURL === "string") return `${api.apiURL.replace(/\/$/, "")}${path}`;
    return path;
}

function createElement(tag, styles = {}, className = "") {
    const element = document.createElement(tag);
    if (className) element.className = className;
    Object.assign(element.style, styles);
    return element;
}

function button(label, kind = "neutral") {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = label;
    element.className = `ths-mask-button ths-mask-button-${kind}`;
    element.setAttribute("aria-label", label);
    return element;
}

// Lucide static 0.468.0 (ISC/MIT); see lucide-LICENSE.txt.
// Pinned local SVGs keep the editor usable offline and match ComfyUI iconography.
const ICONS = {
    "sam_point": '<circle cx="12" cy="12" r="6"/><path d="M12 2v4m0 12v4M2 12h4m12 0h4"/><circle cx="12" cy="12" r="1"/>',
    "sam_add": '<circle cx="12" cy="12" r="9"/><path d="M8 12h8m-4-4v8"/>',
    "sam_remove": '<circle cx="12" cy="12" r="9"/><path d="M8 12h8"/>',
    "brush": "<path d=\"m9.06 11.9 8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08\" />\n  <path d=\"M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z\" />",
    "erase": "<path d=\"m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21\" />\n  <path d=\"M22 21H7\" />\n  <path d=\"m5 11 9 9\" />",
    "rect": "<rect width=\"18\" height=\"18\" x=\"3\" y=\"3\" rx=\"2\" />",
    "pen": "<path d=\"M15.707 21.293a1 1 0 0 1-1.414 0l-1.586-1.586a1 1 0 0 1 0-1.414l5.586-5.586a1 1 0 0 1 1.414 0l1.586 1.586a1 1 0 0 1 0 1.414z\" />\n  <path d=\"m18 13-1.375-6.874a1 1 0 0 0-.746-.776L3.235 2.028a1 1 0 0 0-1.207 1.207L5.35 15.879a1 1 0 0 0 .776.746L13 18\" />\n  <path d=\"m2.3 2.3 7.286 7.286\" />\n  <circle cx=\"11\" cy=\"11\" r=\"2\" />",
    "expand": "<path d=\"M8 3H5a2 2 0 0 0-2 2v3\" />\n  <path d=\"M21 8V5a2 2 0 0 0-2-2h-3\" />\n  <path d=\"M3 16v3a2 2 0 0 0 2 2h3\" />\n  <path d=\"M16 21h3a2 2 0 0 0 2-2v-3\" />",
    "close": "<path d=\"M18 6 6 18\" />\n  <path d=\"m6 6 12 12\" />",
    "fit": "<rect x=\"3\" y=\"3\" width=\"18\" height=\"18\" rx=\"2\" />\n  <circle cx=\"8.5\" cy=\"8.5\" r=\"1.5\" />\n  <path d=\"m21 15-5-5L5 21\" />",
    "download": "<path d=\"M12 3v12m-5-5 5 5 5-5M5 16v4h14v-4\"/>",
    "minus": "<path d=\"M5 12h14\" />",
    "plus": "<path d=\"M5 12h14\" />\n  <path d=\"M12 5v14\" />",
    "undo": "<path d=\"M9 14 4 9l5-5\" />\n  <path d=\"M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11\" />",
    "redo": "<path d=\"m15 14 5-5-5-5\" />\n  <path d=\"M20 9H9.5A5.5 5.5 0 0 0 4 14.5A5.5 5.5 0 0 0 9.5 20H13\" />",
    "first": "<polygon points=\"19 20 9 12 19 4 19 20\" />\n  <line x1=\"5\" x2=\"5\" y1=\"19\" y2=\"5\" />",
    "last": "<polygon points=\"5 4 15 12 5 20 5 4\" />\n  <line x1=\"19\" x2=\"19\" y1=\"5\" y2=\"19\" />",
    "back": "<path d=\"m15 18-6-6 6-6\" />",
    "next": "<path d=\"m9 18 6-6-6-6\" />",
    "play": "<polygon points=\"6 3 20 12 6 21 6 3\" />",
    "pause": "<rect x=\"14\" y=\"4\" width=\"4\" height=\"16\" rx=\"1\" />\n  <rect x=\"6\" y=\"4\" width=\"4\" height=\"16\" rx=\"1\" />",
    "back10": "<path d=\"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8\" />\n  <path d=\"M3 3v5h5\" />",
    "next10": "<path d=\"M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8\" />\n  <path d=\"M21 3v5h-5\" />",
    "previousKey": "<path d=\"m11 17-5-5 5-5\" />\n  <path d=\"m18 17-5-5 5-5\" />",
    "nextKey": "<path d=\"m6 17 5-5-5-5\" />\n  <path d=\"m13 17 5-5-5-5\" />",
    "info": "<circle cx=\"12\" cy=\"12\" r=\"10\" />\n  <path d=\"M12 16v-4\" />\n  <path d=\"M12 8h.01\" />",
    "help": "<circle cx=\"12\" cy=\"12\" r=\"10\" />\n  <path d=\"M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3\" />\n  <path d=\"M12 17h.01\" />"
};

function assignTooltipMetadata(element, tooltip = "", { side = "" } = {}) {
    if (!element) return;
    const clean = String(tooltip ?? "").trim();
    if (element.dataset) {
        element.dataset.tooltip = clean;
        if (side) element.dataset.tipSide = side;
        if (element.dataset.thsTipActive !== "true") {
            if (clean) element.title = clean;
            else element.removeAttribute?.("title");
        }
    } else if (clean) {
        element.title = clean;
    } else {
        element.removeAttribute?.("title");
    }
}

function setButtonIcon(element, name, label, tooltip = label) {
    element.setAttribute("aria-label", label);
    assignTooltipMetadata(element, tooltip);
    if (element.dataset?.icon === name) return;
    const artwork = document.createElement("span");
    artwork.setAttribute("aria-hidden", "true");
    artwork.style.pointerEvents = "none";
    artwork.style.display = "inline-flex";
    artwork.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false">${ICONS[name]}</svg>`;
    const children = [artwork];
    if (name === "back10" || name === "next10") {
        const amount = document.createElement("span");
        amount.textContent = "10"; amount.setAttribute("aria-hidden", "true");
        Object.assign(amount.style, { fontSize: "9px", lineHeight: "1", position: "absolute", right: "3px", bottom: "3px", background: COLORS.panel2, padding: "1px", borderRadius: "2px" });
        children.push(amount);
    }
    replaceChildrenSafe(element, ...children);
    if (element.dataset) element.dataset.icon = name;
}

function iconButton(label, name, tooltip = label, kind = "neutral") {
    const element = button(label, kind);
    element.className += " ths-mask-icon-button";
    setButtonIcon(element, name, label, tooltip);
    return element;
}

function labelControl(labelText, control, readout = null) {
    const wrapper = createElement("label", {}, "ths-mask-label");
    const text = document.createElement("span");
    text.textContent = labelText;
    wrapper.append(text, control);
    if (readout) wrapper.append(readout);
    return wrapper;
}

function addStyles(root, instanceKey = "editor") {
    const scopeSelector = `.ths-mask-editor[data-instance="${instanceKey}"]`;
    const viewAnchor = `--ths-view-${instanceKey}`;
    const toolAnchor = `--ths-tool-${instanceKey}`;
    const tipAnchor = `--ths-tip-${instanceKey}`;
    const style = document.createElement("style");
    style.textContent = `
      .ths-mask-editor { color-scheme: dark; accent-color: ${COLORS.accentStrong}; font: 13px/1.45 Inter, system-ui, sans-serif; color: ${COLORS.text}; container: mask-editor / inline-size; }
      .ths-mask-editor .ths-sr-only { position: absolute !important; inline-size: 1px; block-size: 1px; margin: -1px; padding: 0; border: 0; overflow: hidden; clip-path: inset(50%); white-space: nowrap; pointer-events: none; }
      .ths-mask-editor .ths-mask-button { min-inline-size: 24px; min-block-size: 36px; min-height: 36px; padding: 8px 12px; border: 1px solid color-mix(in oklab, ${COLORS.border} 72%, transparent); border-radius: 7px; background: color-mix(in oklab, ${COLORS.panel2} 88%, ${COLORS.panel}); color: ${COLORS.text}; cursor: pointer; font: 500 13px/1.3 Inter, system-ui, sans-serif; white-space: nowrap; transition: background-color 140ms ease-out, border-color 140ms ease-out, color 140ms ease-out, box-shadow 140ms ease-out, scale 120ms ease-out; }
      .ths-mask-editor .ths-mask-button-primary { background: ${COLORS.accent}; color: ${COLORS.accentText}; border-color: transparent; }
      .ths-mask-editor .ths-mask-button-danger { color: ${COLORS.danger}; border-color: color-mix(in oklab, ${COLORS.danger} 28%, ${COLORS.border}); }
      .ths-mask-editor .ths-mask-button[aria-pressed="true"] { background: color-mix(in oklab, ${COLORS.accentStrong} 20%, ${COLORS.panel2}); border-color: color-mix(in oklab, ${COLORS.accentStrong} 58%, transparent); color: ${COLORS.text}; box-shadow: inset 0 1px 0 color-mix(in oklab, white 10%, transparent); }
      @media (hover: hover) {
        .ths-mask-editor button:hover:not(:disabled) { background: color-mix(in oklab, ${COLORS.text} 8%, ${COLORS.panel2}); border-color: color-mix(in oklab, ${COLORS.border} 82%, white 18%); }
        .ths-mask-editor .ths-mask-button-primary:hover:not(:disabled) { background: color-mix(in oklab, ${COLORS.accent} 86%, white); border-color: transparent; box-shadow: 0 2px 10px color-mix(in oklab, ${COLORS.accentStrong} 28%, transparent); }
        .ths-mask-editor .ths-mask-button-danger:hover:not(:disabled) { background: color-mix(in oklab, ${COLORS.danger} 14%, ${COLORS.panel2}); border-color: color-mix(in oklab, ${COLORS.danger} 52%, transparent); }
      }
      @media (prefers-reduced-motion: no-preference) {
        .ths-mask-editor button:active:not(:disabled) { scale: 0.97; }
      }
      .ths-mask-editor button:disabled { opacity: .45; cursor: default; }
      .ths-mask-editor :is(button,input,select,summary):focus-visible { outline: 2px solid ${COLORS.accentStrong}; outline-offset: 2px; }
      .ths-mask-editor :is(input,select) { accent-color: ${COLORS.accentStrong}; font: inherit; }
      .ths-mask-editor :is(input[type=number],select) { background: ${COLORS.input}; color: ${COLORS.text}; border: 1px solid color-mix(in oklab, ${COLORS.border} 72%, transparent); border-radius: 6px; min-height: 34px; padding: 6px 8px; box-sizing: border-box; min-width: 0; }
      .ths-mask-editor input[type="range"] {
        --slider-track-size: 6px;
        --slider-thumb-size: 16px;
        --slider-track-radius: 999px;
        --slider-thumb-radius: 50%;
        --slider-track-color: #3a3a40;
        --slider-fill-color: ${COLORS.mask};
        --slider-thumb-color: var(--slider-fill-color);
        --slider-direction: to right;
        --attr-min: attr(min type(<number>), 0);
        --attr-max: attr(max type(<number>), 100);
        --control-value: attr(value type(<number>), 0);
        --slider-progress: calc(progress(var(--control-value), var(--attr-min), var(--attr-max)) * 100%);
        appearance: none;
        inline-size: 100%;
        block-size: 28px;
        margin: 0;
        padding: 0;
        border: 0;
        border-radius: 0;
        background: transparent;
        cursor: pointer;
      }
      .ths-mask-editor input[type="range"]:dir(rtl) { --slider-direction: to left; }
      @supports not (opacity: progress(1, 0, 100)) {
        .ths-mask-editor input[type="range"] {
          --slider-progress: calc((var(--control-value) - var(--attr-min)) / max(var(--attr-max) - var(--attr-min), 0.0001) * 100%);
        }
      }
      .ths-mask-editor input[type="range"]::-webkit-slider-runnable-track {
        block-size: var(--slider-track-size);
        border: 0;
        border-radius: var(--slider-track-radius);
        background: linear-gradient(var(--slider-direction), var(--slider-fill-color) var(--slider-progress), var(--slider-track-color) var(--slider-progress));
      }
      .ths-mask-editor input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        inline-size: var(--slider-thumb-size);
        block-size: var(--slider-thumb-size);
        margin-block-start: calc((var(--slider-track-size) - var(--slider-thumb-size)) / 2);
        border: 0;
        border-radius: var(--slider-thumb-radius);
        background: var(--slider-thumb-color);
      }
      .ths-mask-editor input[type="range"]::-moz-range-track {
        block-size: var(--slider-track-size);
        border: 0;
        border-radius: var(--slider-track-radius);
        background: var(--slider-track-color);
      }
      .ths-mask-editor input[type="range"]::-moz-range-progress {
        block-size: var(--slider-track-size);
        border-radius: var(--slider-track-radius);
        background: var(--slider-fill-color);
      }
      .ths-mask-editor input[type="range"]::-moz-range-thumb {
        inline-size: var(--slider-thumb-size);
        block-size: var(--slider-thumb-size);
        border: 0;
        border-radius: var(--slider-thumb-radius);
        background: var(--slider-thumb-color);
      }
      .ths-mask-editor input[type="range"]:focus-visible { outline: 2px solid var(--slider-fill-color); outline-offset: 4px; }
      .ths-mask-editor .ths-mask-icon-button { display: inline-flex; align-items: center; justify-content: center; min-inline-size: 36px; min-block-size: 36px; width: 36px; min-width: 36px; height: 36px; min-height: 36px; padding: 7px; flex: none; position: relative; }
      .ths-mask-editor .ths-mask-label { display: inline-flex; align-items: center; gap: 8px; min-height: 32px; color: ${COLORS.muted}; font-size: 12px; line-height: 1.4; }
      .ths-mask-editor .ths-mask-header { display: flex; align-items: center; gap: 12px; block-size: 36px; min-block-size: 36px; flex-wrap: nowrap; }
      .ths-mask-editor .ths-mask-header > :nth-child(2) { margin-inline-start: auto; min-inline-size: 0; overflow: hidden; white-space: nowrap; max-inline-size: 50%; cursor: default; }
      .ths-mask-editor .ths-mask-source-readout { min-inline-size: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ths-mask-editor .ths-mask-header-actions { display: inline-flex; align-items: center; gap: 4px; flex: none; }
      .ths-mask-editor .ths-mask-header .ths-mask-icon-button,
      .ths-mask-editor .ths-mask-dialog-bar .ths-mask-icon-button { min-inline-size: 32px; min-block-size: 32px; width: 32px; min-width: 32px; height: 32px; min-height: 32px; padding: 6px; }
      .ths-mask-editor .ths-mask-header .ths-mask-header-btn,
      .ths-mask-editor .ths-mask-dialog-bar .ths-mask-header-btn { min-height: 28px; min-block-size: 28px; padding: 4px 10px; font-size: 11px; font-weight: 600; line-height: 1.2; white-space: nowrap; flex: none; }
      .ths-mask-editor .ths-mask-workbench { display: grid; grid-template-rows: minmax(0,1fr); gap: 0; height: 650px; min-width: 0; min-height: 0; }
      .ths-mask-editor .ths-mask-workspace { display: grid; grid-template-columns: minmax(0,1fr) 296px; gap: 16px; min-height: 0; min-width: 0; }
      .ths-mask-editor .ths-mask-viewer { display: grid; grid-template-rows: auto minmax(220px,1fr) auto; gap: 10px; min-width: 0; min-height: 0; }
      .ths-mask-editor .ths-mask-viewbar { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; min-width: 0; }
      .ths-mask-editor .ths-mask-viewgroup { anchor-scope: ${viewAnchor}; position: relative; display: inline-flex; align-items: center; gap: 2px; flex-wrap: nowrap; padding: 3px; border-radius: 8px; background: color-mix(in oklab, ${COLORS.panel} 72%, black); border: 1px solid color-mix(in oklab, ${COLORS.borderSoft} 55%, transparent); }
      ${scopeSelector} .ths-mask-viewgroup::before {
        content: "";
        position: absolute;
        position-anchor: ${viewAnchor};
        inset-block-start: anchor(self-start);
        inset-block-end: anchor(self-end);
        inset-inline-start: anchor(self-start);
        inset-inline-end: anchor(self-end);
        border-radius: 6px;
        background: color-mix(in oklab, ${COLORS.panel2} 84%, white 8%);
        border: 1px solid color-mix(in oklab, ${COLORS.border} 65%, white 14%);
        border-block-end: 2px solid ${COLORS.accentStrong};
        box-shadow: 0 1px 4px rgb(0 0 0 / 0.28);
        pointer-events: none;
        z-index: 0;
      }
      @media (prefers-reduced-motion: no-preference) {
        ${scopeSelector} .ths-mask-viewgroup::before { transition: inset 180ms ease-out; }
      }
      ${scopeSelector} .ths-mask-viewgroup .ths-mask-button {
        position: relative;
        z-index: 1;
        min-block-size: 28px;
        min-height: 28px;
        padding: 4px 11px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 6px;
        border: 1px solid transparent;
        background: transparent;
        color: ${COLORS.muted};
        box-shadow: none;
      }
      ${scopeSelector} .ths-mask-viewgroup .ths-mask-button[aria-pressed="true"] {
        anchor-name: ${viewAnchor};
        color: ${COLORS.text};
        font-weight: 600;
        background: transparent;
        border-color: transparent;
        box-shadow: none;
      }
      @media (hover: hover) {
        ${scopeSelector} .ths-mask-viewgroup .ths-mask-button:not([aria-pressed="true"]):hover:not(:disabled) {
          background: color-mix(in oklab, ${COLORS.text} 6%, transparent);
          border-color: transparent;
          color: ${COLORS.text};
        }
      }
      @supports not (position-anchor: ${viewAnchor}) {
        ${scopeSelector} .ths-mask-viewgroup::before { content: none; }
        ${scopeSelector} .ths-mask-viewgroup .ths-mask-button[aria-pressed="true"] {
          background: color-mix(in oklab, ${COLORS.panel2} 84%, white 8%);
          border: 1px solid color-mix(in oklab, ${COLORS.border} 65%, white 14%);
          border-block-end: 2px solid ${COLORS.accentStrong};
        }
      }
      .ths-mask-editor .ths-mask-viewbar .ths-mask-icon-button { min-inline-size: 32px; min-block-size: 32px; width: 32px; min-width: 32px; height: 32px; min-height: 32px; padding: 6px; }
      .ths-mask-editor .ths-mask-viewbar > :last-child { margin-inline-start: auto; }
      .ths-mask-editor .ths-mask-canvas-area { display: grid; grid-template-columns: 46px minmax(0,1fr); gap: 8px; min-width: 0; min-height: 0; }
      .ths-mask-editor .ths-mask-tools { inline-size: 46px; box-sizing: border-box; anchor-scope: ${toolAnchor}; position: relative; display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 5px 4px; align-self: start; border-radius: 8px; background: color-mix(in oklab, ${COLORS.panel} 72%, black); border: 1px solid color-mix(in oklab, ${COLORS.borderSoft} 55%, transparent); }
      ${scopeSelector} .ths-mask-tools::before {
        content: "";
        position: absolute;
        position-anchor: ${toolAnchor};
        inset-block-start: calc(anchor(self-start) + 6px);
        inset-block-end: calc(anchor(self-end) + 6px);
        inset-inline-start: calc(anchor(self-start) - 4px);
        width: 0;
        border-inline-start: 2.5px solid ${COLORS.accentStrong};
        border-radius: 99px;
        pointer-events: none;
        z-index: 2;
      }
      @media (prefers-reduced-motion: no-preference) {
        ${scopeSelector} .ths-mask-tools::before { transition: inset 180ms ease-out; }
      }
      .ths-mask-editor .ths-mask-toolbar-group { display: flex; flex-direction: column; align-items: center; gap: 4px; }
      ${scopeSelector} .ths-mask-tools .ths-mask-icon-button {
        min-inline-size: 34px;
        min-block-size: 34px;
        width: 34px;
        min-width: 34px;
        height: 34px;
        min-height: 34px;
        padding: 6px;
        border-radius: 6px;
        border: 1px solid transparent;
        background: transparent;
        color: ${COLORS.muted};
      }
      ${scopeSelector} .ths-mask-tools .ths-mask-icon-button[aria-pressed="true"] {
        anchor-name: ${toolAnchor};
        background: color-mix(in oklab, ${COLORS.accentStrong} 18%, ${COLORS.panel2});
        border-color: color-mix(in oklab, ${COLORS.accentStrong} 44%, transparent);
        color: ${COLORS.text};
        box-shadow: inset 0 1px 0 color-mix(in oklab, white 10%, transparent);
      }
      @media (hover: hover) {
        ${scopeSelector} .ths-mask-tools .ths-mask-icon-button:not([aria-pressed="true"]):hover:not(:disabled) {
          background: color-mix(in oklab, ${COLORS.text} 7%, transparent);
          border-color: transparent;
          color: ${COLORS.text};
        }
      }
      @supports not (position-anchor: ${toolAnchor}) {
        ${scopeSelector} .ths-mask-tools::before { content: none; }
        ${scopeSelector} .ths-mask-tools .ths-mask-icon-button[aria-pressed="true"] {
          border-inline-start: 2.5px solid ${COLORS.accentStrong};
        }
      }
      .ths-mask-editor .ths-mask-toolbar-sep { display: block; inline-size: 22px; block-size: 0; margin-block: 3px; border-top: 1px solid color-mix(in oklab, ${COLORS.borderSoft} 65%, transparent); }
      .ths-mask-editor .ths-canvas-wrap { position: relative; min-width: 0; min-height: 220px; background: ${COLORS.panel}; border: 1px solid color-mix(in oklab, ${COLORS.borderSoft} 65%, transparent); border-radius: 7px; overflow: hidden; content-visibility: auto; contain-intrinsic-size: auto none auto 340px; }
      .ths-mask-editor .ths-crop-outline { box-sizing: border-box; position: absolute; z-index: 2; display: none; pointer-events: none; border: 2px dashed #f5b951; border-radius: 2px; box-shadow: inset 0 0 0 1px rgb(13 18 22 / 0.55); }
      .ths-mask-editor .ths-crop-tag { position: absolute; left: 6px; top: 6px; padding: 2px 7px; border-radius: 4px; background: rgb(13 18 22 / 0.88); border: 1px solid rgb(245 185 81 / 0.65); color: #f5b951; font: 600 11px/1.25 Inter, system-ui, sans-serif; font-variant-numeric: tabular-nums; white-space: nowrap; }
      .ths-mask-editor .ths-crop-outline button { position: absolute; width: 16px; height: 16px; min-width: 16px; min-height: 16px; padding: 0; border: 2px solid #f5b951; border-radius: 3px; background: #141b20; pointer-events: auto; cursor: nwse-resize; touch-action: none; -webkit-user-select: none; user-select: none; box-shadow: 0 1px 4px rgb(0 0 0 / 0.6); }
      .ths-mask-editor .ths-crop-nw { left: -2px; top: -2px; }
      .ths-mask-editor .ths-crop-se { right: -2px; bottom: -2px; }
      .ths-mask-editor .ths-mask-inspector .ths-crop-mode { display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center; gap: 8px; min-height: 30px; }
      .ths-mask-editor .ths-mask-inspector .ths-crop-mode select { width: 100%; min-height: 30px; padding: 4px 8px; }
      .ths-mask-editor .ths-crop-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; min-width: 0; }
      .ths-mask-editor .ths-crop-field { display: flex; flex-direction: column; gap: 4px; min-width: 0; color: ${COLORS.muted}; font-size: 11px; line-height: 1.2; }
      .ths-mask-editor .ths-crop-input-wrap { display: flex; align-items: center; gap: 4px; padding-right: 8px; border-radius: 6px; border: 1px solid color-mix(in oklab, ${COLORS.border} 72%, transparent); background: ${COLORS.input}; transition: border-color 140ms ease-out, opacity 140ms ease-out; }
      .ths-mask-editor .ths-crop-input-wrap:focus-within { outline: 2px solid ${COLORS.accentStrong}; outline-offset: -1px; }
      .ths-mask-editor .ths-crop-input-wrap:has(input:disabled) { opacity: 0.45; }
      .ths-mask-editor .ths-crop-input-wrap input[type="number"] { border: 0; background: transparent; width: 100%; min-width: 0; min-height: 30px; padding: 4px 2px 4px 8px; font-variant-numeric: tabular-nums; }
      .ths-mask-editor .ths-crop-input-wrap input[type="number"]:focus-visible { outline: none; }
      .ths-mask-editor .ths-crop-unit { color: ${COLORS.muted}; font-size: 11px; flex: none; user-select: none; }
      .ths-mask-editor .ths-crop-presets { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 4px; }
      .ths-mask-editor .ths-crop-presets .ths-crop-preset { min-height: 24px; min-block-size: 24px; padding: 2px 4px; font-size: 11px; line-height: 1.2; border-radius: 5px; font-variant-numeric: tabular-nums; display: inline-flex; align-items: center; justify-content: center; }
      .ths-mask-editor .ths-crop-presets .ths-crop-preset[aria-pressed="true"] { background: color-mix(in oklab, ${COLORS.mask} 18%, ${COLORS.panel2}); border-color: color-mix(in oklab, ${COLORS.mask} 60%, transparent); color: ${COLORS.mask}; }
      .ths-mask-editor .ths-crop-readout { min-height: 26px; display: flex; align-items: center; padding: 4px 8px; border-radius: 5px; background: color-mix(in oklab, ${COLORS.input} 70%, black); border: 1px solid color-mix(in oklab, ${COLORS.borderSoft} 55%, transparent); color: ${COLORS.muted}; font-size: 11px; line-height: 1.3; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; box-sizing: border-box; }
      .ths-mask-editor .ths-layers-list { display: flex; flex-direction: column; gap: 4px; }
      .ths-mask-editor .ths-layer-item { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: 4px; padding: 4px 5px; border-radius: 6px; border: 1px solid color-mix(in oklab, ${COLORS.borderSoft} 75%, transparent); background: color-mix(in oklab, ${COLORS.panel2} 65%, black); }
      .ths-mask-editor .ths-layer-item[data-active="true"] { border-color: color-mix(in oklab, ${COLORS.accentStrong} 80%, transparent); background: color-mix(in oklab, ${COLORS.accentStrong} 13%, ${COLORS.panel2}); }
      .ths-mask-editor .ths-layer-item[data-disabled="true"] { opacity: 0.55; }
      .ths-mask-editor .ths-layer-main { display: flex; align-items: center; gap: 6px; min-width: 0; border: 0; background: transparent; color: ${COLORS.text}; text-align: left; padding: 2px 4px; min-height: 24px; min-block-size: 24px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 500; }
      .ths-mask-editor .ths-layer-main:hover { background: color-mix(in oklab, ${COLORS.text} 6%, transparent); }
      .ths-mask-editor .ths-layer-dot { width: 6px; height: 6px; border-radius: 50%; background: ${COLORS.dim}; flex: none; }
      .ths-mask-editor .ths-layer-dot[data-has-mask="true"] { background: ${COLORS.mask}; }
      .ths-mask-editor .ths-layer-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ths-mask-editor .ths-layer-meta { color: ${COLORS.muted}; font-size: 10px; white-space: nowrap; margin-left: auto; }
      .ths-mask-editor .ths-layer-actions { display: inline-flex; align-items: center; gap: 3px; }
      .ths-mask-editor .ths-layer-icon,
      .ths-mask-editor .ths-layer-blend { min-width: 24px; min-height: 24px; min-block-size: 24px; padding: 0 5px; font-size: 11px; line-height: 1; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; }
      .ths-mask-editor .ths-layer-blend[data-blend="add"] { color: ${COLORS.mask}; border-color: color-mix(in oklab, ${COLORS.mask} 45%, transparent); }
      .ths-mask-editor .ths-layer-blend[data-blend="subtract"] { color: ${COLORS.danger}; border-color: color-mix(in oklab, ${COLORS.danger} 55%, transparent); }
      .ths-mask-editor .ths-layer-blend[data-blend="intersect"] { color: ${COLORS.warning}; border-color: color-mix(in oklab, ${COLORS.warning} 55%, transparent); }
      .ths-mask-editor .ths-layers-add { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px; }
      .ths-mask-editor .ths-layers-add .ths-layer-add-btn { min-height: 26px; min-block-size: 26px; padding: 3px 4px; font-size: 11px; font-weight: 500; line-height: 1.2; border-radius: 5px; white-space: nowrap; }
      .ths-mask-editor .ths-mask-alert-overlay { position: absolute; z-index: 4; inset-block-start: 8px; inset-inline: 8px; display: none; flex-direction: column; gap: 8px; padding: 10px 12px; border-radius: 8px; background: color-mix(in oklab, ${COLORS.panel} 94%, black); border: 1px solid ${COLORS.border}; box-shadow: 0 6px 20px rgb(0 0 0 / 0.45); color: ${COLORS.text}; font-size: 12px; line-height: 1.4; pointer-events: auto; }
      .ths-mask-editor .ths-mask-alert-overlay[data-state="warn"] { border-color: color-mix(in oklab, ${COLORS.warning} 58%, transparent); }
      .ths-mask-editor .ths-mask-alert-overlay[data-state="error"] { border-color: color-mix(in oklab, ${COLORS.danger} 60%, transparent); color: ${COLORS.danger}; }
      .ths-mask-editor .ths-mask-alert-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
      .ths-mask-editor .ths-mask-alert-text { margin: 0; flex: 1 1 auto; }
      .ths-mask-editor .ths-mask-alert-dismiss { display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; min-width: 24px; min-height: 24px; min-block-size: 24px; padding: 0; line-height: 1; flex: 0 0 auto; }
      .ths-mask-editor .ths-mask-alert-dismiss svg { width: 13px; height: 13px; display: block; }
      .ths-mask-editor .ths-mask-progress { position: absolute; z-index: 3; inset: 0; display: none; place-items: center; padding: 16px; pointer-events: none; }
      .ths-mask-editor .ths-mask-progress-card { box-sizing: border-box; display: grid; gap: 8px; width: min(100%, 300px); padding: 14px 16px; border: 1px solid ${COLORS.border}; border-radius: 8px; background: ${COLORS.panel}; box-shadow: 0 3px 16px #0008; }
      .ths-mask-editor .ths-mask-progress-title { color: ${COLORS.text}; font-size: 13px; font-weight: 650; }
      .ths-mask-editor .ths-mask-progress-track { width: 100%; height: 8px; accent-color: ${COLORS.mask}; }
      @supports not (accent-color: auto) {
        .ths-mask-editor .ths-mask-progress-track { -webkit-appearance: none; -moz-appearance: none; appearance: none; border: 0; border-radius: 4px; background: #3a3a40; }
        .ths-mask-editor .ths-mask-progress-track::-webkit-progress-bar { background-color: #3a3a40; border-radius: 4px; }
        .ths-mask-editor .ths-mask-progress-track::-webkit-progress-value { background-color: ${COLORS.mask}; border-radius: 4px; }
        .ths-mask-editor .ths-mask-progress-track::-moz-progress-bar { background-color: ${COLORS.mask}; border-radius: 4px; }
      }
      .ths-mask-editor .ths-mask-progress-caption { color: ${COLORS.muted}; font-size: 12px; font-variant-numeric: tabular-nums; }
      .ths-mask-editor .ths-mask-progress-time { color: ${COLORS.muted}; font-size: 12px; font-variant-numeric: tabular-nums; }
      .ths-mask-editor .ths-mask-inspector { min-width: 0; min-height: 0; overflow: hidden; background: ${COLORS.panel}; border: 1px solid ${COLORS.borderSoft}; border-radius: 7px; display: flex; flex-direction: column; }
      .ths-mask-editor .ths-mask-inspector-body { flex: 1 1 0; min-height: 0; overflow-y: auto; overscroll-behavior: contain; scrollbar-width: thin; scrollbar-color: color-mix(in oklab, ${COLORS.text} 22%, transparent) transparent; scrollbar-gutter: stable; padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; }
      .ths-mask-editor .ths-mask-inspector-body::-webkit-scrollbar { width: 6px; height: 6px; }
      .ths-mask-editor .ths-mask-inspector-body::-webkit-scrollbar-track { background: transparent; }
      .ths-mask-editor .ths-mask-inspector-body::-webkit-scrollbar-thumb { background: color-mix(in oklab, ${COLORS.text} 18%, transparent); border-radius: 999px; }
      .ths-mask-editor .ths-mask-inspector-body::-webkit-scrollbar-thumb:hover { background: color-mix(in oklab, ${COLORS.text} 32%, transparent); }
      .ths-mask-editor .ths-mask-export { flex: none; padding: 12px 14px; border-top: 1px solid ${COLORS.borderSoft}; }
      .ths-mask-editor .ths-mask-section { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
      .ths-mask-editor .ths-mask-section + .ths-mask-section { border-top: 1px solid ${COLORS.borderSoft}; padding-block-start: 10px; }
      .ths-mask-editor .ths-mask-section-head { display: flex; align-items: center; justify-content: space-between; gap: 6px; min-inline-size: 0; }
      .ths-mask-editor .ths-mask-section-title { margin: 0; color: ${COLORS.text}; font-size: 13px; line-height: 1.4; font-weight: 650; }
      .ths-mask-editor .ths-mask-info-btn { min-inline-size: 24px !important; min-block-size: 24px !important; width: 24px !important; min-width: 24px !important; height: 24px !important; min-height: 24px !important; padding: 3px !important; border: 1px solid transparent !important; background: transparent !important; color: ${COLORS.muted} !important; border-radius: 6px !important; }
      .ths-mask-editor .ths-mask-info-btn svg { width: 15px; height: 15px; }
      .ths-mask-editor .ths-mask-inspector .ths-mask-label { display: grid; grid-template-columns: minmax(0,1fr) auto auto; gap: 8px; }
      .ths-mask-editor .ths-mask-mode { display: flex !important; flex-direction: column; align-items: stretch; gap: 6px !important; }
      .ths-mask-editor .ths-mask-mode select { width: 100%; }
      .ths-mask-editor .ths-mask-settings { display: flex; flex-direction: column; gap: 8px; width: 100%; }
      .ths-mask-editor .ths-mask-setting-row { display: grid; grid-template-columns: 48px minmax(64px, 1fr) 54px auto; align-items: center; gap: 6px; width: 100%; min-width: 0; color: ${COLORS.muted}; font-size: 12px; line-height: 1.4; }
      .ths-mask-editor .ths-mask-setting-row input[type="range"] { width: 100%; min-width: 0; }
      .ths-mask-editor .ths-mask-setting-row input[type="number"] { width: 54px; min-width: 0; font-variant-numeric: tabular-nums; }
      .ths-mask-editor .ths-tip-wrap { display: block; width: 100%; }
      .ths-mask-editor .ths-tip-wrap > button:disabled { pointer-events: none; }
      .ths-mask-editor .ths-mask-actions { display: flex; flex-direction: column; gap: 8px; }
      .ths-mask-editor .ths-mask-actions .ths-mask-button-primary { background: ${COLORS.panel2}; color: ${COLORS.text}; border-color: ${COLORS.border}; }
      .ths-mask-editor .ths-mask-export button[data-next-action=true] { background: ${COLORS.accent}; color: ${COLORS.accentText}; border-color: transparent; }
      .ths-mask-editor .ths-mask-export > .ths-tip-wrap,
      .ths-mask-editor .ths-mask-export > .ths-mask-button { margin-block-start: 8px; }
      .ths-mask-editor .ths-mask-actions button, .ths-mask-editor .ths-mask-export button { width: 100%; align-items: center; justify-content: center; }
      .ths-mask-editor .ths-mask-range { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; color: ${COLORS.muted}; font-size: 12px; }
      .ths-mask-editor .ths-mask-range input { width: 70px; }
      .ths-mask-editor .ths-mask-range > :last-child { width: 100%; }
      .ths-mask-editor .ths-mask-timeline { display: flex; flex-direction: column; gap: 10px; padding: 12px; border: 1px solid ${COLORS.borderSoft}; border-radius: 6px; background: ${COLORS.panel}; min-width: 0; }
      .ths-mask-editor .ths-mask-transport { display: flex; align-items: center; flex-wrap: wrap; gap: 5px; }
      .ths-mask-editor .ths-mask-transport-tools { display: inline-flex; align-items: center; flex-wrap: wrap; gap: 5px; }
      .ths-mask-editor .ths-mask-transport .ths-mask-icon-button { min-inline-size: 32px; min-block-size: 32px; width: 32px; min-width: 32px; height: 32px; min-height: 32px; padding: 6px; }
      .ths-mask-editor .ths-mask-transport .ths-mask-icon-button svg { width: 18px; height: 18px; }
      .ths-mask-editor .ths-mask-timeline-meta { display: flex; align-items: center; justify-content: space-between; gap: 10px; white-space: nowrap; color: ${COLORS.muted}; font: 500 11px/1.2 Inter, Arial, sans-serif; font-variant-numeric: tabular-nums; }
      .ths-mask-editor .ths-mask-key-readout { min-inline-size: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ths-mask-editor .ths-mask-secondary { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; width: 100%; }
      .ths-mask-editor .ths-mask-secondary .ths-mask-button { width: 100%; display: inline-flex; align-items: center; justify-content: center; font-size: 11.5px; padding: 6px 8px; min-block-size: 32px; min-height: 32px; border-radius: 6px; overflow: hidden; text-overflow: ellipsis; }
      .ths-mask-editor .ths-mask-secondary .ths-mask-btn-wide { grid-column: 1 / -1; }
      .ths-mask-editor .ths-mask-secondary .ths-mask-button-danger { background: color-mix(in oklab, ${COLORS.danger} 8%, ${COLORS.panel2}); border-color: color-mix(in oklab, ${COLORS.danger} 34%, transparent); }
      @media (hover: hover) {
        .ths-mask-editor .ths-mask-secondary .ths-mask-button-danger:hover:not(:disabled) { background: color-mix(in oklab, ${COLORS.danger} 16%, ${COLORS.panel2}); border-color: color-mix(in oklab, ${COLORS.danger} 58%, transparent); }
      }
      .ths-mask-editor .ths-key-marker { width: 14px; height: 22px; min-height: 22px; padding: 0; border: 0; border-radius: 2px; background: transparent; position: absolute; top: -4px; transform: translateX(-50%); }
      .ths-mask-editor .ths-key-marker::before { content: ''; display: block; width: 3px; height: 14px; margin: 4px auto; border-radius: 1px; background: ${COLORS.mask}; }
      ${scopeSelector} [data-tooltip][data-ths-tip-active="true"] { anchor-name: ${tipAnchor}; }
      ${scopeSelector} .ths-mask-tooltip {
        position: fixed;
        padding: 0;
        border: 0;
        background: transparent;
        color: ${COLORS.text};
        max-inline-size: 270px;
        overflow: visible;
        z-index: 20;
        pointer-events: auto;
        opacity: 0;
        transition-property: opacity, transform, display, overlay;
        transition-duration: 120ms;
        transition-timing-function: ease-out;
        transition-behavior: allow-discrete;
      }
      ${scopeSelector} .ths-mask-tooltip:is(:popover-open, .\:popover-open, [data-open="true"]) { display: block; opacity: 1; }
      @supports not selector(:popover-open) {
        ${scopeSelector} .ths-mask-tooltip:not(.\:popover-open):not([data-open="true"]) { display: none; }
      }
      @starting-style {
        ${scopeSelector} .ths-mask-tooltip:is(:popover-open, [data-open="true"]) { opacity: 0; transform: translateY(2px); }
      }
      ${scopeSelector} .ths-mask-tooltip-body {
        position: relative;
        padding: 6px 10px;
        border-radius: 7px;
        background: color-mix(in oklab, ${COLORS.panel} 94%, black);
        border: 1px solid color-mix(in oklab, ${COLORS.border} 78%, white 10%);
        box-shadow: 0 8px 24px rgb(0 0 0 / 0.48);
        color: ${COLORS.text};
        font: 500 12px/1.4 Inter, system-ui, sans-serif;
        overflow-wrap: break-word;
      }
      @supports (position-area: block-start) {
        ${scopeSelector} .ths-mask-tooltip {
          position-anchor: ${tipAnchor};
          inset: auto;
          margin: 0;
          position-area: block-start;
          position-try-fallbacks: flip-block, flip-inline, flip-block flip-inline;
          margin-block: 7px;
        }
        ${scopeSelector} .ths-mask-tooltip[data-tip-side="inline-end"] {
          position-area: inline-end;
          position-try-fallbacks: flip-inline, flip-block;
          margin-block: 0;
          margin-inline: 8px;
        }
      }
      @supports (container-type: anchored) {
        ${scopeSelector} .ths-mask-tooltip { container-type: anchored; }
        ${scopeSelector} .ths-mask-tooltip-body::before {
          content: "";
          position: absolute;
          inline-size: 7px;
          block-size: 7px;
          background: color-mix(in oklab, ${COLORS.panel} 94%, black);
          border-inline-end: 1px solid color-mix(in oklab, ${COLORS.border} 78%, white 10%);
          border-block-end: 1px solid color-mix(in oklab, ${COLORS.border} 78%, white 10%);
          transform: rotate(45deg);
          inset-block-end: -4px;
          inset-inline-start: calc(50% - 3.5px);
        }
        ${scopeSelector} .ths-mask-tooltip[data-tip-side="inline-end"] .ths-mask-tooltip-body::before {
          inset-block-end: auto;
          inset-block-start: calc(50% - 3.5px);
          inset-inline-start: -4px;
          transform: rotate(135deg);
        }
        @container anchored(fallback: flip-block) {
          ${scopeSelector} .ths-mask-tooltip-body::before {
            inset-block-end: auto;
            inset-block-start: -4px;
            transform: rotate(225deg);
          }
        }
        @container anchored(fallback: flip-inline) {
          ${scopeSelector} .ths-mask-tooltip[data-tip-side="inline-end"] .ths-mask-tooltip-body::before {
            inset-inline-start: auto;
            inset-inline-end: -4px;
            transform: rotate(-45deg);
          }
        }
      }
      .ths-mask-editor .ths-mask-dialog {
        position: fixed; inset: 0; margin: 0; color: ${COLORS.text}; background: ${COLORS.page}; border: 0; border-radius: 0;
        padding: 16px 20px; width: 100%; height: 100vh; height: 100dvh; max-width: none; max-height: none; box-sizing: border-box;
        overflow: hidden; box-shadow: none; opacity: 0;
        transition-property: opacity, transform, display, overlay;
        transition-duration: 0.22s;
        transition-timing-function: ease-out;
        transition-behavior: allow-discrete;
      }
      .ths-mask-editor .ths-mask-dialog[open] { display: flex; flex-direction: column; gap: 12px; opacity: 1; }
      .ths-mask-editor .ths-mask-dialog[data-closing] { opacity: 0; }
      @media (prefers-reduced-motion: no-preference) {
        .ths-mask-editor .ths-mask-dialog { transform: translateY(10px) scale(0.99); }
        .ths-mask-editor .ths-mask-dialog[open] { transform: none; }
        .ths-mask-editor .ths-mask-dialog[data-closing] { transform: translateY(8px) scale(0.99); }
      }
      @starting-style {
        .ths-mask-editor .ths-mask-dialog[open] { opacity: 0; transform: translateY(10px) scale(0.99); }
        .ths-mask-editor .ths-mask-dialog[open]::backdrop { opacity: 0; }
      }
      .ths-mask-editor .ths-mask-dialog:modal { overscroll-behavior: contain; }
      .ths-mask-editor .ths-mask-dialog::backdrop {
        background: rgba(0,0,0,.72); overflow: hidden; overscroll-behavior: contain; opacity: 0;
        transition-property: opacity, display, overlay; transition-duration: 0.22s; transition-behavior: allow-discrete;
      }
      .ths-mask-editor .ths-mask-dialog[open]::backdrop { opacity: 1; }
      .ths-mask-editor .ths-mask-dialog[data-closing]::backdrop { opacity: 0; }
      .ths-mask-editor .ths-mask-dialog-bar { display: flex; align-items: center; gap: 12px; min-height: 40px; flex: none; }
      .ths-mask-editor .ths-mask-dialog-bar > :nth-child(2) { margin-inline-start: auto; min-width: 0; overflow: hidden; white-space: nowrap; }
      .ths-mask-editor .ths-mask-dialog-inner { flex: 1 1 0; min-height: 0; min-width: 0; container: mask-editor / inline-size; }
      .ths-mask-editor .ths-mask-dialog-inner .ths-mask-workbench { height: 100%; grid-template-rows: minmax(0,1fr); }
      .ths-mask-editor .ths-mask-dialog-inner .ths-canvas-wrap { min-height: 80px; content-visibility: visible; }
      @supports (container-type: inline-size) {
        @container mask-editor (max-width: 960px) {
          .ths-mask-editor .ths-mask-workspace { grid-template-columns: minmax(0,1fr) 260px; gap: 12px; }
        }
        @container mask-editor (max-width: 760px) {
          .ths-mask-editor .ths-mask-workspace { grid-template-columns: minmax(0,1fr); }
          .ths-mask-editor .ths-mask-workbench { grid-template-rows: auto; height: auto; }
          .ths-mask-editor .ths-mask-viewer { min-height: 420px; }
          .ths-mask-editor .ths-mask-inspector { overflow: visible; }
          .ths-mask-editor .ths-mask-inspector-body { overflow: visible; flex: none; scrollbar-gutter: auto; }
          .ths-mask-editor .ths-mask-dialog-inner { overflow-y: auto; overscroll-behavior: contain; }
          .ths-mask-editor .ths-mask-dialog-inner .ths-mask-workbench { height: auto; grid-template-rows: auto; }
        }
      }
      @media (pointer: coarse) {
        .ths-mask-editor .ths-mask-button { min-inline-size: 44px; min-block-size: 44px; }
        .ths-mask-editor .ths-mask-viewbar .ths-mask-button { min-block-size: 40px; }
        .ths-mask-editor input[type="range"] { block-size: 40px; }
      }
      @media (forced-colors: active) {
        .ths-mask-editor .ths-mask-button { border: 1px solid ButtonBorder; color: ButtonText; background: ButtonFace; }
        .ths-mask-editor .ths-mask-button[aria-pressed="true"] { background: Highlight; color: HighlightText; border-color: Highlight; }
        ${scopeSelector} .ths-mask-viewgroup::before { border: 1px solid Highlight; border-block-end: 2px solid Highlight; }
        ${scopeSelector} .ths-mask-tools::before { border-inline-start-color: Highlight; }
        ${scopeSelector} .ths-mask-tooltip-body { background: Canvas; color: CanvasText; border-color: CanvasText; }
        .ths-mask-editor .ths-mask-toolbar-sep { border-top-color: ButtonBorder; }
        .ths-mask-editor .ths-mask-dialog { border: 1px solid CanvasText; }
        .ths-mask-editor :is(button,input,select,summary):focus-visible { outline: 2px solid CanvasText; }
      }
      @media (prefers-reduced-motion: reduce) {
        .ths-mask-editor button, ${scopeSelector} .ths-mask-tooltip { transition: none; transform: none; }
        .ths-mask-editor .ths-mask-dialog, .ths-mask-editor .ths-mask-dialog::backdrop { transition-duration: 0.08s; transform: none; }
      }
    `;
    root.append(style);
}

/** Install the editor on a Comfy node, or on a harness fake node. */
export function createUniversalMaskEditor(node, options = {}) {
    if (!node || node.__thsUniversalMaskEditor) return node?.__thsUniversalMaskEditor;
    const app = options.app || comfyApp;
    const queueApi = options.api || comfyApi;
    hideTechnicalWidget(node, "mask_data");
    hideTechnicalWidget(node, "tracking_mode");
    hideTechnicalWidget(node, "grow");
    hideTechnicalWidget(node, "feather");
    hideTechnicalWidget(node, "start_frame");
    hideTechnicalWidget(node, "end_frame");

    const instanceKey = `${String(node.id ?? "node").replace(/[^a-z0-9_-]/gi, "_")}-${++nextEditorInstance}`;
    const root = createElement("section", {
        display: "flex", flexDirection: "column", gap: "8px", padding: "12px",
        width: "100%", boxSizing: "border-box", color: COLORS.text, background: COLORS.page,
        border: `1px solid ${COLORS.borderSoft}`, borderRadius: "7px", fontFamily: "Inter, Arial, sans-serif",
    }, "ths-mask-editor");
    root.tabIndex = 0;
    root.setAttribute("aria-label", "Редактор маски видео");
    root.setAttribute("data-instance", instanceKey);
    addStyles(root, instanceKey);

    const header = createElement("div", { display: "flex", alignItems: "center", justifyContent: "space-between", gap: "8px", minHeight: "26px" });
    const title = createElement("div", { font: "600 14px/1.2 Inter, Arial, sans-serif", letterSpacing: "-.1px" });
    title.textContent = "Маска видео";
    const headerMeta = createElement("div", { display: "flex", alignItems: "center", gap: "7px", color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif", fontVariantNumeric: "tabular-nums" });
    const dirtyDot = createElement("span", { width: "7px", height: "7px", borderRadius: "50%", background: COLORS.dim, display: "inline-block", flex: "none" });
    const sourceReadout = createElement("span", {}, "ths-mask-source-readout");
    headerMeta.append(dirtyDot, sourceReadout);
    const newVideoButton = button("Новое видео");
    newVideoButton.className = `${newVideoButton.className || ""} ths-mask-header-btn`.trim();
    assignTooltipMetadata(newVideoButton, "Загрузить новое видео со входа images и начать чистую маску (остальной граф не запускается)");
    const helpButton = iconButton("Управление и горячие клавиши", "help", "Space — воспроизведение / пауза · ← / → — ±1 кадр (Shift ±10) · Home / End — границы · Ctrl+колесо — зум · Средняя кнопка или Alt+перетаскивание — панорама · Ctrl+Z / Ctrl+Shift+Z — отмена/повтор · Esc — отмена");
    const expand = iconButton("Развернуть", "expand", "Развернуть редактор");
    const debugDownload = iconButton("Скачать debug-лог", "download", "Скачать debug-лог · действия, ошибки и размеры редактора");
    const headerActions = createElement("div", { display: "flex", gap: "4px", alignItems: "center" }, "ths-mask-header-actions");
    headerActions.append(newVideoButton, helpButton, debugDownload, expand);
    header.append(title, headerMeta, headerActions);
    root.append(header);

    const status = createElement("div", {}, "ths-mask-status ths-sr-only");
    status.setAttribute("role", "status");
    const recoverSource = button("Начать маску для нового клипа", "danger");
    recoverSource.style.display = "none";
    recoverSource.style.alignSelf = "flex-start";
    const restorePrevious = button("Вернуть прежнюю маску");
    restorePrevious.style.display = "none";
    restorePrevious.style.alignSelf = "flex-start";
    const recoveryActions = createElement("div", { display: "none", flexWrap: "wrap", gap: "6px", alignItems: "center" }, "ths-mask-recovery-actions");
    recoveryActions.append(recoverSource, restorePrevious);
    const alertBanner = createElement("div", {}, "ths-mask-alert-overlay");
    alertBanner.setAttribute("role", "alert");
    const alertHead = createElement("div", {}, "ths-mask-alert-head");
    const alertText = createElement("div", {}, "ths-mask-alert-text");
    const dismissAlert = iconButton("Скрыть уведомление", "close", "Скрыть");
    dismissAlert.className = `${dismissAlert.className || ""} ths-mask-alert-dismiss`.trim();
    dismissAlert.style.display = "none";
    alertHead.append(alertText, dismissAlert);
    alertBanner.append(alertHead, recoveryActions);

    const canvasPlaceholder = createElement("span", { display: "none" });
    const canvasWrap = createElement("div", { position: "relative", width: "100%", height: "clamp(210px, 32vw, 340px)", minHeight: "190px", background: COLORS.page, borderRadius: "5px", overflow: "hidden", border: `1px solid ${COLORS.border}`, resize: "vertical", boxSizing: "border-box" }, "ths-canvas-wrap");
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", "Холст маски");
    canvas.tabIndex = 0;
    // Keep the bitmap's intrinsic size out of the flex layout. Otherwise a
    // redraw can resize its own container and shift/crop portrait footage.
    Object.assign(canvas.style, { position: "absolute", inset: "0", display: "block", width: "100%", height: "100%", cursor: "crosshair", touchAction: "none", userSelect: "none" });
    const empty = createElement("div", { position: "absolute", inset: "0", display: "grid", placeItems: "center", textAlign: "center", padding: "20px", boxSizing: "border-box", color: COLORS.dim, font: "500 12px/1.4 Inter, Arial, sans-serif", pointerEvents: "none" });
    empty.textContent = "Подготовьте превью, чтобы открыть кадры";
    const canvasBadge = createElement("div", { position: "absolute", left: "8px", top: "8px", padding: "3px 6px", borderRadius: "3px", background: COLORS.panel, color: COLORS.muted, font: "600 10px/1.2 Inter, Arial, sans-serif", pointerEvents: "none" });
    const brushCursor = createElement("div", { position: "absolute", display: "none", pointerEvents: "none", border: `1px solid ${COLORS.mask}`, borderRadius: "50%", transform: "translate(-50%, -50%)", boxSizing: "border-box", zIndex: "2" }, "ths-mask-brush-cursor");
    const progressOverlay = createElement("div", {}, "ths-mask-progress");
    const progressCard = createElement("div", {}, "ths-mask-progress-card");
    const progressTitle = createElement("div", {}, "ths-mask-progress-title");
    progressTitle.setAttribute("role", "status");
    const progressTrack = createElement("progress", {}, "ths-mask-progress-track");
    const progressCaption = createElement("div", {}, "ths-mask-progress-caption");
    const progressTime = createElement("div", {}, "ths-mask-progress-time");
    progressCard.append(progressTitle, progressTrack, progressCaption, progressTime);
    progressOverlay.append(progressCard);
    canvasWrap.append(canvas, empty, canvasBadge, brushCursor, progressOverlay, alertBanner);
    let progressTimer = null;
    root.append(status, canvasWrap, canvasPlaceholder);

    const viewRow = createElement("div", { display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "7px" });
    const viewButtons = {};
    const viewGroup = createElement("div", {}, "ths-mask-viewgroup");
    viewGroup.setAttribute("role", "toolbar");
    viewGroup.setAttribute("aria-orientation", "horizontal");
    viewGroup.setAttribute("aria-label", "Режим просмотра");
    viewGroup.setAttribute("focusgroup", "toolbar wrap inline");
    for (const [key, label, tip] of [
        ["overlay", "Наложение", "Исходное видео с полупрозрачной зелёной маской"],
        ["source", "Источник", "Только исходное видео без наложения маски"],
        ["mask", "Маска", "Чёрно-белая альфа-маска текущего кадра"],
    ]) {
        const control = button(label, "neutral");
        control.setAttribute("aria-pressed", key === "overlay" ? "true" : "false");
        control.dataset && (control.dataset.view = key);
        assignTooltipMetadata(control, tip);
        viewButtons[key] = control;
        viewGroup.append(control);
    }
    const opacity = document.createElement("input");
    opacity.type = "range"; opacity.min = "0"; opacity.max = "1"; opacity.step = ".01"; opacity.value = ".38"; opacity.style.width = "92px";
    opacity.setAttribute("aria-label", "Прозрачность маски");
    assignTooltipMetadata(opacity, "Прозрачность наложения зелёной маски поверх видео");
    syncSliderMetrics(opacity, "38%");
    const opacityReadout = createElement("span", { minWidth: "34px", textAlign: "right", color: COLORS.muted, fontSize: "11px", fontVariantNumeric: "tabular-nums" }); opacityReadout.textContent = "38%";
    viewRow.append(viewGroup, labelControl("Прозрачность", opacity, opacityReadout));
    root.append(viewRow);

    const toolPanel = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "6px", padding: "6px 0 2px", borderTop: `1px solid ${COLORS.borderSoft}` }, "ths-mask-tools");
    const toolLabel = createElement("span", { color: COLORS.dim, font: "600 10px/1.2 Inter, Arial, sans-serif", textTransform: "uppercase", letterSpacing: ".07em", marginRight: "2px" }); toolLabel.textContent = "Рисование";
    const tools = {};
    for (const [key, label, hint] of [
        ["sam_point", "Выбрать объект", "Клик по объекту добавляет его к общей маске на любом кадре."],
        ["sam_add", "Добавить к объекту", "Клик по пропущенной части выбранного объекта → уточнить предпросмотр."],
        ["sam_remove", "Исключить из объекта", "Клик по лишней области → исключить её из выделения SAM3."],
    ]) { tools[key] = iconButton(label, key, hint); tools[key].setAttribute("aria-pressed", "false"); assignTooltipMetadata(tools[key], hint, { side: "inline-end" }); toolPanel.append(tools[key]); }
    for (const [key, label, hint] of [
        ["brush", "Кисть", "Кисть: добавляет область в маску на текущем кадре"],
        ["erase", "Ластик", "Ластик: стирает маску на текущем кадре"],
        ["rect", "Прямоугольник", "Прямоугольник: протяните область для маски или рамку выделения"],
        ["pen", "Перо", "Перо: клик — угол, перетаскивание — кривая Безье; клик по первой точке, двойной клик или Enter — замкнуть; Backspace — шаг назад"],
    ]) { tools[key] = iconButton(label, key, hint); tools[key].setAttribute("aria-pressed", "false"); assignTooltipMetadata(tools[key], hint, { side: "inline-end" }); toolPanel.append(tools[key]); }
    const fit = iconButton("Вписать", "fit", "Вписать весь кадр · сбросить масштаб и положение");
    const zoomOut = iconButton("Уменьшить", "minus", "Уменьшить масштаб · Ctrl+колесо (или колесо в полном экране)");
    const zoomReadout = createElement("span", { minWidth: "42px", textAlign: "center", color: COLORS.muted, fontSize: "11px", fontVariantNumeric: "tabular-nums" }); zoomReadout.textContent = "100%";
    const zoomIn = iconButton("Увеличить", "plus", "Увеличить масштаб · Ctrl+колесо (или колесо в полном экране)");
    assignTooltipMetadata(zoomReadout, "Масштаб относительно вписанного кадра. Ctrl+колесо — зум; средняя кнопка или Alt+перетаскивание — панорама.");
    const zoomGroup = createElement("span", { display: "inline-flex", alignItems: "center", gap: "3px", marginLeft: "auto" }); zoomGroup.append(zoomOut, zoomReadout, zoomIn, fit);
    toolPanel.append(zoomGroup);
    root.append(toolPanel);

    const settingsRow = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "7px", padding: "4px 0 2px" });
    const brushInput = document.createElement("input"); brushInput.type = "range"; brushInput.min = ".002"; brushInput.max = ".2"; brushInput.step = ".001"; brushInput.value = ".035"; brushInput.style.width = "92px";
    brushInput.setAttribute("aria-label", "Размер кисти");
    assignTooltipMetadata(brushInput, "Диаметр кисти и ластика относительно короткой стороны кадра");
    syncSliderMetrics(brushInput, "3.5%");
    const brushReadout = createElement("span", { minWidth: "46px", textAlign: "right", color: COLORS.muted, fontSize: "11px", fontVariantNumeric: "tabular-nums" }); brushReadout.textContent = "3.5%";
    const brushLabel = labelControl("Размер", brushInput, brushReadout);
    settingsRow.append(brushLabel);
    const modeSelect = document.createElement("select");
    for (const [value, text] of [["sam3", "Объект · SAM3"], ["optical_flow", "Область · Optical Flow"], ["keyframes", "Ручные ключи"], ["static", "Неподвижная маска"]]) { const option = document.createElement("option"); option.value = value; option.textContent = text; modeSelect.append(option); }
    modeSelect.setAttribute("aria-label", "Трекинг");
    assignTooltipMetadata(modeSelect, "SAM3 — объект с двухэтапным расчётом; Optical Flow — область по движению; Ручные ключи; Неподвижная маска");
    const maskSettingsGroup = createElement("div", { display: "inline-flex", alignItems: "center", flexWrap: "wrap", gap: "12px" }, "ths-mask-settings");
    maskSettingsGroup.setAttribute("role", "group");
    maskSettingsGroup.setAttribute("aria-label", "Настройки маски");
    const modeLabel = labelControl("Режим", modeSelect);
    modeLabel.className += " ths-mask-mode";
    const directionSelect = document.createElement("select");
    for (const [value, text] of [["forward", "вперёд"], ["backward", "назад"], ["both", "оба"]]) { const option = document.createElement("option"); option.value = value; option.textContent = text; directionSelect.append(option); }
    directionSelect.setAttribute("aria-label", "Направление");
    assignTooltipMetadata(directionSelect, "Направление трекинга SAM3; диапазон кадров задаётся ниже");
    const directionLabel = labelControl("Направление", directionSelect);

    const maskSettingsInputs = {};
    const maskSettingsRanges = {};
    const maskSettingsDefaults = { grow: 8, feather: 4 };
    for (const [name, label, description] of [["grow", "Grow", "Расширение маски"], ["feather", "Feather", "Растушёвка края маски"]]) {
        const row = createElement("div", {}, "ths-mask-setting-row");
        const title = document.createElement("span");
        title.textContent = label;
        const range = document.createElement("input");
        range.type = "range"; range.min = "0"; range.max = "256"; range.step = "1";
        range.value = String(maskSettingsDefaults[name]);
        range.setAttribute("aria-label", `Ползунок ${label}`);
        assignTooltipMetadata(range, `${description}, 0–256 пикселей.`);
        syncSliderMetrics(range, `${maskSettingsDefaults[name]} px`);
        const input = document.createElement("input");
        input.type = "number"; input.min = "0"; input.max = "256"; input.step = "1";
        input.setAttribute("aria-label", label);
        assignTooltipMetadata(input, `${description}, 0–256 пикселей.`);
        const unit = createElement("span", { color: COLORS.muted, fontSize: "11px" }); unit.textContent = "px";
        maskSettingsInputs[name] = input;
        maskSettingsRanges[name] = range;
        row.append(title, range, input, unit);
        maskSettingsGroup.append(row);
    }
    viewRow.append(maskSettingsGroup);
    const refine = button("Выделять объект");
    assignTooltipMetadata(refine, "Прямоугольник вокруг объекта должен захватить объект и немного окружающего фона; после расчёта уточните кистью или ластиком.");
    refine.setAttribute("aria-pressed", "false");
    settingsRow.append(refine);
    const refineMethod = document.createElement("select");
    refineMethod.setAttribute("aria-label", "Метод выделения");
    assignTooltipMetadata(refineMethod, "Метод для новых прямоугольников. GrabCut — захватите немного фона; SAM3 — обведите объект близко к границам. Уже созданные маски сохраняют свой метод.");
    for (const [value, text] of [["grabcut", "GrabCut"], ["sam3", "SAM3"]]) {
        const option = document.createElement("option"); option.value = value; option.textContent = text; refineMethod.append(option);
    }
    const refineMethodLabel = labelControl("Метод", refineMethod);
    settingsRow.append(refineMethodLabel);
    root.append(settingsRow);

    const timeline = createElement("div", { display: "flex", flexDirection: "column", gap: "6px", paddingTop: "4px", borderTop: `1px solid ${COLORS.borderSoft}` }, "ths-mask-timeline");
    const navRow = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "4px" });
    const first = iconButton("Первый кадр", "first", "Первый кадр · Home");
    const back10 = iconButton("Назад на 10 кадров", "back10", "Назад на 10 кадров · Shift + ←"); back10.className += " ths-nav-wide";
    const back = iconButton("Предыдущий кадр", "back", "Предыдущий кадр · ←");
    const play = iconButton("Воспроизвести", "play", "Воспроизвести · Space", "primary");
    const next = iconButton("Следующий кадр", "next", "Следующий кадр · →");
    const next10 = iconButton("Вперёд на 10 кадров", "next10", "Вперёд на 10 кадров · Shift + →"); next10.className += " ths-nav-wide";
    const last = iconButton("Последний кадр", "last", "Последний кадр · End");
    const previousKeyButton = iconButton("Пред. ключ", "previousKey", "Предыдущий ключевой кадр"); previousKeyButton.className += " ths-nav-wide";
    const nextKeyButton = iconButton("След. ключ", "nextKey", "Следующий ключевой кадр"); nextKeyButton.className += " ths-nav-wide";
    const direct = document.createElement("input"); direct.type = "number"; direct.min = "1"; direct.value = "1"; direct.style.width = "64px"; direct.setAttribute("aria-label", "Номер кадра");
    assignTooltipMetadata(direct, "Перейти к указанному номеру кадра (Enter)");
    const frameReadout = createElement("span", { minWidth: "76px", whiteSpace: "nowrap", color: COLORS.text, font: "600 12px/1.2 Inter, Arial, sans-serif", fontVariantNumeric: "tabular-nums", marginLeft: "3px" });
    const timeReadout = createElement("span", { minWidth: "96px", textAlign: "right", whiteSpace: "nowrap", color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif", fontVariantNumeric: "tabular-nums", marginLeft: "auto" });
    const transportToolbar = createElement("div", {}, "ths-mask-transport-tools");
    transportToolbar.setAttribute("role", "toolbar");
    transportToolbar.setAttribute("aria-orientation", "horizontal");
    transportToolbar.setAttribute("aria-label", "Навигация по кадрам");
    transportToolbar.setAttribute("focusgroup", "toolbar wrap inline");
    transportToolbar.append(first, back10, back, play, next, next10, last, previousKeyButton, nextKeyButton);
    navRow.append(transportToolbar, direct, frameReadout, timeReadout);
    const speed = document.createElement("select"); speed.setAttribute("aria-label", "Скорость воспроизведения"); speed.style.width = "68px";
    assignTooltipMetadata(speed, "Скорость предпросмотра воспроизведения");
    for (const value of ["0.25", "0.5", "1", "1.5", "2"]) { const option = document.createElement("option"); option.value = value; option.textContent = `${value}×`; if (value === "1") option.selected = true; speed.append(option); }
    navRow.append(labelControl("", speed));
    timeline.append(navRow);

    const sliderWrap = createElement("div", { position: "relative", padding: "2px 4px 0" });
    const slider = document.createElement("input"); slider.type = "range"; slider.min = "0"; slider.max = "0"; slider.step = "1"; slider.value = "0"; slider.style.width = "100%"; slider.setAttribute("aria-label", "Лента кадров");
    assignTooltipMetadata(slider, "Перемотка по кадрам клипа; зелёные метки показывают ключевые кадры");
    const markerLayer = createElement("div", { position: "absolute", left: "9px", right: "9px", top: "0", height: "20px", pointerEvents: "none" });
    sliderWrap.append(slider, markerLayer);
    const timelineLine = createElement("div", {}, "ths-mask-timeline-meta");
    const rangeReadout = document.createElement("span");
    const keyReadout = createElement("span", {}, "ths-mask-key-readout");
    timelineLine.append(rangeReadout, keyReadout);
    timeline.append(sliderWrap, timelineLine);
    root.append(timeline);

    const actionRow = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "6px", paddingTop: "2px" });
    const undo = iconButton("Отменить", "undo", "Отменить · Ctrl+Z"); assignTooltipMetadata(undo, "Отменить · Ctrl+Z", { side: "inline-end" });
    const redo = iconButton("Повторить", "redo", "Повторить · Ctrl+Shift+Z / Ctrl+Y"); assignTooltipMetadata(redo, "Повторить · Ctrl+Shift+Z / Ctrl+Y", { side: "inline-end" });
    const clear = button("Очистить кадр"); assignTooltipMetadata(clear, "Удалить правки и выделение на текущем кадре");
    const copyPrevious = button("Скопировать предыдущий ключ"); assignTooltipMetadata(copyPrevious, "Скопировать маску с предыдущего ключевого кадра на текущий");
    copyPrevious.className += " ths-mask-btn-wide";
    const reset = button("Сбросить маску", "danger"); assignTooltipMetadata(reset, "Очистить всю маску клипа (можно отменить через Ctrl+Z)");
    actionRow.append(undo, redo, copyPrevious, clear, reset);
    root.append(actionRow);

    const rangeRow = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "7px", color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif" });
    const startInput = document.createElement("input"); startInput.type = "number"; startInput.min = "1"; startInput.value = "1"; startInput.style.width = "59px"; startInput.setAttribute("aria-label", "Начало диапазона");
    assignTooltipMetadata(startInput, "Первый кадр диапазона трекинга");
    const endInput = document.createElement("input"); endInput.type = "number"; endInput.min = "1"; endInput.value = "1"; endInput.style.width = "59px"; endInput.setAttribute("aria-label", "Конец диапазона");
    assignTooltipMetadata(endInput, "Последний кадр диапазона трекинга (пусто — до конца клипа)");
    const endInfinity = createElement("span", { color: COLORS.muted }); endInfinity.textContent = "до конца";
    const fpsReadout = document.createElement("span");
    const rangeLabel = createElement("span"); rangeLabel.textContent = "Кадры";
    const rangeDash = createElement("span"); rangeDash.textContent = "—";
    rangeRow.append(rangeLabel, startInput, rangeDash, endInput, endInfinity, fpsReadout);
    root.append(rangeRow);

    const wrapAction = (control, tip = "") => {
        const wrapper = createElement("span", {}, "ths-tip-wrap");
        assignTooltipMetadata(control, tip);
        assignTooltipMetadata(wrapper, tip);
        wrapper.append(control);
        return wrapper;
    };
    const actionButtons = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "7px", paddingTop: "2px" }, "ths-mask-actions");
    const prepare = button("Загрузить кадры", "primary");
    const prepareWrap = wrapAction(prepare, "Рассчитать маску и трекинг для просмотра всех кадров. Остальной граф не запускается.");
    const calculateFrame = button("Рассчитать кадр", "primary");
    const calculateFrameWrap = wrapAction(calculateFrame, "Рассчитать SAM3 только на выбранном кадре. Остальной граф и трекинг видео не запускаются.");
    const trackVideo = button("Отследить видео", "primary");
    const trackVideoWrap = wrapAction(trackVideo, "Отследить SAM3 по выбранному направлению и диапазону. Сначала рассчитайте кадр.");
    const cancelRequest = button("Отменить расчёт", "danger");
    const cancelRequestWrap = wrapAction(cancelRequest, "Отменить только свой запрос, если ComfyUI поддерживает адресную отмену.");
    cancelRequest.style.display = "none";
    cancelRequestWrap.style.display = "none";
    const queue = button("Запустить граф");
    const queueWrap = wrapAction(queue, "Запустить граф ComfyUI с готовой маской");
    const queueHint = createElement("span", {}, "ths-mask-queue-hint ths-sr-only"); queueHint.textContent = "Подготовка не запускает полный граф";
    actionButtons.append(prepareWrap, calculateFrameWrap, trackVideoWrap, cancelRequestWrap);
    root.append(actionButtons);

    // Keep every mask control in the primary workbench while showing all
    // guidance cleanly on hover/focus via the Top-Layer tooltip controller.
    const workbenchPlaceholder = createElement("span", { display: "none" });

    // A single DOM tree is shared by inline and fullscreen views. Only its
    // outer wrapper moves; controls retain position, state and event handlers.
    function restyle(element, className) {
        element.removeAttribute?.("style");
        if (className) element.className = className;
        return element;
    }
    function section(label, className = "", infoControl = null) {
        const element = createElement("section", {}, `ths-mask-section ${className}`);
        element.setAttribute("aria-label", label);
        const headingRow = createElement("div", {}, "ths-mask-section-head");
        const heading = createElement("h3", {}, "ths-mask-section-title");
        heading.textContent = label;
        headingRow.append(heading);
        if (infoControl) headingRow.append(infoControl);
        element.append(headingRow);
        return element;
    }
    restyle(header, "ths-mask-header");
    restyle(viewRow, "ths-mask-viewbar");
    restyle(toolPanel, "ths-mask-tools");
    restyle(canvasWrap, "ths-canvas-wrap");
    restyle(timeline, "ths-mask-timeline");
    restyle(navRow, "ths-mask-transport");
    restyle(actionRow, "ths-mask-secondary");
    restyle(rangeRow, "ths-mask-range");
    restyle(actionButtons, "ths-mask-actions");
    restyle(maskSettingsGroup, "ths-mask-settings");
    const modeInfoButton = iconButton("О режиме выделения", "info", "Слои маски можно комбинировать (+ добавить, − вычесть, ∩ пересечь). SAM3: прицел добавляет объекты; ＋ и − уточняют активный объект.");
    modeInfoButton.className += " ths-mask-info-btn";
    const selectionPanel = section("1. Слои маски", "", modeInfoButton);
    const layersContainer = createElement("div", {}, "ths-layers-list");
    layersContainer.setAttribute("role", "list");
    layersContainer.setAttribute("aria-label", "Слои маски");
    const layersAddRow = createElement("div", {}, "ths-layers-add");
    const addLayerButtons = {};
    for (const [layerType, shortLabel, tip] of [
        ["sam3", "+ SAM3", "Добавить слой SAM3 (выделение объекта по точкам/рамке)"],
        ["optical_flow", "+ Flow", "Добавить слой Optical Flow (трекинг области по движению)"],
        ["keyframes", "+ Ключи", "Добавить слой ручных ключей"],
        ["static", "+ Статика", "Добавить неподвижный слой маски"],
    ]) {
        const addBtn = button(shortLabel);
        addBtn.className = `${addBtn.className || ""} ths-layer-add-btn`.trim();
        addBtn.setAttribute("data-layer-type", layerType);
        assignTooltipMetadata(addBtn, tip);
        addLayerButtons[layerType] = addBtn;
        layersAddRow.append(addBtn);
    }
    const modeHint = createElement("p", {}, "ths-mask-section-copy ths-sr-only");
    const objectSelect = document.createElement("select");
    objectSelect.setAttribute("aria-label", "Активный объект SAM3");
    assignTooltipMetadata(objectSelect, "Выбор объекта SAM3 (до 8 независимых объектов)");
    const objectLabel = labelControl("Активный объект SAM3", objectSelect);
    const objectControls = createElement("div", { display: "none", flexWrap: "wrap", alignItems: "center", gap: "6px" });
    objectControls.append(objectLabel);
    selectionPanel.append(layersContainer, layersAddRow, modeLabel, modeHint, objectControls);
    const edgesPanel = section("2. Настройки маски");
    edgesPanel.append(maskSettingsGroup, brushLabel, refine, refineMethodLabel);
    const trackingPanel = section("3. Диапазон трекинга");
    trackingPanel.append(directionLabel, rangeRow);
    timelineLine.append(fpsReadout);
    const exportPanel = section("Расчёт", "ths-mask-export");
    exportPanel.append(actionButtons, queueWrap, queueHint);
    const editPanel = section("Правки кадра");
    editPanel.append(actionRow);
    const inspector = createElement("aside", {}, "ths-mask-inspector");
    inspector.setAttribute("aria-label", "Настройки маски и расчёт");
    const inspectorBody = createElement("div", {}, "ths-mask-inspector-body");
    inspectorBody.append(selectionPanel, edgesPanel, trackingPanel);
    inspector.append(inspectorBody, exportPanel);
    viewRow.append(zoomGroup);
    const samToolGroup = createElement("div", {}, "ths-mask-toolbar-group");
    samToolGroup.setAttribute("role", "group");
    samToolGroup.setAttribute("aria-label", "Инструменты SAM3");
    samToolGroup.append(tools.sam_point, tools.sam_add, tools.sam_remove);
    const samSeparator = createElement("span", {}, "ths-mask-toolbar-sep");
    samSeparator.setAttribute("role", "separator");
    samSeparator.setAttribute("aria-orientation", "horizontal");
    const drawToolGroup = createElement("div", {}, "ths-mask-toolbar-group");
    drawToolGroup.setAttribute("role", "group");
    drawToolGroup.setAttribute("aria-label", "Инструменты рисования");
    drawToolGroup.append(tools.brush, tools.erase, tools.rect, tools.pen);
    const historySeparator = createElement("span", {}, "ths-mask-toolbar-sep");
    historySeparator.setAttribute("role", "separator");
    historySeparator.setAttribute("aria-orientation", "horizontal");
    const historyToolGroup = createElement("div", {}, "ths-mask-toolbar-group");
    historyToolGroup.setAttribute("role", "group");
    historyToolGroup.setAttribute("aria-label", "История правок");
    historyToolGroup.append(undo, redo);
    replaceChildrenSafe(toolPanel, samToolGroup, samSeparator, drawToolGroup, historySeparator, historyToolGroup);
    toolPanel.setAttribute("role", "toolbar");
    toolPanel.setAttribute("aria-orientation", "vertical");
    toolPanel.setAttribute("aria-label", "Инструменты маски");
    toolPanel.setAttribute("focusgroup", "toolbar wrap block");
    const toolToolbarController = installToolbarRovingTabindex(toolPanel, "vertical");
    const viewToolbarController = installToolbarRovingTabindex(viewGroup, "horizontal");
    const transportToolbarController = installToolbarRovingTabindex(transportToolbar, "horizontal");
    const canvasArea = createElement("div", {}, "ths-mask-canvas-area");
    canvasArea.append(toolPanel, canvasWrap);
    const viewer = createElement("div", {}, "ths-mask-viewer");
    viewer.append(viewRow, canvasArea, timeline);
    const workspace = createElement("div", {}, "ths-mask-workspace");
    workspace.append(viewer, inspector);
    const workbench = createElement("div", {}, "ths-mask-workbench");
    workbench.append(workspace);
    // Remove obsolete empty row wrappers after moving their children.
    settingsRow.remove?.(); canvasPlaceholder.remove?.();
    root.append(workbenchPlaceholder, workbench);

    const tooltip = createElement("div", {}, "ths-mask-tooltip");
    tooltip.id = `ths-mask-tooltip-${instanceKey}`;
    tooltip.setAttribute("popover", "hint");
    tooltip.setAttribute("role", "tooltip");
    const tooltipBody = createElement("div", {}, "ths-mask-tooltip-body");
    tooltip.append(tooltipBody);
    root.append(tooltip);

    // Feature-detect native popover="hint" and fallback placement support.
    let hintPopoverSupported = false;
    try {
        const probe = document.createElement("div");
        probe.popover = "hint";
        hintPopoverSupported = probe.popover === "hint";
    } catch (_) { hintPopoverSupported = false; }
    if (!hintPopoverSupported && typeof document !== "undefined") {
        tooltip.setAttribute("popover", "auto");
    }

    let activeTipTrigger = null;
    let tipOpen = false;
    let tipShowTimer = null;
    let tipHideTimer = null;
    let tipClosedAt = 0;
    let tipPointerDownAt = 0;
    let tipOwnedDescribedBy = false;

    function clearTipTimers() {
        if (tipShowTimer) { clearTimeout(tipShowTimer); tipShowTimer = null; }
        if (tipHideTimer) { clearTimeout(tipHideTimer); tipHideTimer = null; }
    }
    function elementAttribute(element, name) {
        return typeof element?.getAttribute === "function" ? element.getAttribute(name) : element?.attributes?.[name];
    }
    function normalizedText(value) {
        return String(value ?? "").replace(/\s+/g, " ").trim().toLowerCase();
    }
    function triggerAccessibleName(trigger) {
        return normalizedText(elementAttribute(trigger, "aria-label") || trigger?.textContent || "");
    }
    function findTipTrigger(target) {
        if (!target || target === canvas) return null;
        if (typeof target.closest === "function") {
            const matched = target.closest("[data-tooltip]");
            if (!matched || matched === canvas || (root.contains && !root.contains(matched))) return null;
            return matched.dataset?.tooltip ? matched : null;
        }
        return target.dataset?.tooltip ? target : null;
    }
    function positionTooltipFallback(trigger) {
        const triggerRect = trigger?.getBoundingClientRect?.();
        if (!triggerRect) return;
        const tipRect = tooltip.getBoundingClientRect?.();
        const tipWidth = tipRect?.width || 220;
        const tipHeight = tipRect?.height || 34;
        const side = trigger.dataset?.tipSide || "block-start";
        const vw = typeof window !== "undefined" ? (window.innerWidth || 1280) : 1280;
        const vh = typeof window !== "undefined" ? (window.innerHeight || 800) : 800;
        let left = triggerRect.left + triggerRect.width / 2 - tipWidth / 2;
        let top = triggerRect.top - tipHeight - 8;
        if (side === "inline-end") {
            left = triggerRect.right + 8;
            top = triggerRect.top + triggerRect.height / 2 - tipHeight / 2;
            if (left + tipWidth > vw - 8) left = triggerRect.left - tipWidth - 8;
        } else if (top < 8) {
            top = triggerRect.bottom + 8;
        }
        left = Math.max(8, Math.min(left, vw - tipWidth - 8));
        top = Math.max(8, Math.min(top, vh - tipHeight - 8));
        tooltip.style.inset = "auto";
        tooltip.style.margin = "0";
        tooltip.style.left = `${Math.round(left)}px`;
        tooltip.style.top = `${Math.round(top)}px`;
    }
    function closeTooltip({ keepWarm = false } = {}) {
        clearTipTimers();
        if (activeTipTrigger) {
            const prev = activeTipTrigger;
            activeTipTrigger = null;
            if (prev.dataset) delete prev.dataset.thsTipActive;
            const saved = prev.dataset?.tooltip || "";
            if (saved) prev.title = saved;
            if (tipOwnedDescribedBy && elementAttribute(prev, "aria-describedby") === tooltip.id) {
                prev.removeAttribute?.("aria-describedby");
            }
            tipOwnedDescribedBy = false;
        }
        if (tipOpen) {
            if (typeof tooltip.hidePopover === "function") {
                try { tooltip.hidePopover(); } catch (_) { /* Already hidden. */ }
            } else {
                tooltip.removeAttribute?.("data-open");
            }
            tipOpen = false;
            tipClosedAt = keepWarm ? Date.now() : 0;
        } else if (!keepWarm) {
            tipClosedAt = 0;
        }
    }
    function openTooltip(trigger) {
        const text = String(trigger?.dataset?.tooltip || trigger?.title || "").trim();
        if (!text || state.destroyed) { closeTooltip(); return; }
        clearTipTimers();
        if (activeTipTrigger && activeTipTrigger !== trigger) closeTooltip();
        activeTipTrigger = trigger;
        if (trigger.dataset) trigger.dataset.thsTipActive = "true";
        if (elementAttribute(trigger, "title")) trigger.removeAttribute?.("title");
        if (trigger.title) trigger.title = "";
        tipOwnedDescribedBy = false;
        if (!elementAttribute(trigger, "aria-describedby") && normalizedText(text) !== triggerAccessibleName(trigger)) {
            trigger.setAttribute?.("aria-describedby", tooltip.id);
            tipOwnedDescribedBy = true;
        }
        tooltipBody.textContent = text;
        tooltip.setAttribute("data-tip-side", trigger.dataset?.tipSide || "block-start");
        if (typeof tooltip.showPopover === "function") {
            try { tooltip.showPopover(); } catch (_) { tooltip.setAttribute("data-open", "true"); }
        } else {
            tooltip.setAttribute("data-open", "true");
        }
        tipOpen = true;
        const anchorSupported = typeof CSS !== "undefined" && Boolean(CSS.supports?.("position-area: block-start"));
        if (!anchorSupported) positionTooltipFallback(trigger);
    }
    function setTooltip(element, text, opts = {}) {
        assignTooltipMetadata(element, text, opts);
        if (element && element === activeTipTrigger) {
            const clean = String(text ?? "").trim();
            if (clean) tooltipBody.textContent = clean;
            else closeTooltip();
        }
    }
    function setActionVisible(control, visible, wrap = null) {
        control.style.display = visible ? "inline-flex" : "none";
        if (wrap) wrap.style.display = visible ? "block" : "none";
    }
    function hasCurrentMaskWork() {
        if (state.metadata?.has_mask || state.localEdits) return true;
        if (sam3Mode() && (sam3Objects().length > 0 || state.sam3?.frameReady || state.sam3?.trackReady)) return true;
        return Object.values(documentState.frames || {}).some((bucket) => parseEvents(bucket).length > 0);
    }
    function canShowRestoreBackupBanner() {
        return Boolean(state.previousMaskBackup && !state.previousMaskDismissed && !hasCurrentMaskWork());
    }
    function syncAlertBanner() {
        const tone = state.lastStatusTone || "normal";
        const message = String(status.textContent || "").trim();
        const isAlertTone = tone === "error" || tone === "warn";
        const showPending = Boolean(state.pendingSource);
        const showRestore = canShowRestoreBackupBanner();
        const showToneAlert = Boolean(message && isAlertTone && state.dismissedAlertKey !== `${tone}:${message}`);
        const visible = Boolean(showPending || showRestore || showToneAlert);
        const showRecoverBtn = showPending;
        const showRestoreBtn = Boolean(state.previousMaskBackup && (showPending || showRestore));
        recoverSource.style.display = showRecoverBtn ? "inline-flex" : "none";
        restorePrevious.style.display = showRestoreBtn ? "inline-flex" : "none";
        recoveryActions.style.display = showRecoverBtn || showRestoreBtn ? "flex" : "none";
        dismissAlert.style.display = visible && !showPending ? "inline-flex" : "none";
        alertBanner.style.display = visible ? "flex" : "none";
        if (!visible) return;
        const visualState = showPending ? "warn" : (isAlertTone ? tone : "info");
        if (alertBanner.dataset) alertBanner.dataset.state = visualState;
        alertBanner.setAttribute?.("data-state", visualState);
        alertText.textContent = message;
        alertText.style.display = message ? "block" : "none";
    }
    root.addEventListener("pointerover", (event) => {
        if (state.destroyed) return;
        const trigger = findTipTrigger(event.target);
        if (!trigger) return;
        if (trigger === activeTipTrigger && tipOpen) {
            if (tipHideTimer) { clearTimeout(tipHideTimer); tipHideTimer = null; }
            return;
        }
        if (tipHideTimer) { clearTimeout(tipHideTimer); tipHideTimer = null; }
        if (tipShowTimer) { clearTimeout(tipShowTimer); tipShowTimer = null; }
        const delay = tipOpen || (tipClosedAt > 0 && Date.now() - tipClosedAt < 250) ? TOOLTIP_WARM_DELAY_MS : TOOLTIP_SHOW_DELAY_MS;
        tipShowTimer = setTimeout(() => { tipShowTimer = null; openTooltip(trigger); }, delay);
        tipShowTimer?.unref?.();
    });
    root.addEventListener("pointerout", (event) => {
        const leaving = findTipTrigger(event.target);
        if (!activeTipTrigger) {
            const entering = findTipTrigger(event.relatedTarget);
            if (tipShowTimer && leaving && entering !== leaving) { clearTimeout(tipShowTimer); tipShowTimer = null; }
            return;
        }
        if (leaving !== activeTipTrigger) return;
        const next = event.relatedTarget;
        if (next && ((activeTipTrigger.contains?.(next) || next === activeTipTrigger) || (tooltip.contains?.(next) || next === tooltip))) return;
        if (tipShowTimer) { clearTimeout(tipShowTimer); tipShowTimer = null; }
        tipHideTimer = setTimeout(() => { tipHideTimer = null; closeTooltip({ keepWarm: true }); }, 110);
        tipHideTimer?.unref?.();
    });
    tooltip.addEventListener("pointerenter", () => {
        if (tipHideTimer) { clearTimeout(tipHideTimer); tipHideTimer = null; }
    });
    tooltip.addEventListener("pointerleave", (event) => {
        const next = event.relatedTarget;
        if (next && activeTipTrigger && (activeTipTrigger.contains?.(next) || next === activeTipTrigger)) return;
        tipHideTimer = setTimeout(() => { tipHideTimer = null; closeTooltip({ keepWarm: true }); }, 110);
        tipHideTimer?.unref?.();
    });
    root.addEventListener("focusin", (event) => {
        if (state.destroyed || Date.now() - tipPointerDownAt < 350) return;
        const trigger = findTipTrigger(event.target);
        if (trigger) openTooltip(trigger);
    });
    root.addEventListener("focusout", (event) => {
        if (!activeTipTrigger) return;
        const next = event.relatedTarget;
        if (next && (activeTipTrigger.contains?.(next) || tooltip.contains?.(next))) return;
        closeTooltip();
    });
    root.addEventListener("pointerdown", () => {
        tipPointerDownAt = Date.now();
        closeTooltip();
    }, true);
    root.addEventListener("wheel", () => closeTooltip(), { passive: true, capture: true });
    inspectorBody.addEventListener("scroll", () => closeTooltip(), { passive: true });

    const dialog = document.createElement("dialog");
    dialog.className = "ths-mask-dialog";
    dialog.setAttribute("closedby", "closerequest");
    const dialogBar = createElement("div", {}, "ths-mask-dialog-bar");
    const dialogTitle = createElement("strong");
    dialogTitle.id = `ths-mask-dialog-title-${instanceKey}`;
    dialogTitle.textContent = "Маска видео";
    dialog.setAttribute("aria-labelledby", dialogTitle.id);
    const closeDialogButton = iconButton("Закрыть", "close", "Закрыть редактор · Esc");
    closeDialogButton.setAttribute("autofocus", "");
    dialogBar.append(dialogTitle, closeDialogButton);
    const dialogInner = createElement("div", {}, "ths-mask-dialog-inner");
    dialog.append(dialogBar, dialogInner);
    root.append(dialog);

    const ctx = canvas.getContext?.("2d");
    const maskCanvas = document.createElement("canvas");
    const maskCtx = maskCanvas.getContext?.("2d");
    const maskPostprocessor = createMaskPostprocessor();
    const documentState = readDocument(node);
    const hadLoadedLayers = Array.isArray(documentState.layers) && documentState.layers.length > 0;
    ensureDocumentLayers(documentState, {
        mode: safeMode(node, documentState),
        startFrame: widgetValue(node, "start_frame", 0),
        endFrame: widgetValue(node, "end_frame", -1),
    });
    if (hadLoadedLayers) {
        const initActiveLayer = activeLayerOf(documentState);
        if (initActiveLayer) {
            projectLayerToDocument(documentState, initActiveLayer);
            setWidget(node, "tracking_mode", initActiveLayer.type);
            setWidget(node, "start_frame", initActiveLayer.start_frame ?? 0);
            setWidget(node, "end_frame", initActiveLayer.end_frame ?? -1);
        }
    }
    const initialSource = sourceInfo(documentState);
    const state = {
        mode: "brush", view: "overlay", frame: 0, frameCount: initialSource?.frames || 1,
        fps: number(widgetValue(node, "fps", documentState.fps || 24), 24), source: initialSource,
        metadata: {}, metadataAnchors: [], frameItems: new Map(), previews: new Map(), imageCache: new Map(), imagePending: new Map(),
        loadGeneration: 0, sourceGeneration: 0, missingFrame: false,
        anchors: explicitAnchors(documentState), localEdits: false, dirty: false, previewStale: false, resetBlank: false, previewSignature: "", editRevision: 0, pendingPreview: null, widgetSnapshot: widgetSnapshot(node), modeSnapshot: String(widgetValue(node, "tracking_mode", "")), serverMaskSettings: null,
        active: null, penPath: [], penHover: null, penSnapClose: false, ghost: null, grabcut: false, refineMethod: node.properties?.__thsRefineMethod === "sam3" ? "sam3" : "grabcut", dpr: 1, zoom: 1, panX: 0, panY: 0, panning: null,
        playbackSpeed: 1, playing: false, timer: null, history: [], historyIndex: -1, dialogOpen: false, dialogReturnFocus: null,
        markerKey: "", previewUse: 0, lastPointer: null, destroyed: false, pendingSource: null,
        sourceMismatch: null, lastMismatch: null, mismatchBlocked: false, previousMaskBackup: null, previousMaskDismissed: false, dismissedAlertKey: "",
        refinementBusy: false, refinementError: "", pendingQueue: null, pendingPromptId: null,
        sam3: { frameReady: false, trackReady: false, frame: null, seedToken: "", direction: "both", computedFrames: [], error: "", sourceKey: "", frameSignature: "", trackSignature: "", pendingAction: "" },
        layerSam3: new Map(), layerMaskFlags: new Map(),
        serverActiveLayerId: "", serverRefinedFrames: new Map(),
        pendingRequestKind: "", apiListeners: [],
    };
    if (node.properties?.__thsPreviousMask && typeof node.properties.__thsPreviousMask === "object") state.previousMaskBackup = clone(node.properties.__thsPreviousMask);

    let cropControls = null;
    cropControls = createCropControls({
        state,
        documentState,
        section,
        labelControl,
        iconButton,
        canvasWrap,
        geometry: () => geometry(),
        getExtent: () => {
            const { grow, feather } = snapshotMaskSettings();
            return effectiveCropExtent(state.metadata, grow, feather, state.source);
        },
        assignTooltipMetadata,
        setTooltip,
        onCropChange(nextCrop) {
            if (state.mismatchBlocked) return;
            const preservedSam3 = { ...state.sam3 };
            const wasStale = state.previewStale;
            const wasDirty = state.dirty;
            const wasLocalEdits = state.localEdits;
            documentState.crop = clone(nextCrop);
            const multNote = nextCrop.multiple > 1 ? `, ×${nextCrop.multiple}` : "";
            pushHistory(`Автокроп: ${nextCrop.mode === "auto" ? `авто (${nextCrop.margin} px${multNote})` : nextCrop.mode === "manual" ? `рамка вручную${nextCrop.multiple > 1 ? ` (×${nextCrop.multiple})` : ""}` : "весь кадр"}`);
            commit({ markDirty: false });
            state.previewStale = wasStale;
            state.dirty = wasDirty;
            state.localEdits = wasLocalEdits;
            state.sam3 = preservedSam3;
            updateDirtyIndicator();
            updateReadouts();
            updateActionAvailability();
            draw();
            setStatus(
                nextCrop.mode === "full"
                    ? "Автокроп выключен: выходы используют полный кадр."
                    : "Окно автокропа обновлено — выходы cropped_images и cropped_mask готовы.",
                "ready",
            );
        },
    });
    inspectorBody.append(cropControls.panel, editPanel);

    function diagnosticState() {
        const source = (item) => item ? { frames: item.frames, width: item.width, height: item.height, fingerprint: item.fingerprint } : null;
        return {
            frame: state.frame, frameCount: state.frameCount, mode: state.mode, view: state.view,
            zoom: state.zoom, pan: [state.panX, state.panY], fullscreen: state.dialogOpen,
            source: source(state.source), documentSource: source(sourceInfo(documentState)),
            dirty: state.dirty, localEdits: state.localEdits, revision: state.editRevision,
            stale: state.previewStale, mismatch: state.mismatchBlocked, hasBackup: Boolean(state.previousMaskBackup),
            refinement: state.grabcut, refineMethod: state.refineMethod, refinementBusy: state.refinementBusy, refinementError: Boolean(state.refinementError),
            pendingKind: state.pendingRequestKind, promptId: state.pendingPromptId,
            history: [state.historyIndex, state.history.length], keyframes: Object.keys(documentState.frames || {}).length,
            descriptors: state.frameItems.size, previews: state.previews.size, imagesPending: state.imagePending.size,
            tracking: safeMode(node, documentState), grow: number(widgetValue(node, "grow", 0)), feather: number(widgetValue(node, "feather", 0)),
            range: [number(widgetValue(node, "start_frame", 0)), number(widgetValue(node, "end_frame", -1), -1)],
            sam3: { frameReady: Boolean(state.sam3.frameReady), trackReady: Boolean(state.sam3.trackReady), frame: state.sam3.frame, direction: state.sam3.direction, computedFrames: state.sam3.computedFrames.length, error: Boolean(state.sam3.error) },
            queueDisabled: Boolean(queue.disabled), previewDisabled: Boolean(prepare.disabled), frameDisabled: Boolean(calculateFrame.disabled), trackDisabled: Boolean(trackVideo.disabled),
        };
    }
    let diagnosticStorage = null;
    try { diagnosticStorage = window.sessionStorage; } catch (_) { /* Storage disabled. */ }
    const diagnostics = createMaskDiagnostics({ nodeId: node.id, snapshot: diagnosticState, storage: diagnosticStorage });
    function layoutSnapshot() {
        const measure = (element) => {
            const rect = element.getBoundingClientRect(); const style = window.getComputedStyle?.(element);
            return { x: Math.round(rect.x ?? rect.left), y: Math.round(rect.y ?? rect.top), width: Math.round(rect.width), height: Math.round(rect.height),
                scrollHeight: element.scrollHeight, clientHeight: element.clientHeight,
                position: style?.position, transform: style?.transform, zoom: style?.zoom, display: style?.display };
        };
        const metrics = canvasMetrics();
        return { viewport: [window.innerWidth, window.innerHeight], dpr: window.devicePixelRatio,
            canvasLocalSize: [metrics.width, metrics.height], graphScale: [metrics.clientScaleX, metrics.clientScaleY],
            dialog: measure(dialog), workbench: measure(workbench), wrap: measure(canvasWrap), canvas: measure(canvas),
            bitmap: [canvas.width, canvas.height] };
    }
    debugDownload.addEventListener("click", () => {
        diagnostics.record("debug.export", { layout: layoutSnapshot() }); diagnostics.flush();
        try {
            const url = URL.createObjectURL(new Blob([JSON.stringify(diagnostics.report(), null, 2)], { type: "application/json" }));
            const link = document.createElement("a"); link.href = url;
            link.download = `ths-mask-debug-node-${String(node.id).replace(/[^a-z0-9_-]/gi, "_")}-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
            link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch (error) { setStatus(`Не удалось скачать журнал: ${diagnosticText(error?.message)}`, "error"); }
    });
    const recordAction = (event) => {
        const target = event.target?.closest?.("button,input,select,summary") || event.target;
        if (!target || !["BUTTON", "INPUT", "SELECT", "SUMMARY"].includes(target.tagName)) return;
        const control = diagnosticText(target.getAttribute?.("aria-label") || target.title || target.textContent);
        const value = ["INPUT", "SELECT"].includes(target.tagName) ? diagnosticText(target.value) : undefined;
        diagnostics.record(`ui.${event.type}`, { control, value, layout: event.type === "click" ? layoutSnapshot() : undefined });
        // Capture post-handler state as well as the state before the click.
        Promise.resolve().then(() => { if (!state.destroyed) diagnostics.record("ui.settled", { control }); });
    };
    root.addEventListener("click", recordAction, true); root.addEventListener("change", recordAction, true);
    canvas.addEventListener("pointerdown", (event) => diagnostics.record("gesture.start", { button: event.button }));
    canvas.addEventListener("pointerup", (event) => Promise.resolve().then(() => diagnostics.record("gesture.end", { button: event.button })));
    canvas.addEventListener("pointercancel", () => diagnostics.record("gesture.cancel"));
    let wheelLogTimer = null;
    canvas.addEventListener("wheel", () => {
        if (wheelLogTimer) clearTimeout(wheelLogTimer);
        wheelLogTimer = setTimeout(() => { wheelLogTimer = null; diagnostics.record("view.zoom", { layout: layoutSnapshot() }); }, 200);
    }, { passive: true });
    const recordPageError = (event) => {
        const message = event.message || event.reason?.message || event.reason || "Unknown browser error";
        diagnostics.record(event.type || "browser.error", { message: diagnosticText(message),
            stack: diagnosticText(event.error?.stack || event.reason?.stack), line: event.lineno, column: event.colno, layout: layoutSnapshot() });
        diagnostics.flush();
    };
    window.addEventListener?.("error", recordPageError); window.addEventListener?.("unhandledrejection", recordPageError);
    window.addEventListener?.("pagehide", diagnostics.flush);
    diagnostics.record("editor.created", { layout: layoutSnapshot() });

    function sam3Mode() { return safeMode(node, documentState) === "sam3"; }
    function configuredRefinementIntent() {
        if (node.properties?.__thsRefineEnabled === false) return "";
        const persisted = node.properties?.__thsRefineMethod;
        if (persisted === "sam3" || persisted === "grabcut") return persisted;
        if (node.properties?.__thsRefineEnabled === true) return "grabcut";
        for (const bucket of Object.values(documentState.frames || {})) {
            for (const event of parseEvents(bucket)) {
                if (event?.refine_method === "sam3") return "sam3";
                if (event?.refine_method === "grabcut" || event?.refine === true) return "grabcut";
            }
        }
        return "";
    }
    function currentSourceKey(source = state.source || sourceInfo(documentState)) {
        if (!source) return "";
        return [source.frames, source.width, source.height, source.fingerprint || ""].join("|");
    }
    function sam3Objects() {
        const sam3 = documentState.sam3;
        return Array.isArray(sam3?.objects) ? sam3.objects : sam3?.prompt ? [sam3.prompt] : [];
    }
    function isSam3Box(event) {
        return event?.type === "rect" && (event.refine_method === "sam3" || event.sam3_prompt);
    }
    function sam3SeedEvents(frame = state.frame) {
        const objects = sam3Objects();
        if (objects.length) {
            const hits = [];
            for (const item of objects) {
                if (Number(item.frame) === Number(frame)) hits.push(item);
                for (const ref of item.refinements || []) {
                    if (Number(ref.frame) === Number(frame)) hits.push(ref);
                }
            }
            return hits;
        }
        return eventsAt(frame).filter(isSam3Box);
    }
    function hasSam3Seed(frame = state.frame) { return sam3SeedEvents(frame).length > 0; }
    function sam3SeedSignature(frame = state.frame) {
        try { return JSON.stringify({ source: currentSourceKey(), frame, events: sam3SeedEvents(frame) }); } catch (_) { return `${currentSourceKey()}|${frame}`; }
    }
    function sam3AllSeedSignature() {
        const objects = sam3Objects();
        const legacyBoxes = {};
        for (const key of Object.keys(documentState.frames || {}).sort((a, b) => Number(a) - Number(b))) {
            const boxes = parseEvents(documentState.frames[key]).filter(isSam3Box);
            if (boxes.length) legacyBoxes[key] = boxes;
        }
        try { return JSON.stringify({ source: currentSourceKey(), objects, legacyBoxes }); } catch (_) { return currentSourceKey(); }
    }
    function ensureSam3Document() {
        if (!sam3Mode()) return;
        if (!documentState.sam3 || typeof documentState.sam3 !== "object") documentState.sam3 = {};
        const stored = documentState.sam3;
        if (!["frame", "track"].includes(stored.action)) stored.action = "frame";
        stored.frame = Number.isFinite(Number(stored.frame)) ? Math.max(0, Math.round(Number(stored.frame))) : state.frame;
        stored.direction = sam3Direction(stored.direction || directionSelect.value || "both");
        if (!stored.seed_token && hasSam3Seed(stored.frame)) stored.seed_token = makeSeedToken();
        if (stored.seed_token) state.sam3.seedToken = String(stored.seed_token);
        directionSelect.value = stored.direction;
    }
    function sam3ReadinessSignature({ frame = state.sam3.frame, direction = state.sam3.direction, range = true } = {}) {
        const start = range ? number(widgetValue(node, "start_frame", 0), 0) : 0;
        const end = range ? number(widgetValue(node, "end_frame", -1), -1) : -1;
        const signature = { source: currentSourceKey(), seed: state.sam3.seedToken || documentState.sam3?.seed_token || "", frame: Number.isFinite(Number(frame)) ? Number(frame) : null, edits: sam3AllSeedSignature() };
        if (range) Object.assign(signature, { direction: sam3Direction(direction), start, end });
        return JSON.stringify(signature);
    }
    function invalidateSam3({ frame = false, track = true, reason = "" } = {}) {
        if (!sam3Mode()) return;
        if (frame) state.sam3.frameReady = false;
        if (track) state.sam3.trackReady = false;
        if (frame || track) {
            if (track) state.sam3.trackSignature = "";
            if (frame) state.sam3.frameSignature = "";
            state.sam3.error = reason || "";
            state.metadata = { ...state.metadata, sam3: { ...(state.metadata?.sam3 || {}), frame_ready: frame ? false : Boolean(state.sam3.frameReady), track_ready: false, error: reason || undefined } };
        }
        updateActionAvailability();
    }
    function sam3FrameReady() {
        if (!sam3Mode() || !state.sam3.frameReady) return false;
        return Boolean(state.sam3.frameSignature && state.sam3.frameSignature === sam3ReadinessSignature({ frame: state.sam3.frame, direction: state.sam3.direction, range: false }));
    }
    function sam3TrackReady() {
        if (!sam3Mode() || !state.sam3.trackReady) return false;
        return Boolean(state.sam3.trackSignature && state.sam3.trackSignature === sam3ReadinessSignature());
    }
    function pendingSam3Action() { return state.pendingPreview?.sam3Action || state.sam3.pendingAction || ""; }

    function actionBlocked() {
        return Boolean(state.mismatchBlocked || state.refinementError || state.refinementBusy || state.pendingPreview);
    }

    function requestLabel(pending) {
        if (pending === state.pendingQueue) return "Запуск графа";
        if (pending.sam3Action === "frame") return "Расчёт кадра SAM3";
        if (pending.sam3Action === "track") return "Трекинг SAM3";
        if (pending.kind === "refinement") return "Уточнение маски";
        return safeMode(node, documentState) === "optical_flow" ? "Расчёт Optical Flow" : "Подготовка кадров и маски";
    }

    function durationLabel(milliseconds) {
        const seconds = Math.max(0, Math.floor(milliseconds / 1000));
        return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    }

    function showRequestProgress() {
        const pending = state.pendingPreview || state.pendingQueue;
        progressOverlay.style.display = pending ? "grid" : "none";
        if (!pending) {
            if (progressTimer !== null) clearInterval(progressTimer);
            progressTimer = null;
            return;
        }
        if (progressTimer === null) {
            progressTimer = setInterval(showRequestProgress, 1000);
            progressTimer?.unref?.();
        }
        const now = Date.now();
        const sample = pending.frameProgress;
        const label = sample?.phase === "optical_flow_corrections" ? "Optical Flow · учёт правок" : requestLabel(pending);
        if (progressTitle.textContent !== label) {
            progressTitle.textContent = label;
            progressTrack.setAttribute("aria-label", label);
        }
        if (sample) {
            progressTrack.max = sample.total;
            progressTrack.value = sample.value;
            const percent = Math.floor(100 * sample.value / sample.total);
            progressCaption.textContent = sample.value === sample.total
                ? `${sample.value} из ${sample.total} кадров · подготовка результата…`
                : `${sample.value} из ${sample.total} кадров · ${percent}%`;
        } else {
            progressTrack.removeAttribute?.("value");
            progressCaption.textContent = pending.startedAt ? "Подготовка…" : pending.promptId ? "Ожидание в очереди…" : "Отправка в очередь…";
        }
        const elapsed = `Прошло ${durationLabel(now - (pending.startedAt || pending.submittedAt))}`;
        const eta = sample?.finishAt;
        progressTime.textContent = eta > now && now - sample.at < 15000
            ? `Осталось ≈ ${durationLabel(eta - now + 999)}`
            : elapsed;
    }

    function updateMaskSettingTooltips() {
        const sam3 = sam3Mode();
        const hint = sam3
            ? "Меняется в предпросмотре сразу. Для нового трека SAM3 нажмите «Отследить видео»."
            : "Меняется в предпросмотре сразу и применяется к итоговой маске.";
        for (const [name, input] of Object.entries(maskSettingsInputs)) {
            const description = name === "grow" ? "Расширение маски" : "Растушёвка края маски";
            const tip = `${description}, 0–256 пикселей. ${hint}`;
            setTooltip(input, tip);
            if (maskSettingsRanges[name]) setTooltip(maskSettingsRanges[name], tip);
        }
    }

    function currentLayers() {
        if (!Array.isArray(documentState.layers) || documentState.layers.length === 0) {
            return ensureDocumentLayers(documentState, {
                mode: safeMode(node, documentState),
                startFrame: widgetValue(node, "start_frame", 0),
                endFrame: widgetValue(node, "end_frame", -1),
            });
        }
        if (!documentState.active_layer_id || !documentState.layers.some((layer) => layer?.id === String(documentState.active_layer_id))) {
            documentState.active_layer_id = String(documentState.layers[0].id || "layer-1");
        }
        return documentState.layers;
    }
    function activeLayer() {
        currentLayers();
        return activeLayerOf(documentState);
    }
    function hasOtherEnabledLayers() {
        const layers = currentLayers();
        const active = activeLayer();
        if (!active || layers.length <= 1) return false;
        return layers.some((layer) => layer.id !== active.id && layer.enabled !== false);
    }
    function layerSam3Prompted(layer) {
        if (!layer || layer.type !== "sam3") return false;
        if (Array.isArray(layer.sam3?.objects) && layer.sam3.objects.length > 0) return true;
        const prompt = layer.sam3?.prompt;
        if (prompt && ((Array.isArray(prompt.points) && prompt.points.length > 0) || (Number(prompt.w) > 0 && Number(prompt.h) > 0))) return true;
        for (const bucket of Object.values(layer.frames || {})) {
            if (parseEvents(bucket).some(isSam3Box)) return true;
        }
        return false;
    }
    function hasUntrackedSam3Layer() {
        const layers = currentLayers();
        const active = activeLayer();
        if (layers.length <= 1) {
            return sam3Mode() && !sam3TrackReady();
        }
        for (const layer of layers) {
            if (layer.enabled === false || layer.type !== "sam3") continue;
            if (layer.id === active?.id) {
                if (layerSam3Prompted(layer) && !sam3TrackReady()) return true;
            } else {
                const saved = state.layerSam3.get(layer.id);
                const ready = Boolean(saved?.trackReady && (!saved.sourceKey || saved.sourceKey === currentSourceKey()));
                if (layerSam3Prompted(layer) && !ready) return true;
            }
        }
        return false;
    }
    function saveCurrentLayerState() {
        const active = syncActiveLayerToList(documentState, {
            mode: safeMode(node, documentState),
            startFrame: widgetValue(node, "start_frame", 0),
            endFrame: widgetValue(node, "end_frame", -1),
        });
        if (active) state.layerSam3.set(active.id, clone(state.sam3));
        return active;
    }
    function activateLayerEntry(target) {
        if (!target) return null;
        projectLayerToDocument(documentState, target);
        setWidget(node, "tracking_mode", target.type);
        setWidget(node, "start_frame", target.start_frame ?? 0);
        setWidget(node, "end_frame", target.end_frame ?? -1);
        modeSelect.value = target.type;
        const savedSam3 = state.layerSam3.get(target.id);
        if (target.type === "sam3" && savedSam3) {
            state.sam3 = clone(savedSam3);
        } else {
            state.sam3 = {
                frameReady: false,
                trackReady: false,
                frame: Number.isFinite(Number(target.sam3?.frame)) ? Math.max(0, Math.round(Number(target.sam3.frame))) : state.frame,
                seedToken: String(target.sam3?.seed_token || ""),
                direction: sam3Direction(target.sam3?.direction || directionSelect.value),
                computedFrames: [],
                error: "",
                sourceKey: String(savedSam3?.sourceKey || ""),
                frameSignature: "",
                trackSignature: "",
                pendingAction: target.sam3?.action || "",
            };
        }
        directionSelect.value = state.sam3.direction;
        syncConfiguredToolState({ resetLegacy: true });
        state.anchors = explicitAnchors(documentState);
        state.widgetSnapshot = widgetSnapshot(node);
        state.modeSnapshot = String(widgetValue(node, "tracking_mode", ""));
        state.maskSettingsSnapshot = JSON.stringify(snapshotMaskSettings());
        return target;
    }
    function selectLayer(layerId) {
        if (state.mismatchBlocked || state.pendingPreview || state.pendingQueue) return false;
        if (activeLayer()?.id === layerId) return false;
        if (!currentLayers().some((item) => item.id === layerId)) return false;
        saveCurrentLayerState();
        const target = currentLayers().find((item) => item.id === layerId);
        if (!target) return false;
        activateLayerEntry(target);
        pushHistory(`Слой: ${target.name}`);
        commit({ markDirty: false });
        updateRefineControls();
        updateReadouts();
        updateActionAvailability();
        draw();
        return true;
    }
    function addLayer(type) {
        if (state.mismatchBlocked || state.pendingPreview || state.pendingQueue) return false;
        if (currentLayers().length >= MAX_MASK_LAYERS) {
            setStatus(`Достигнут лимит слоёв (${MAX_MASK_LAYERS}).`, "warn");
            return false;
        }
        saveCurrentLayerState();
        const layers = currentLayers();
        const cleanType = ["sam3", "optical_flow", "keyframes", "static"].includes(type) ? type : "optical_flow";
        const created = {
            id: nextLayerId(layers),
            name: nextLayerName(layers, cleanType),
            type: cleanType,
            blend: "add",
            enabled: true,
            start_frame: 0,
            end_frame: -1,
            frames: {},
            ...(cleanType === "sam3" ? { sam3: { action: "frame", frame: state.frame, direction: sam3Direction(directionSelect.value), seed_token: "" } } : {}),
        };
        layers.push(created);
        activateLayerEntry(created);
        pushHistory(`Добавлен слой: ${created.name}`);
        commit({ markDirty: true });
        updateRefineControls();
        updateReadouts();
        updateActionAvailability();
        draw();
        setStatus(`Добавлен слой «${created.name}». Предыдущие слои сохранены.`, "ready");
        return true;
    }
    function toggleLayerEnabled(layerId) {
        if (state.mismatchBlocked || state.pendingPreview || state.pendingQueue) return false;
        saveCurrentLayerState();
        const layers = currentLayers();
        const target = layers.find((item) => item.id === layerId);
        if (!target) return false;
        target.enabled = target.enabled === false;
        pushHistory(`${target.enabled ? "Включён" : "Скрыт"} слой: ${target.name}`);
        commit({ markDirty: true });
        updateReadouts();
        updateActionAvailability();
        draw();
        return true;
    }
    function cycleLayerBlend(layerId) {
        if (state.mismatchBlocked || state.pendingPreview || state.pendingQueue) return false;
        saveCurrentLayerState();
        const layers = currentLayers();
        const target = layers.find((item) => item.id === layerId);
        if (!target) return false;
        target.blend = nextBlendMode(target.blend);
        pushHistory(`Смешивание (${target.name}): ${blendLabel(target.blend)}`);
        commit({ markDirty: true });
        updateReadouts();
        updateActionAvailability();
        draw();
        return true;
    }
    function moveLayer(layerId, delta) {
        if (state.mismatchBlocked || state.pendingPreview || state.pendingQueue) return false;
        saveCurrentLayerState();
        const layers = currentLayers();
        const idx = layers.findIndex((item) => item.id === layerId);
        const nextIdx = idx + delta;
        if (idx < 0 || nextIdx < 0 || nextIdx >= layers.length) return false;
        const [item] = layers.splice(idx, 1);
        layers.splice(nextIdx, 0, item);
        pushHistory(`Порядок слоёв: ${item.name}`);
        commit({ markDirty: true });
        updateActionAvailability();
        draw();
        return true;
    }
    function deleteLayer(layerId) {
        if (state.mismatchBlocked || state.pendingPreview || state.pendingQueue) return false;
        if (currentLayers().length <= 1) return false;
        const activeId = activeLayer()?.id;
        saveCurrentLayerState();
        const layers = currentLayers();
        const idx = layers.findIndex((item) => item.id === layerId);
        if (idx < 0) return false;
        const [removed] = layers.splice(idx, 1);
        state.layerSam3.delete(removed.id);
        state.layerMaskFlags.delete(removed.id);
        if (activeId === removed.id) {
            const fallback = layers[Math.min(idx, layers.length - 1)];
            activateLayerEntry(fallback);
        }
        pushHistory(`Удалён слой: ${removed.name}`);
        commit({ markDirty: true });
        updateRefineControls();
        updateReadouts();
        updateActionAvailability();
        draw();
        return true;
    }
    function renderLayersList() {
        const layers = currentLayers();
        const active = activeLayer();
        const busy = Boolean(state.pendingPreview || state.pendingQueue || state.mismatchBlocked);
        const rows = layers.map((layer, idx) => {
            const isActive = layer.id === active?.id;
            const disabledLayer = layer.enabled === false;
            const hasLayerMask = Boolean(state.layerMaskFlags.get(layer.id)) || layerHasContent(layer)
                || (isActive && layer.type === "sam3" && (sam3FrameReady() || sam3TrackReady()));

            const row = createElement("div", {}, "ths-layer-item");
            row.setAttribute("role", "listitem");
            row.setAttribute("data-layer-id", layer.id);
            row.setAttribute("data-active", String(isActive));
            row.setAttribute("data-disabled", String(disabledLayer));
            if (row.dataset) {
                row.dataset.layerId = layer.id;
                row.dataset.active = String(isActive);
                row.dataset.disabled = String(disabledLayer);
            }

            const mainBtn = createElement("button", {}, "ths-layer-main");
            mainBtn.type = "button";
            mainBtn.setAttribute("aria-pressed", String(isActive));
            mainBtn.setAttribute("aria-label", `Выбрать слой ${layer.name}`);
            const dot = createElement("span", {}, "ths-layer-dot");
            dot.setAttribute("data-has-mask", String(hasLayerMask));
            const nameSpan = createElement("span", {}, "ths-layer-name");
            nameSpan.textContent = layer.name;
            const metaSpan = createElement("span", {}, "ths-layer-meta");
            metaSpan.textContent = layerShortLabel(layer.type);
            mainBtn.append(dot, nameSpan, metaSpan);
            mainBtn.disabled = busy;
            mainBtn.addEventListener("click", () => selectLayer(layer.id));

            const actions = createElement("div", {}, "ths-layer-actions");
            const blendBtn = button(blendSymbol(layer.blend));
            blendBtn.className = `${blendBtn.className || ""} ths-layer-blend`.trim();
            blendBtn.setAttribute("data-blend", layer.blend);
            if (blendBtn.dataset) blendBtn.dataset.blend = layer.blend;
            blendBtn.setAttribute("aria-label", "Режим смешивания слоя");
            setTooltip(blendBtn, `${blendLabel(layer.blend)} — нажмите для переключения (+ / − / ∩)`);
            blendBtn.disabled = busy;
            blendBtn.addEventListener("click", () => cycleLayerBlend(layer.id));

            const visBtn = button(disabledLayer ? "○" : "👁");
            visBtn.className = `${visBtn.className || ""} ths-layer-icon`.trim();
            visBtn.setAttribute("aria-label", "Видимость слоя");
            visBtn.setAttribute("aria-pressed", String(!disabledLayer));
            setTooltip(visBtn, disabledLayer ? "Включить слой" : "Скрыть слой без удаления");
            visBtn.disabled = busy;
            visBtn.addEventListener("click", () => toggleLayerEnabled(layer.id));

            actions.append(blendBtn, visBtn);
            if (layers.length > 1) {
                const upBtn = button("↑");
                upBtn.className = `${upBtn.className || ""} ths-layer-icon`.trim();
                upBtn.setAttribute("aria-label", "Слой выше");
                setTooltip(upBtn, "Переместить слой выше");
                upBtn.disabled = busy || idx === 0;
                upBtn.addEventListener("click", () => moveLayer(layer.id, -1));

                const downBtn = button("↓");
                downBtn.className = `${downBtn.className || ""} ths-layer-icon`.trim();
                downBtn.setAttribute("aria-label", "Слой ниже");
                setTooltip(downBtn, "Переместить слой ниже");
                downBtn.disabled = busy || idx === layers.length - 1;
                downBtn.addEventListener("click", () => moveLayer(layer.id, 1));

                const delBtn = button("×");
                delBtn.className = `${delBtn.className || ""} ths-layer-icon`.trim();
                delBtn.setAttribute("aria-label", "Удалить слой");
                setTooltip(delBtn, "Удалить слой");
                delBtn.disabled = busy;
                delBtn.addEventListener("click", () => deleteLayer(layer.id));
                actions.append(upBtn, downBtn, delBtn);
            }

            row.append(mainBtn, actions);
            return row;
        });
        replaceChildrenSafe(layersContainer, ...rows);
        for (const btn of Object.values(addLayerButtons)) {
            btn.disabled = busy || layers.length >= MAX_MASK_LAYERS;
        }
    }

    function updateActionAvailability() {
        const modeIsSam3 = sam3Mode();
        const modeDescription = modeIsSam3
            ? "SAM3: прицел добавляет объекты. ＋ и − уточняют активный объект. Правый клик удаляет точку или объект."
            : safeMode(node, documentState) === "optical_flow"
                ? "Optical Flow: нарисуйте область — движение перенесёт её по кадрам."
                : safeMode(node, documentState) === "keyframes"
                    ? "Ручные ключи: кадры удерживают маску ближайшего ключа без расчёта движения."
                    : "Неподвижная маска: одна нарисованная маска на весь диапазон кадров.";
        modeHint.textContent = modeDescription;
        setTooltip(modeInfoButton, modeDescription);
        setTooltip(modeSelect, modeDescription);
        const objects = sam3Objects();
        const activeObject = clamp(state.activeSam3Object || 0, 0, Math.max(0, objects.length - 1));
        state.activeSam3Object = activeObject;
        objectControls.style.display = modeIsSam3 && objects.length ? "flex" : "none";
        const optionSig = objects.map((obj, idx) => `${idx}:${Number(obj?.frame ?? 0)}`).join(",");
        if (state.sam3OptionSignature !== optionSig) {
            const options = [];
            for (let i = 0; i < objects.length; i += 1) {
                const option = document.createElement("option");
                const objFrame = Number.isFinite(Number(objects[i]?.frame)) ? Math.round(Number(objects[i].frame)) : 0;
                option.value = String(i); option.textContent = `Объект ${i + 1} · к.${objFrame + 1}`;
                options.push(option);
            }
            replaceChildrenSafe(objectSelect, ...options);
            state.sam3OptionCount = objects.length;
            state.sam3OptionSignature = optionSig;
        }
        objectSelect.value = String(activeObject);
        objectSelect.disabled = Boolean(state.pendingPreview || state.pendingQueue || state.mismatchBlocked);
        samToolGroup.style.display = modeIsSam3 ? "flex" : "none";
        samSeparator.style.display = modeIsSam3 ? "block" : "none";
        for (const name of ["sam_point", "sam_add", "sam_remove"]) {
            tools[name].style.display = modeIsSam3 ? "inline-flex" : "none";
            tools[name].disabled = Boolean(state.pendingPreview || state.pendingQueue || state.mismatchBlocked);
        }
        const frameReady = sam3FrameReady();
        const trackReady = sam3TrackReady();
        const untrackedSam3 = hasUntrackedSam3Layer();
        calculateFrame.setAttribute("data-next-action", String(modeIsSam3 && !frameReady));
        trackVideo.setAttribute("data-next-action", String(modeIsSam3 && frameReady && !trackReady));
        prepare.setAttribute("data-next-action", String(!modeIsSam3));
        queue.setAttribute("data-next-action", String(!untrackedSam3 && (modeIsSam3 ? trackReady : Boolean(state.metadata?.has_mask))));
        queue.disabled = Boolean(actionBlocked() || state.pendingQueue || untrackedSam3);
        const sam3Pending = Boolean(state.pendingPreview && String(state.pendingPreview.kind || "").startsWith("sam3_"));
        const sourceLoaded = Boolean(state.source) || state.frameItems.size > 0;
        const frameActionLabel = sourceLoaded ? "Рассчитать кадр" : "Загрузить кадры";
        calculateFrame.textContent = frameActionLabel;
        calculateFrame.setAttribute("aria-label", frameActionLabel);
        const frameActionTip = sourceLoaded
            ? "Рассчитать SAM3 только на выбранном кадре. Остальной граф и трекинг видео не запускаются."
            : "Загрузить кадры и подготовить выбранный кадр SAM3. Остальной граф и трекинг видео не запускаются.";
        setTooltip(calculateFrame, frameActionTip);
        setTooltip(calculateFrameWrap, frameActionTip);
        setActionVisible(calculateFrame, modeIsSam3, calculateFrameWrap);
        setActionVisible(trackVideo, modeIsSam3, trackVideoWrap);
        directionLabel.style.display = modeIsSam3 ? "inline-flex" : "none";
        refine.style.display = modeIsSam3 ? "none" : "inline-flex";
        refineMethodLabel.style.display = modeIsSam3 ? "none" : "inline-flex";
        setActionVisible(prepare, !modeIsSam3, prepareWrap);
        setActionVisible(cancelRequest, Boolean(state.pendingPreview || state.pendingQueue), cancelRequestWrap);
        calculateFrame.disabled = Boolean(actionBlocked() || sam3Pending || state.pendingQueue || state.mismatchBlocked);
        trackVideo.disabled = Boolean(actionBlocked() || sam3Pending || state.pendingQueue || state.mismatchBlocked || !frameReady);
        const trackTip = !frameReady
            ? "Сначала рассчитайте опорный кадр SAM3, чтобы разблокировать трекинг видео."
            : "Отследить SAM3 по выбранному направлению и диапазону.";
        setTooltip(trackVideo, trackTip);
        setTooltip(trackVideoWrap, trackTip);
        if (state.mismatchBlocked) {
            prepare.disabled = Boolean(state.pendingPreview);
        } else if (state.pendingPreview) {
            prepare.disabled = true;
        }
        newVideoButton.disabled = Boolean(state.pendingPreview || state.pendingQueue);
        setTooltip(recoverSource, "Принять новый клип и начать пустую маску. Старая маска и настройки будут сохранены для восстановления.");
        setTooltip(restorePrevious, "Восстановить сохранённую маску и настройки. Затем выберите прежний источник видео.");
        refineMethod.disabled = Boolean(state.pendingPreview || state.pendingQueue || modeIsSam3);
        updateMaskSettingTooltips();
        toolToolbarController.refresh();
        const queueTip = untrackedSam3 && !modeIsSam3
            ? "Есть неотслеженный слой SAM3 — выберите его и нажмите «Отследить видео»"
            : modeIsSam3
                ? (trackReady ? "Трек готов · нажмите, чтобы запустить граф ComfyUI" : !untrackedSam3 ? "Можно запускать граф (активный слой SAM3 пуст, сработают остальные слои)" : frameReady ? "Кадр готов · сначала нажмите «Отследить видео»" : "Сначала рассчитайте кадр SAM3 и выполните трекинг")
                : "Запустить граф ComfyUI с готовой маской (подготовка выше не запускает полный граф)";
        queueHint.textContent = untrackedSam3 && !modeIsSam3
            ? "Есть неотслеженный слой SAM3 — выберите его и нажмите «Отследить видео»"
            : modeIsSam3
                ? (trackReady ? "Трек готов · можно запускать граф" : !untrackedSam3 ? "Можно запускать граф" : frameReady ? "Кадр готов · нажмите «Отследить видео»" : "Сначала рассчитайте кадр SAM3")
                : "Подготовка не запускает полный граф";
        setTooltip(queue, queueTip);
        setTooltip(queueWrap, queueTip);
        cropControls?.refresh(Boolean(state.pendingPreview || state.pendingQueue));
        renderLayersList();
        syncAlertBanner();
        showRequestProgress();
    }

    function clearPreviewCaches() {
        state.sourceGeneration += 1;
        state.loadGeneration += 1;
        state.frameItems.clear();
        state.previews.clear();
        state.imageCache.clear();
        state.imagePending.clear();
        state.serverRefinedFrames.clear();
        state.missingFrame = false;
        state.previewUse = 0;
        maskPostprocessor.invalidate();
    }

    function capturePreviousMask(source = state.source || sourceInfo(documentState)) {
        const properties = node.properties && typeof node.properties === "object" ? node.properties : (node.properties = {});
        const existing = properties.__thsPreviousMask;
        const boundSource = sourceInfo(documentState) || sourceInfo({ source }) || sourceInfo({ source: state.source });
        const existingSource = sourceInfo({ source: existing?.source });
        const sameSource = existingSource && boundSource && existingSource.frames === boundSource.frames && existingSource.width === boundSource.width && existingSource.height === boundSource.height && (!existingSource.fingerprint || !boundSource.fingerprint || existingSource.fingerprint === boundSource.fingerprint);
        if (existing && typeof existing === "object" && sameSource) { state.previousMaskBackup = clone(existing); state.previousMaskDismissed = false; return state.previousMaskBackup; }
        const rawMask = widgetValue(node, "mask_data", "{}");
        const backup = {
            mask_data: typeof rawMask === "string" ? rawMask : clone(rawMask),
            document: snapshotDocument(),
            tracking_mode: widgetValue(node, "tracking_mode", safeMode(node, documentState)),
            grow: widgetValue(node, "grow", maskSettingsDefaults.grow),
            feather: widgetValue(node, "feather", maskSettingsDefaults.feather),
            start_frame: widgetValue(node, "start_frame", 0),
            end_frame: widgetValue(node, "end_frame", -1),
            source: clone(sourceInfo({ source }) || source || sourceInfo(documentState)),
        };
        properties.__thsPreviousMask = clone(backup);
        state.previousMaskBackup = backup;
        state.previousMaskDismissed = false;
        return backup;
    }

    function frameBucket(frame, create = false, event = null) {
        const frames = documentState.frames && typeof documentState.frames === "object" ? documentState.frames : (documentState.frames = {});
        const key = String(Math.max(0, Math.round(frame)));
        if (!frames[key] && create) frames[key] = { events: [], inherit: shouldInherit(frame, event) };
        return frames[key];
    }

    function hasIndependentAnchor() {
        if (sam3Mode() && sam3Objects().length) return true;
        return Object.values(documentState.frames || {}).some((bucket) => {
            const events = parseEvents(bucket);
            return events.length > 0 && bucket?.inherit !== true;
        });
    }

    function hasReliableEditBase(frame) {
        const item = state.previews.get(frame) || state.frameItems.get(frame);
        const descriptor = item?.edit_base || item?.editBase || item?.edit_base_descriptor;
        return Boolean(descriptor || item?.editBaseImage?.naturalWidth);
    }

    function shouldInherit(frame, event = null) {
        // A correction can inherit only after an independently painted anchor
        // exists and the backend supplied its raw edit baseline.  Source
        // metadata alone is not a mask and must never turn the first stroke
        // into an inherited correction.
        if (safeMode(node, documentState) === "static" || !hasIndependentAnchor()) return false;
        if (hasReliableEditBase(frame)) return true;
        const item = state.previews.get(frame) || state.frameItems.get(frame);
        const hasFrameMask = Boolean(
            item?.editBaseImage?.naturalWidth
            || item?.rawMaskImage?.naturalWidth
            || item?.maskImage?.naturalWidth
            || item?.raw_mask
            || item?.mask
            || item?.mask_url
        );
        if (sam3Mode() && (sam3FrameReady() || sam3TrackReady())) return hasFrameMask;
        if (event?.mode === "erase" || state.mode === "erase") {
            return Boolean(
                item?.rawMaskImage?.naturalWidth
                || item?.maskImage?.naturalWidth
                || item?.raw_mask
                || item?.mask
                || item?.mask_url
            );
        }
        return false;
    }

    function eventsAt(frame, create = false, event = null) {
        const bucket = frameBucket(frame, create, event);
        if (!bucket) return [];
        if (!Array.isArray(bucket.events)) bucket.events = parseEvents(bucket);
        return bucket.events;
    }

    function totalPoints() {
        let total = 0;
        for (const bucket of Object.values(documentState.frames || {})) for (const event of parseEvents(bucket)) total += Array.isArray(event.points) ? event.points.length : 0;
        return total;
    }

    function snapshotDocument() {
        syncActiveLayerToList(documentState, {
            mode: safeMode(node, documentState),
            startFrame: widgetValue(node, "start_frame", 0),
            endFrame: widgetValue(node, "end_frame", -1),
        });
        const snapshot = { ...clone(documentState), tracking_mode: safeMode(node, documentState) };
        if (snapshot.tracking_mode === "sam3") {
            const sam3 = { ...(snapshot.sam3 || {}) };
            sam3.action = ["frame", "track"].includes(sam3.action) ? sam3.action : "frame";
            sam3.frame = Number.isFinite(Number(sam3.frame)) ? Math.max(0, Math.round(Number(sam3.frame))) : state.frame;
            sam3.direction = sam3Direction(sam3.direction || directionSelect.value);
            if (state.sam3.seedToken) sam3.seed_token = state.sam3.seedToken;
            snapshot.sam3 = sam3;
        }
        return snapshot;
    }

    function snapshotMaskSettings() {
        return Object.fromEntries(Object.entries(maskSettingsDefaults).map(([name, fallback]) => [name, clamp(Math.round(number(widgetValue(node, name, fallback), fallback)), 0, 256)]));
    }

    function syncMaskSettingsInputs({ skipInput = null } = {}) {
        const values = snapshotMaskSettings();
        for (const [name, input] of Object.entries(maskSettingsInputs)) {
            const str = String(values[name]);
            const disabled = !findWidget(node, name) || Boolean(node.inputs?.some((socket) => socket.name === name && socket.link != null));
            if (input !== skipInput) input.value = str;
            input.disabled = disabled;
            const range = maskSettingsRanges[name];
            if (range) {
                range.value = str;
                range.disabled = disabled;
                syncSliderMetrics(range, `${str} px`);
            }
        }
    }

    function documentSignature() {
        try {
            const { crop: _ignoredCrop, ...maskOnly } = documentState;
            return JSON.stringify(maskOnly);
        } catch (_) { return ""; }
    }

    function pushHistory(label = "Изменение") {
        const value = snapshotDocument();
        const settings = snapshotMaskSettings();
        const previous = state.history[state.historyIndex];
        if (previous && JSON.stringify(previous.document) === JSON.stringify(value) && JSON.stringify(previous.settings) === JSON.stringify(settings)) return;
        state.history = state.history.slice(0, state.historyIndex + 1);
        state.history.push({ document: value, settings, label });
        if (state.history.length > MAX_HISTORY) state.history.shift();
        state.historyIndex = state.history.length - 1;
        state.localEdits = true; state.dirty = true; state.previewStale = true; state.editRevision += 1;
        updateDirtyIndicator();
    }

    function restoreHistory(index) {
        if (state.mismatchBlocked) { setStatus("Источник не совпадает с сохранённой маской. Сначала примите новый клип или восстановите прежний источник.", "warn"); return false; }
        const entry = state.history[index];
        if (!entry) return false;
        const documentChanged = JSON.stringify(entry.document) !== JSON.stringify(snapshotDocument());
        const wasStale = state.previewStale;
        const wasClean = !wasStale && (!state.previewSignature || state.previewSignature === documentSignature());
        const wasSam3 = { ...state.sam3 };
        const approvedSeed = sam3FrameReady() ? {
            frame: state.sam3.frame, token: state.sam3.seedToken,
            signature: sam3SeedSignature(state.sam3.frame),
        } : null;
        const historyAction = index < state.historyIndex ? "Изменение отменено" : "Изменение повторено";
        const previousActiveType = activeLayer()?.type;
        if (documentChanged) replaceDocument(documentState, entry.document);
        for (const [name, value] of Object.entries(entry.settings || {})) {
            if (Object.prototype.hasOwnProperty.call(maskSettingsDefaults, name)) setWidget(node, name, value);
        }
        syncMaskSettingsInputs();
        if (documentChanged) {
            ensureDocumentLayers(documentState, {
                mode: documentState.tracking_mode,
                startFrame: widgetValue(node, "start_frame", 0),
                endFrame: widgetValue(node, "end_frame", -1),
            });
            const restoredActiveLayer = activeLayerOf(documentState);
            if (restoredActiveLayer) {
                projectLayerToDocument(documentState, restoredActiveLayer);
                setWidget(node, "start_frame", restoredActiveLayer.start_frame ?? 0);
                setWidget(node, "end_frame", restoredActiveLayer.end_frame ?? -1);
            }
            if (["sam3", "optical_flow", "static", "keyframes"].includes(documentState.tracking_mode)) {
                setWidget(node, "tracking_mode", documentState.tracking_mode);
                modeSelect.value = documentState.tracking_mode;
            }
            if (documentState.tracking_mode === "sam3") {
                state.sam3 = { ...state.sam3, ...(clone(documentState.sam3) || {}) };
                state.sam3.direction = sam3Direction(state.sam3.direction);
                directionSelect.value = state.sam3.direction;
            }
            state.historyIndex = index;
            state.resetBlank = Object.keys(documentState.frames || {}).length === 0
                && !(sam3Mode() && sam3Objects().length > 0)
                && !hasOtherEnabledLayers();
            state.localEdits = true; state.dirty = true; state.previewStale = true; state.editRevision += 1;
            state.source = sourceInfo(documentState) || state.source;
            state.frameCount = state.source?.frames || state.frameCount;
            state.anchors = explicitAnchors(documentState);
            if (documentState.tracking_mode === "sam3") {
                state.sam3.frame = Number.isFinite(Number(documentState.sam3?.frame)) ? Math.max(0, Math.round(Number(documentState.sam3.frame))) : state.sam3.frame;
                state.sam3.seedToken = String(documentState.sam3?.seed_token || state.sam3.seedToken || "");
                state.sam3.direction = sam3Direction(documentState.sam3?.direction || state.sam3.direction);
                // History may predate the server's canonical token. Reuse a live
                // verified seed only when its source, frame and prompt boxes match.
                // Brush edits/settings require a new track, not a new object seed.
                const sameSeed = Boolean(approvedSeed && approvedSeed.frame === state.sam3.frame
                    && approvedSeed.signature === sam3SeedSignature(state.sam3.frame));
                if (sameSeed) {
                    state.sam3.seedToken = approvedSeed.token;
                    documentState.sam3.seed_token = approvedSeed.token;
                }
                state.sam3.frameReady = sameSeed; state.sam3.trackReady = false;
                state.sam3.frameSignature = sameSeed ? sam3ReadinessSignature({ range: false }) : "";
                state.sam3.trackSignature = "";
                state.sam3.error = "";
                state.metadata = { ...state.metadata, sam3: { ...(state.metadata?.sam3 || {}), frame_ready: sameSeed, track_ready: false } };
                directionSelect.value = state.sam3.direction;
            }
            if (restoredActiveLayer && restoredActiveLayer.type !== previousActiveType) {
                syncConfiguredToolState({ resetLegacy: true });
                updateRefineControls();
            }
            state.widgetSnapshot = widgetSnapshot(node);
            state.modeSnapshot = String(widgetValue(node, "tracking_mode", ""));
            state.maskSettingsSnapshot = JSON.stringify(snapshotMaskSettings());
        } else {
            state.historyIndex = index;
            state.localEdits = true; state.dirty = true; state.previewStale = wasStale; state.editRevision += 1;
            state.sam3 = wasSam3;
        }
        updateReadouts(); updateActionAvailability(); cropControls?.refresh?.(); draw(); commit({ markDirty: false });
        if (!documentChanged) {
            if (wasClean) {
                state.previewStale = false;
                state.previewSignature = documentSignature();
            } else {
                state.previewStale = wasStale;
            }
            state.sam3 = wasSam3;
            updateDirtyIndicator();
            updateActionAvailability();
            cropControls?.refresh?.();
            draw();
        }
        setStatus(
            !documentChanged
                ? `${historyAction} — предпросмотр обновлён.`
                : `${historyAction}. ${sam3FrameReady() ? "Нажмите «Отследить видео», чтобы применить правки." : "Обновите маску для проверки результата."}`,
            !documentChanged && !wasStale ? "ready" : "dirty",
        );
        return true;
    }

    function commit({ markDirty = false, sam3Action = null } = {}) {
        if (state.mismatchBlocked) {
            updateActionAvailability();
            return false;
        }
        syncFromConfiguredWidget();
        const frames = {};
        for (const [key, bucket] of Object.entries(documentState.frames || {})) {
            const events = parseEvents(bucket);
            const inherit = bucket?.inherit;
            const inheritPresent = Object.prototype.hasOwnProperty.call(bucket || {}, "inherit");
            if (events.length || inheritPresent) {
                frames[String(key)] = { events: clone(events) };
                if (inheritPresent && typeof inherit === "boolean") frames[String(key)].inherit = inherit;
            }
        }
        const payload = { version: 1, frames };
        const source = state.source || sourceInfo(documentState);
        if (source) payload.source = clone(source);
        if (Object.prototype.hasOwnProperty.call(documentState, "anchors") || Object.prototype.hasOwnProperty.call(documentState, "keyframes") || Object.prototype.hasOwnProperty.call(documentState, "keys")) payload.anchors = explicitAnchors(documentState);
        payload.tracking_mode = safeMode(node, documentState);
        if (payload.tracking_mode === "sam3") {
            ensureSam3Document();
            const existing = documentState.sam3 && typeof documentState.sam3 === "object" ? documentState.sam3 : {};
            payload.sam3 = {
                action: ["frame", "track"].includes(sam3Action) ? sam3Action : (["frame", "track"].includes(existing.action) ? existing.action : "frame"),
                frame: Number.isFinite(Number(existing.frame)) ? Math.max(0, Math.round(Number(existing.frame))) : state.frame,
                direction: sam3Direction(existing.direction || directionSelect.value),
                seed_token: String(existing.seed_token || state.sam3.seedToken || ""),
                ...(existing.prompt ? { prompt: clone(existing.prompt) } : {}),
                ...(existing.objects ? { objects: clone(existing.objects) } : {}),
            };
            state.sam3.seedToken = payload.sam3.seed_token;
        }
        documentState.frames = frames; documentState.version = 1; documentState.source = payload.source;
        if (payload.anchors) documentState.anchors = clone(payload.anchors);
        delete documentState.keyframes; delete documentState.keys; documentState.tracking_mode = payload.tracking_mode;
        if (documentState.crop && typeof documentState.crop === "object") payload.crop = clone(documentState.crop);
        if (payload.sam3) documentState.sam3 = clone(payload.sam3); else delete documentState.sam3;
        syncActiveLayerToList(documentState, {
            mode: payload.tracking_mode,
            startFrame: widgetValue(node, "start_frame", 0),
            endFrame: widgetValue(node, "end_frame", -1),
        });
        const layers = currentLayers();
        if (shouldSerializeLayers(layers, {
            startFrame: widgetValue(node, "start_frame", 0),
            endFrame: widgetValue(node, "end_frame", -1),
        })) {
            payload.layers = clone(layers);
            payload.active_layer_id = String(documentState.active_layer_id || layers[0].id);
        }
        if (payload.tracking_mode === "sam3") {
            const active = activeLayer();
            if (active) state.layerSam3.set(active.id, clone(state.sam3));
        }
        setWidget(node, "mask_data", JSON.stringify(payload));
        state.widgetSnapshot = widgetSnapshot(node); state.modeSnapshot = String(widgetValue(node, "tracking_mode", "")); state.anchors = explicitAnchors(documentState);
        state.maskSettingsSnapshot = JSON.stringify(snapshotMaskSettings());
        if (markDirty) { state.localEdits = true; state.dirty = true; }
        updateDirtyIndicator();
        node.__thsUniversalMaskDirty = Boolean(state.dirty);
        updateActionAvailability();
        return true;
    }

    function updateDirtyIndicator() {
        const tone = state.lastStatusTone || "normal";
        dirtyDot.style.background = tone === "error" || state.metadata?.status === "error"
            ? COLORS.danger
            : (state.dirty || tone === "warn") ? COLORS.warning
            : (tone === "ok" || tone === "ready") ? COLORS.mask
            : COLORS.dim;
        sourceReadout.textContent = state.source ? `${state.source.width}×${state.source.height} · ${state.frameCount} кадр.` : "Источник не загружен";
    }

    function setStatus(message, tone = "normal") {
        const nextKey = `${tone}:${String(message || "").trim()}`;
        if (state.dismissedAlertKey && state.dismissedAlertKey !== nextKey) state.dismissedAlertKey = "";
        if (status.textContent !== message || status.dataset?.state !== tone) diagnostics.record("status", { tone, message: diagnosticText(message) });
        state.lastStatusTone = tone;
        status.textContent = message;
        status.dataset && (status.dataset.state = tone);
        status.setAttribute?.("data-state", tone);
        setTooltip(headerMeta, message || "Готов к работе");
        updateDirtyIndicator();
        syncAlertBanner();
    }

    function selectedPreview() {
        return state.previews.get(state.frame) || state.frameItems.get(state.frame) || null;
    }

    function frameHasDraft(frame) {
        const bucket = documentState.frames?.[String(Math.max(0, Math.round(frame)))];
        return Boolean((sam3Mode() && hasSam3Seed(frame)) || bucket && (Object.prototype.hasOwnProperty.call(bucket, "inherit") || parseEvents(bucket).length));
    }

    function frameNeedsDraft(frame) {
        if (state.resetBlank) return true;
        if (!frameHasDraft(frame)) return Boolean(state.active || state.ghost || state.penPath.length) && frame === state.frame;
        return state.previewStale || state.previewSignature !== documentSignature() || (frame === state.frame && Boolean(state.active || state.ghost || state.penPath.length));
    }

    function displayAnchors() {
        if (state.mismatchBlocked) return [];
        const derived = Object.keys(documentState.frames || {}).filter((key) => {
            const bucket = documentState.frames[key];
            return Object.prototype.hasOwnProperty.call(bucket || {}, "inherit") || parseEvents(bucket).length;
        }).map(Number);
        if (sam3Mode()) {
            for (const object of sam3Objects()) {
                derived.push(Number(object.frame));
                for (const ref of object.refinements || []) derived.push(Number(ref.frame));
            }
        }
        return [...new Set([...explicitAnchors(documentState), ...(state.metadataAnchors || []), ...derived])]
            .filter((frame) => Number.isFinite(frame) && frame >= 0 && frame < state.frameCount)
            .sort((a, b) => a - b);
    }

    function updateBrushReadout() {
        const size = Math.min(number(state.source?.width, 0), number(state.source?.height, 0));
        const label = size > 0
            ? `${Math.max(1, Math.round(number(brushInput.value, .035) * size * 2))} px`
            : `${(number(brushInput.value, .035) * 100).toFixed(1)}%`;
        brushReadout.textContent = label;
        syncSliderMetrics(brushInput, label);
    }

    function canvasMetrics() {
        const rect = canvas.getBoundingClientRect();
        const style = window.getComputedStyle?.(canvas);
        // getBoundingClientRect includes Comfy's graph transform. Canvas
        // painting and the HTML brush cursor both need untransformed CSS px.
        const cssPixels = (value) => typeof value === "string" && value.endsWith("px") ? number(parseFloat(value), 0) : 0;
        const width = cssPixels(style?.width) || canvas.clientWidth || rect.width;
        const height = cssPixels(style?.height) || canvas.clientHeight || rect.height;
        return { rect, width, height, clientScaleX: width ? rect.width / width : 1, clientScaleY: height ? rect.height / height : 1 };
    }

    function geometry() {
        const preview = selectedPreview();
        if (!preview?.sourceImage?.naturalWidth || !preview.sourceImage.naturalHeight) return null;
        const { rect, width: viewportWidth, height: viewportHeight, clientScaleX, clientScaleY } = canvasMetrics();
        if (!rect.width || !rect.height) return null;
        const width = preview.sourceImage.naturalWidth, height = preview.sourceImage.naturalHeight;
        const fitScale = Math.min(viewportWidth / width, viewportHeight / height);
        const scale = fitScale * state.zoom;
        const drawnWidth = width * scale, drawnHeight = height * scale;
        const offsetX = (viewportWidth - drawnWidth) / 2 + state.panX;
        const offsetY = (viewportHeight - drawnHeight) / 2 + state.panY;
        return { rect, viewportWidth, viewportHeight, clientScaleX, clientScaleY, width, height, fitScale, scale, drawnWidth, drawnHeight, offsetX, offsetY };
    }

    function pointFromEvent(event) {
        const box = geometry(); if (!box) return null;
        const x = (event.clientX - box.rect.left) / box.clientScaleX - box.offsetX;
        const y = (event.clientY - box.rect.top) / box.clientScaleY - box.offsetY;
        if (x < 0 || y < 0 || x > box.drawnWidth || y > box.drawnHeight) return null;
        return { x: clamp(x / box.drawnWidth), y: clamp(y / box.drawnHeight) };
    }

    function rawPointFromEvent(event) {
        const box = geometry(); if (!box) return null;
        const x = (event.clientX - box.rect.left) / box.clientScaleX - box.offsetX;
        const y = (event.clientY - box.rect.top) / box.clientScaleY - box.offsetY;
        return { x: x / box.drawnWidth, y: y / box.drawnHeight };
    }

    function cursorForMode(mode) {
        return mode === "brush" || mode === "erase" || mode === "pen" ? "crosshair" : "cell";
    }

    function isNearPenAnchor(point, anchor, box, thresholdPx = 10) {
        if (!point || !anchor || !box) return false;
        const dx = (point.x - anchor.x) * box.drawnWidth * box.clientScaleX;
        const dy = (point.y - anchor.y) * box.drawnHeight * box.clientScaleY;
        return Math.hypot(dx, dy) <= thresholdPx;
    }

    function eventPoint(point, box) { return [box.offsetX + point[0] * box.drawnWidth, box.offsetY + point[1] * box.drawnHeight]; }

    function drawEvent(target, event, box, maskOnly = false, options = {}) {
        if (!target || !event) return;
        const px = (point) => eventPoint(point, box);
        const add = event.mode !== "erase";
        target.save?.();
        target.globalAlpha = maskOnly ? 1 : (add ? .42 : .32);
        target.fillStyle = add ? (maskOnly ? "#fff" : COLORS.mask) : "rgba(0,0,0,0)";
        target.strokeStyle = add ? (maskOnly ? "#fff" : COLORS.mask) : "rgba(0,0,0,0)";
        target.lineCap = "round"; target.lineJoin = "round";
        if (event.type === "stroke") {
            const points = Array.isArray(event.points) ? event.points : []; if (!points.length) { target.restore?.(); return; }
            target.lineWidth = Math.max(2, number(event.radius, .02) * Math.min(box.drawnWidth, box.drawnHeight) * 2);
            if (points.length === 1) {
                const [x, y] = px(points[0]);
                target.beginPath(); target.arc?.(x, y, target.lineWidth / 2, 0, Math.PI * 2);
                if (add) target.fill(); else { target.globalCompositeOperation = "destination-out"; target.fillStyle = "rgba(0,0,0,1)"; target.fill(); }
            } else {
                target.beginPath(); points.forEach((point, index) => { const [x, y] = px(point); if (!index) target.moveTo(x, y); else target.lineTo(x, y); });
                if (add) target.stroke(); else { target.globalCompositeOperation = "destination-out"; target.strokeStyle = "rgba(0,0,0,1)"; target.stroke(); }
            }
        } else if (event.type === "rect") {
            const [x, y] = px([event.x, event.y]); const w = event.w * box.drawnWidth, h = event.h * box.drawnHeight;
            // A refined rectangle is only a selection until the user calculates
            // the mask. Never present the entire draft box as segmented foreground.
            const outlineOnly = Boolean(options.outlineOnly || event.refine);
            if (add && outlineOnly) {
                target.save?.();
                target.globalAlpha = 1;
                target.strokeStyle = maskOnly ? "#fff" : COLORS.warning;
                target.lineWidth = Math.max(1, Math.min(3, box.scale * 2));
                target.setLineDash?.([6, 4]);
                target.strokeRect(x, y, w, h);
                target.setLineDash?.([]);
                target.restore?.();
            } else if (add) { target.fillRect(x, y, w, h); target.strokeRect(x, y, w, h); } else { target.globalCompositeOperation = "destination-out"; target.fillStyle = "rgba(0,0,0,1)"; target.fillRect(x, y, w, h); }
        } else if (event.type === "polygon") {
            const points = Array.isArray(event.points) ? event.points : []; if (points.length < 3) { target.restore?.(); return; }
            target.beginPath(); points.forEach((point, index) => { const [x, y] = px(point); if (!index) target.moveTo(x, y); else target.lineTo(x, y); }); target.closePath();
            if (add) target.fill(); else { target.globalCompositeOperation = "destination-out"; target.fillStyle = "rgba(0,0,0,1)"; target.fill(); }
        }
        target.restore?.();
    }

    function drawPenOverlay(box) {
        if (!state.penPath.length) return;
        const toX = (nx) => box.offsetX + nx * box.drawnWidth;
        const toY = (ny) => box.offsetY + ny * box.drawnHeight;
        const traceSegment = (a, b) => {
            const bx = toX(b.x), by = toY(b.y);
            if ((a.outX || a.outY || b.inX || b.inY) && typeof ctx.bezierCurveTo === "function") {
                ctx.bezierCurveTo(
                    toX(a.x + a.outX), toY(a.y + a.outY),
                    toX(b.x + b.inX), toY(b.y + b.inY),
                    bx, by
                );
            } else {
                ctx.lineTo(bx, by);
            }
        };
        ctx.save();
        if (state.penPath.length >= 2) {
            ctx.beginPath();
            ctx.moveTo(toX(state.penPath[0].x), toY(state.penPath[0].y));
            for (let index = 0; index < state.penPath.length - 1; index += 1) traceSegment(state.penPath[index], state.penPath[index + 1]);
            ctx.lineCap = "round"; ctx.lineJoin = "round";
            ctx.strokeStyle = "rgba(15, 23, 42, 0.72)"; ctx.lineWidth = 3; ctx.stroke();
            ctx.strokeStyle = COLORS.mask; ctx.lineWidth = 1.6; ctx.stroke();
        }
        if (state.penHover && !state.active) {
            const prev = state.penPath.at(-1);
            const target = state.penSnapClose ? state.penPath[0] : { x: state.penHover.x, y: state.penHover.y, inX: 0, inY: 0 };
            ctx.beginPath();
            ctx.moveTo(toX(prev.x), toY(prev.y));
            traceSegment(prev, target);
            ctx.setLineDash?.([4, 4]);
            ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
            ctx.lineWidth = 1.25;
            ctx.stroke();
            ctx.setLineDash?.([]);
        }
        for (const [index, anchor] of state.penPath.entries()) {
            const ax = toX(anchor.x), ay = toY(anchor.y);
            const hasHandles = penAnchorHasHandles(anchor);
            if (hasHandles) {
                const inPx = toX(anchor.x + anchor.inX), inPy = toY(anchor.y + anchor.inY);
                const outPx = toX(anchor.x + anchor.outX), outPy = toY(anchor.y + anchor.outY);
                ctx.beginPath();
                ctx.moveTo(inPx, inPy); ctx.lineTo(ax, ay); ctx.lineTo(outPx, outPy);
                ctx.strokeStyle = "rgba(147, 197, 253, 0.85)"; ctx.lineWidth = 1; ctx.stroke();
                for (const [hx, hy] of [[inPx, inPy], [outPx, outPy]]) {
                    ctx.beginPath(); ctx.arc(hx, hy, 3, 0, Math.PI * 2);
                    ctx.fillStyle = "#0f172a"; ctx.fill();
                    ctx.strokeStyle = "#60a5fa"; ctx.lineWidth = 1.25; ctx.stroke();
                }
            }
            if (index === 0 && state.penSnapClose) {
                ctx.beginPath(); ctx.arc(ax, ay, 8, 0, Math.PI * 2);
                ctx.strokeStyle = COLORS.mask; ctx.lineWidth = 2; ctx.stroke();
            }
            const isLast = index === state.penPath.length - 1;
            ctx.fillStyle = isLast || (index === 0 && state.penSnapClose) ? COLORS.mask : "#fff";
            ctx.strokeStyle = isLast ? "#fff" : "#0f172a";
            ctx.lineWidth = 1.5;
            if (hasHandles) {
                ctx.beginPath(); ctx.arc(ax, ay, 3.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
            } else {
                ctx.fillRect(ax - 3, ay - 3, 6, 6);
                ctx.strokeRect(ax - 3, ay - 3, 6, 6);
            }
        }
        ctx.restore();
    }

    function draw() {
        const preview = selectedPreview();
        if (!ctx || !preview?.sourceImage?.naturalWidth || state.missingFrame) {
            ctx?.clearRect?.(0, 0, canvas.width, canvas.height);
            empty.style.display = "grid"; brushCursor.style.display = "none"; cropControls?.place?.(null); return;
        }
        empty.style.display = "none";
        const { width: viewportWidth, height: viewportHeight } = canvasMetrics(); state.dpr = Math.min(2, window.devicePixelRatio || 1);
        canvas.width = Math.max(1, Math.round(viewportWidth * state.dpr)); canvas.height = Math.max(1, Math.round(viewportHeight * state.dpr));
        ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0); ctx.clearRect(0, 0, viewportWidth, viewportHeight);
        const box = geometry(); if (!box) return;
        ctx.fillStyle = window.getComputedStyle?.(canvasWrap)?.backgroundColor || "#202020"; ctx.fillRect(0, 0, viewportWidth, viewportHeight);
        if (state.view !== "mask") ctx.drawImage(preview.sourceImage, box.offsetX, box.offsetY, box.drawnWidth, box.drawnHeight);
        const imgW = preview.sourceImage.naturalWidth, imgH = preview.sourceImage.naturalHeight;
        const workScale = Math.min(1, 640 / Math.max(1, imgW, imgH));
        const workW = Math.max(1, Math.round(imgW * workScale)), workH = Math.max(1, Math.round(imgH * workScale));
        const srcW = number(state.source?.width, imgW), srcH = number(state.source?.height, imgH);
        const pxScale = Math.min(workW / Math.max(1, srcW), workH / Math.max(1, srcH));
        const maskBox = { offsetX: 0, offsetY: 0, drawnWidth: workW, drawnHeight: workH, scale: pxScale };
        maskCanvas.width = workW; maskCanvas.height = workH;
        if (maskCtx) {
            maskCtx.setTransform(1, 0, 0, 1, 0, 0); maskCtx.clearRect(0, 0, workW, workH);
            const mismatchBlocked = Boolean(state.mismatchBlocked);
            const refinementFailed = Boolean(state.refinementError || state.metadata?.refinement_error);
            const draft = mismatchBlocked || refinementFailed ? true : frameNeedsDraft(state.frame);
            const bucket = frameBucket(state.frame);
            const currentLayer = activeLayer();
            const layerMode = safeMode(node, documentState);
            const otherLayersActive = hasOtherEnabledLayers();
            const activeBlend = currentLayer?.blend || "add";
            const staticEvents = layerMode === "static"
                ? Object.keys(documentState.frames || {}).sort((a, b) => Number(a) - Number(b)).flatMap((key) => parseEvents(documentState.frames[key]))
                : null;
            const localEvents = layerMode === "static" ? staticEvents : eventsAt(state.frame);
            const explicitClearBucket = Boolean(bucket && bucket.inherit === false && parseEvents(bucket).length === 0);
            const sam3ActiveBase = Boolean(
                sam3Mode()
                && !state.resetBlank
                && !explicitClearBucket
                && (sam3FrameReady() || sam3TrackReady())
                && (preview.editBaseImage?.naturalWidth || preview.rawMaskImage?.naturalWidth || preview.maskImage?.naturalWidth)
            );
            const inherits = bucket ? (bucket.inherit === true || sam3ActiveBase) : (shouldInherit(state.frame) || sam3ActiveBase);
            const drawEventForLayer = (ev) => (activeBlend === "subtract" && ev?.mode !== "erase" ? { ...ev, mode: "erase" } : ev);
            const currentSettings = snapshotMaskSettings();
            const serverSettings = state.serverMaskSettings;
            const settingsDifferFromServer = Boolean(
                serverSettings && (
                    currentSettings.grow !== serverSettings.grow ||
                    currentSettings.feather !== serverSettings.feather
                )
            );
            let baseline = null;
            let applyPostprocess = true;
            let replayLocal = false;
            let effGrow = currentSettings.grow;
            let effFeather = currentSettings.feather;
            let baselineKind = "none";

            const forcedOutline = refinementFailed || state.refinementBusy;
            const isOutlineRect = (event) => Boolean(event && event.type === "rect" && event.mode !== "erase" && (forcedOutline || event.refine));
            const savedRefined = (draft && !inherits && !otherLayersActive && !mismatchBlocked && !refinementFailed && !state.resetBlank)
                ? state.serverRefinedFrames.get(layerMode === "static" ? "__static__" : String(state.frame))
                : null;
            const refinedMatch = Boolean(
                savedRefined
                && localEvents.length >= savedRefined.baseCount
                && JSON.stringify(localEvents.slice(0, savedRefined.baseCount)) === savedRefined.prefixJson
                && !localEvents.slice(savedRefined.baseCount).some(isOutlineRect)
                && (preview.rawMaskImage?.naturalWidth || preview.maskImage?.naturalWidth)
            );

            if (mismatchBlocked || refinementFailed || state.resetBlank) {
                baseline = null;
                applyPostprocess = true;
                replayLocal = draft && !mismatchBlocked;
                baselineKind = "none";
            } else if (draft) {
                if (otherLayersActive && currentLayer?.id && state.serverActiveLayerId && currentLayer.id !== state.serverActiveLayerId) {
                    baseline = preview.rawMaskImage || preview.maskImage || preview.editBaseImage;
                } else if (inherits || otherLayersActive) {
                    baseline = preview.editBaseImage || preview.rawMaskImage || preview.maskImage;
                } else if (refinedMatch) {
                    baseline = preview.rawMaskImage || preview.maskImage;
                } else {
                    baseline = null;
                }
                applyPostprocess = true;
                replayLocal = true;
                baselineKind = baseline === preview.editBaseImage ? "editBase" : baseline === preview.rawMaskImage ? "rawMask" : baseline ? "maskFallback" : "none";
            } else if (!settingsDifferFromServer) {
                baseline = preview.maskImage;
                applyPostprocess = false;
                replayLocal = false;
                baselineKind = baseline ? "mask" : "none";
            } else if (preview.rawMaskImage?.naturalWidth) {
                baseline = preview.rawMaskImage;
                applyPostprocess = true;
                replayLocal = false;
                baselineKind = "rawMask";
            } else if (sam3Mode() && preview.editBaseImage?.naturalWidth) {
                baseline = preview.editBaseImage;
                applyPostprocess = true;
                replayLocal = true;
                baselineKind = "editBase";
            } else if (eventsAt(state.frame).length === 0 && preview.editBaseImage?.naturalWidth) {
                baseline = preview.editBaseImage;
                applyPostprocess = true;
                replayLocal = false;
                baselineKind = "editBase";
            } else if (preview.maskImage?.naturalWidth) {
                baseline = preview.maskImage;
                applyPostprocess = true;
                replayLocal = false;
                effGrow = Math.max(0, currentSettings.grow - (serverSettings?.grow ?? 0));
                effFeather = Math.max(0, currentSettings.feather - (serverSettings?.feather ?? 0));
                baselineKind = "maskFallback";
            }

            const deferredOutlines = [];
            if (baseline?.naturalWidth) maskCtx.drawImage(baseline, 0, 0, workW, workH);
            if (replayLocal && !mismatchBlocked) {
                const replayEvents = refinedMatch ? localEvents.slice(savedRefined.baseCount) : localEvents;
                for (const event of replayEvents) {
                    if (isOutlineRect(event)) {
                        if (draft) deferredOutlines.push(event);
                    } else {
                        drawEvent(maskCtx, drawEventForLayer(event), maskBox, true);
                    }
                }
            }
            if (state.ghost && !mismatchBlocked) {
                if (state.ghost.type === "rect" && state.ghost.mode !== "erase" && (state.refinementBusy || state.ghost.refine)) {
                    deferredOutlines.push(state.ghost);
                } else {
                    drawEvent(maskCtx, drawEventForLayer(state.ghost), maskBox, true);
                }
            }
            if (state.penPath.length >= 2) {
                const penPreviewPoints = flattenPenPath(state.penPath, { closed: true });
                if (penPreviewPoints.length >= 3) drawEvent(maskCtx, { type: "polygon", mode: activeBlend === "subtract" ? "erase" : "add", points: penPreviewPoints }, maskBox, true);
            }
            if (state.view !== "source") {
                const maskOnly = state.view === "mask";
                const alpha = number(opacity.value, .38);
                let imageData = null;
                if (typeof maskCtx.getImageData === "function" && typeof maskCtx.putImageData === "function") {
                    try { imageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height); } catch (_) { imageData = null; }
                }
                if (imageData?.data) {
                    const postData = imageData.width && imageData.height
                        ? imageData
                        : { data: imageData.data, width: maskCanvas.width, height: maskCanvas.height };
                    const baselineTag = baseline === preview.rawMaskImage ? "raw"
                        : baseline === preview.editBaseImage ? "base"
                        : baseline === preview.maskImage ? "mask"
                        : "none";
                    const isInteractiveGesture = Boolean(state.active || state.ghost || state.penPath.length);
                    const rawKey = isInteractiveGesture
                        ? ""
                        : `${state.sourceGeneration}:${state.loadGeneration}:${state.frame}:${state.editRevision}:${Number(preview.lastUsed || 0)}:${draft ? 1 : 0}:${replayLocal ? 1 : 0}:${state.resetBlank ? 1 : 0}:${baselineKind}:${baselineTag}:${currentLayer?.id || ""}:${activeBlend}:${otherLayersActive ? 1 : 0}:${refinedMatch ? 1 : 0}:${workW}x${workH}`;
                    maskPostprocessor.processImageData(postData, {
                        rawKey,
                        growPx: effGrow * pxScale,
                        featherPx: effFeather * pxScale,
                        maskOnly,
                        alpha,
                        applyPostprocess,
                    });
                    try { maskCtx.putImageData(imageData, 0, 0); } catch (_) { /* ignore */ }
                }
                ctx.save(); ctx.globalAlpha = 1; ctx.drawImage(maskCanvas, box.offsetX, box.offsetY, box.drawnWidth, box.drawnHeight); ctx.restore();
            }
            for (const outlineEvent of deferredOutlines) {
                drawEvent(ctx, outlineEvent, box, state.view === "mask", { outlineOnly: true });
            }
        }
        const sam3Badge = sam3Mode() ? ` · SAM3 ${sam3TrackReady() ? "трек" : sam3FrameReady() ? "кадр" : "черновик"}` : "";
        const pointObjects = sam3Mode() ? sam3Objects() : [];
        const drawPointMarker = (xPos, yPos, positive, active, badgeIndex = null) => {
            const x = box.offsetX + xPos * box.drawnWidth, y = box.offsetY + yPos * box.drawnHeight;
            ctx.fillStyle = positive ? "#247b45" : "#a32b35"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(x, y, active ? 8 : 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(x - 3, y); ctx.lineTo(x + 3, y);
            if (positive) { ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3); }
            ctx.stroke();
            if (badgeIndex !== null && pointObjects.length > 1) {
                ctx.fillStyle = "#fff";
                ctx.fillText?.(String(badgeIndex + 1), x + 10, y + 4);
            }
        };
        for (const [objectIndex, pointPrompt] of pointObjects.entries()) {
            const active = objectIndex === state.activeSam3Object;
            const onPrimaryFrame = Number(pointPrompt.frame) === state.frame;
            const refEntry = (pointPrompt.refinements || []).find((item) => Number(item.frame) === state.frame);
            if (!onPrimaryFrame && !refEntry) continue;
            ctx.save();
            ctx.globalAlpha = active ? 1 : .65;
            ctx.font = "bold 12px sans-serif";
            if (onPrimaryFrame) {
                if (!pointPrompt.points && frameNeedsDraft(state.frame)) {
                    const x = box.offsetX + pointPrompt.x * box.drawnWidth;
                    const y = box.offsetY + pointPrompt.y * box.drawnHeight;
                    ctx.strokeStyle = "#fff"; ctx.fillStyle = "#fff";
                    ctx.setLineDash?.([4, 4]);
                    ctx.strokeRect?.(x, y, pointPrompt.w * box.drawnWidth, pointPrompt.h * box.drawnHeight);
                    ctx.fillText?.(String(objectIndex + 1), x + 4, y + 14);
                }
                for (const [index, point] of (pointPrompt.points || []).entries()) {
                    drawPointMarker(point[0], point[1], pointPrompt.labels?.[index] !== 0, active, index === 0 ? objectIndex : null);
                }
            }
            if (refEntry) {
                for (const [index, point] of (refEntry.points || []).entries()) {
                    drawPointMarker(point[0], point[1], refEntry.labels?.[index] !== 0, active, index === 0 ? objectIndex : null);
                }
            }
            ctx.restore();
        }
        drawPenOverlay(box);
        cropControls?.place?.(box);
        canvasBadge.textContent = `${state.view === "overlay" ? "Наложение" : state.view === "mask" ? "Маска" : "Источник"}${sam3Badge} · Масштаб ${Math.round(state.zoom * 100)}%`;
        zoomReadout.textContent = `${Math.round(state.zoom * 100)}%`;
        updateBrushCursor(state.lastPointer);
    }

    function updateBrushCursor(event) {
        state.lastPointer = event;
        if (!event || state.mode !== "brush" && state.mode !== "erase" || state.playing || state.panning) { brushCursor.style.display = "none"; return; }
        const box = geometry(); const point = pointFromEvent(event); if (!box || !point) { brushCursor.style.display = "none"; return; }
        const size = Math.max(3, number(brushInput.value, .035) * Math.min(box.drawnWidth, box.drawnHeight) * 2);
        brushCursor.style.display = "block"; brushCursor.style.width = `${size}px`; brushCursor.style.height = `${size}px`;
        brushCursor.style.left = `${box.offsetX + point.x * box.drawnWidth}px`; brushCursor.style.top = `${box.offsetY + point.y * box.drawnHeight}px`;
    }

    function updateReadouts() {
        const count = Math.max(1, state.frameCount); const frame = clamp(Math.round(state.frame), 0, count - 1);
        frameReadout.textContent = `Кадр ${frame + 1} / ${count}`; if (document.activeElement !== direct) direct.value = String(frame + 1); slider.max = String(count - 1); slider.value = String(frame);
        syncSliderMetrics(slider, `Кадр ${frame + 1} из ${count}`);
        const seconds = state.fps > 0 ? frame / state.fps : 0; const duration = state.fps > 0 ? (count - 1) / state.fps : 0;
        timeReadout.textContent = `${seconds.toFixed(2)} с / ${duration.toFixed(2)} с`;
        state.anchors = displayAnchors();
        keyReadout.textContent = state.anchors.length ? `Ключи: ${state.anchors.map((key) => key + 1).join(", ")}` : "Ключи: нет";
        setTooltip(keyReadout, keyReadout.textContent);
        const start = Math.max(0, Math.round(number(widgetValue(node, "start_frame", 0)))); const endRaw = Math.round(number(widgetValue(node, "end_frame", -1), -1)); const end = endRaw < 0 ? count - 1 : clamp(endRaw, 0, count - 1);
        rangeReadout.textContent = `Диапазон: ${start + 1}—${end + 1}`; if (document.activeElement !== startInput) startInput.value = String(start + 1); if (document.activeElement !== endInput) endInput.value = String(end + 1); endInfinity.textContent = endRaw < 0 ? "до конца" : "";
        fpsReadout.textContent = `FPS ${state.fps.toFixed(2)}`;
        first.disabled = frame <= 0; back10.disabled = frame <= 0; back.disabled = frame <= 0;
        next.disabled = frame >= count - 1; next10.disabled = frame >= count - 1; last.disabled = frame >= count - 1;
        previousKeyButton.disabled = !state.anchors.some((key) => key < frame);
        nextKeyButton.disabled = !state.anchors.some((key) => key > frame);
        undo.disabled = state.historyIndex <= 0; redo.disabled = state.historyIndex < 0 || state.historyIndex >= state.history.length - 1;
        copyPrevious.disabled = !Object.keys(documentState.frames || {}).some((key) => Number(key) < frame && parseEvents(documentState.frames[key]).length > 0);
        updateBrushReadout();
        const markerKey = `${count}:${state.anchors.join(",")}`;
        if (markerKey !== state.markerKey) {
            const markers = [];
            for (const key of state.anchors) {
                const marker = button(`Ключевой кадр ${key + 1}`); marker.textContent = ""; marker.className = "ths-key-marker"; setTooltip(marker, `Ключевой кадр ${key + 1}`); marker.style.left = `${count <= 1 ? 0 : (key / (count - 1)) * 100}%`; marker.style.pointerEvents = "auto"; marker.addEventListener("click", () => setFrame(key, { source: "key" })); markers.push(marker);
            }
            replaceChildrenSafe(markerLayer, ...markers);
            state.markerKey = markerKey;
        }
        transportToolbarController.refresh();
        toolToolbarController.refresh();
        updateDirtyIndicator();
    }

    function chooseMode(mode) {
        if (mode !== "pen" && state.penPath.length) {
            state.penPath = []; state.penHover = null; state.penSnapClose = false;
            draw();
        }
        state.mode = mode;
        for (const [key, control] of Object.entries(tools)) { control.setAttribute("aria-pressed", String(key === mode)); }
        toolToolbarController.refresh(tools[mode]);
        canvas.style.cursor = cursorForMode(mode);
        if (mode.startsWith("sam_")) {
            setStatus(mode === "sam_remove" ? "Кликните по лишней области активного объекта." : mode === "sam_add" ? "Кликните по пропущенной части активного объекта." : "Кликните по объекту, чтобы добавить его к общей маске. Трекинг запускается отдельно.");
            return;
        }
        setStatus(mode === "pen" ? "Перо: клик — угол, перетаскивание — кривая Безье; клик по первой точке, двойной клик или Enter — замкнуть; Backspace — шаг назад." : mode === "rect" ? (sam3Mode() ? "SAM3: обведите объект рамкой, затем нажмите «Рассчитать кадр»." : state.grabcut ? state.refineMethod === "sam3" ? "SAM3: обведите объект близко к границам, затем нажмите «Рассчитать маску»." : "GrabCut: обведите объект и немного фона, затем нажмите «Рассчитать маску»." : "Прямоугольник: протяните область для маски.") : mode === "erase" ? "Ластик: стирает зелёную маску; правка сохранится для повторного трекинга." : "Кисть: добавляет область в маску; правка сохранится для повторного трекинга.");
    }

    function syncConfiguredToolState({ resetLegacy = false } = {}) {
        if (sam3Mode()) {
            state.grabcut = true;
            state.refineMethod = "sam3";
            chooseMode("sam_point");
            return;
        }
        if (resetLegacy) {
            const intent = configuredRefinementIntent();
            state.refineMethod = intent || "grabcut";
            state.grabcut = Boolean(intent);
        }
        chooseMode(state.grabcut ? "rect" : "brush");
    }

    function setFrame(frame, { source = "timeline", load = true } = {}) {
        const requested = clamp(Math.round(number(frame)), 0, Math.max(0, state.frameCount - 1));
        state.frame = requested; state.penPath = []; state.penHover = null; state.penSnapClose = false; state.ghost = null; state.active = null; state.missingFrame = false;
        if (!state.playing) diagnostics.record("frame.selected", { reason: source });
        updateReadouts(); updateActionAvailability(); draw();
        if (load) loadFrame(requested, { reason: source });
        return requested;
    }

    function cacheImage(key, image) {
        if (!image) return;
        state.imageCache.delete(key); state.imageCache.set(key, image);
        while (state.imageCache.size > MAX_PREVIEWS * 4) state.imageCache.delete(state.imageCache.keys().next().value);
    }

    function loadImage(value, key = "") {
        const url = imageUrl(value, queueApi); if (!url) return Promise.resolve(null);
        if (state.imageCache.has(url)) { const image = state.imageCache.get(url); state.imageCache.delete(url); state.imageCache.set(url, image); return Promise.resolve(image); }
        if (state.imagePending.has(url)) return state.imagePending.get(url);
        if (typeof Image === "undefined") return Promise.resolve(null);
        const promise = new Promise((resolve) => {
            const image = new Image(); image.onload = () => { cacheImage(url, image); state.imagePending.delete(url); resolve(image); }; image.onerror = () => { state.imagePending.delete(url); resolve(null); }; image.src = url;
        });
        state.imagePending.set(url, promise); return promise;
    }

    function descriptorFor(frame) {
        const item = state.frameItems.get(frame) || state.previews.get(frame); if (!item) return null;
        return {
            source: item.source || item.sourceDescriptor || item,
            mask: item.mask || item.maskDescriptor || item.mask_url,
            raw_mask: item.raw_mask || item.rawMask || item.raw_mask_url,
            edit_base: item.edit_base || item.editBase || item.edit_base_descriptor,
            ...(item.sourceImage ? { sourceImage: item.sourceImage } : {}),
            ...(item.maskImage ? { maskImage: item.maskImage } : {}),
            ...(item.rawMaskImage ? { rawMaskImage: item.rawMaskImage } : {}),
            ...(item.editBaseImage ? { editBaseImage: item.editBaseImage } : {}),
        };
    }

    function cachePreview(frame, item) {
        state.previews.set(frame, { ...item, lastUsed: ++state.previewUse });
        while (state.previews.size > MAX_PREVIEWS) {
            let victim = null; let oldest = Infinity;
            for (const [key, value] of state.previews) {
                if (key === state.frame) continue;
                if (number(value?.lastUsed, 0) < oldest) { oldest = number(value?.lastUsed, 0); victim = key; }
            }
            if (victim === null) break;
            state.previews.delete(victim);
        }
    }

    async function loadFrame(frame, { prefetch = false, reason = "timeline", generation = state.loadGeneration } = {}) {
        if (!prefetch && !state.playing) diagnostics.record("frame.load", { frame, generation, reason });
        const expectedSource = state.source?.fingerprint || state.sourceGeneration;
        const descriptor = descriptorFor(frame);
        if (!descriptor) {
            if (!prefetch && generation === state.loadGeneration && frame === state.frame) { state.missingFrame = true; empty.textContent = `Кадр ${frame + 1} недоступен. Подготовьте обновлённое превью.`; setStatus(`Кадр ${frame + 1} не найден — соседний кадр не показывается.`, "warn"); draw(); }
            return null;
        }
        const sourceImage = descriptor.sourceImage || await loadImage(descriptor.source, `source:${frame}`);
        const maskImage = descriptor.maskImage || await loadImage(descriptor.mask, `mask:${frame}`);
        const rawMaskImage = descriptor.rawMaskImage || await loadImage(descriptor.raw_mask, `raw_mask:${frame}`);
        const editBaseImage = descriptor.editBaseImage || await loadImage(descriptor.edit_base, `edit_base:${frame}`);
        if (generation !== state.loadGeneration || expectedSource !== (state.source?.fingerprint || state.sourceGeneration)) {
            if (!prefetch) diagnostics.record("frame.discarded", { frame, generation });
            return null;
        }
        if (!sourceImage) {
            if (!prefetch && frame === state.frame) { state.missingFrame = true; empty.textContent = `Не удалось загрузить кадр ${frame + 1}. Проверьте preview /view.`; setStatus(`Ошибка загрузки кадра ${frame + 1}.`, "error"); draw(); }
            return null;
        }
        cachePreview(frame, { sourceImage, maskImage, rawMaskImage, editBaseImage, edit_base: descriptor.edit_base });
        if (!prefetch && frame === state.frame) state.missingFrame = false;
        if (!prefetch && frame === state.frame) {
            const needsUpdate = state.previewStale || state.previewSignature !== documentSignature();
            const refinementFailure = state.refinementError || (state.metadata?.refinement_error ? refinementMessage(state.metadata.refinement_error) : "");
            empty.textContent = "";
            if (state.mismatchBlocked && state.sourceMismatch) setStatus(mismatchMessage(state.sourceMismatch), "warn");
            else if (sam3Mode() && sam3TrackReady()) setStatus("Трекинг SAM3 готов. Проверьте кадры и запускайте граф.", "ready");
            else if (sam3Mode() && sam3FrameReady()) setStatus("Кадр SAM3 готов. Нажмите «Отследить видео» для полного трека.", "ready");
            else {
                const refreshAction = sam3Mode() ? "Отследить видео" : "Рассчитать маску";
                const emptyHint = sam3Mode() ? "Кликните по объекту, чтобы увидеть выделение. Трекинг запускается отдельно." : "Нарисуйте маску на опорном кадре, затем нажмите «Рассчитать маску».";
                setStatus(refinementFailure ? `Уточнение не удалось: ${refinementFailure}. Можно отменить рамку, выбрать другой метод или рисовать кистью.` : needsUpdate ? `Маска изменена. Нажмите «${refreshAction}» для проверки результата.` : state.metadata?.has_mask ? "Маска рассчитана. Пролистайте кадры или включите воспроизведение для проверки." : emptyHint, refinementFailure ? "error" : needsUpdate ? "dirty" : "ready");
            }
            updateReadouts(); draw();
            if (!state.playing) diagnostics.record("frame.ready", { frame, sourceSize: [sourceImage.naturalWidth, sourceImage.naturalHeight], hasMask: Boolean(maskImage), layout: layoutSnapshot() });
            const nextFrame = frame + 1 < state.frameCount ? frame + 1 : frame - 1;
            if (nextFrame >= 0 && nextFrame < state.frameCount) loadFrame(nextFrame, { prefetch: true, generation });
        }
        return state.previews.get(frame);
    }

    function ingestDescriptors(output = {}, { allowMask = true } = {}) {
        const descriptors = Array.isArray(output.universal_mask_frames) ? output.universal_mask_frames : [];
        for (const item of descriptors) {
            const frame = Math.max(0, Math.round(number(item?.frame, 0)));
            const rawMask = allowMask ? (item.raw_mask || item.rawMask || item.raw_mask_url) : null;
            const descriptor = {
                ...item,
                source: item.source || item,
                mask: allowMask ? (item.mask || item.mask_url) : null,
                edit_base: allowMask ? item.edit_base : null,
                ...(rawMask ? { raw_mask: rawMask } : {}),
            };
            if (!allowMask) for (const key of ["mask_url", "maskDescriptor", "raw_mask", "rawMask", "raw_mask_url", "edit_base_descriptor", "editBase", "edit_base"]) delete descriptor[key];
            state.frameItems.set(frame, descriptor);
        }
        const previews = Array.isArray(output.universal_mask_preview) ? output.universal_mask_preview : [];
        for (const item of previews.slice(0, MAX_PREVIEWS)) {
            const frame = Math.max(0, Math.round(number(item?.frame, 0)));
            const existing = state.frameItems.get(frame) || {};
            const rawMask = allowMask ? (item.raw_mask || item.rawMask || item.raw_mask_url || existing.raw_mask || existing.rawMask || existing.raw_mask_url) : null;
            const editBase = allowMask ? (item.edit_base || item.editBase || item.edit_base_descriptor || existing.edit_base || existing.editBase || existing.edit_base_descriptor || null) : null;
            const descriptor = {
                ...existing,
                ...item,
                source: item.data_url || item.source || existing.source || item,
                mask: allowMask ? (item.mask_url || item.mask || existing.mask || null) : null,
                edit_base: editBase,
                ...(rawMask ? { raw_mask: rawMask } : {}),
            };
            if (!allowMask) for (const key of ["mask_url", "maskDescriptor", "raw_mask", "rawMask", "raw_mask_url", "edit_base_descriptor", "editBase", "edit_base"]) delete descriptor[key];
            state.frameItems.set(frame, descriptor);
        }
    }

    function applyFramePatch(patch) {
        if (patch?.keep_existing_mask) return;
        // Keep the original source descriptors and decoded images. Drop
        // masks from the previous selection/track, including on other frames.
        state.loadGeneration += 1;
        for (const [frame, item] of state.frameItems) {
            const nextItem = {
                source: item.source || item,
                mask: frame === patch.frame ? patch.mask : null,
                edit_base: frame === patch.frame ? patch.edit_base : null,
            };
            if (frame === patch.frame && patch.raw_mask) nextItem.raw_mask = patch.raw_mask;
            state.frameItems.set(frame, nextItem);
        }
        for (const [frame, preview] of state.previews) {
            state.previews.set(frame, { sourceImage: preview.sourceImage, lastUsed: preview.lastUsed });
        }
    }

    function mismatchMessage(details) {
        return `Источник изменился (${sourceMismatchLabel(details.reason)}): было ${sourceSummary(details.previousSource)}, стало ${sourceSummary(details.source)}. Старые правки сохранены; выберите «Начать маску для нового клипа».`;
    }

    function applySam3Metadata(rawSam3, completedRequestKind = "") {
        if (!sam3Mode()) return;
        const meta = rawSam3 && typeof rawSam3 === "object" ? rawSam3 : {};
        const pendingAction = completedRequestKind === "sam3_frame" ? "frame" : completedRequestKind === "sam3_track" ? "track" : "";
        const returnedToken = String(meta.seed_token || "");
        const expectedToken = String(state.sam3.seedToken || documentState.sam3?.seed_token || "");
        const tokenMatches = Boolean(returnedToken && expectedToken && returnedToken === expectedToken);
        const requestedFrame = Number.isFinite(Number(state.sam3.frame)) ? Math.max(0, Math.round(Number(state.sam3.frame))) : state.frame;
        const returnedFrame = Number.isFinite(Number(meta.frame)) ? Math.max(0, Math.round(Number(meta.frame))) : state.sam3.frame;
        state.metadata.sam3 = clone(meta);
        state.sam3.error = typeof meta.error === "string" ? meta.error : "";
        state.sam3.computedFrames = Array.isArray(meta.computed_frames) ? [...new Set(meta.computed_frames.map((item) => Math.max(0, Math.round(number(item, 0)))))] : [];
        state.sam3.frame = returnedFrame;
        // Frame calculation has no propagation direction in the backend (it
        // reports a compatibility default); retain the user's selected
        // direction until a real track response confirms it.
        state.sam3.direction = pendingAction === "frame"
            ? sam3Direction(state.sam3.direction || directionSelect.value)
            : sam3Direction(meta.direction || state.sam3.direction || directionSelect.value);
        state.sam3.sourceKey = currentSourceKey();
        directionSelect.value = state.sam3.direction;
        if (returnedToken && (pendingAction !== "track" || tokenMatches)) state.sam3.seedToken = returnedToken;
        // Backend readiness is authoritative, but only for the exact source,
        // seed token and request signature that is currently in the editor.
        if (pendingAction === "frame") {
            // The frame step mints the authoritative backend token.  A
            // client-side placeholder token (or an older token) must not make
            // a valid fresh frame look stale; the strict equality check is
            // required on the subsequent track step.
            state.sam3.frameReady = Boolean(meta.frame_ready && returnedToken && returnedFrame === requestedFrame);
            state.sam3.trackReady = false;
            state.sam3.frameSignature = state.sam3.frameReady ? sam3ReadinessSignature({ frame: returnedFrame, direction: state.sam3.direction, range: false }) : "";
            state.sam3.trackSignature = "";
            state.sam3.pendingAction = "frame";
            documentState.sam3 = { ...(documentState.sam3 || {}), action: "frame", frame: returnedFrame, direction: state.sam3.direction, seed_token: state.sam3.seedToken };
            commit({ markDirty: false, sam3Action: "frame" });
        } else if (pendingAction === "track") {
            state.sam3.frameReady = Boolean(meta.frame_ready && tokenMatches);
            state.sam3.trackReady = Boolean(meta.track_ready && tokenMatches && state.sam3.frameReady);
            state.sam3.frameSignature = state.sam3.frameReady ? sam3ReadinessSignature({ frame: returnedFrame, direction: state.sam3.direction, range: false }) : "";
            state.sam3.trackSignature = state.sam3.trackReady ? sam3ReadinessSignature({ frame: returnedFrame, direction: state.sam3.direction, range: true }) : "";
            state.sam3.pendingAction = state.sam3.trackReady ? "track" : "frame";
            documentState.sam3 = { ...(documentState.sam3 || {}), action: state.sam3.trackReady ? "track" : "frame", frame: returnedFrame, direction: state.sam3.direction, seed_token: state.sam3.seedToken };
            commit({ markDirty: false, sam3Action: state.sam3.trackReady ? "track" : "frame" });
        } else if (meta.frame_ready || meta.track_ready) {
            // A configured workflow can reopen with metadata from a previous
            // run. Accept it only when its token/source still match.
            state.sam3.frameReady = Boolean(meta.frame_ready && tokenMatches);
            state.sam3.trackReady = Boolean(meta.track_ready && tokenMatches && state.sam3.frameReady);
            state.sam3.frameSignature = state.sam3.frameReady ? sam3ReadinessSignature({ frame: returnedFrame, direction: state.sam3.direction, range: false }) : "";
            state.sam3.trackSignature = state.sam3.trackReady ? sam3ReadinessSignature({ frame: returnedFrame, direction: state.sam3.direction, range: true }) : "";
        }
        if (state.sam3.error) { state.sam3.frameReady = false; state.sam3.trackReady = false; state.sam3.frameSignature = ""; state.sam3.trackSignature = ""; }
    }

    function ingestOutput(output = {}, options = {}) {
        if (state.destroyed) return { ignored: true };
        const rawMeta = Array.isArray(output.universal_mask_meta) ? output.universal_mask_meta[0] : output.universal_mask_meta;
        const framePatch = output.universal_mask_frame_patch?.[0];
        diagnostics.record("preview.received", { source: sourceInfo(rawMeta), hasMask: Boolean(rawMeta?.has_mask), mismatch: Boolean(rawMeta?.source_mismatch), reason: rawMeta?.source_mismatch_reason,
            refinementError: diagnosticText(rawMeta?.refinement_error), descriptors: output.universal_mask_frames?.length || 0, acceptMismatch: Boolean(options.acceptMismatch) });
        const nextSource = sourceInfo(rawMeta);
        if (framePatch && !state.pendingPreview?.framePatch) {
            diagnostics.record("preview.patch_discarded", { frame: framePatch.frame });
            return { stale: true, hasPreview: state.previews.size > 0, hasMask: false };
        }
        if (state.pendingPreview?.promptId && output?.prompt_id && String(state.pendingPreview.promptId) !== String(output.prompt_id)) {
            diagnostics.record("preview.stale_prompt", { expected: state.pendingPreview.promptId, received: output.prompt_id });
            return { stale: true, hasPreview: state.previews.size > 0, hasMask: false };
        }
        const previousSource = sourceInfo(documentState) || state.sourceMismatch?.previousSource || state.source || sourceInfo(rawMeta?.previous_source ? { source: rawMeta.previous_source } : null);
        const rawMismatch = Boolean(rawMeta?.source_mismatch);
        const oldFingerprint = previousSource?.fingerprint || "";
        const newFingerprint = nextSource?.fingerprint || "";
        const legacyMismatch = Boolean(nextSource && oldFingerprint && newFingerprint && oldFingerprint !== newFingerprint && state.localEdits && !options.acceptMismatch);
        if ((rawMismatch || legacyMismatch) && !options.acceptMismatch) {
            const mismatch = {
                reason: sourceMismatchReason(previousSource, nextSource, rawMeta?.source_mismatch_reason),
                previousSource: sourceInfo(rawMeta?.previous_source ? { source: rawMeta.previous_source } : null) || clone(previousSource),
                source: clone(nextSource),
            };
            capturePreviousMask(mismatch.previousSource);
            state.pendingSource = { source: clone(nextSource), output, mismatch };
            state.sourceMismatch = mismatch;
            state.lastMismatch = clone(mismatch);
            state.mismatchBlocked = true;
            state.pendingPreview = null;
            state.pendingQueue = null;
            state.pendingPromptId = null;
            state.pendingRequestKind = "";
            state.refinementBusy = false;
            state.refinementError = "";
            state.previewStale = true;
            setPrepareBusy(false);
            clearPreviewCaches();
            if (nextSource) {
                state.source = clone(nextSource);
                state.frameCount = nextSource.frames;
                state.frame = clamp(state.frame, 0, Math.max(0, state.frameCount - 1));
                state.zoom = 1; state.panX = 0; state.panY = 0;
            }
            state.metadata = rawMeta && typeof rawMeta === "object" ? clone(rawMeta) : {};
            state.metadata.has_mask = false;
            state.metadataAnchors = [];
            state.sam3.frameReady = false; state.sam3.trackReady = false; state.sam3.frameSignature = ""; state.sam3.trackSignature = ""; state.sam3.error = "Источник изменился"; state.sam3.sourceKey = currentSourceKey();
            state.fps = number(rawMeta?.fps, number(widgetValue(node, "fps", 24), 24));
            // The backend may return source-only descriptors for the new clip.
            // Keep them visible, but never retain a stale mask/edit baseline.
            ingestDescriptors(output, { allowMask: false });
            updateReadouts();
            setStatus(mismatchMessage(mismatch), "warn");
            updateActionAvailability();
            if (state.frameItems.size) { empty.textContent = "Загрузка кадра…"; loadFrame(state.frame, { generation: state.loadGeneration }); }
            else { empty.textContent = "Выберите новый клип или восстановите прежнюю маску"; draw(); }
            return { sourceMismatch: true, pendingSource: true, hasPreview: state.frameItems.size > 0, hasMask: false, metadata: state.metadata };
        }
        if (state.pendingPreview && (state.pendingPreview.revision !== state.editRevision || state.pendingPreview.snapshot !== widgetSnapshot(node) || (state.pendingPreview.sourceKey && state.pendingPreview.sourceKey !== currentSourceKey()) || (state.pendingPreview.sam3Signature && state.pendingPreview.sam3Signature !== sam3ReadinessSignature()))) {
            state.pendingPreview = null; state.refinementBusy = false; setPrepareBusy(false); updateActionAvailability();
            setStatus("Получено устаревшее превью — правки изменились после запуска подготовки. Запустите обновление ещё раз.", "warn");
            return { stale: true, hasPreview: state.previews.size > 0, hasMask: false };
        }
        if (framePatch && (framePatch.frame !== state.pendingPreview.seedFrame || !state.frameItems.has(framePatch.frame) ||
            currentSourceKey(nextSource) !== currentSourceKey())) {
            finishPending("preview");
            setStatus("Предпросмотр относится к другому кадру или источнику. Загрузите кадры заново.", "warn");
            return { stale: true, hasPreview: state.previews.size > 0, hasMask: false };
        }
        const completedRequestKind = state.pendingRequestKind;
        state.pendingPreview = null;
        state.pendingPromptId = null;
        state.pendingRequestKind = "";
        state.refinementBusy = false;
        if (framePatch) applyFramePatch(framePatch);
        else clearPreviewCaches();
        state.previewStale = false;
        state.resetBlank = false;
        if (nextSource) {
            state.source = nextSource;
            state.frameCount = nextSource.frames;
            if (!state.localEdits) documentState.source = clone(nextSource);
        }
        state.metadata = rawMeta && typeof rawMeta === "object"
            ? { ...(framePatch ? state.metadata : {}), ...clone(rawMeta) }
            : {};
        state.serverMaskSettings = rawMeta?.mask_postprocess
            ? {
                grow: clamp(Math.round(number(rawMeta.mask_postprocess.grow, 0)), 0, 256),
                feather: clamp(Math.round(number(rawMeta.mask_postprocess.feather, 0)), 0, 256),
            }
            : snapshotMaskSettings();
        maskPostprocessor.invalidate();
        state.metadataAnchors = Array.isArray(rawMeta?.anchors) ? explicitAnchors({ anchors: rawMeta.anchors }) : [];
        state.fps = number(rawMeta?.fps, number(widgetValue(node, "fps", 24), 24));
        state.refinementError = typeof rawMeta?.refinement_error === "string" ? refinementMessage(rawMeta.refinement_error) : "";
        if (state.refinementError) state.previewStale = true;
        applySam3Metadata(rawMeta?.sam3, completedRequestKind);
        const currentActiveLayer = activeLayer();
        if (currentActiveLayer?.type === "sam3") {
            state.layerSam3.set(currentActiveLayer.id, clone(state.sam3));
        }
        if (Array.isArray(rawMeta?.layers)) {
            const layers = currentLayers();
            for (const item of rawMeta.layers) {
                if (!item || typeof item !== "object") continue;
                const layerId = String(item.id || "");
                state.layerMaskFlags.set(layerId, Boolean(item.has_mask));
                const localLayer = layers.find((entry) => entry.id === layerId);
                if (localLayer?.type === "sam3" && item.sam3 && typeof item.sam3 === "object") {
                    const returnedToken = String(item.sam3.seed_token || "");
                    if (returnedToken) {
                        localLayer.sam3 = { ...(localLayer.sam3 || {}), seed_token: returnedToken };
                    }
                    if (layerId !== currentActiveLayer?.id) {
                        const prev = state.layerSam3.get(layerId) || {};
                        state.layerSam3.set(layerId, {
                            ...prev,
                            frameReady: Boolean(item.sam3.frame_ready),
                            trackReady: Boolean(item.sam3.track_ready),
                            frame: Number.isFinite(Number(item.sam3.frame)) ? Math.max(0, Math.round(Number(item.sam3.frame))) : (prev.frame ?? 0),
                            seedToken: returnedToken || String(prev.seedToken || ""),
                            direction: sam3Direction(item.sam3.direction || localLayer.sam3?.direction || prev.direction || "both"),
                            computedFrames: Array.isArray(item.sam3.computed_frames) ? clone(item.sam3.computed_frames) : (prev.computedFrames || []),
                            error: typeof item.sam3.error === "string" ? item.sam3.error : "",
                            sourceKey: currentSourceKey(nextSource || state.source),
                            pendingAction: item.sam3.track_ready ? "track" : "frame",
                        });
                    }
                }
            }
        }
        if (framePatch?.object_map) {
            state.sam3HitMap = { url: framePatch.object_map, signature: sam3SeedSignature(framePatch.frame) };
        }
        ingestDescriptors(output, { allowMask: !state.refinementError });
        state.serverActiveLayerId = String(rawMeta?.active_layer_id || activeLayer()?.id || "");
        if (!state.refinementError && state.metadata.has_mask) {
            const isRefineRect = (ev) => Boolean(ev && ev.type === "rect" && ev.mode !== "erase" && ev.refine);
            if (!framePatch) state.serverRefinedFrames.clear();
            if (!framePatch && safeMode(node, documentState) === "static") {
                const allEvents = Object.keys(documentState.frames || {})
                    .sort((a, b) => Number(a) - Number(b))
                    .flatMap((key) => parseEvents(documentState.frames[key]));
                if (allEvents.some(isRefineRect)) {
                    state.serverRefinedFrames.set("__static__", {
                        baseCount: allEvents.length,
                        prefixJson: JSON.stringify(allEvents),
                    });
                }
            } else {
                const entries = framePatch
                    ? [[String(framePatch.frame), documentState.frames?.[String(framePatch.frame)]]]
                    : Object.entries(documentState.frames || {});
                for (const [key, bucket] of entries) {
                    const evs = parseEvents(bucket);
                    if (evs.some(isRefineRect)) {
                        state.serverRefinedFrames.set(String(key), {
                            baseCount: evs.length,
                            prefixJson: JSON.stringify(evs),
                        });
                    } else if (framePatch) {
                        state.serverRefinedFrames.delete(String(key));
                    }
                }
            }
        } else if (!framePatch) {
            state.serverRefinedFrames.clear();
        }
        setPrepareBusy(false);
        state.frame = clamp(state.frame, 0, Math.max(0, state.frameCount - 1));
        if (completedRequestKind === "refinement" && state.metadata.has_mask && !state.refinementError) {
            state.view = "overlay";
            for (const [name, item] of Object.entries(viewButtons)) item.style.background = name === "overlay" ? COLORS.selected : COLORS.panel2;
        }
        state.previewSignature = state.refinementError ? "" : documentSignature();
        state.sourceMismatch = null;
        state.mismatchBlocked = false;
        state.pendingSource = null;
        updateReadouts();
        updateActionAvailability();
        if (state.refinementError) {
            setStatus(`Уточнение не удалось: ${state.refinementError}. Можно отменить рамку, выбрать другой метод или рисовать кистью.`, "error");
            if (state.frameItems.size) { empty.textContent = "Загрузка кадра…"; loadFrame(state.frame, { generation: state.loadGeneration }); }
            else { empty.textContent = "Подготовьте превью, чтобы открыть кадры"; draw(); }
        } else if (completedRequestKind === "sam3_frame") {
            empty.textContent = state.frameItems.size ? "Загрузка кадра…" : "Рассчитайте выделение SAM3 на выбранном кадре";
            const frameStatusMsg = !state.sam3.frameReady
                ? (state.sam3.error || "Кадр SAM3 не готов: кликните по объекту или задайте рамку и повторите расчёт.")
                : rawMeta?.sam3?.keep_existing_mask
                    ? `Точка уточнения сохранена на кадре ${state.frame + 1}. Нажмите «Отследить видео», чтобы применить правку.`
                    : "Объект выделен. Уточните точками +/− или нажмите «Отследить видео».";
            setStatus(frameStatusMsg, state.sam3.frameReady ? "ready" : "warn");
            if (state.frameItems.size) loadFrame(state.frame, { generation: state.loadGeneration }); else draw();
        } else if (completedRequestKind === "sam3_track") {
            empty.textContent = state.frameItems.size ? "Загрузка кадра…" : "Рассчитайте трекинг SAM3 заново";
            setStatus(state.sam3.trackReady ? `Трекинг SAM3 готов: ${state.sam3.computedFrames.length || "все"} кадров. Можно запускать граф.` : state.sam3.error || "Трекинг SAM3 не готов: повторите расчёт.", state.sam3.trackReady ? "ready" : "warn");
            if (state.frameItems.size) loadFrame(state.frame, { generation: state.loadGeneration }); else draw();
        } else if (state.frameItems.size) {
            empty.textContent = "Загрузка кадра…";
            setStatus(state.metadata.has_mask ? "Маска рассчитана. Проверьте результат в режиме «Маска»." : "Загрузка выбранного кадра…");
            loadFrame(state.frame, { generation: state.loadGeneration });
        } else {
            empty.textContent = "Подготовьте превью, чтобы открыть кадры";
            setStatus(rawMeta?.status || (sam3Mode() ? "Нет кадров. Нажмите «Загрузить кадры»." : "Нет кадров. Нажмите «Подготовить / обновить маску»."));
            draw();
        }
        return { hasPreview: state.frameItems.size > 0, hasMask: Boolean(rawMeta?.has_mask) && !state.refinementError, metadata: state.metadata };
    }

    async function applyExecuted(output = {}) {
        if (output?.execution_error || output?.execution_interrupted) {
            runtimeEvent({ ...output, prompt_id: output.prompt_id || output.promptId }, output.execution_interrupted ? "execution_interrupted" : "execution_error");
            return { execution_error: true };
        }
        if (state.pendingQueue && output?.prompt_id && String(output.prompt_id) === String(state.pendingQueue.promptId)) finishPending("queue");
        return ingestOutput(output);
    }

    function addEvent(event, label = "Изменение") {
        if (state.mismatchBlocked) { setStatus("Источник не совпадает с сохранённой маской. Примите новый клип или восстановите прежний источник.", "warn"); return false; }
        const events = eventsAt(state.frame, true, event);
        if (events.length >= MAX_EVENTS || totalPoints() + (event.points?.length || 0) > MAX_TOTAL_POINTS) { setStatus("Лимит правок достигнут: отмените или очистите кадр.", "warn"); return false; }
        if (event.points && event.points.length > MAX_POINTS_PER_STROKE) event.points = event.points.slice(0, MAX_POINTS_PER_STROKE);
        const isSam3Seed = sam3Mode() && event?.type === "rect";
        if (isSam3Seed) {
            const hasAnyLegacySam3Box = Object.keys(documentState.frames || {}).some((key) => parseEvents(documentState.frames[key]).some(isSam3Box));
            if (sam3Objects().length || hasAnyLegacySam3Box) {
                const objects = editableSam3Objects();
                if (!objects || !canAddSam3Object(objects)) return false;
                const { x, y, w, h } = event;
                objects.push({ frame: state.frame, x, y, w, h });
                writeSam3Objects(objects, objects.length - 1, "Добавлена рамка SAM3");
                setStatus("Нажмите «Рассчитать кадр», чтобы увидеть общую маску.", "dirty");
                return true;
            }
            event = { ...event, refine: true, refine_method: "sam3" };
            // A new seed changes the SAM3 identity. Manual brush/erase events
            // never rotate this token, so retracking can preserve them.
            state.sam3.seedToken = makeSeedToken();
            documentState.sam3 = { ...(documentState.sam3 || {}), action: "frame", frame: state.frame, direction: sam3Direction(directionSelect.value), seed_token: state.sam3.seedToken };
            invalidateSam3({ frame: true, track: true, reason: "Новая рамка SAM3 требует расчёта кадра." });
        }
        const bucket = frameBucket(state.frame, true, event);
        events.push(event);
        if (bucket.inherit === undefined) bucket.inherit = shouldInherit(state.frame, event);
        if (sam3Mode() && !isSam3Seed && (sam3FrameReady() || sam3TrackReady() || hasReliableEditBase(state.frame))) {
            bucket.inherit = true;
        }
        state.refinementError = "";
        pushHistory(label); commit({ markDirty: true }); updateReadouts(); updateActionAvailability(); draw();
        if (sam3Mode() && !isSam3Seed) invalidateSam3({ track: true, reason: "Покадровая правка требует повторного трекинга." });
        diagnostics.record("mask.edit", { type: event.type, mode: event.mode, refine: Boolean(event.refine), refineMethod: event.refine ? event.refine_method || "grabcut" : undefined, points: event.points?.length || 0 }); return true;
    }

    function editableSam3Objects() {
        const explicit = sam3Objects();
        if (explicit.length) return clone(explicit);
        const legacy = [];
        for (const key of Object.keys(documentState.frames || {}).sort((a, b) => Number(a) - Number(b))) {
            const frameNum = Number(key);
            for (const { x, y, w, h } of parseEvents(documentState.frames[key]).filter(isSam3Box)) {
                legacy.push({ frame: frameNum, x, y, w, h });
            }
        }
        return legacy;
    }

    function canAddSam3Object(objects) {
        if (objects.length < MAX_SAM3_OBJECTS) return true;
        setStatus(`В одной маске доступно до ${MAX_SAM3_OBJECTS} объектов SAM3.`, "warn");
        return false;
    }

    function writeSam3Objects(objects, active, historyLabel, { keepMask = false } = {}) {
        // Promote old rectangle prompts across all frames into the explicit
        // object list. Leave all manual per-frame edit events in place.
        for (const key of Object.keys(documentState.frames || {})) {
            const bucket = documentState.frames[key];
            if (!bucket) continue;
            bucket.events = parseEvents(bucket).filter(event => !isSam3Box(event));
            if (bucket.events.length === 0 && bucket.inherit === false) {
                delete documentState.frames[key];
            }
        }
        state.sam3.seedToken = objects.length ? makeSeedToken() : "";
        state.sam3.frame = state.frame;
        state.activeSam3Object = active;
        const sam3 = { action: "frame", frame: state.frame, direction: sam3Direction(directionSelect.value), seed_token: state.sam3.seedToken };
        if (objects.length === 1) sam3.prompt = objects[0];
        else if (objects.length) sam3.objects = objects;
        documentState.sam3 = sam3;
        state.refinementError = ""; state.previewStale = true; state.resetBlank = !keepMask;
        state.view = "overlay";
        for (const [name, item] of Object.entries(viewButtons)) item.setAttribute("aria-pressed", String(name === "overlay"));
        viewToolbarController.refresh(viewButtons.overlay);
        invalidateSam3({ frame: true, track: true });
        pushHistory(historyLabel);
        commit({ markDirty: true, sam3Action: "frame" }); updateReadouts(); draw();
        updateActionAvailability();
    }

    function selectSam3Point(point, tool) {
        if (!sam3Mode() || state.destroyed || state.pendingPreview || state.pendingQueue || state.mismatchBlocked) return false;
        const objects = editableSam3Objects();
        if (!objects) return false;
        const append = tool !== "sam_point";
        if (append && !objects.length) {
            setStatus("Сначала выберите объект прицелом, затем уточняйте точками «+» и «−».", "warn");
            return false;
        }
        const active = append ? clamp(state.activeSam3Object || 0, 0, Math.max(0, objects.length - 1)) : objects.length;
        if (!append && !canAddSam3Object(objects)) return false;
        const target = append ? objects[active] : { frame: state.frame, points: [], labels: [] };
        const existingRefPoints = (target.refinements || []).reduce((acc, item) => acc + (item.points?.length || 0), 0);
        const totalTargetPoints = (target.points?.length || 0) + existingRefPoints;
        if (totalTargetPoints >= 64) { setStatus("Достигнут лимит 64 точек на объект.", "warn"); return false; }

        let keepMask = false;
        const coord = [Number(point.x.toFixed(6)), Number(point.y.toFixed(6))];
        const label = tool === "sam_remove" ? 0 : 1;
        if (!append || Number(target.frame) === state.frame) {
            if (!target?.points) {
                setStatus("Для точек «+» и «−» выберите объект, созданный прицелом.", "warn");
                return false;
            }
            target.points.push(coord);
            target.labels.push(label);
            objects[active] = target;
        } else {
            const refs = Array.isArray(target.refinements) ? target.refinements : (target.refinements = []);
            let ref = refs.find((item) => Number(item.frame) === state.frame);
            if (!ref) {
                ref = { frame: state.frame, points: [], labels: [] };
                refs.push(ref);
                refs.sort((a, b) => Number(a.frame) - Number(b.frame));
            }
            ref.points.push(coord);
            ref.labels.push(label);
            objects[active] = target;
            keepMask = !ref.labels.includes(1) && !objects.some((item) => Number(item.frame) === state.frame);
        }
        writeSam3Objects(objects, active, tool === "sam_remove" ? "Точка исключения SAM3" : append ? "Точка уточнения SAM3" : "Добавлен объект SAM3", { keepMask });
        diagnostics.record("sam3.point", { tool, points: totalTargetPoints + 1, frame: state.frame, objects: objects.length });
        requestSam3("frame");
        return true;
    }

    async function removeSam3At(event) {
        if (!sam3Mode() || event.button !== 2 || state.playing) return;
        const point = pointFromEvent(event);
        if (!point) return;
        event.preventDefault();
        event.stopPropagation?.();
        if (state.pendingPreview || state.pendingQueue || state.mismatchBlocked) return;
        const objects = editableSam3Objects();
        if (!objects?.length) return;

        const box = geometry();
        let nearest = null;
        for (const [objectIndex, obj] of objects.entries()) {
            if (Number(obj.frame) === state.frame) {
                for (const [index, [x, y]] of (obj.points || []).entries()) {
                    const distance = Math.hypot((point.x - x) * box.drawnWidth * box.clientScaleX, (point.y - y) * box.drawnHeight * box.clientScaleY);
                    if (distance <= 12 && (!nearest || distance < nearest.distance || (distance === nearest.distance && objectIndex === state.activeSam3Object))) {
                        nearest = { object: objectIndex, refIndex: -1, index, distance };
                    }
                }
            }
            for (const [refIndex, ref] of (obj.refinements || []).entries()) {
                if (Number(ref.frame) !== state.frame) continue;
                for (const [index, [x, y]] of (ref.points || []).entries()) {
                    const distance = Math.hypot((point.x - x) * box.drawnWidth * box.clientScaleX, (point.y - y) * box.drawnHeight * box.clientScaleY);
                    if (distance <= 12 && (!nearest || distance < nearest.distance || (distance === nearest.distance && objectIndex === state.activeSam3Object))) {
                        nearest = { object: objectIndex, refIndex, index, distance };
                    }
                }
            }
        }
        let object = nearest?.object;
        let removeObject = !nearest;
        if (nearest) {
            if (nearest.refIndex >= 0) {
                const refs = objects[object].refinements || [];
                const ref = refs[nearest.refIndex];
                ref.points.splice(nearest.index, 1);
                ref.labels.splice(nearest.index, 1);
                if (!ref.points.length) refs.splice(nearest.refIndex, 1);
                if (!refs.length) delete objects[object].refinements;
                removeObject = false;
            } else {
                const { points, labels } = objects[object];
                points.splice(nearest.index, 1);
                labels.splice(nearest.index, 1);
                removeObject = !labels.includes(1);
            }
        } else {
            if (state.view === "source") return;
            const frame = state.frame;
            const signature = sam3SeedSignature(frame);
            if (state.sam3HitMap?.signature !== signature) {
                setStatus("Нажмите «Рассчитать кадр», затем кликните по области ещё раз.", "warn");
                return;
            }
            const image = await loadImage(state.sam3HitMap.url);
            if (!image || state.destroyed || frame !== state.frame || state.pendingPreview || state.pendingQueue || state.mismatchBlocked || sam3SeedSignature(frame) !== signature) return;
            const sampler = document.createElement("canvas");
            sampler.width = 1; sampler.height = 1;
            const context = sampler.getContext("2d");
            if (!context) return;
            const x = Math.min(image.naturalWidth - 1, Math.floor(point.x * image.naturalWidth));
            const y = Math.min(image.naturalHeight - 1, Math.floor(point.y * image.naturalHeight));
            context.drawImage(image, x, y, 1, 1, 0, 0, 1, 1);
            const bits = context.getImageData(0, 0, 1, 1).data[0];
            object = state.activeSam3Object;
            if (!(bits & (1 << object))) object = objects.findLastIndex((_, index) => bits & (1 << index));
            if (object < 0) return;
        }
        if (removeObject) objects.splice(object, 1);
        const keepMaskOnDelete = Boolean(nearest && nearest.refIndex >= 0 && !objects.some((item) => Number(item.frame) === state.frame));
        writeSam3Objects(objects, clamp(object, 0, Math.max(0, objects.length - 1)), removeObject ? "Объект SAM3 удалён" : "Точка SAM3 удалена", { keepMask: keepMaskOnDelete });
        if (objects.length && hasSam3Seed(state.frame)) requestSam3("frame");
        else if (objects.length) {
            invalidateSam3({ track: true, reason: "Точка уточнения удалена — обновите трекинг." });
            setStatus("Точка уточнения удалена. Нажмите «Отследить видео», чтобы обновить маску.", "dirty");
        } else setStatus("Выделение очищено. Кликните по следующему объекту.", "dirty");
    }

    function finishPen() {
        state.penHover = null; state.penSnapClose = false;
        if (state.active?.type === "pen_anchor" || state.active?.type === "pen_close") {
            canvas.releasePointerCapture?.(state.active.pointerId);
            state.active = null;
        }
        if (!canClosePenPath(state.penPath)) { state.penPath = []; draw(); return; }
        const points = flattenPenPath(state.penPath, { closed: true });
        if (points.length < 3) { state.penPath = []; draw(); return; }
        const accepted = addEvent({ type: "polygon", mode: "add", points }, "Контур пером добавлен");
        state.penPath = [];
        if (accepted) {
            setStatus(`Контур пером сохранён на кадре ${state.frame + 1}. Нажмите «${sam3Mode() ? "Отследить видео" : "Рассчитать маску"}», чтобы проверить результат.`, "dirty");
        }
        draw();
    }

    function finishPointer(event) {
        if (!state.active || state.playing || (event.pointerId !== undefined && event.pointerId !== state.active.pointerId)) return;
        const active = state.active; state.active = null; canvas.releasePointerCapture?.(active.pointerId);
        if (active.type === "pen_anchor") { draw(); return; }
        if (active.type === "pen_close") { finishPen(); return; }
        if (active.type === "sam_click") {
            const moved = Math.hypot(event.clientX - active.clientX, event.clientY - active.clientY);
            if (!active.moved && moved <= 5 && pointFromEvent(event)) selectSam3Point(active.start, active.tool);
            draw(); return;
        }
        if (active.type === "stroke" && active.points.length) {
            const accepted = addEvent(active, active.mode === "erase" ? "Ластик" : "Кисть");
            if (accepted) {
                setStatus(`Правка сохранена на кадре ${state.frame + 1}. Нажмите «${sam3Mode() ? "Отследить видео" : "Рассчитать маску"}», чтобы проверить результат.`, "dirty");
            }
        } else if (active.type === "rect" && state.ghost) {
            const rect = state.ghost;
            if (rect.w > .002 && rect.h > .002) {
                const accepted = addEvent(rect, rect.refine ? "Рамка выделения добавлена" : "Прямоугольник добавлен");
                if (accepted && rect.refine) setStatus(`Рамка сохранена. Проверьте её или отмените правку, затем нажмите «${sam3Mode() ? "Рассчитать кадр" : "Рассчитать маску"}».`, "dirty");
            }
        }
        state.ghost = null; draw();
    }

    function cancelGesture() {
        if (state.panning) canvas.releasePointerCapture?.(state.panning.pointerId);
        if (state.active) canvas.releasePointerCapture?.(state.active.pointerId);
        state.panning = null; canvas.style.cursor = cursorForMode(state.mode);
        state.active = null; state.ghost = null; state.penPath = []; state.penHover = null; state.penSnapClose = false;
        draw();
    }

    canvas.addEventListener("pointerdown", (event) => {
        if (state.playing || state.mismatchBlocked || event.button !== 0 || event.altKey || event.shiftKey || state.panning) return;
        canvas.focus?.({ preventScroll: true }); const point = pointFromEvent(event); if (!point) return;
        if (state.mode.startsWith("sam_")) {
            if (!sam3Mode() || state.pendingPreview || state.pendingQueue) { setStatus("Дождитесь предпросмотра или отмените текущий запрос.", "busy"); return; }
            state.active = { type: "sam_click", pointerId: event.pointerId, start: point, tool: state.mode, clientX: event.clientX, clientY: event.clientY, moved: false };
            canvas.setPointerCapture?.(event.pointerId); event.preventDefault?.(); return;
        }
        if (state.mode === "pen") {
            if (state.active?.type === "pen_anchor" || state.active?.type === "pen_close") {
                canvas.releasePointerCapture?.(state.active.pointerId);
                state.active = null;
            }
            const box = geometry();
            if (box && canClosePenPath(state.penPath) && isNearPenAnchor(point, state.penPath[0], box, 10)) {
                state.penSnapClose = true;
                state.active = { type: "pen_close", pointerId: event.pointerId, anchor: state.penPath[0], startClientX: number(event.clientX, 0), startClientY: number(event.clientY, 0), moved: false };
                canvas.setPointerCapture?.(event.pointerId); event.preventDefault?.(); draw(); return;
            }
            if (state.penPath.length < MAX_POINTS_PER_STROKE) {
                const anchor = { x: roundCoord(point.x), y: roundCoord(point.y), inX: 0, inY: 0, outX: 0, outY: 0 };
                state.penPath.push(anchor);
                state.penSnapClose = false;
                state.active = { type: "pen_anchor", pointerId: event.pointerId, anchor, startClientX: number(event.clientX, 0), startClientY: number(event.clientY, 0), moved: false };
                canvas.setPointerCapture?.(event.pointerId);
            }
            event.preventDefault?.(); draw(); return;
        }
        state.active = { pointerId: event.pointerId, type: state.mode === "rect" ? "rect" : "stroke", mode: state.mode === "erase" ? "erase" : "add", radius: number(brushInput.value, .035), points: [[Number(point.x.toFixed(6)), Number(point.y.toFixed(6))]], start: point };
        const rectangleSeed = sam3Mode() || state.grabcut;
        state.ghost = state.active.type === "stroke" ? state.active : { type: "rect", mode: "add", x: point.x, y: point.y, w: 0, h: 0, refine: rectangleSeed, ...(rectangleSeed ? { refine_method: sam3Mode() ? "sam3" : state.refineMethod } : {}) };
        canvas.setPointerCapture?.(event.pointerId); event.preventDefault?.(); draw();
    });
    canvas.addEventListener("pointermove", (event) => {
        updateBrushCursor(event); if (state.playing) return;
        if (!state.active) {
            if (!state.panning && state.mode === "pen" && state.penPath.length) {
                const hover = pointFromEvent(event); const box = geometry();
                if (hover && box) {
                    state.penHover = { x: roundCoord(hover.x), y: roundCoord(hover.y) };
                    state.penSnapClose = canClosePenPath(state.penPath) && isNearPenAnchor(hover, state.penPath[0], box, 10);
                    draw();
                } else if (state.penHover || state.penSnapClose) {
                    state.penHover = null; state.penSnapClose = false; draw();
                }
            }
            return;
        }
        if (state.active.type === "pen_anchor" || state.active.type === "pen_close") {
            const pt = rawPointFromEvent(event); if (!pt) return;
            const moved = Math.hypot(number(event.clientX, state.active.startClientX) - state.active.startClientX, number(event.clientY, state.active.startClientY) - state.active.startClientY);
            if (!state.active.moved && moved <= 3) return;
            state.active.moved = true;
            const anchor = state.active.anchor;
            const dx = roundOffset(pt.x - anchor.x), dy = roundOffset(pt.y - anchor.y);
            if (state.active.type === "pen_close") {
                anchor.inX = roundOffset(-dx); anchor.inY = roundOffset(-dy);
            } else {
                anchor.outX = dx; anchor.outY = dy;
                anchor.inX = roundOffset(-dx); anchor.inY = roundOffset(-dy);
            }
            draw(); return;
        }
        const point = pointFromEvent(event); if (!point) return;
        if (state.active.type === "sam_click") { if (Math.hypot(event.clientX - state.active.clientX, event.clientY - state.active.clientY) > 5) state.active.moved = true; return; }
        if (state.active.type === "stroke") { const previous = state.active.points.at(-1); if ((!previous || Math.hypot(point.x - previous[0], point.y - previous[1]) > .002) && state.active.points.length < MAX_POINTS_PER_STROKE) { state.active.points.push([Number(point.x.toFixed(6)), Number(point.y.toFixed(6))]); state.ghost = state.active; draw(); } }
        else { const x = Math.min(state.active.start.x, point.x), y = Math.min(state.active.start.y, point.y); const rectangleSeed = sam3Mode() || state.grabcut; state.ghost = { type: "rect", mode: "add", x, y, w: Math.abs(point.x - state.active.start.x), h: Math.abs(point.y - state.active.start.y), refine: rectangleSeed, ...(rectangleSeed ? { refine_method: sam3Mode() ? "sam3" : state.refineMethod } : {}) }; draw(); }
    });
    canvas.addEventListener("pointerup", finishPointer); canvas.addEventListener("pointercancel", cancelGesture);
    canvas.addEventListener("pointerleave", () => {
        state.lastPointer = null; state.penHover = null; state.penSnapClose = false;
        if (!state.active) {
            brushCursor.style.display = "none";
            if (state.mode === "pen" && state.penPath.length) draw();
        }
    });
    canvas.addEventListener("dblclick", () => {
        if (state.mode !== "pen") return;
        if (state.penPath.length >= 2) {
            const box = geometry();
            const tail = state.penPath.at(-1);
            const prev = state.penPath.at(-2);
            if (box && tail && prev && !penAnchorHasHandles(tail) && isNearPenAnchor(tail, prev, box, 4)) state.penPath.pop();
        }
        finishPen();
    });

    function zoomBy(factor, event = null) {
        const box = geometry();
        const nextZoom = clamp(state.zoom * factor, .25, 8);
        if (box) {
            const x = event ? (number(event.clientX, box.rect.left + box.rect.width / 2) - box.rect.left) / box.clientScaleX : box.viewportWidth / 2;
            const y = event ? (number(event.clientY, box.rect.top + box.rect.height / 2) - box.rect.top) / box.clientScaleY : box.viewportHeight / 2;
            const u = (x - box.offsetX) / box.drawnWidth, v = (y - box.offsetY) / box.drawnHeight;
            const width = box.width * box.fitScale * nextZoom, height = box.height * box.fitScale * nextZoom;
            state.panX = x - (box.viewportWidth - width) / 2 - u * width;
            state.panY = y - (box.viewportHeight - height) / 2 - v * height;
        }
        state.zoom = nextZoom; draw();
    }
    canvas.addEventListener("wheel", (event) => {
        if (!state.dialogOpen && !event.ctrlKey && !event.metaKey) return;
        event.preventDefault?.(); event.stopPropagation?.();
        if (state.active || state.panning || !event.deltaY) return;
        zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12, event);
    }, { passive: false });
    canvas.addEventListener("pointerdown", (event) => {
        if (event.button !== 1 && !(event.button === 0 && (event.altKey || event.shiftKey))) return;
        event.preventDefault?.(); event.stopPropagation?.();
        const box = geometry(); if (!box || state.active) return;
        state.panning = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY, scaleX: box.clientScaleX, scaleY: box.clientScaleY };
        brushCursor.style.display = "none"; canvas.style.cursor = "grabbing";
        canvas.setPointerCapture?.(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
        if (!state.panning || event.pointerId !== state.panning.pointerId) return;
        event.preventDefault?.(); event.stopPropagation?.(); brushCursor.style.display = "none";
        state.panX = state.panning.panX + (event.clientX - state.panning.x) / state.panning.scaleX;
        state.panY = state.panning.panY + (event.clientY - state.panning.y) / state.panning.scaleY; draw();
    });
    canvas.addEventListener("pointerup", (event) => {
        if (state.panning?.pointerId !== event.pointerId) return;
        event.preventDefault?.(); event.stopPropagation?.(); state.panning = null;
        canvas.style.cursor = cursorForMode(state.mode);
        canvas.releasePointerCapture?.(event.pointerId);
    });
    canvas.addEventListener("auxclick", (event) => { if (event.button === 1) { event.preventDefault?.(); event.stopPropagation?.(); } });
    canvas.addEventListener("contextmenu", removeSam3At);

    function stopPlayback() { if (state.timer) { clearTimeout(state.timer); state.timer = null; } state.playing = false; setButtonIcon(play, "play", "Воспроизвести", "Воспроизвести · Space"); updateBrushCursor(state.lastPointer); }
    async function playbackTick() {
        if (!state.playing || state.destroyed) return; const nextFrame = state.frame + 1 >= state.frameCount ? 0 : state.frame + 1;
        const expected = nextFrame; const loaded = await loadFrame(expected, { reason: "playback", generation: state.loadGeneration });
        if (!state.playing || expected !== (state.frame + 1 >= state.frameCount ? 0 : state.frame + 1)) return;
        if (!loaded) { stopPlayback(); setStatus(`Воспроизведение остановлено: кадр ${expected + 1} недоступен.`, "warn"); return; }
        setFrame(expected, { source: "playback", load: false }); state.timer = setTimeout(playbackTick, Math.max(16, 1000 / Math.max(.1, state.fps * state.playbackSpeed)));
    }
    function togglePlayback() { if (state.playing) stopPlayback(); else { if (state.frameItems.size === 0) { setStatus("Сначала подготовьте превью.", "warn"); return; } state.playing = true; setButtonIcon(play, "pause", "Пауза", "Пауза · Space"); playbackTick(); } }

    function move(delta) { stopPlayback(); setFrame(state.frame + delta, { source: "keyboard" }); }
    function previousKey(direction) { const keys = state.anchors.filter((frame) => direction < 0 ? frame < state.frame : frame > state.frame).sort((a, b) => direction < 0 ? b - a : a - b); if (keys.length) setFrame(keys[0], { source: "key" }); else setStatus(direction < 0 ? "Предыдущего ключа нет." : "Следующего ключа нет.", "warn"); }

    for (const tool of ["sam_point", "sam_add", "sam_remove"]) tools[tool].addEventListener("click", () => { if (sam3Mode()) chooseMode(tool); });
    objectSelect.addEventListener("change", () => {
        state.activeSam3Object = Number(objectSelect.value);
        const frame = sam3Objects()[state.activeSam3Object]?.frame;
        if (frame != null && frame !== state.frame) { stopPlayback(); setFrame(frame, { source: "sam3-object" }); }
        else draw();
    });
    tools.brush.addEventListener("click", () => chooseMode("brush")); tools.erase.addEventListener("click", () => chooseMode("erase")); tools.rect.addEventListener("click", () => chooseMode("rect")); tools.pen.addEventListener("click", () => chooseMode("pen"));
    function updateRefineControls() {
        const modeIsSam3 = sam3Mode();
        refineMethod.value = state.refineMethod;
        refine.textContent = "Выделять объект";
        refine.setAttribute("aria-pressed", String(state.grabcut));
        setTooltip(refine, modeIsSam3 ? "SAM3: прямоугольник задаёт объект для выделения. После расчёта кисть и ластик исправляют маску текущего кадра." : state.grabcut
            ? state.refineMethod === "sam3" ? "SAM3: обведите объект близко к границам и нажмите «Рассчитать маску»." : "GrabCut: обведите объект с небольшим запасом фона и нажмите «Рассчитать маску»."
            : "Выключено: прямоугольник станет обычной прямоугольной маской.");
        refine.style.display = modeIsSam3 ? "none" : "inline-flex";
        refineMethodLabel.style.display = modeIsSam3 ? "none" : "inline-flex";
        directionLabel.style.display = modeIsSam3 ? "inline-flex" : "none";
    }
    refine.addEventListener("click", () => {
        state.grabcut = !state.grabcut;
        (node.properties ||= {}).__thsRefineEnabled = state.grabcut;
        node.graph?.setDirtyCanvas?.(true, true);
        updateRefineControls();
        if (state.grabcut || state.mode === "rect") chooseMode("rect");
    });
    refineMethod.addEventListener("change", () => {
        if (sam3Mode()) return;
        state.refineMethod = refineMethod.value === "sam3" ? "sam3" : "grabcut";
        (node.properties ||= {}).__thsRefineMethod = state.refineMethod;
        node.properties.__thsRefineEnabled = true;
        node.graph?.setDirtyCanvas?.(true, true);
        state.grabcut = true; updateRefineControls(); chooseMode("rect");
    });
    brushInput.addEventListener("input", () => { updateBrushReadout(); updateBrushCursor(state.lastPointer); draw(); });
    opacity.addEventListener("input", () => { const label = `${Math.round(number(opacity.value) * 100)}%`; opacityReadout.textContent = label; syncSliderMetrics(opacity, label); draw(); });
    for (const [key, control] of Object.entries(viewButtons)) control.addEventListener("click", () => { state.view = key; for (const [name, item] of Object.entries(viewButtons)) item.setAttribute("aria-pressed", String(name === key)); viewToolbarController.refresh(control); draw(); });
    for (const [layerType, btn] of Object.entries(addLayerButtons)) {
        btn.addEventListener("click", () => addLayer(layerType));
    }
    modeSelect.value = safeMode(node, documentState); updateRefineControls(); modeSelect.addEventListener("change", () => {
        if (state.mismatchBlocked) { modeSelect.value = safeMode(node, documentState); setStatus("Источник не совпадает с сохранённой маской. Настройки заблокированы до восстановления.", "warn"); return; }
        saveCurrentLayerState();
        const previousMode = safeMode(node, documentState);
        const nextMode = modeSelect.value;
        const active = activeLayer();
        const crossingSam3 = (previousMode === "sam3") !== (nextMode === "sam3");
        const activeHasMask = Boolean(layerHasContent(active) || (previousMode === "sam3" && (sam3FrameReady() || sam3TrackReady())));
        if (crossingSam3 && activeHasMask) {
            if (addLayer(nextMode)) {
                setStatus(`Добавлен слой «${layerTypeLabel(nextMode)}» поверх текущей маски — прежний слой сохранён.`, "ready");
            } else {
                modeSelect.value = previousMode;
            }
            return;
        }
        if (active) {
            if (isDefaultLayerName(active.name, active.type)) active.name = layerTypeLabel(nextMode);
            active.type = nextMode;
        }
        setWidget(node, "tracking_mode", nextMode); documentState.tracking_mode = nextMode;
        if (nextMode === "sam3") {
            ensureSam3Document(); state.sam3.frameReady = false; state.sam3.trackReady = false; state.sam3.frameSignature = ""; state.sam3.trackSignature = ""; state.sam3.error = ""; state.sam3.sourceKey = currentSourceKey();
            syncConfiguredToolState();
        } else if (previousMode === "sam3") {
            state.sam3.frameReady = false; state.sam3.trackReady = false; state.sam3.frameSignature = ""; state.sam3.trackSignature = ""; state.sam3.error = "";
            syncConfiguredToolState({ resetLegacy: true });
        } else {
            chooseMode(state.grabcut ? "rect" : "brush");
        }
        pushHistory("Режим отслеживания"); commit({ markDirty: true }); updateRefineControls(); updateReadouts(); updateActionAvailability();
    });
    directionSelect.value = sam3Direction(documentState.sam3?.direction || "both");
    directionSelect.addEventListener("change", () => {
        if (!sam3Mode() || state.mismatchBlocked) { directionSelect.value = sam3Direction(documentState.sam3?.direction || "both"); return; }
        const direction = sam3Direction(directionSelect.value); documentState.sam3 = { ...(documentState.sam3 || {}), action: documentState.sam3?.action || "frame", frame: state.sam3.frame ?? state.frame, direction, seed_token: state.sam3.seedToken || documentState.sam3?.seed_token || "" }; state.sam3.direction = direction;
        invalidateSam3({ track: true, reason: "Направление изменилось — повторите трекинг видео." }); pushHistory("Направление трекинга"); commit({ markDirty: true }); updateReadouts(); updateActionAvailability();
    });
    for (const [name, input] of Object.entries(maskSettingsInputs)) {
        const range = maskSettingsRanges[name];
        const label = name === "grow" ? "Grow" : "Feather";
        const previewSetting = (rawValue, { fromRange = false } = {}) => {
            if (state.mismatchBlocked || input.disabled) return;
            if (!fromRange && String(rawValue).trim() === "") return;
            const parsed = number(rawValue, NaN);
            if (!Number.isFinite(parsed)) return;
            const value = clamp(Math.round(parsed), 0, 256);
            if (!state.serverMaskSettings) state.serverMaskSettings = snapshotMaskSettings();
            if (value !== snapshotMaskSettings()[name]) setWidget(node, name, value);
            syncMaskSettingsInputs({ skipInput: fromRange ? null : input });
            cropControls?.refresh?.();
            draw();
        };
        const commitSetting = (rawValue) => {
            if (state.mismatchBlocked) { syncMaskSettingsInputs(); setStatus("Источник не совпадает с сохранённой маской. Настройки заблокированы до восстановления.", "warn"); return; }
            if (input.disabled) return;
            const current = snapshotMaskSettings()[name];
            const value = String(rawValue).trim() === "" ? current : clamp(Math.round(number(rawValue, current)), 0, 256);
            if (!state.serverMaskSettings) state.serverMaskSettings = snapshotMaskSettings();
            if (value !== current) setWidget(node, name, value);
            syncMaskSettingsInputs();
            const committedValue = state.history[state.historyIndex]?.settings?.[name];
            if (value === committedValue) {
                cropControls?.refresh?.();
                draw();
                return;
            }
            const wasStale = state.previewStale;
            const wasClean = !wasStale && (!state.previewSignature || state.previewSignature === documentSignature());
            const wasSam3 = { ...state.sam3 };
            pushHistory(`${label}: ${value} px`);
            commit({ markDirty: true });
            if (wasClean) {
                state.previewStale = false;
                state.previewSignature = documentSignature();
            } else {
                state.previewStale = wasStale;
            }
            state.sam3 = wasSam3;
            updateDirtyIndicator();
            updateActionAvailability();
            cropControls?.refresh?.();
            updateReadouts();
            draw();
            setStatus(`${label}: ${value} px — предпросмотр обновлён.`, wasStale ? "dirty" : "ready");
        };
        input.addEventListener("input", () => previewSetting(input.value, { fromRange: false }));
        input.addEventListener("change", () => commitSetting(input.value));
        input.addEventListener("blur", () => commitSetting(input.value));
        input.addEventListener("keydown", (event) => {
            if (event.isComposing || event.keyCode === 229) return;
            if (event.key !== "Enter") return;
            commitSetting(input.value);
            event.preventDefault?.();
            event.stopPropagation?.();
        });
        range?.addEventListener("input", () => previewSetting(range.value, { fromRange: true }));
        range?.addEventListener("change", () => commitSetting(range.value));
    }
    slider.addEventListener("input", () => setFrame(slider.value));
    direct.addEventListener("change", () => setFrame(number(direct.value, 1) - 1));
    direct.addEventListener("keydown", (event) => {
        if (event.isComposing || event.keyCode === 229) return;
        if (event.key !== "Enter") return;
        setFrame(number(direct.value, 1) - 1, { source: "direct" });
        event.preventDefault?.(); event.stopPropagation?.();
    });
    first.addEventListener("click", () => setFrame(0, { source: "nav" })); last.addEventListener("click", () => setFrame(state.frameCount - 1, { source: "nav" })); back.addEventListener("click", () => move(-1)); next.addEventListener("click", () => move(1)); back10.addEventListener("click", () => move(-10)); next10.addEventListener("click", () => move(10)); previousKeyButton.addEventListener("click", () => { stopPlayback(); previousKey(-1); }); nextKeyButton.addEventListener("click", () => { stopPlayback(); previousKey(1); }); play.addEventListener("click", togglePlayback); speed.addEventListener("change", () => { state.playbackSpeed = number(speed.value, 1); });
    fit.addEventListener("click", () => { state.zoom = 1; state.panX = 0; state.panY = 0; draw(); }); zoomOut.addEventListener("click", () => zoomBy(.8)); zoomIn.addEventListener("click", () => zoomBy(1.25));
    undo.addEventListener("click", () => { if (state.historyIndex > 0) restoreHistory(state.historyIndex - 1); else setStatus("Отменять больше нечего.", "warn"); }); redo.addEventListener("click", () => { if (state.historyIndex + 1 < state.history.length) restoreHistory(state.historyIndex + 1); else setStatus("Повторять больше нечего.", "warn"); });
    clear.addEventListener("click", () => {
        if (state.mismatchBlocked) return;
        const bucket = frameBucket(state.frame, true);
        bucket.events = [];
        bucket.inherit = false;
        delete bucket.cleared;
        if (sam3Mode()) {
            const objects = editableSam3Objects();
            const remaining = [];
            let removedAnySam3 = false;
            for (const obj of objects) {
                if (Number(obj.frame) === state.frame) {
                    removedAnySam3 = true;
                    continue;
                }
                const cloned = clone(obj);
                if (Array.isArray(cloned.refinements)) {
                    const filteredRefs = cloned.refinements.filter((ref) => Number(ref.frame) !== state.frame);
                    if (filteredRefs.length !== cloned.refinements.length) removedAnySam3 = true;
                    if (filteredRefs.length) cloned.refinements = filteredRefs;
                    else delete cloned.refinements;
                }
                remaining.push(cloned);
            }
            if (remaining.length === 0 && (removedAnySam3 || state.sam3.frame === state.frame)) {
                if (documentState.sam3) {
                    delete documentState.sam3.prompt;
                    delete documentState.sam3.objects;
                    documentState.sam3.seed_token = "";
                }
                state.sam3.seedToken = "";
                state.resetBlank = true;
                invalidateSam3({ frame: true, track: true, reason: "Выделение опорного кадра очищено." });
            } else if (removedAnySam3) {
                if (documentState.sam3) {
                    delete documentState.sam3.prompt;
                    delete documentState.sam3.objects;
                }
                documentState.sam3 = { ...(documentState.sam3 || {}), action: "frame", frame: remaining[0].frame, direction: sam3Direction(directionSelect.value), seed_token: "" };
                if (remaining.length === 1) documentState.sam3.prompt = remaining[0];
                else documentState.sam3.objects = remaining;
                state.sam3.seedToken = "";
                state.sam3.frame = remaining[0].frame;
                state.activeSam3Object = clamp(state.activeSam3Object || 0, 0, remaining.length - 1);
                state.resetBlank = true;
                invalidateSam3({ frame: true, track: true, reason: "Выделение на текущем кадре очищено." });
            } else {
                invalidateSam3({ track: true, reason: "Покадровая правка требует повторного трекинга." });
            }
        }
        pushHistory(`Кадр ${state.frame + 1} очищен`);
        commit({ markDirty: true }); updateReadouts(); updateActionAvailability(); draw();
        setStatus(`Кадр ${state.frame + 1} очищен.`, "dirty");
    });
    copyPrevious.addEventListener("click", () => {
        if (state.mismatchBlocked) return;
        const candidates = Object.keys(documentState.frames || {}).map(Number).filter((frame) => frame < state.frame && parseEvents(documentState.frames[String(frame)]).length > 0).sort((a, b) => b - a);
        if (!candidates.length) { setStatus("Предыдущего ключа с правкой нет.", "warn"); return; }
        const sourceFrame = candidates[0];
        documentState.frames[String(state.frame)] = clone(documentState.frames[String(sourceFrame)]);
        if (sam3Mode()) {
            const copiedSeed = eventsAt(state.frame).some(event => event.type === "rect" && (event.refine_method === "sam3" || event.refine === true));
            if (copiedSeed && !sam3Objects().length) {
                state.sam3.seedToken = makeSeedToken();
                documentState.sam3 = { ...(documentState.sam3 || {}), action: "frame", frame: state.frame, seed_token: state.sam3.seedToken };
                delete documentState.sam3.prompt;
                invalidateSam3({ frame: true, track: true, reason: "Скопирована новая рамка SAM3." });
            } else invalidateSam3({ track: true, reason: "Скопированы ручные правки — обновите трекинг." });
        }
        pushHistory(`Ключ ${sourceFrame + 1} скопирован в ${state.frame + 1}`);
        commit({ markDirty: true }); updateReadouts(); draw();
    });
    reset.addEventListener("click", () => {
        if (state.mismatchBlocked) { setStatus("Источник не совпадает с сохранённой маской. Сначала примите новый клип или восстановите прежнюю маску.", "warn"); return; }
        state.layerSam3.clear();
        state.layerMaskFlags.clear();
        replaceDocument(documentState, { version: 1, frames: {}, source: state.source || undefined, tracking_mode: safeMode(node, documentState), ...(sam3Mode() ? { sam3: { action: "frame", frame: state.frame, direction: sam3Direction(directionSelect.value), seed_token: "" } } : {}) });
        ensureDocumentLayers(documentState, {
            mode: safeMode(node, documentState),
            startFrame: widgetValue(node, "start_frame", 0),
            endFrame: widgetValue(node, "end_frame", -1),
        });
        state.metadataAnchors = []; state.previewStale = true; state.resetBlank = true; state.pendingPreview = null; state.refinementError = ""; state.serverRefinedFrames.clear(); state.serverActiveLayerId = "";
        if (sam3Mode()) { state.sam3.frameReady = false; state.sam3.trackReady = false; state.sam3.seedToken = ""; state.sam3.frameSignature = ""; state.sam3.trackSignature = ""; state.sam3.error = ""; }
        state.localEdits = true; state.dirty = true; pushHistory("Маска сброшена"); commit({ markDirty: true }); updateReadouts(); draw();
        setStatus("Маска сброшена. Превью источника сохранено; обновите маску перед рисованием.", "dirty");
    });
    function markRangeDirty() {
        if (state.mismatchBlocked) return;
        state.localEdits = true; state.dirty = true; state.previewStale = true; state.editRevision += 1; updateDirtyIndicator();
    }
    const syncRangeToActiveLayer = () => {
        syncActiveLayerToList(documentState, {
            mode: safeMode(node, documentState),
            startFrame: widgetValue(node, "start_frame", 0),
            endFrame: widgetValue(node, "end_frame", -1),
        });
        if (currentLayers().length > 1) commit({ markDirty: true });
    };
    const applyStartRange = () => { if (state.mismatchBlocked) { updateReadouts(); return; } const value = clamp(Math.round(number(startInput.value, 1)) - 1, 0, state.frameCount - 1); setWidget(node, "start_frame", value); markRangeDirty(); syncRangeToActiveLayer(); if (sam3Mode()) invalidateSam3({ track: true, reason: "Диапазон изменился — повторите трекинг видео." }); updateReadouts(); updateActionAvailability(); };
    const applyEndRange = () => { if (state.mismatchBlocked) { updateReadouts(); return; } const value = clamp(Math.round(number(endInput.value, state.frameCount)) - 1, 0, state.frameCount - 1); setWidget(node, "end_frame", value); markRangeDirty(); syncRangeToActiveLayer(); if (sam3Mode()) invalidateSam3({ track: true, reason: "Диапазон изменился — повторите трекинг видео." }); updateReadouts(); updateActionAvailability(); };
    startInput.addEventListener("change", applyStartRange);
    endInput.addEventListener("change", applyEndRange);
    startInput.addEventListener("keydown", (event) => { if (event.isComposing || event.keyCode === 229) return; if (event.key === "Enter") { applyStartRange(); event.preventDefault?.(); event.stopPropagation?.(); } });
    endInput.addEventListener("keydown", (event) => { if (event.isComposing || event.keyCode === 229) return; if (event.key === "Enter") { applyEndRange(); event.preventDefault?.(); event.stopPropagation?.(); } });

    function setPrepareBusy(busy) {
        prepare.disabled = Boolean(busy || state.pendingPreview || state.mismatchBlocked);
        prepare.textContent = busy ? "Рассчитываем…" : (state.frameItems.size ? "Рассчитать маску" : "Загрузить кадры");
        prepare.setAttribute("aria-label", prepare.textContent);
        if (!busy) updateActionAvailability();
    }
    function promptIdFromResult(result) {
        if (typeof result === "string" || typeof result === "number") return String(result);
        return String(result?.prompt_id ?? result?.promptId ?? result?.id ?? "") || null;
    }
    function setPendingPromptId(kind, result) {
        const promptId = promptIdFromResult(result);
        const pending = kind === "queue" ? state.pendingQueue : state.pendingPreview;
        if (!pending || !promptId) return;
        pending.promptId = promptId;
        state.pendingPromptId = promptId;
    }
    function sam3TrackFailureInvalidatesFrame(detail = {}, message = "", eventType = "execution_error") {
        // A cancellation or an unrelated runtime failure does not revoke the
        // already approved frame token.  Only an explicit backend indication
        // that the source/seed is stale may invalidate that proof.
        if (eventType === "execution_interrupted") return false;
        const outputMeta = Array.isArray(detail?.universal_mask_meta) ? detail.universal_mask_meta[0] : detail?.universal_mask_meta;
        const nestedOutputMeta = Array.isArray(detail?.output?.universal_mask_meta) ? detail.output.universal_mask_meta[0] : detail?.output?.universal_mask_meta;
        const meta = detail?.sam3 || detail?.sam3_meta || detail?.metadata?.sam3 || outputMeta?.sam3 || nestedOutputMeta?.sam3;
        if (meta && Object.prototype.hasOwnProperty.call(meta, "frame_ready")) return meta.frame_ready === false;
        const text = [detail?.exception_type, detail?.exception_message, detail?.message, detail?.error, message]
            .filter(Boolean).join(" ");
        return /(?:source|fingerprint|seed(?:[ _-]?token)?|frame seed|stale token|invalid token)/i.test(text);
    }
    function finishPending(kind) {
        if (kind === "queue") {
            state.pendingQueue = null;
            state.pendingPromptId = state.pendingPreview?.promptId || null;
        } else {
            state.pendingPreview = null;
            state.pendingPromptId = state.pendingQueue?.promptId || null;
            state.refinementBusy = false;
            state.sam3.pendingAction = "";
            setPrepareBusy(false);
        }
        state.pendingRequestKind = state.pendingQueue ? "queue" : state.pendingPreview ? (state.pendingPreview.kind || "preview") : "";
        updateActionAvailability();
    }
    function runtimeEvent(detail = {}, eventType = "execution_error") {
        if (state.destroyed) return false;
        const pending = state.pendingPreview || state.pendingQueue;
        if (!pending) return false;
        const eventPromptId = promptIdFromResult(detail);
        const eventNodeId = String(detail?.node_id ?? detail?.node ?? detail?.nodeId ?? "");
        const targetNodeId = String(node.id);
        const promptMatches = Boolean(eventPromptId && pending.promptId && eventPromptId === String(pending.promptId));
        const targetMatches = Boolean(eventNodeId && eventNodeId === targetNodeId);
        // queuePrompt can resolve after a fast cached failure. In that narrow
        // window there is no prompt id yet; a target-node error still belongs
        // to this editor. Upstream failures are matched by prompt_id above.
        if (!promptMatches && !(targetMatches && !pending.promptId)) return false;
        diagnostics.record(eventType, { promptId: eventPromptId, nodeId: eventNodeId, exception: diagnosticText(detail.exception_type), message: diagnosticText(detail.exception_message || detail.message) });
        const message = String(detail?.exception_message || detail?.message || detail?.error || (eventType === "execution_interrupted" ? "Выполнение прервано." : "Ошибка выполнения."));
        const kind = state.pendingPreview ? "preview" : "queue";
        const sam3Action = pending.sam3Action || "";
        finishPending(kind);
        if (kind === "preview") {
            state.refinementBusy = false;
            // SAM3 frame/track failures are request-scoped.  Keep the normal
            // refinement error gate for GrabCut/preview, but let Track retry
            // after a transient SAM3 failure without hiding the valid frame.
            state.refinementError = sam3Action ? "" : refinementMessage(message);
            state.previewStale = true;
            state.metadata = { ...state.metadata, execution_error: message, execution_interrupted: eventType === "execution_interrupted" };
            if (sam3Action) {
                state.sam3.error = message;
                if (sam3Action === "frame") {
                    state.sam3.frameReady = false;
                    state.sam3.frameSignature = "";
                    state.sam3.trackReady = false;
                    state.sam3.trackSignature = "";
                } else if (sam3Action === "track") {
                    state.sam3.trackReady = false;
                    state.sam3.trackSignature = "";
                    if (sam3TrackFailureInvalidatesFrame(detail, message, eventType)) {
                        state.sam3.frameReady = false;
                        state.sam3.frameSignature = "";
                    }
                }
            }
            draw();
        }
        setStatus(`${eventType === "execution_interrupted" ? "Выполнение прервано" : "Ошибка выполнения"}: ${message}`, "error");
        updateActionAvailability();
        return true;
    }
    async function requestPreview(kind = "manual", sam3Action = null) {
        diagnostics.record("preview.request", { kind });
        if (state.destroyed || state.pendingPreview || state.pendingQueue || (state.mismatchBlocked && kind !== "manual")) { diagnostics.record("preview.blocked", { kind }); return false; }
        stopPlayback();
        if (!state.mismatchBlocked && !commit({ sam3Action })) return false;
        const seedFrame = state.sam3.frame;
        const framePatch = sam3Action === "frame" && Boolean(state.source?.fingerprint)
            && state.frameItems.has(seedFrame) && hasSam3Seed(seedFrame);
        const pending = { revision: state.editRevision, snapshot: widgetSnapshot(node), kind, sam3Action, sam3Signature: sam3Action ? sam3ReadinessSignature() : "", sourceKey: currentSourceKey(), promptId: null, framePatch, seedFrame, submittedAt: Date.now() };
        state.pendingPreview = pending;
        state.pendingRequestKind = kind;
        state.refinementBusy = kind === "refinement" || Boolean(sam3Action);
        state.sam3.pendingAction = sam3Action || "";
        setPrepareBusy(true);
        updateActionAvailability();
        if (app?.graphToPrompt && queueApi?.queuePrompt) {
            setStatus(kind === "refinement" ? "Уточняем объект…" : sam3Action === "frame" ? "Рассчитываем выбранный кадр SAM3…" : sam3Action === "track" ? "Отслеживаем видео SAM3…" : "Подготовка кадров и обновление маски…");
            try {
                const prompt = await app.graphToPrompt(app.graph || app.graph?.rootGraph);
                if (state.pendingPreview !== pending) return false;
                const scoped = previewPrompt(prompt, node.id);
                if (framePatch) {
                    const id = String(node.id);
                    scoped.output[id] = {
                        ...scoped.output[id],
                        inputs: {
                            ...scoped.output[id].inputs,
                            mask_data: JSON.stringify({ ...JSON.parse(pending.snapshot), editor_request: SAM3_FRAME_PREVIEW }),
                        },
                    };
                }
                const result = await queueApi.queuePrompt(0, scoped, { partialExecutionTargets: [String(node.id)] });
                if (state.pendingPreview !== pending) return false;
                setPendingPromptId("preview", result);
                showRequestProgress();
                diagnostics.record("preview.queued", { kind, promptId: promptIdFromResult(result) });
                if (!pending.startedAt) setStatus(kind === "refinement" ? "Уточнение поставлено в очередь. Остальной граф не запускается." : sam3Action === "frame" ? "Расчёт кадра поставлен в очередь. Остальной граф не запускается." : sam3Action === "track" ? "Трекинг поставлен в очередь. Остальной граф не запускается." : "Предпросмотр поставлен в очередь. Этот запрос рассчитывает только маску.");
            } catch (error) {
                if (state.pendingPreview !== pending) return false;
                diagnostics.record("preview.failed", { kind, message: diagnosticText(error?.message), stack: diagnosticText(error?.stack) });
                finishPending("preview");
                state.refinementError = kind === "refinement" ? String(error?.message || error) : "";
                setStatus(`${kind === "refinement" ? "Ошибка уточнения" : "Ошибка подготовки"}: ${error?.message || error}`, "error");
            }
        } else {
            finishPending("preview");
            setStatus("Preview API недоступен в автономном режиме. mask_data уже сохранён.", "warn");
        }
        return true;
    }
    function requestPrepare() { requestPreview("manual"); }
    function requestSam3(action) {
        if (!sam3Mode() || state.destroyed || state.pendingPreview || state.pendingQueue || state.mismatchBlocked) return false;
        const seedFrame = Number.isFinite(Number(state.sam3.frame)) ? Math.max(0, Math.round(Number(state.sam3.frame))) : state.frame;
        const currentHasSeed = hasSam3Seed(state.frame);
        const hasAnySeed = Boolean(sam3Objects().length) || Object.keys(documentState.frames || {}).some((key) => hasSam3Seed(Number(key)));
        if (action === "frame" && !currentHasSeed && hasAnySeed) {
            setStatus(`Выделение SAM3 находится на кадре ${seedFrame + 1}. Выберите его или кликните по объекту на текущем кадре.`, "warn");
            return false;
        }
        if (action === "track" && !sam3FrameReady()) {
            setStatus("Сначала нажмите «Рассчитать кадр» и дождитесь корректного результата SAM3.", "warn");
            return false;
        }
        if (currentHasSeed) {
            state.sam3.frame = state.frame;
            if (!state.sam3.seedToken) state.sam3.seedToken = makeSeedToken();
        }
        ensureSam3Document();
        documentState.sam3 = { ...(documentState.sam3 || {}), action, frame: state.sam3.frame ?? state.frame, direction: sam3Direction(directionSelect.value), seed_token: state.sam3.seedToken || documentState.sam3?.seed_token || "" };
        state.sam3.direction = sam3Direction(directionSelect.value); state.sam3.pendingAction = action;
        commit({ sam3Action: action });
        return requestPreview(`sam3_${action}`, action);
    }
    async function cancelOwnPending() {
        const pending = state.pendingPreview || state.pendingQueue;
        const promptId = pending?.promptId;
        if (!pending || !promptId) { setStatus("Адресная отмена недоступна: идентификатор запроса ещё не получен.", "warn"); return false; }
        let cancelled = false;
        try {
            if (typeof queueApi?.cancelPrompt === "function") { await queueApi.cancelPrompt(promptId); cancelled = true; }
            else if (typeof queueApi?.interrupt === "function" && queueApi.interrupt.length >= 1) { await queueApi.interrupt(promptId); cancelled = true; }
            else if (typeof queueApi?.fetchApi === "function") { await queueApi.fetchApi("/interrupt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt_id: String(promptId) }) }); cancelled = true; }
        } catch (error) {
            diagnostics.record("request.cancel.failed", { promptId, message: diagnosticText(error?.message || error) });
            setStatus(`Не удалось отменить свой запрос: ${error?.message || error}`, "error");
            return false;
        }
        if (!cancelled) { setStatus("ComfyUI не поддерживает адресную отмену этого запроса.", "warn"); return false; }
        diagnostics.record("request.cancel", { promptId });
        setStatus("Отмена своего запроса отправлена…", "warn");
        return true;
    }
    async function requestQueue() {
        if (state.destroyed) return false;
        if (state.pendingQueue || state.pendingPreview || state.refinementBusy) {
            setStatus("Дождитесь завершения текущего расчёта.", "warn");
            return false;
        }
        if (actionBlocked()) return false;
        if (hasUntrackedSam3Layer()) {
            const nextStep = sam3Mode() && sam3FrameReady() ? "«Отследить видео»" : "«Рассчитать кадр», затем «Отследить видео»";
            setStatus(`В редакторе маски нажмите ${nextStep}.`, "warn");
            return false;
        }
        stopPlayback();
        if (!commit()) return false;
        if (!app?.queuePrompt) {
            setStatus("Очередь ComfyUI недоступна. Маска сохранена.", "warn");
            return false;
        }
        state.pendingQueue = { promptId: null, revision: state.editRevision, snapshot: widgetSnapshot(node), submittedAt: Date.now() };
        state.pendingRequestKind = "queue";
        updateActionAvailability();
        try {
            const result = await app.queuePrompt(0);
            if (result === false) throw new Error("Проверьте ошибки графа в ComfyUI.");
            setPendingPromptId("queue", result);
            if (!state.pendingQueue?.promptId) finishPending("queue");
            setStatus("Граф поставлен в очередь.", "ok");
            return true;
        } catch (error) {
            finishPending("queue");
            setStatus(`Очередь графа: ${error?.message || error}`, "error");
            return false;
        }
    }
    prepare.addEventListener("click", requestPrepare); calculateFrame.addEventListener("click", () => requestSam3("frame")); trackVideo.addEventListener("click", () => requestSam3("track")); cancelRequest.addEventListener("click", cancelOwnPending); queue.addEventListener("click", requestQueue);
    recoverSource.addEventListener("click", () => {
        diagnostics.record("source.accept");
        if (!state.pendingSource) return;
        const pending = state.pendingSource;
        capturePreviousMask(pending.mismatch?.previousSource || state.source);
        replaceDocument(documentState, { version: 1, frames: {}, source: clone(pending.source), tracking_mode: safeMode(node, documentState), ...(safeMode(node, documentState) === "sam3" ? { sam3: { action: "frame", frame: 0, direction: sam3Direction(directionSelect.value), seed_token: "" } } : {}) });
        state.layerSam3.clear();
        state.layerMaskFlags.clear();
        ensureDocumentLayers(documentState, { mode: safeMode(node, documentState), startFrame: 0, endFrame: -1 });
        setWidget(node, "start_frame", 0);
        setWidget(node, "end_frame", -1);
        state.localEdits = false; state.dirty = false; state.pendingSource = null; state.sourceMismatch = null; state.mismatchBlocked = false;
        state.previousMaskDismissed = false; state.dismissedAlertKey = "";
        state.refinementError = ""; state.pendingPreview = null; state.pendingQueue = null; state.pendingPromptId = null; state.pendingRequestKind = "";
        state.serverActiveLayerId = "";
        state.history = []; state.historyIndex = -1; state.frame = 0; state.zoom = 1; state.panX = 0; state.panY = 0; state.sam3 = { ...state.sam3, frameReady: false, trackReady: false, frame: 0, seedToken: "", direction: sam3Direction(directionSelect.value), computedFrames: [], error: "", frameSignature: "", trackSignature: "", pendingAction: "frame" };
        clearPreviewCaches();
        // The mismatch response intentionally remains source-only. Do not
        // let its old metadata reintroduce a mask over the newly accepted clip.
        const accepted = clone(pending.output || {});
        if (accepted.universal_mask_meta) {
            const meta = Array.isArray(accepted.universal_mask_meta) ? accepted.universal_mask_meta[0] : accepted.universal_mask_meta;
            const cleanMeta = { ...(meta || {}) };
            delete cleanMeta.source_mismatch; delete cleanMeta.source_mismatch_reason; delete cleanMeta.previous_source;
            cleanMeta.has_mask = false; accepted.universal_mask_meta = Array.isArray(accepted.universal_mask_meta) ? [cleanMeta] : cleanMeta;
        }
        ingestOutput(accepted, { acceptMismatch: true });
        state.history = [{ document: snapshotDocument(), settings: snapshotMaskSettings(), label: "Новый источник" }]; state.historyIndex = 0;
        commit(); updateReadouts(); updateActionAvailability();
        setStatus("Новый клип принят. Диапазон сброшен; начните пустую маску.", "ready");
    });
    restorePrevious.addEventListener("click", () => {
        diagnostics.record("source.restore");
        const backup = state.previousMaskBackup || node.properties?.__thsPreviousMask;
        if (!backup) return;
        const activeSource = clone(state.source);
        const currentRawMask = widgetValue(node, "mask_data", "{}");
        const currentBackup = {
            mask_data: typeof currentRawMask === "string" ? currentRawMask : clone(currentRawMask),
            document: snapshotDocument(),
            tracking_mode: widgetValue(node, "tracking_mode", safeMode(node, documentState)),
            grow: widgetValue(node, "grow", maskSettingsDefaults.grow),
            feather: widgetValue(node, "feather", maskSettingsDefaults.feather),
            start_frame: widgetValue(node, "start_frame", 0),
            end_frame: widgetValue(node, "end_frame", -1),
            source: clone(sourceInfo(documentState) || activeSource),
        };
        node.properties.__thsPreviousMask = clone(currentBackup);
        state.previousMaskBackup = currentBackup;
        const restoredDocument = backup.document || (typeof backup.mask_data === "string" ? readDocument({ widgets: [{ name: "mask_data", value: backup.mask_data }] }) : backup.mask_data);
        replaceDocument(documentState, restoredDocument || { version: 1, frames: {} });
        state.layerSam3.clear();
        state.layerMaskFlags.clear();
        if (backup.mask_data !== undefined) setWidget(node, "mask_data", typeof backup.mask_data === "string" ? backup.mask_data : JSON.stringify(backup.mask_data));
        for (const name of ["tracking_mode", "grow", "feather", "start_frame", "end_frame"]) if (backup[name] !== undefined) setWidget(node, name, backup[name]);
        ensureDocumentLayers(documentState, {
            mode: safeMode(node, documentState),
            startFrame: widgetValue(node, "start_frame", 0),
            endFrame: widgetValue(node, "end_frame", -1),
        });
        if (Array.isArray(restoredDocument?.layers) && restoredDocument.layers.length > 0) {
            const restoredActive = activeLayerOf(documentState);
            if (restoredActive) {
                projectLayerToDocument(documentState, restoredActive);
                setWidget(node, "tracking_mode", restoredActive.type);
                setWidget(node, "start_frame", restoredActive.start_frame ?? 0);
                setWidget(node, "end_frame", restoredActive.end_frame ?? -1);
            }
        }
        state.source = sourceInfo(documentState) || sourceInfo({ source: backup.source }) || state.source;
        state.frameCount = state.source?.frames || 1; state.frame = 0; state.zoom = 1; state.panX = 0; state.panY = 0;
        state.localEdits = false; state.dirty = false; state.previewStale = true; state.resetBlank = false;
        state.pendingPreview = null; state.pendingQueue = null; state.pendingPromptId = null; state.pendingRequestKind = ""; state.refinementBusy = false; state.refinementError = "";
        state.sam3 = { ...state.sam3, frameReady: false, trackReady: false, seedToken: String(documentState.sam3?.seed_token || ""), frame: Number.isFinite(Number(documentState.sam3?.frame)) ? Math.max(0, Math.round(Number(documentState.sam3.frame))) : 0, direction: sam3Direction(documentState.sam3?.direction || directionSelect.value), computedFrames: [], error: "Источник восстановлен; требуется новый preview.", frameSignature: "", trackSignature: "", pendingAction: documentState.sam3?.action || "" };
        const currentSource = activeSource;
        const restoredSource = sourceInfo(documentState) || sourceInfo({ source: backup.source });
        state.sourceMismatch = {
            reason: sourceMismatchReason(restoredSource, currentSource, state.lastMismatch?.reason),
            previousSource: clone(restoredSource),
            source: clone(currentSource),
        };
        state.lastMismatch = clone(state.sourceMismatch);
        state.pendingSource = { source: clone(currentSource), output: null, mismatch: clone(state.sourceMismatch) };
        state.mismatchBlocked = true;
        clearPreviewCaches();
        state.metadata = {};
        state.history = [{ document: snapshotDocument(), settings: snapshotMaskSettings(), label: "Прежняя маска восстановлена" }]; state.historyIndex = 0;
        modeSelect.value = safeMode(node, documentState); directionSelect.value = state.sam3.direction; updateRefineControls(); syncMaskSettingsInputs(); updateReadouts(); updateActionAvailability(); draw();
        setStatus("Прежняя маска восстановлена. Выберите прежний источник видео, чтобы снова открыть её превью.", "warn");
    });
    dismissAlert.addEventListener("click", () => {
        if (state.pendingSource) return;
        const tone = state.lastStatusTone || "normal";
        const message = String(status.textContent || "").trim();
        if (message && (tone === "error" || tone === "warn")) {
            state.dismissedAlertKey = `${tone}:${message}`;
        }
        if (state.previousMaskBackup) {
            state.previousMaskDismissed = true;
            state.previousMaskBackup = null;
            if (node.properties && typeof node.properties === "object") delete node.properties.__thsPreviousMask;
        }
        updateActionAvailability();
    });

    function startNewVideo() {
        if (state.destroyed || state.pendingPreview || state.pendingQueue) return false;
        diagnostics.record("source.new_video");
        stopPlayback();
        closeTooltip();
        cancelGesture();
        if (state.source && hasCurrentMaskWork()) capturePreviousMask(state.source);
        const mode = safeMode(node, documentState);
        const currentCrop = documentState.crop && typeof documentState.crop === "object" ? documentState.crop : null;
        const preservedCrop = currentCrop
            ? {
                  mode: currentCrop.mode === "manual" ? "auto" : (currentCrop.mode || "full"),
                  margin: currentCrop.margin ?? 64,
                  multiple: currentCrop.multiple ?? 1,
              }
            : undefined;
        state.layerSam3.clear();
        state.layerMaskFlags.clear();
        replaceDocument(documentState, {
            version: 1,
            frames: {},
            tracking_mode: mode,
            ...(preservedCrop ? { crop: preservedCrop } : {}),
            ...(mode === "sam3" ? { sam3: { action: "frame", frame: 0, direction: sam3Direction(directionSelect.value), seed_token: "" } } : {}),
        });
        ensureDocumentLayers(documentState, { mode, startFrame: 0, endFrame: -1 });
        setWidget(node, "start_frame", 0);
        setWidget(node, "end_frame", -1);
        state.widgetSnapshot = widgetSnapshot(node);
        state.maskSettingsSnapshot = JSON.stringify(snapshotMaskSettings());
        state.modeSnapshot = String(widgetValue(node, "tracking_mode", ""));
        state.source = null;
        state.frameCount = 1;
        state.frame = 0;
        state.zoom = 1;
        state.panX = 0;
        state.panY = 0;
        state.localEdits = false;
        state.dirty = false;
        state.previewStale = false;
        state.resetBlank = false;
        state.previewSignature = "";
        state.metadata = {};
        state.metadataAnchors = [];
        state.pendingSource = null;
        state.sourceMismatch = null;
        state.mismatchBlocked = false;
        state.previousMaskDismissed = true;
        state.dismissedAlertKey = "";
        state.refinementError = "";
        state.serverRefinedFrames.clear();
        state.serverActiveLayerId = "";
        state.sam3 = {
            ...state.sam3,
            frameReady: false,
            trackReady: false,
            frame: 0,
            seedToken: "",
            direction: sam3Direction(directionSelect.value),
            computedFrames: [],
            error: "",
            sourceKey: "",
            frameSignature: "",
            trackSignature: "",
            pendingAction: "",
        };
        clearPreviewCaches();
        state.history = [{ document: snapshotDocument(), settings: snapshotMaskSettings(), label: "Новое видео" }];
        state.historyIndex = 0;
        commit({ markDirty: false });
        cropControls?.refresh?.();
        updateReadouts();
        updateActionAvailability();
        empty.textContent = "Загрузка нового видео…";
        draw();
        return requestPreview("manual");
    }
    newVideoButton.addEventListener("click", startNewVideo);

    function supportsTopLayerExit() {
        return typeof CSS !== "undefined"
            && Boolean(CSS.supports?.("transition-behavior: allow-discrete"))
            && Boolean(CSS.supports?.("selector(@starting-style)"));
    }

    function openDialog() {
        if (state.dialogOpen) return;
        if (dialog.dataset) delete dialog.dataset.closing;
        closeTooltip(); cancelGesture(); state.lastPointer = null; brushCursor.style.display = "none";
        state.dialogReturnFocus = document.activeElement || expand;
        state.dialogOpen = true;
        dialogInner.append(workbench);
        dialogBar.append(headerMeta, newVideoButton, helpButton, debugDownload, closeDialogButton);
        if (typeof dialog.showModal === "function") {
            dialog.showModal();
        } else {
            dialog.setAttribute("open", "");
            closeDialogButton.focus?.();
        }
        draw();
        diagnostics.record("dialog.open", { layout: layoutSnapshot() });
    }
    function closeDialog({ fromCloseEvent = false } = {}) {
        if (!state.dialogOpen) return;
        state.dialogOpen = false; closeTooltip(); stopPlayback(); cancelGesture(); state.lastPointer = null; brushCursor.style.display = "none";
        header.append(headerMeta, headerActions);
        headerActions.append(newVideoButton, helpButton, debugDownload, expand);
        if (workbenchPlaceholder.parentNode && workbenchPlaceholder.after) workbenchPlaceholder.after(workbench); else root.append(workbench);
        dialog.returnValue = "";
        if (!fromCloseEvent && typeof dialog.close === "function") {
            if (supportsTopLayerExit() || typeof dialog.getAnimations !== "function") {
                dialog.close();
            } else {
                if (dialog.dataset) dialog.dataset.closing = "";
                Promise.allSettled(dialog.getAnimations().map((animation) => animation.finished)).then(() => {
                    if (dialog.dataset) delete dialog.dataset.closing;
                    if (!state.dialogOpen && dialog.open) dialog.close();
                });
            }
        } else {
            dialog.removeAttribute?.("open");
        }
        (state.dialogReturnFocus || expand)?.focus?.(); state.dialogReturnFocus = null; draw();
        diagnostics.record("dialog.close", { layout: layoutSnapshot() });
    }
    expand.addEventListener("click", openDialog);
    closeDialogButton.addEventListener("click", () => closeDialog());
    dialog.addEventListener("cancel", (event) => { event.preventDefault?.(); closeDialog(); });
    dialog.addEventListener("close", () => { dialog.returnValue = ""; if (state.dialogOpen) closeDialog({ fromCloseEvent: true }); });

    function inTextInput(target) { const tag = String(target?.tagName || "").toLowerCase(); return tag === "input" || tag === "textarea" || tag === "select" || target?.isContentEditable; }
    function consumeKey(event) { event.preventDefault?.(); event.stopPropagation?.(); }
    root.addEventListener("keydown", (event) => {
        if (isImeComposing(event) || inTextInput(event.target) && event.target !== canvas) return;
        if (event.key === "Escape") {
            if (state.penPath.length || state.active) cancelGesture();
            else if (tipOpen) closeTooltip();
            else if (state.dialogOpen) closeDialog();
            else cancelGesture();
            consumeKey(event);
            return;
        }
        if (event.key === "Enter" && state.mode === "pen" && state.penPath.length) { finishPen(); consumeKey(event); return; }
        if ((event.key === "Backspace" || event.key === "Delete") && state.mode === "pen" && state.penPath.length) {
            if (state.active?.type === "pen_anchor" || state.active?.type === "pen_close") {
                canvas.releasePointerCapture?.(state.active.pointerId);
                state.active = null;
            }
            state.penPath.pop();
            state.penSnapClose = false;
            draw();
            consumeKey(event);
            return;
        }
        if (event.key === " " || event.code === "Space") { togglePlayback(); consumeKey(event); return; }
        if (event.key === "ArrowLeft") { move(event.shiftKey ? -10 : -1); consumeKey(event); return; }
        if (event.key === "ArrowRight") { move(event.shiftKey ? 10 : 1); consumeKey(event); return; }
        if (event.key === "Home") { setFrame(0, { source: "keyboard" }); consumeKey(event); return; }
        if (event.key === "End") { setFrame(state.frameCount - 1, { source: "keyboard" }); consumeKey(event); return; }
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.shiftKey ? redo.click() : undo.click(); consumeKey(event); }
        else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") { redo.click(); consumeKey(event); }
    });

    function syncFromConfiguredWidget(force = false) {
        if (!force && state.mismatchBlocked) return false;
        state.refineMethod = node.properties?.__thsRefineMethod === "sam3" ? "sam3" : "grabcut";
        updateRefineControls();
        syncMaskSettingsInputs();
        const settingsSnapshot = JSON.stringify(snapshotMaskSettings());
        const snapshot = widgetSnapshot(node); const modeSnapshot = String(widgetValue(node, "tracking_mode", ""));
        if (!force && state.localEdits) return false; if (!force && snapshot === state.widgetSnapshot && settingsSnapshot === state.maskSettingsSnapshot && modeSnapshot === state.modeSnapshot) return false;
        const maskChanged = snapshot !== state.widgetSnapshot;
        const modeChanged = modeSnapshot !== state.modeSnapshot;
        diagnostics.record("widget.sync", { force, maskChanged, settingsChanged: settingsSnapshot !== state.maskSettingsSnapshot, modeChanged });
        state.layerSam3.clear();
        state.layerMaskFlags.clear();
        const loadedDoc = readDocument(node);
        const hadLayers = Array.isArray(loadedDoc.layers) && loadedDoc.layers.length > 0;
        replaceDocument(documentState, loadedDoc);
        ensureDocumentLayers(documentState, {
            mode: safeMode(node, documentState),
            startFrame: widgetValue(node, "start_frame", 0),
            endFrame: widgetValue(node, "end_frame", -1),
        });
        if (hadLayers) {
            const loadedActive = activeLayerOf(documentState);
            if (loadedActive) {
                projectLayerToDocument(documentState, loadedActive);
                setWidget(node, "tracking_mode", loadedActive.type);
                setWidget(node, "start_frame", loadedActive.start_frame ?? 0);
                setWidget(node, "end_frame", loadedActive.end_frame ?? -1);
            }
        }
        state.source = sourceInfo(documentState); state.frameCount = state.source?.frames || state.frameCount || 1; state.anchors = explicitAnchors(documentState); state.metadataAnchors = []; state.frame = clamp(state.frame, 0, Math.max(0, state.frameCount - 1)); modeSelect.value = safeMode(node, documentState); ensureSam3Document(); state.sam3 = { ...state.sam3, frameReady: false, trackReady: false, frame: documentState.sam3?.frame ?? state.sam3.frame, seedToken: String(documentState.sam3?.seed_token || state.sam3.seedToken || ""), direction: sam3Direction(documentState.sam3?.direction || directionSelect.value), computedFrames: [], error: "", sourceKey: currentSourceKey(), frameSignature: "", trackSignature: "", pendingAction: documentState.sam3?.action || "" }; directionSelect.value = state.sam3.direction; syncConfiguredToolState({ resetLegacy: force || modeChanged || maskChanged }); state.widgetSnapshot = snapshot; state.modeSnapshot = String(widgetValue(node, "tracking_mode", "")); state.maskSettingsSnapshot = JSON.stringify(snapshotMaskSettings()); state.serverMaskSettings = null; state.previewSignature = ""; state.previewStale = true; state.resetBlank = false; state.loadGeneration += 1; state.frameItems.clear(); state.previews.clear(); state.imageCache.clear(); state.imagePending.clear(); state.serverRefinedFrames.clear(); state.serverActiveLayerId = ""; maskPostprocessor.invalidate(); state.history = [{ document: snapshotDocument(), settings: snapshotMaskSettings(), label: "Исходное состояние" }]; state.historyIndex = 0; updateRefineControls(); updateReadouts(); updateActionAvailability(); draw(); return true;
    }

    // The frame also resizes when toolbars wrap or details open, without any
    // window resize. Redraw the bitmap at that size instead of stretching it.
    let layoutTimer = null;
    const layoutObserver = typeof ResizeObserver === "function" ? new ResizeObserver(() => {
        if (state.destroyed) return;
        draw();
        if (layoutTimer) clearTimeout(layoutTimer);
        layoutTimer = setTimeout(() => { layoutTimer = null; if (!state.destroyed) diagnostics.record("layout.resize", { layout: layoutSnapshot() }); }, 250);
    }) : null;
    layoutObserver?.observe(canvasWrap);
    const onContentVisibilityState = (event) => {
        if (state.destroyed) return;
        if (event?.skipped) stopPlayback();
    };
    canvasWrap.addEventListener?.("contentvisibilityautostatechange", onContentVisibilityState);
    let visibilityFallbackObserver = null;
    const contentVisibilitySupported = typeof document !== "undefined"
        && Boolean(document.documentElement?.style)
        && ("contentVisibility" in document.documentElement.style);
    if (!contentVisibilitySupported && typeof IntersectionObserver === "function") {
        visibilityFallbackObserver = new IntersectionObserver((entries) => {
            if (state.destroyed) return;
            for (const entry of entries) if (!entry.isIntersecting) stopPlayback();
        }, { rootMargin: "200px" });
        visibilityFallbackObserver.observe(canvasWrap);
    }
    const onDocumentVisibility = () => {
        if (typeof document !== "undefined" && document.hidden) stopPlayback();
    };
    document.addEventListener?.("visibilitychange", onDocumentVisibility);

    const addApiListener = (type, handler) => {
        if (typeof queueApi?.addEventListener !== "function") return;
        queueApi.addEventListener(type, handler);
        state.apiListeners.push([type, handler]);
    };
    addApiListener("execution_error", ({ detail } = {}) => runtimeEvent(detail || {}, "execution_error"));
    addApiListener("execution_interrupted", ({ detail } = {}) => runtimeEvent(detail || {}, "execution_interrupted"));
    function requestForEvent(detail) {
        const pending = state.pendingPreview || state.pendingQueue;
        return pending?.promptId && pending.promptId === promptIdFromResult(detail) ? pending : null;
    }
    function beginRequest(detail) {
        const pending = requestForEvent(detail);
        if (!pending || pending.startedAt) return;
        pending.startedAt = Date.now();
        setStatus(`${requestLabel(pending)}…`, "busy");
        showRequestProgress();
    }
    function updateFrameProgress(detail) {
        const pending = requestForEvent(detail);
        if (pending !== state.pendingPreview || !pending || String(detail.node) !== String(node.id)) return;
        const { phase, value, max: total } = detail;
        if (!["sam3_track", "optical_flow", "optical_flow_corrections"].includes(phase)) return;
        if (!Number.isInteger(value) || !Number.isInteger(total) || total < 1 || value < 0 || value > total) return;
        const previous = pending.frameProgress;
        const samePhase = previous?.phase === phase && previous.total === total;
        if (samePhase && value <= previous.value) return;
        const now = Date.now();
        const firstAt = samePhase ? previous.firstAt : now;
        const firstValue = samePhase ? previous.firstValue : value;
        const samples = value - firstValue;
        let finishAt = null;
        if (samples >= 2 && now - firstAt >= 1000 && value < total) {
            finishAt = now + (now - firstAt) / samples * (total - value);
        }
        beginRequest(detail);
        pending.frameProgress = { phase, value, total, at: now, firstAt, firstValue, finishAt };
        diagnostics.record("preview.progress", { kind: pending.kind, phase, value, max: total });
        showRequestProgress();
    }
    addApiListener("execution_start", ({ detail } = {}) => beginRequest(detail));
    addApiListener("executing", ({ detail } = {}) => {
        if (detail?.node != null) beginRequest(detail);
    });
    addApiListener("progress", ({ detail } = {}) => beginRequest(detail));
    addApiListener("universal_inpaint_mask_progress", ({ detail } = {}) => updateFrameProgress(detail));
    addApiListener("execution_success", ({ detail } = {}) => {
        const pending = requestForEvent(detail);
        if (!pending) return;
        finishPending(pending === state.pendingQueue ? "queue" : "preview");
        setStatus("Расчёт завершён.", "ok");
    });
    function destroy() {
        if (state.destroyed) return;
        state.destroyed = true;
        for (const [type, handler] of state.apiListeners.splice(0)) queueApi?.removeEventListener?.(type, handler);
        layoutObserver?.disconnect();
        visibilityFallbackObserver?.disconnect();
        canvasWrap.removeEventListener?.("contentvisibilityautostatechange", onContentVisibilityState);
        document.removeEventListener?.("visibilitychange", onDocumentVisibility);
        stopPlayback(); closeTooltip(); clearTipTimers(); state.loadGeneration += 1;
        if (layoutTimer) clearTimeout(layoutTimer);
        if (wheelLogTimer) clearTimeout(wheelLogTimer);
        window.removeEventListener?.("error", recordPageError); window.removeEventListener?.("unhandledrejection", recordPageError);
        window.removeEventListener?.("pagehide", diagnostics.flush);
        state.pendingPreview = null; state.pendingQueue = null; state.pendingPromptId = null; state.refinementBusy = false;
        showRequestProgress();
        closeDialog(); window.removeEventListener?.("resize", draw); root.remove?.();
        diagnostics.record("editor.destroyed"); diagnostics.dispose();
    }
    window.addEventListener?.("resize", draw);
    node.addDOMWidget?.("universal_mask_editor", "THSUniversalMaskEditor", root, { serialize: false, getMinHeight: () => 740, getMaxHeight: () => 1100 });
    const editor = { root, canvas, status, applyExecuted, handleRuntimeEvent: runtimeEvent, setFrame, commit, requestQueue, syncFromConfiguredWidget, destroy, loadFrame, ingestOutput, debugReport: diagnostics.report, get state() { return state; }, get document() { return documentState; } };
    node.__thsUniversalMaskEditor = editor;
    if (!node.__thsUniversalMaskConfigureSync) {
        const previousOnConfigure = node.onConfigure;
        node.onConfigure = function onUniversalMaskConfigure(...args) {
            const result = typeof previousOnConfigure === "function" ? previousOnConfigure.apply(this, args) : undefined;
            if (result && typeof result.then === "function") return result.then((value) => { syncFromConfiguredWidget(); return value; });
            syncFromConfiguredWidget(); return result;
        };
        node.__thsUniversalMaskConfigureSync = true;
    }
    if (!node.__thsUniversalMaskRemovedHook) {
        const previousRemoved = node.onRemoved;
        node.onRemoved = function onUniversalMaskRemoved(...args) { editor.destroy(); return typeof previousRemoved === "function" ? previousRemoved.apply(this, args) : undefined; };
        node.__thsUniversalMaskRemovedHook = true;
    }
    syncFromConfiguredWidget(true);
    if (sam3Mode()) { state.grabcut = true; state.refineMethod = "sam3"; chooseMode("sam_point"); }
    else chooseMode("brush");
    updateReadouts(); updateActionAvailability(); draw();
    return editor;
}

if (comfyApp?.registerExtension) {
    comfyApp.registerExtension({
        name: "UniversalInpaint.UniversalMaskEditor",
        async nodeCreated(node) { if (node.comfyClass === NODE_CLASS) createUniversalMaskEditor(node); },
    });
    comfyApi?.addEventListener?.("executed", ({ detail }) => {
        const node = comfyApp.graph?.getNodeById?.(Number(detail?.node));
        if (!node || node.comfyClass !== NODE_CLASS) return;
        const editor = node.__thsUniversalMaskEditor || createUniversalMaskEditor(node);
        editor?.applyExecuted({ ...(detail?.output || {}), prompt_id: detail?.prompt_id || detail?.promptId, node_id: detail?.node || detail?.node_id });
    });
}

export { NODE_CLASS, parseEvents, sourceInfo, imageUrl };
