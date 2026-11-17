/* global document, window, Image */
import { previewPrompt } from "./preview_prompt.js";

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
const EDITOR_BUILD = "20261118-r11-inspector-layout";

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
            events: JSON.parse(JSON.stringify(entries)) };
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

function clone(value) {
    try { return JSON.parse(JSON.stringify(value)); } catch (_) { return value; }
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
    "brush": "<path d=\"m9.06 11.9 8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08\" />\n  <path d=\"M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z\" />",
    "erase": "<path d=\"m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21\" />\n  <path d=\"M22 21H7\" />\n  <path d=\"m5 11 9 9\" />",
    "rect": "<rect width=\"18\" height=\"18\" x=\"3\" y=\"3\" rx=\"2\" />",
    "polygon": "<path d=\"M10.83 2.38a2 2 0 0 1 2.34 0l8 5.74a2 2 0 0 1 .73 2.25l-3.04 9.26a2 2 0 0 1-1.9 1.37H7.04a2 2 0 0 1-1.9-1.37L2.1 10.37a2 2 0 0 1 .73-2.25z\" />",
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
    "nextKey": "<path d=\"m6 17 5-5-5-5\" />\n  <path d=\"m13 17 5-5-5-5\" />"
};

function setButtonIcon(element, name, label, tooltip = label) {
    element.setAttribute("aria-label", label);
    element.title = tooltip;
    if (element.dataset?.icon === name) return;
    element.textContent = "";
    const artwork = document.createElement("span");
    artwork.setAttribute("aria-hidden", "true");
    artwork.style.pointerEvents = "none";
    artwork.style.display = "inline-flex";
    artwork.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false">${ICONS[name]}</svg>`;
    element.append(artwork);
    if (name === "back10" || name === "next10") {
        const amount = document.createElement("span");
        amount.textContent = "10"; amount.setAttribute("aria-hidden", "true");
        Object.assign(amount.style, { fontSize: "9px", lineHeight: "1", position: "absolute", right: "3px", bottom: "3px", background: COLORS.panel2, padding: "1px", borderRadius: "2px" });
        element.append(amount);
    }
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

function addStyles(root) {
    const style = document.createElement("style");
    style.textContent = `
      .ths-mask-editor { font: 13px/1.45 Inter, system-ui, sans-serif; color: ${COLORS.text}; container: mask-editor / inline-size; }
      .ths-mask-editor .ths-mask-button { min-height: 36px; padding: 8px 12px; border: 1px solid ${COLORS.border}; border-radius: 6px; background: ${COLORS.panel2}; color: ${COLORS.text}; cursor: pointer; font: 500 13px/1.3 Inter, system-ui, sans-serif; white-space: nowrap; transition: background-color 120ms ease, border-color 120ms ease; }
      .ths-mask-editor .ths-mask-button-primary { background: ${COLORS.accent}; color: ${COLORS.accentText}; border-color: transparent; }
      .ths-mask-editor .ths-mask-button-danger { color: ${COLORS.danger}; }
      .ths-mask-editor button:hover:not(:disabled) { border-color: ${COLORS.accentStrong}; filter: brightness(1.12); }
      .ths-mask-editor button:disabled { opacity: .45; cursor: default; }
      .ths-mask-editor :is(button,input,select,summary):focus-visible { outline: 2px solid ${COLORS.accentStrong}; outline-offset: 3px; }
      .ths-mask-editor :is(input,select) { accent-color: ${COLORS.accent}; font: inherit; }
      .ths-mask-editor :is(input[type=number],select) { background: ${COLORS.input}; color: ${COLORS.text}; border: 1px solid ${COLORS.border}; border-radius: 5px; min-height: 34px; padding: 6px 8px; box-sizing: border-box; min-width: 0; }
      .ths-mask-editor .ths-mask-icon-button { display: inline-flex; align-items: center; justify-content: center; width: 38px; min-width: 38px; height: 38px; min-height: 38px; padding: 8px; flex: none; position: relative; }
      .ths-mask-editor .ths-mask-label { display: inline-flex; align-items: center; gap: 8px; min-height: 32px; color: ${COLORS.muted}; font-size: 12px; line-height: 1.4; }
      .ths-mask-editor .ths-mask-header { display: flex; align-items: center; gap: 16px; min-height: 40px; }
      .ths-mask-editor .ths-mask-header > :nth-child(2) { margin-inline-start: auto; min-width: 0; overflow: hidden; }
      .ths-mask-editor .ths-mask-workbench { display: grid; grid-template-rows: auto minmax(0,1fr); gap: 12px; height: 650px; min-width: 0; min-height: 0; }
      .ths-mask-editor .ths-mask-status { box-sizing: border-box; min-height: 36px; padding: 9px 12px; border-radius: 5px; background: ${COLORS.panel2}; font-size: 12px; line-height: 1.5; }
      .ths-mask-editor .ths-mask-status-area { display: grid; gap: 8px; }
      .ths-mask-editor .ths-mask-workspace { display: grid; grid-template-columns: minmax(0,1fr) 296px; gap: 16px; min-height: 0; min-width: 0; }
      .ths-mask-editor .ths-mask-viewer { display: grid; grid-template-rows: auto minmax(220px,1fr) auto; gap: 10px; min-width: 0; min-height: 0; }
      .ths-mask-editor .ths-mask-viewbar { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; min-width: 0; }
      .ths-mask-editor .ths-mask-viewbar .ths-mask-button { min-height: 32px; padding: 6px 10px; font-size: 12px; }
      .ths-mask-editor .ths-mask-viewbar .ths-mask-icon-button { width: 32px; min-width: 32px; height: 32px; padding: 6px; }
      .ths-mask-editor .ths-mask-viewbar > :last-child { margin-inline-start: auto; }
      .ths-mask-editor .ths-mask-canvas-area { display: grid; grid-template-columns: 42px minmax(0,1fr); gap: 8px; min-width: 0; min-height: 0; }
      .ths-mask-editor .ths-mask-tools { display: flex; flex-direction: column; align-items: center; gap: 6px; padding-block: 4px; }
      .ths-mask-editor .ths-mask-tools > :nth-child(5) { margin-block-start: 12px; }
      .ths-mask-editor .ths-canvas-wrap { position: relative; min-width: 0; min-height: 220px; background: ${COLORS.panel}; border: 1px solid ${COLORS.borderSoft}; border-radius: 6px; overflow: hidden; }
      .ths-mask-editor .ths-mask-inspector { min-width: 0; min-height: 0; overflow: hidden; background: ${COLORS.panel}; border: 1px solid ${COLORS.borderSoft}; border-radius: 7px; display: flex; flex-direction: column; }
      .ths-mask-editor .ths-mask-inspector-body { flex: 1 1 0; min-height: 0; overflow-y: auto; overscroll-behavior: contain; scrollbar-width: thin; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
      .ths-mask-editor .ths-mask-export { flex: none; padding: 12px 16px; border-top: 1px solid ${COLORS.borderSoft}; }
      .ths-mask-editor .ths-mask-section { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
      .ths-mask-editor .ths-mask-section + .ths-mask-section { border-top: 1px solid ${COLORS.borderSoft}; padding-block-start: 12px; }
      .ths-mask-editor .ths-mask-section-title { margin: 0; color: ${COLORS.text}; font-size: 13px; line-height: 1.4; font-weight: 650; }
      .ths-mask-editor .ths-mask-section-copy { margin: 0; color: ${COLORS.muted}; font-size: 12px; line-height: 1.5; }
      .ths-mask-editor .ths-mask-inspector .ths-mask-label { display: grid; grid-template-columns: minmax(0,1fr) auto auto; gap: 8px; }
      .ths-mask-editor .ths-mask-mode { display: flex !important; flex-direction: column; align-items: stretch; gap: 6px !important; }
      .ths-mask-editor .ths-mask-mode select { width: 100%; }
      .ths-mask-editor .ths-mask-settings { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      .ths-mask-editor .ths-mask-settings .ths-mask-label { display: flex; flex-wrap: wrap; gap: 6px; }
      .ths-mask-editor .ths-mask-settings .ths-mask-label > :first-child { width: 100%; }
      .ths-mask-editor .ths-mask-settings input { width: 68px; font-variant-numeric: tabular-nums; }
      .ths-mask-editor .ths-mask-actions { display: flex; flex-direction: column; gap: 8px; }
      .ths-mask-editor .ths-mask-actions .ths-mask-button-primary { background: ${COLORS.panel2}; color: ${COLORS.text}; border-color: ${COLORS.border}; }
      .ths-mask-editor .ths-mask-export button[data-next-action=true] { background: ${COLORS.accent}; color: ${COLORS.accentText}; border-color: transparent; }
      .ths-mask-editor .ths-mask-export > .ths-mask-button { margin-block-start: 4px; }
      .ths-mask-editor .ths-mask-actions button, .ths-mask-editor .ths-mask-export button { width: 100%; align-items: center; justify-content: center; }
      .ths-mask-editor .ths-mask-queue-hint { color: ${COLORS.muted}; font-size: 12px; line-height: 1.5; }
      .ths-mask-editor .ths-mask-range { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; color: ${COLORS.muted}; font-size: 12px; }
      .ths-mask-editor .ths-mask-range input { width: 70px; }
      .ths-mask-editor .ths-mask-range > :last-child { width: 100%; }
      .ths-mask-editor .ths-mask-timeline { display: flex; flex-direction: column; gap: 10px; padding: 12px; border: 1px solid ${COLORS.borderSoft}; border-radius: 6px; background: ${COLORS.panel}; min-width: 0; }
      .ths-mask-editor .ths-mask-transport { display: flex; align-items: center; flex-wrap: wrap; gap: 5px; }
      .ths-mask-editor .ths-mask-transport .ths-mask-icon-button { width: 32px; min-width: 32px; height: 32px; min-height: 32px; padding: 6px; }
      .ths-mask-editor .ths-mask-transport .ths-mask-icon-button svg { width: 18px; height: 18px; }
      .ths-mask-editor .ths-mask-secondary { display: flex; flex-wrap: wrap; gap: 6px; }
      .ths-mask-editor .ths-mask-secondary .ths-mask-button { font-size: 11px; padding: 6px 8px; min-height: 30px; }
      .ths-mask-editor .ths-key-marker { width: 14px; height: 22px; min-height: 22px; padding: 0; border: 0; border-radius: 2px; background: transparent; position: absolute; top: -4px; transform: translateX(-50%); }
      .ths-mask-editor .ths-key-marker::before { content: ''; display: block; width: 3px; height: 14px; margin: 4px auto; border-radius: 1px; background: ${COLORS.mask}; }
      .ths-mask-editor details { color: ${COLORS.muted}; font-size: 12px; }
      .ths-mask-editor summary { cursor: pointer; padding-block: 4px; }
      .ths-mask-editor .ths-mask-dialog { position: fixed; inset: 0; margin: 0; color: ${COLORS.text}; background: ${COLORS.page}; border: 0; border-radius: 0; padding: 16px 20px; width: 100%; height: 100vh; height: 100dvh; max-width: none; max-height: none; box-sizing: border-box; overflow: hidden; box-shadow: none; transform: none; }
      .ths-mask-editor .ths-mask-dialog[open] { display: flex; flex-direction: column; gap: 12px; }
      .ths-mask-editor .ths-mask-dialog:modal { overscroll-behavior: contain; }
      .ths-mask-editor .ths-mask-dialog::backdrop { background: rgba(0,0,0,.72); overflow: hidden; overscroll-behavior: contain; }
      .ths-mask-editor .ths-mask-dialog-bar { display: flex; align-items: center; gap: 12px; min-height: 40px; flex: none; }
      .ths-mask-editor .ths-mask-dialog-bar > :nth-child(2) { margin-inline-start: auto; min-width: 0; overflow: hidden; }
      .ths-mask-editor .ths-mask-dialog-inner { flex: 1 1 0; min-height: 0; min-width: 0; container: mask-editor / inline-size; }
      .ths-mask-editor .ths-mask-dialog-inner .ths-mask-workbench { height: 100%; grid-template-rows: auto minmax(0,1fr); }
      .ths-mask-editor .ths-mask-dialog-inner .ths-canvas-wrap { min-height: 80px; }
      @container mask-editor (max-width: 760px) {
        .ths-mask-editor .ths-mask-workspace { grid-template-columns: minmax(0,1fr); }
        .ths-mask-editor .ths-mask-workbench { grid-template-rows: auto auto; height: auto; }
        .ths-mask-editor .ths-mask-viewer { min-height: 420px; }
        .ths-mask-editor .ths-mask-inspector { overflow: visible; }
        .ths-mask-editor .ths-mask-inspector-body { overflow: visible; flex: none; }
        .ths-mask-editor .ths-mask-dialog-inner { overflow-y: auto; overscroll-behavior: contain; }
        .ths-mask-editor .ths-mask-dialog-inner .ths-mask-workbench { height: auto; grid-template-rows: auto auto; }
      }
      @media (prefers-reduced-motion: reduce) { .ths-mask-editor button { transition: none; } }
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

    const root = createElement("section", {
        display: "flex", flexDirection: "column", gap: "8px", padding: "12px",
        width: "100%", boxSizing: "border-box", color: COLORS.text, background: COLORS.page,
        border: `1px solid ${COLORS.borderSoft}`, borderRadius: "7px", fontFamily: "Inter, Arial, sans-serif",
    }, "ths-mask-editor");
    root.tabIndex = 0;
    root.setAttribute("aria-label", "Редактор маски видео");
    addStyles(root);

    const header = createElement("div", { display: "flex", alignItems: "center", justifyContent: "space-between", gap: "8px", minHeight: "26px" });
    const title = createElement("div", { font: "600 14px/1.2 Inter, Arial, sans-serif", letterSpacing: "-.1px" });
    title.textContent = "Маска видео";
    const headerMeta = createElement("div", { display: "flex", alignItems: "center", gap: "7px", color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif", fontVariantNumeric: "tabular-nums" });
    const dirtyDot = createElement("span", { width: "7px", height: "7px", borderRadius: "50%", background: COLORS.dim, display: "inline-block" });
    const sourceReadout = document.createElement("span");
    headerMeta.append(dirtyDot, sourceReadout);
    const expand = iconButton("Развернуть", "expand", "Развернуть редактор");
    const debugDownload = iconButton("Скачать debug-лог", "download", "Скачать debug-лог · действия, ошибки и размеры редактора");
    const headerActions = createElement("div", { display: "flex", gap: "4px", alignItems: "center" });
    headerActions.append(debugDownload, expand);
    header.append(title, headerMeta, headerActions);
    root.append(header);

    const status = createElement("div", { minHeight: "24px", padding: "4px 8px", boxSizing: "border-box", borderRadius: "4px", background: COLORS.panel2, color: COLORS.muted, font: "500 12px/1.35 Inter, Arial, sans-serif" }, "ths-mask-status");
    status.setAttribute("role", "status");
    root.append(status);
    const recoverSource = button("Начать маску для нового клипа", "danger");
    recoverSource.title = "Принять новый клип и начать пустую маску. Старая маска и настройки будут сохранены для восстановления.";
    recoverSource.style.display = "none";
    recoverSource.style.alignSelf = "flex-start";
    const restorePrevious = button("Вернуть прежнюю маску");
    restorePrevious.title = "Восстановить сохранённую маску и настройки. Затем выберите прежний источник видео.";
    restorePrevious.style.display = "none";
    restorePrevious.style.alignSelf = "flex-start";
    const recoveryActions = createElement("div", { display: "none", flexWrap: "wrap", gap: "6px", alignItems: "center" }, "ths-mask-recovery-actions");
    recoveryActions.append(recoverSource, restorePrevious);
    root.append(recoveryActions);

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
    canvasWrap.append(canvas, empty, canvasBadge, brushCursor);
    root.append(canvasWrap, canvasPlaceholder);

    const viewRow = createElement("div", { display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "7px" });
    const viewButtons = {};
    const viewGroup = createElement("div", { display: "inline-flex", alignItems: "center", gap: "4px", flexWrap: "wrap" });
    for (const [key, label] of [["overlay", "Наложение"], ["source", "Источник"], ["mask", "Маска"]]) {
        const control = button(label, "neutral");
        control.dataset && (control.dataset.view = key);
        viewButtons[key] = control;
        viewGroup.append(control);
    }
    const opacity = document.createElement("input");
    opacity.type = "range"; opacity.min = "0"; opacity.max = "1"; opacity.step = ".01"; opacity.value = ".38"; opacity.style.width = "92px";
    const opacityReadout = createElement("span", { minWidth: "31px", color: COLORS.muted, fontSize: "11px", fontVariantNumeric: "tabular-nums" }); opacityReadout.textContent = "38%";
    viewRow.append(viewGroup, labelControl("Прозрачность", opacity, opacityReadout));
    root.append(viewRow);

    const toolPanel = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "6px", padding: "6px 0 2px", borderTop: `1px solid ${COLORS.borderSoft}` }, "ths-mask-tools");
    const toolLabel = createElement("span", { color: COLORS.dim, font: "600 10px/1.2 Inter, Arial, sans-serif", textTransform: "uppercase", letterSpacing: ".07em", marginRight: "2px" }); toolLabel.textContent = "Рисование";
    const tools = {};
    for (const [key, label] of [["brush", "Кисть"], ["erase", "Ластик"], ["rect", "Прямоугольник"], ["polygon", "Полигон"]]) { tools[key] = iconButton(label, key); toolPanel.append(tools[key]); }
    const fit = iconButton("Вписать", "fit", "Вписать весь кадр · сбросить масштаб и положение");
    const zoomOut = iconButton("Уменьшить", "minus", "Уменьшить масштаб · колесо в полноэкранном просмотре");
    const zoomReadout = createElement("span", { minWidth: "42px", textAlign: "center", color: COLORS.muted, fontSize: "11px", fontVariantNumeric: "tabular-nums" });
    const zoomIn = iconButton("Увеличить", "plus", "Увеличить масштаб · колесо в полноэкранном просмотре");
    canvas.title = "Полный экран: колесо — зум под курсором; зажатое колёсико — перемещение кадра. В ноде: Ctrl + колесо — зум кадра.";
    zoomReadout.title = "Масштаб относительно вписанного кадра. Колесо — зум; зажатое колёсико — перемещение.";
    const zoomGroup = createElement("span", { display: "inline-flex", alignItems: "center", gap: "3px", marginLeft: "auto" }); zoomGroup.append(zoomOut, zoomReadout, zoomIn, fit);
    toolPanel.append(zoomGroup);
    root.append(toolPanel);

    const settingsRow = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "7px", padding: "4px 0 2px" });
    const brushInput = document.createElement("input"); brushInput.type = "range"; brushInput.min = ".002"; brushInput.max = ".2"; brushInput.step = ".001"; brushInput.value = ".035"; brushInput.style.width = "92px";
    const brushReadout = createElement("span", { minWidth: "34px", color: COLORS.muted, fontSize: "11px", fontVariantNumeric: "tabular-nums" }); brushReadout.textContent = "3.5%";
    const brushLabel = labelControl("Размер", brushInput, brushReadout);
    settingsRow.append(brushLabel);
    const modeSelect = document.createElement("select");
    for (const [value, text] of [["sam3", "Объект · SAM3"], ["optical_flow", "Область · Optical Flow"], ["keyframes", "Ручные ключи"], ["static", "Неподвижная маска"]]) { const option = document.createElement("option"); option.value = value; option.textContent = text; modeSelect.append(option); }
    modeSelect.setAttribute("aria-label", "Трекинг");
    modeSelect.title = "SAM3 — объект с двухэтапным расчётом; Optical Flow — область по движению; Ручные ключи; Неподвижная маска";
    const maskSettingsGroup = createElement("div", { display: "inline-flex", alignItems: "center", flexWrap: "wrap", gap: "12px" }, "ths-mask-settings");
    maskSettingsGroup.setAttribute("role", "group");
    maskSettingsGroup.setAttribute("aria-label", "Настройки маски");
    const modeLabel = labelControl("Режим", modeSelect);
    modeLabel.className += " ths-mask-mode";
    const directionSelect = document.createElement("select");
    for (const [value, text] of [["forward", "вперёд"], ["backward", "назад"], ["both", "оба"]]) { const option = document.createElement("option"); option.value = value; option.textContent = text; directionSelect.append(option); }
    directionSelect.setAttribute("aria-label", "Направление");
    directionSelect.title = "Направление трекинга SAM3; диапазон задаётся ниже";
    const directionLabel = labelControl("Направление", directionSelect);

    const maskSettingsInputs = {};
    const maskSettingsDefaults = { grow: 8, feather: 4 };
    for (const [name, label, description] of [["grow", "Grow", "Расширение маски"], ["feather", "Feather", "Растушёвка края маски"]]) {
        const input = document.createElement("input");
        input.type = "number"; input.min = "0"; input.max = "256"; input.step = "1";
        input.style.width = "60px";
        input.setAttribute("aria-label", label);
        input.title = `${description}, 0–256 пикселей.`;
        const unit = createElement("span", { color: COLORS.muted, fontSize: "11px" }); unit.textContent = "px";
        maskSettingsInputs[name] = input;
        maskSettingsGroup.append(labelControl(label, input, unit));
    }
    viewRow.append(maskSettingsGroup);
    const refine = button("Выделять объект");
    refine.title = "Прямоугольник вокруг объекта должен захватить объект и немного окружающего фона; после расчёта уточните кистью или ластиком.";
    refine.setAttribute("aria-pressed", "false");
    settingsRow.append(refine);
    const refineMethod = document.createElement("select");
    refineMethod.setAttribute("aria-label", "Метод выделения");
    refineMethod.title = "Метод для новых прямоугольников. GrabCut — захватите немного фона; SAM3 — обведите объект близко к границам. Уже созданные маски сохраняют свой метод.";
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
    const frameReadout = createElement("span", { color: COLORS.text, font: "600 12px/1.2 Inter, Arial, sans-serif", fontVariantNumeric: "tabular-nums", marginLeft: "3px" });
    const timeReadout = createElement("span", { color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif", fontVariantNumeric: "tabular-nums", marginLeft: "auto" });
    navRow.append(first, back10, back, play, next, next10, last, previousKeyButton, nextKeyButton, direct, frameReadout, timeReadout);
    const speed = document.createElement("select"); speed.setAttribute("aria-label", "Скорость воспроизведения"); speed.style.width = "68px";
    for (const value of ["0.25", "0.5", "1", "1.5", "2"]) { const option = document.createElement("option"); option.value = value; option.textContent = `${value}×`; if (value === "1") option.selected = true; speed.append(option); }
    navRow.append(labelControl("", speed));
    timeline.append(navRow);

    const sliderWrap = createElement("div", { position: "relative", padding: "2px 4px 0" });
    const slider = document.createElement("input"); slider.type = "range"; slider.min = "0"; slider.max = "0"; slider.step = "1"; slider.value = "0"; slider.style.width = "100%"; slider.setAttribute("aria-label", "Лента кадров");
    const markerLayer = createElement("div", { position: "absolute", left: "9px", right: "9px", top: "0", height: "20px", pointerEvents: "none" });
    sliderWrap.append(slider, markerLayer);
    const timelineLine = createElement("div", { display: "flex", justifyContent: "space-between", color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif", fontVariantNumeric: "tabular-nums" });
    const rangeReadout = document.createElement("span");
    const keyReadout = document.createElement("span");
    timelineLine.append(rangeReadout, keyReadout);
    timeline.append(sliderWrap, timelineLine);
    root.append(timeline);

    const actionRow = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "6px", paddingTop: "2px" });
    const undo = iconButton("Отменить", "undo", "Отменить · Ctrl+Z"); const redo = iconButton("Повторить", "redo", "Повторить · Ctrl+Shift+Z / Ctrl+Y"); const clear = button("Очистить кадр"); const copyPrevious = button("Скопировать предыдущий ключ"); const reset = button("Сбросить маску", "danger");
    actionRow.append(undo, redo, clear, copyPrevious, reset);
    toolPanel.append(undo, redo, brushLabel);
    root.append(actionRow);

    const rangeRow = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "7px", color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif" });
    const startInput = document.createElement("input"); startInput.type = "number"; startInput.min = "1"; startInput.value = "1"; startInput.style.width = "59px"; startInput.setAttribute("aria-label", "Начало диапазона");
    const endInput = document.createElement("input"); endInput.type = "number"; endInput.min = "1"; endInput.value = "1"; endInput.style.width = "59px"; endInput.setAttribute("aria-label", "Конец диапазона");
    const endInfinity = createElement("span", { color: COLORS.muted }); endInfinity.textContent = "до конца";
    const fpsReadout = document.createElement("span");
    const rangeLabel = createElement("span"); rangeLabel.textContent = "Кадры";
    const rangeDash = createElement("span"); rangeDash.textContent = "—";
    rangeRow.append(rangeLabel, startInput, rangeDash, endInput, endInfinity, fpsReadout);
    root.append(rangeRow);

    const actionButtons = createElement("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "7px", paddingTop: "2px" }, "ths-mask-actions");
    const prepare = button("Загрузить кадры", "primary");
    prepare.title = "Рассчитать маску и трекинг для просмотра всех кадров. Этот запрос не запускает генерацию H3.";
    const calculateFrame = button("Рассчитать кадр", "primary");
    calculateFrame.title = "Рассчитать SAM3 только на выбранном кадре. H3 и трекинг видео не запускаются.";
    const trackVideo = button("Отследить видео", "primary");
    trackVideo.title = "Отследить SAM3 по выбранному направлению и диапазону. Сначала рассчитайте кадр.";
    const cancelRequest = button("Отменить расчёт", "danger");
    cancelRequest.title = "Отменить только свой запрос, если ComfyUI поддерживает адресную отмену.";
    cancelRequest.style.display = "none";
    const queue = button("Запустить H3");
    queue.title = "Запустить полный workflow H3";
    const queueHint = createElement("span", { color: COLORS.dim, font: "500 11px/1.25 Inter, Arial, sans-serif" }, "ths-mask-queue-hint"); queueHint.textContent = "Подготовка не запускает полный граф";
    actionButtons.append(prepare, calculateFrame, trackVideo, cancelRequest, queue, queueHint);
    root.append(actionButtons);

    // Keep every mask control in the primary workbench.  Older revisions put
    // tracking, grow/feather, brush refinement and the frame range inside a
    // collapsed <details>, which made the main editor look complete while
    // silently hiding the settings required for a safe repair.  The details
    // element remains as a compact explanatory affordance, but owns no state
    // controls; fullscreen and inline views therefore expose the same controls.
    const advanced = document.createElement("details");
    const advancedSummary = document.createElement("summary"); advancedSummary.textContent = "Подсказки: режимы, диапазон и правки доступны выше";
    const advancedNote = createElement("div", { color: COLORS.muted, font: "500 11px/1.35 Inter, Arial, sans-serif", padding: "4px 0" });
    advancedNote.textContent = "Static, Keyframes, Optical Flow, Grow, Feather, кисть и уточнение применяются в основном редакторе.";
    advanced.append(advancedSummary, advancedNote);

    // The inline node view and the modal share one complete workbench.  Moving
    // the wrapper (rather than only the canvas) keeps navigation, tools and
    // actions available while the large viewer is open.
    const workbenchPlaceholder = createElement("span", { display: "none" });

    // A single DOM tree is shared by inline and fullscreen views. Only its
    // outer wrapper moves; controls retain position, state and event handlers.
    function restyle(element, className) {
        element.removeAttribute?.("style");
        if (className) element.className = className;
        return element;
    }
    function section(label, className = "") {
        const element = createElement("section", {}, `ths-mask-section ${className}`);
        element.setAttribute("aria-label", label);
        const heading = createElement("h3", {}, "ths-mask-section-title");
        heading.textContent = label; element.append(heading); return element;
    }
    restyle(header, "ths-mask-header");
    restyle(status, "ths-mask-status");
    restyle(viewRow, "ths-mask-viewbar");
    restyle(toolPanel, "ths-mask-tools");
    restyle(canvasWrap, "ths-canvas-wrap");
    restyle(timeline, "ths-mask-timeline");
    restyle(navRow, "ths-mask-transport");
    restyle(actionRow, "ths-mask-secondary");
    restyle(rangeRow, "ths-mask-range");
    restyle(actionButtons, "ths-mask-actions");
    restyle(maskSettingsGroup, "ths-mask-settings");
    restyle(queueHint, "ths-mask-queue-hint");
    const selectionPanel = section("1. Выделение");
    const modeHint = createElement("p", {}, "ths-mask-section-copy");
    selectionPanel.append(modeLabel, modeHint);
    const edgesPanel = section("2. Настройки маски");
    edgesPanel.append(maskSettingsGroup, brushLabel, refine, refineMethodLabel);
    const trackingPanel = section("3. Диапазон трекинга");
    trackingPanel.append(directionLabel, rangeRow);
    timelineLine.append(fpsReadout);
    const exportPanel = section("Расчёт", "ths-mask-export");
    exportPanel.append(actionButtons, queue, queueHint);
    const editPanel = section("Правки кадра");
    editPanel.append(actionRow, advanced);
    const inspector = createElement("aside", {}, "ths-mask-inspector");
    inspector.setAttribute("aria-label", "Настройки маски и расчёт");
    const inspectorBody = createElement("div", {}, "ths-mask-inspector-body");
    inspectorBody.append(selectionPanel, edgesPanel, trackingPanel, editPanel);
    inspector.append(inspectorBody, exportPanel);
    viewRow.append(zoomGroup);
    toolPanel.append(tools.brush, tools.erase, tools.rect, tools.polygon, undo, redo);
    toolPanel.setAttribute("role", "toolbar");
    toolPanel.setAttribute("aria-label", "Инструменты рисования");
    const canvasArea = createElement("div", {}, "ths-mask-canvas-area");
    canvasArea.append(toolPanel, canvasWrap);
    const viewer = createElement("div", {}, "ths-mask-viewer");
    viewer.append(viewRow, canvasArea, timeline);
    const workspace = createElement("div", {}, "ths-mask-workspace");
    workspace.append(viewer, inspector);
    const statusArea = createElement("div", {}, "ths-mask-status-area");
    statusArea.append(status, recoveryActions);
    const workbench = createElement("div", {}, "ths-mask-workbench");
    workbench.append(statusArea, workspace);
    // Remove obsolete empty row wrappers after moving their children.
    settingsRow.remove?.(); canvasPlaceholder.remove?.();
    root.append(workbenchPlaceholder, workbench);

    const dialog = document.createElement("dialog");
    dialog.className = "ths-mask-dialog";
    dialog.setAttribute("aria-label", "Редактор маски видео — большой экран");
    const dialogBar = createElement("div", {}, "ths-mask-dialog-bar");
    const dialogTitle = createElement("strong"); dialogTitle.textContent = "Маска видео";
    const closeDialogButton = iconButton("Закрыть", "close", "Закрыть редактор · Esc");
    closeDialogButton.setAttribute("autofocus", "");
    dialogBar.append(dialogTitle, closeDialogButton);
    const dialogInner = createElement("div", {}, "ths-mask-dialog-inner");
    dialog.append(dialogBar, dialogInner);
    root.append(dialog);

    const ctx = canvas.getContext?.("2d");
    const maskCanvas = document.createElement("canvas");
    const maskCtx = maskCanvas.getContext?.("2d");
    const documentState = readDocument(node);
    const initialSource = sourceInfo(documentState);
    const state = {
        mode: "brush", view: "overlay", frame: 0, frameCount: initialSource?.frames || 1,
        fps: number(widgetValue(node, "fps", documentState.fps || 24), 24), source: initialSource,
        metadata: {}, metadataAnchors: [], frameItems: new Map(), previews: new Map(), imageCache: new Map(), imagePending: new Map(),
        loadGeneration: 0, sourceGeneration: 0, missingFrame: false,
        anchors: explicitAnchors(documentState), localEdits: false, dirty: false, previewStale: false, resetBlank: false, previewSignature: "", editRevision: 0, pendingPreview: null, widgetSnapshot: widgetSnapshot(node), modeSnapshot: String(widgetValue(node, "tracking_mode", "")),
        active: null, polygon: [], ghost: null, grabcut: false, refineMethod: node.properties?.__thsRefineMethod === "sam3" ? "sam3" : "grabcut", dpr: 1, zoom: 1, panX: 0, panY: 0, panning: null,
        playbackSpeed: 1, playing: false, timer: null, history: [], historyIndex: -1, dialogOpen: false, dialogReturnFocus: null,
        markerKey: "", previewUse: 0, lastPointer: null, destroyed: false, pendingSource: null,
        sourceMismatch: null, lastMismatch: null, mismatchBlocked: false, previousMaskBackup: null,
        refinementBusy: false, refinementError: "", pendingH3: null, pendingPromptId: null,
        sam3: { frameReady: false, trackReady: false, frame: null, seedToken: "", direction: "both", computedFrames: [], error: "", sourceKey: "", frameSignature: "", trackSignature: "", pendingAction: "" },
        pendingRequestKind: "", apiListeners: [],
    };
    if (node.properties?.__thsPreviousMask && typeof node.properties.__thsPreviousMask === "object") state.previousMaskBackup = clone(node.properties.__thsPreviousMask);

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
            h3Disabled: Boolean(queue.disabled), previewDisabled: Boolean(prepare.disabled), frameDisabled: Boolean(calculateFrame.disabled), trackDisabled: Boolean(trackVideo.disabled),
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
    function currentSourceKey() {
        const source = state.source || sourceInfo(documentState);
        if (!source) return "";
        return [source.frames, source.width, source.height, source.fingerprint || ""].join("|");
    }
    function sam3SeedEvents(frame = state.frame) {
        return eventsAt(frame).filter((event) => event?.type === "rect" && (event.refine_method === "sam3" || event.refine === true));
    }
    function hasSam3Seed(frame = state.frame) { return sam3SeedEvents(frame).length > 0; }
    function sam3SeedSignature(frame = state.frame) {
        try { return JSON.stringify({ source: currentSourceKey(), frame, events: sam3SeedEvents(frame) }); } catch (_) { return `${currentSourceKey()}|${frame}`; }
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
        const signature = { source: currentSourceKey(), seed: state.sam3.seedToken || documentState.sam3?.seed_token || "", frame: Number.isFinite(Number(frame)) ? Number(frame) : null, edits: sam3SeedSignature(frame) };
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

    function updateMaskSettingTooltips() {
        const sam3 = sam3Mode();
        const hint = sam3
            ? "Для SAM3 значение применяется к сохранённому кэшу трекинга; для нового трека нажмите «Отследить видео»."
            : "Для предпросмотра нажмите «Рассчитать маску».";
        for (const [name, input] of Object.entries(maskSettingsInputs)) {
            const description = name === "grow" ? "Расширение маски" : "Растушёвка края маски";
            input.title = `${description}, 0–256 пикселей. ${hint}`;
        }
    }

    function updateActionAvailability() {
        const modeIsSam3 = sam3Mode();
        modeHint.textContent = modeIsSam3 ? "Рамка вокруг объекта → маска кадра → трекинг видео." : safeMode(node, documentState) === "optical_flow" ? "Нарисуйте область. Optical Flow перенесёт её по движению." : safeMode(node, documentState) === "keyframes" ? "Маски ближайших ручных ключей, без расчёта движения." : "Одна неподвижная маска для всего диапазона.";
        const frameReady = sam3FrameReady();
        const trackReady = sam3TrackReady();
        calculateFrame.setAttribute("data-next-action", String(modeIsSam3 && !frameReady));
        trackVideo.setAttribute("data-next-action", String(modeIsSam3 && frameReady && !trackReady));
        prepare.setAttribute("data-next-action", String(!modeIsSam3));
        queue.setAttribute("data-next-action", String(modeIsSam3 && trackReady));
        queue.disabled = Boolean(actionBlocked() || state.pendingH3 || (modeIsSam3 && !trackReady));
        const sam3Pending = Boolean(state.pendingPreview && String(state.pendingPreview.kind || "").startsWith("sam3_"));
        const sourceLoaded = Boolean(state.source) || state.frameItems.size > 0;
        const frameActionLabel = sourceLoaded ? "Рассчитать кадр" : "Загрузить кадры";
        calculateFrame.textContent = frameActionLabel;
        calculateFrame.setAttribute("aria-label", frameActionLabel);
        calculateFrame.title = sourceLoaded
            ? "Рассчитать SAM3 только на выбранном кадре. H3 и трекинг видео не запускаются."
            : "Загрузить кадры и подготовить выбранный кадр SAM3. H3 и трекинг видео не запускаются.";
        calculateFrame.style.display = modeIsSam3 ? "inline-flex" : "none";
        trackVideo.style.display = modeIsSam3 ? "inline-flex" : "none";
        directionLabel.style.display = modeIsSam3 ? "inline-flex" : "none";
        refine.style.display = modeIsSam3 ? "none" : "inline-flex";
        refineMethodLabel.style.display = modeIsSam3 ? "none" : "inline-flex";
        prepare.style.display = modeIsSam3 ? "none" : "inline-flex";
        cancelRequest.style.display = state.pendingPreview || state.pendingH3 ? "inline-flex" : "none";
        calculateFrame.disabled = Boolean(actionBlocked() || sam3Pending || state.pendingH3 || state.mismatchBlocked);
        trackVideo.disabled = Boolean(actionBlocked() || sam3Pending || state.pendingH3 || state.mismatchBlocked || !frameReady);
        if (state.mismatchBlocked) {
            prepare.disabled = Boolean(state.pendingPreview);
        } else if (state.pendingPreview) {
            prepare.disabled = true;
        }
        recoverSource.style.display = state.pendingSource ? "inline-flex" : "none";
        restorePrevious.style.display = state.previousMaskBackup ? "inline-flex" : "none";
        recoveryActions.style.display = state.pendingSource || state.previousMaskBackup ? "flex" : "none";
        refineMethod.disabled = Boolean(state.pendingPreview || state.pendingH3 || modeIsSam3);
        updateMaskSettingTooltips();
        if (modeIsSam3) {
            queueHint.textContent = trackReady ? "Трек готов · H3 доступен" : frameReady ? "Кадр готов · нажмите «Отследить видео»" : "Сначала рассчитайте кадр SAM3";
        } else queueHint.textContent = "Подготовка не запускает полный граф";
    }

    function clearPreviewCaches() {
        state.sourceGeneration += 1;
        state.loadGeneration += 1;
        state.frameItems.clear();
        state.previews.clear();
        state.imageCache.clear();
        state.imagePending.clear();
        state.missingFrame = false;
        state.previewUse = 0;
    }

    function capturePreviousMask(source = state.source || sourceInfo(documentState)) {
        const properties = node.properties && typeof node.properties === "object" ? node.properties : (node.properties = {});
        const existing = properties.__thsPreviousMask;
        const boundSource = sourceInfo(documentState) || sourceInfo({ source }) || sourceInfo({ source: state.source });
        const existingSource = sourceInfo({ source: existing?.source });
        const sameSource = existingSource && boundSource && existingSource.frames === boundSource.frames && existingSource.width === boundSource.width && existingSource.height === boundSource.height && (!existingSource.fingerprint || !boundSource.fingerprint || existingSource.fingerprint === boundSource.fingerprint);
        if (existing && typeof existing === "object" && sameSource) { state.previousMaskBackup = clone(existing); return state.previousMaskBackup; }
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
        return backup;
    }

    function frameBucket(frame, create = false) {
        const frames = documentState.frames && typeof documentState.frames === "object" ? documentState.frames : (documentState.frames = {});
        const key = String(Math.max(0, Math.round(frame)));
        if (!frames[key] && create) frames[key] = { events: [], inherit: shouldInherit(frame) };
        return frames[key];
    }

    function hasIndependentAnchor() {
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

    function shouldInherit(frame) {
        // A correction can inherit only after an independently painted anchor
        // exists and the backend supplied its raw edit baseline.  Source
        // metadata alone is not a mask and must never turn the first stroke
        // into an inherited correction.
        return safeMode(node, documentState) !== "static" && hasIndependentAnchor() && hasReliableEditBase(frame);
    }

    function eventsAt(frame, create = false) {
        const bucket = frameBucket(frame, create);
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

    function syncMaskSettingsInputs() {
        const values = snapshotMaskSettings();
        for (const [name, input] of Object.entries(maskSettingsInputs)) {
            input.value = String(values[name]);
            input.disabled = !findWidget(node, name) || Boolean(node.inputs?.some((socket) => socket.name === name && socket.link != null));
        }
    }

    function documentSignature() {
        try { return JSON.stringify(documentState); } catch (_) { return ""; }
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
        const historyAction = index < state.historyIndex ? "Изменение отменено" : "Изменение повторено";
        replaceDocument(documentState, entry.document);
        for (const [name, value] of Object.entries(entry.settings || {})) {
            if (Object.prototype.hasOwnProperty.call(maskSettingsDefaults, name)) setWidget(node, name, value);
        }
        syncMaskSettingsInputs();
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
        state.resetBlank = Object.keys(documentState.frames || {}).length === 0;
        state.localEdits = true; state.dirty = true; state.previewStale = true; state.editRevision += 1;
        state.source = sourceInfo(documentState) || state.source;
        state.frameCount = state.source?.frames || state.frameCount;
        state.anchors = explicitAnchors(documentState);
        if (documentState.tracking_mode === "sam3") {
            state.sam3.frame = Number.isFinite(Number(documentState.sam3?.frame)) ? Math.max(0, Math.round(Number(documentState.sam3.frame))) : state.sam3.frame;
            state.sam3.seedToken = String(documentState.sam3?.seed_token || state.sam3.seedToken || "");
            state.sam3.direction = sam3Direction(documentState.sam3?.direction || state.sam3.direction);
            state.sam3.frameReady = false; state.sam3.trackReady = false; state.sam3.frameSignature = ""; state.sam3.trackSignature = "";
            directionSelect.value = state.sam3.direction;
        }
        updateReadouts(); draw(); commit({ markDirty: false });
        setStatus(`${historyAction}. Обновите маску для проверки результата.`, "dirty");
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
            };
            state.sam3.seedToken = payload.sam3.seed_token;
        }
        documentState.frames = frames; documentState.version = 1; documentState.source = payload.source;
        if (payload.anchors) documentState.anchors = clone(payload.anchors);
        delete documentState.keyframes; delete documentState.keys; documentState.tracking_mode = payload.tracking_mode;
        if (payload.sam3) documentState.sam3 = clone(payload.sam3); else delete documentState.sam3;
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
        dirtyDot.style.background = state.dirty ? COLORS.warning : state.metadata?.status === "error" ? COLORS.danger : COLORS.dim;
        sourceReadout.textContent = state.source ? `${state.source.width}×${state.source.height} · ${state.frameCount} кадр.` : "Источник не загружен";
    }

    function setStatus(message, tone = "normal") {
        if (status.textContent !== message || status.dataset?.state !== tone) diagnostics.record("status", { tone, message: diagnosticText(message) });
        status.textContent = message;
        status.dataset && (status.dataset.state = tone);
        status.style.background = COLORS.panel2;
        status.style.color = tone === "error" ? COLORS.danger : COLORS.muted;
    }

    function selectedPreview() {
        return state.previews.get(state.frame) || state.frameItems.get(state.frame) || null;
    }

    function frameHasDraft(frame) {
        const bucket = documentState.frames?.[String(Math.max(0, Math.round(frame)))];
        return Boolean(bucket && (Object.prototype.hasOwnProperty.call(bucket, "inherit") || parseEvents(bucket).length));
    }

    function frameNeedsDraft(frame) {
        if (state.resetBlank) return true;
        if (!frameHasDraft(frame)) return Boolean(state.active || state.ghost || state.polygon.length) && frame === state.frame;
        return state.previewStale || state.previewSignature !== documentSignature() || (frame === state.frame && Boolean(state.active || state.ghost || state.polygon.length));
    }

    function displayAnchors() {
        if (state.mismatchBlocked) return [];
        const derived = Object.keys(documentState.frames || {}).filter((key) => {
            const bucket = documentState.frames[key];
            return Object.prototype.hasOwnProperty.call(bucket || {}, "inherit") || parseEvents(bucket).length;
        }).map(Number);
        return [...new Set([...explicitAnchors(documentState), ...(state.metadataAnchors || []), ...derived])]
            .filter((frame) => Number.isFinite(frame) && frame >= 0 && frame < state.frameCount)
            .sort((a, b) => a - b);
    }

    function updateBrushReadout() {
        const size = Math.min(number(state.source?.width, 0), number(state.source?.height, 0));
        brushReadout.textContent = size > 0
            ? `${Math.max(1, Math.round(number(brushInput.value, .035) * size * 2))} px`
            : `${(number(brushInput.value, .035) * 100).toFixed(1)}%`;
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

    function tintMaskCanvas(maskOnly, alpha) {
        if (!maskCtx || typeof maskCtx.getImageData !== "function" || typeof maskCtx.putImageData !== "function") return false;
        let imageData;
        try { imageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height); } catch (_) { return false; }
        if (!imageData?.data) return false;
        const pixels = imageData.data;
        for (let index = 0; index < pixels.length; index += 4) {
            const luminance = (0.2126 * pixels[index] + 0.7152 * pixels[index + 1] + 0.0722 * pixels[index + 2]) / 255;
            const sourceAlpha = pixels[index + 3] / 255;
            if (maskOnly) {
                const value = Math.round(luminance * sourceAlpha * 255);
                pixels[index] = value; pixels[index + 1] = value; pixels[index + 2] = value; pixels[index + 3] = 255;
            } else {
                pixels[index] = 79; pixels[index + 1] = 227; pixels[index + 2] = 139;
                pixels[index + 3] = Math.round(luminance * sourceAlpha * 255 * alpha);
            }
        }
        try { maskCtx.putImageData(imageData, 0, 0); } catch (_) { return false; }
        return true;
    }

    function draw() {
        const preview = selectedPreview();
        if (!ctx || !preview?.sourceImage?.naturalWidth || state.missingFrame) {
            ctx?.clearRect?.(0, 0, canvas.width, canvas.height);
            empty.style.display = "grid"; brushCursor.style.display = "none"; return;
        }
        empty.style.display = "none";
        const { width: viewportWidth, height: viewportHeight } = canvasMetrics(); state.dpr = Math.min(2, window.devicePixelRatio || 1);
        canvas.width = Math.max(1, Math.round(viewportWidth * state.dpr)); canvas.height = Math.max(1, Math.round(viewportHeight * state.dpr));
        ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0); ctx.clearRect(0, 0, viewportWidth, viewportHeight);
        const box = geometry(); if (!box) return;
        ctx.fillStyle = window.getComputedStyle?.(canvasWrap)?.backgroundColor || "#202020"; ctx.fillRect(0, 0, viewportWidth, viewportHeight);
        if (state.view !== "mask") ctx.drawImage(preview.sourceImage, box.offsetX, box.offsetY, box.drawnWidth, box.drawnHeight);
        maskCanvas.width = Math.max(1, Math.round(viewportWidth * state.dpr)); maskCanvas.height = Math.max(1, Math.round(viewportHeight * state.dpr));
        if (maskCtx) {
            maskCtx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0); maskCtx.clearRect(0, 0, viewportWidth, viewportHeight);
            const mismatchBlocked = Boolean(state.mismatchBlocked);
            const refinementFailed = Boolean(state.refinementError || state.metadata?.refinement_error);
            const draft = mismatchBlocked || refinementFailed ? true : frameNeedsDraft(state.frame);
            const bucket = frameBucket(state.frame);
            const inherits = bucket ? bucket.inherit === true : shouldInherit(state.frame);
            const baseline = mismatchBlocked || refinementFailed || state.resetBlank ? null : draft
                ? (inherits ? preview.editBaseImage : null)
                : preview.maskImage;
            if (baseline?.naturalWidth) maskCtx.drawImage(baseline, box.offsetX, box.offsetY, box.drawnWidth, box.drawnHeight);
            if (draft && !mismatchBlocked) for (const event of eventsAt(state.frame)) drawEvent(maskCtx, event, box, true, { outlineOnly: refinementFailed || state.refinementBusy });
            if (state.ghost && !mismatchBlocked) drawEvent(maskCtx, state.ghost, box, true, { outlineOnly: state.refinementBusy || state.ghost.refine });
            if (state.polygon.length) drawEvent(maskCtx, { type: "polygon", mode: "add", points: state.polygon }, box, true);
            if (state.view !== "source") {
                const maskOnly = state.view === "mask";
                const alpha = number(opacity.value, .38);
                tintMaskCanvas(maskOnly, alpha);
                ctx.save(); ctx.globalAlpha = 1; ctx.drawImage(maskCanvas, 0, 0, viewportWidth, viewportHeight); ctx.restore();
            }
        }
        const sam3Badge = sam3Mode() ? ` · SAM3 ${sam3TrackReady() ? "трек" : sam3FrameReady() ? "кадр" : "черновик"}` : "";
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
        const seconds = state.fps > 0 ? frame / state.fps : 0; const duration = state.fps > 0 ? (count - 1) / state.fps : 0;
        timeReadout.textContent = `${seconds.toFixed(2)} с / ${duration.toFixed(2)} с`;
        state.anchors = displayAnchors();
        keyReadout.textContent = state.anchors.length ? `Ключи: ${state.anchors.map((key) => key + 1).join(", ")}` : "Ключи: нет";
        const start = Math.max(0, Math.round(number(widgetValue(node, "start_frame", 0)))); const endRaw = Math.round(number(widgetValue(node, "end_frame", -1), -1)); const end = endRaw < 0 ? count - 1 : clamp(endRaw, 0, count - 1);
        rangeReadout.textContent = `Диапазон: ${start + 1}—${end + 1}`; if (document.activeElement !== startInput) startInput.value = String(start + 1); if (document.activeElement !== endInput) endInput.value = String(end + 1); endInfinity.textContent = endRaw < 0 ? "до конца" : "";
        fpsReadout.textContent = `FPS ${state.fps.toFixed(2)}`;
        first.disabled = frame <= 0; back10.disabled = frame <= 0; back.disabled = frame <= 0;
        next.disabled = frame >= count - 1; next10.disabled = frame >= count - 1; last.disabled = frame >= count - 1;
        previousKeyButton.disabled = !state.anchors.some((key) => key < frame);
        nextKeyButton.disabled = !state.anchors.some((key) => key > frame);
        undo.disabled = state.historyIndex <= 0; redo.disabled = state.historyIndex < 0 || state.historyIndex >= state.history.length - 1;
        copyPrevious.disabled = !state.anchors.some((key) => key < frame && frameHasDraft(key));
        updateBrushReadout();
        const markerKey = `${count}:${state.anchors.join(",")}`;
        if (markerKey !== state.markerKey) {
            markerLayer.textContent = "";
            for (const key of state.anchors) {
                const marker = button(""); marker.className = "ths-key-marker"; marker.title = `Ключевой кадр ${key + 1}`; marker.style.left = `${count <= 1 ? 0 : (key / (count - 1)) * 100}%`; marker.style.pointerEvents = "auto"; marker.addEventListener("click", () => setFrame(key, { source: "key" })); markerLayer.append(marker);
            }
            state.markerKey = markerKey;
        }
        updateDirtyIndicator();
    }

    function chooseMode(mode) {
        state.mode = mode;
        for (const [key, control] of Object.entries(tools)) { control.setAttribute("aria-pressed", String(key === mode)); control.style.outline = key === mode ? `2px solid ${COLORS.accentStrong}` : "none"; control.style.background = key === mode ? COLORS.selected : COLORS.panel2; }
        canvas.style.cursor = mode === "brush" || mode === "erase" ? "crosshair" : "cell";
        setStatus(mode === "polygon" ? "Полигон: кликайте вершины, Enter или двойной клик — завершить." : mode === "rect" ? (sam3Mode() ? "SAM3: обведите объект рамкой, затем нажмите «Рассчитать кадр»." : state.grabcut ? state.refineMethod === "sam3" ? "SAM3: обведите объект близко к границам, затем нажмите «Рассчитать маску»." : "GrabCut: обведите объект и немного фона, затем нажмите «Рассчитать маску»." : "Прямоугольник: протяните область для маски.") : mode === "erase" ? "Ластик: стирает зелёную маску; правка сохранится для повторного трекинга." : "Кисть: добавляет область в маску; правка сохранится для повторного трекинга.");
    }

    function syncConfiguredToolState({ resetLegacy = false } = {}) {
        if (sam3Mode()) {
            state.grabcut = true;
            state.refineMethod = "sam3";
            chooseMode("rect");
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
        state.frame = requested; state.polygon = []; state.ghost = null; state.active = null; state.missingFrame = false;
        if (!state.playing) diagnostics.record("frame.selected", { reason: source });
        updateReadouts(); updateActionAvailability(); draw();
        if (load) loadFrame(requested, { reason: source });
        return requested;
    }

    function cacheImage(key, image) {
        if (!image) return;
        state.imageCache.delete(key); state.imageCache.set(key, image);
        while (state.imageCache.size > MAX_PREVIEWS * 2) state.imageCache.delete(state.imageCache.keys().next().value);
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
        return { source: item.source || item.sourceDescriptor || item, mask: item.mask || item.maskDescriptor || item.mask_url, edit_base: item.edit_base || item.editBase || item.edit_base_descriptor };
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
        const editBaseImage = descriptor.editBaseImage || await loadImage(descriptor.edit_base, `edit_base:${frame}`);
        if (generation !== state.loadGeneration || expectedSource !== (state.source?.fingerprint || state.sourceGeneration)) {
            if (!prefetch) diagnostics.record("frame.discarded", { frame, generation });
            return null;
        }
        if (!sourceImage) {
            if (!prefetch && frame === state.frame) { state.missingFrame = true; empty.textContent = `Не удалось загрузить кадр ${frame + 1}. Проверьте preview /view.`; setStatus(`Ошибка загрузки кадра ${frame + 1}.`, "error"); draw(); }
            return null;
        }
        cachePreview(frame, { sourceImage, maskImage, editBaseImage, edit_base: descriptor.edit_base });
        if (!prefetch && frame === state.frame) state.missingFrame = false;
        if (!prefetch && frame === state.frame) {
            const needsUpdate = state.previewStale || state.previewSignature !== documentSignature();
            const refinementFailure = state.refinementError || (state.metadata?.refinement_error ? refinementMessage(state.metadata.refinement_error) : "");
            empty.textContent = "";
            if (state.mismatchBlocked && state.sourceMismatch) setStatus(mismatchMessage(state.sourceMismatch), "warn");
            else if (sam3Mode() && sam3TrackReady()) setStatus("Трекинг SAM3 готов. Проверьте кадры и запускайте H3.", "ready");
            else if (sam3Mode() && sam3FrameReady()) setStatus("Кадр SAM3 готов. Нажмите «Отследить видео» для полного трека.", "ready");
            else {
                const refreshAction = sam3Mode() ? "Отследить видео" : "Рассчитать маску";
                const emptyHint = sam3Mode() ? "Нарисуйте рамку на опорном кадре, затем нажмите «Рассчитать кадр»." : "Нарисуйте маску на опорном кадре, затем нажмите «Рассчитать маску».";
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
            const descriptor = {
                ...item,
                source: item.source || item,
                mask: allowMask ? (item.mask || item.mask_url) : null,
                edit_base: allowMask ? item.edit_base : null,
            };
            if (!allowMask) for (const key of ["mask_url", "maskDescriptor", "edit_base_descriptor", "editBase", "edit_base"]) delete descriptor[key];
            state.frameItems.set(frame, descriptor);
        }
        const previews = Array.isArray(output.universal_mask_preview) ? output.universal_mask_preview : [];
        for (const item of previews.slice(0, MAX_PREVIEWS)) {
            const frame = Math.max(0, Math.round(number(item?.frame, 0)));
            const descriptor = {
                ...(state.frameItems.get(frame) || {}),
                ...item,
                source: item.data_url || item.source || item,
                mask: allowMask ? (item.mask_url || item.mask) : null,
                edit_base: allowMask ? item.edit_base : null,
            };
            if (!allowMask) for (const key of ["mask_url", "maskDescriptor", "edit_base_descriptor", "editBase", "edit_base"]) delete descriptor[key];
            state.frameItems.set(frame, descriptor);
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
        diagnostics.record("preview.received", { source: sourceInfo(rawMeta), hasMask: Boolean(rawMeta?.has_mask), mismatch: Boolean(rawMeta?.source_mismatch), reason: rawMeta?.source_mismatch_reason,
            refinementError: diagnosticText(rawMeta?.refinement_error), descriptors: output.universal_mask_frames?.length || 0, acceptMismatch: Boolean(options.acceptMismatch) });
        const nextSource = sourceInfo(rawMeta);
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
            state.pendingH3 = null;
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
        if (state.pendingPreview && state.pendingPreview.promptId && output?.prompt_id && String(state.pendingPreview.promptId) !== String(output.prompt_id)) {
            diagnostics.record("preview.stale_prompt", { expected: state.pendingPreview.promptId, received: output.prompt_id });
            return { stale: true, hasPreview: state.previews.size > 0, hasMask: false };
        }
        if (state.pendingPreview && (state.pendingPreview.revision !== state.editRevision || state.pendingPreview.snapshot !== widgetSnapshot(node) || (state.pendingPreview.sourceKey && state.pendingPreview.sourceKey !== currentSourceKey()) || (state.pendingPreview.sam3Signature && state.pendingPreview.sam3Signature !== sam3ReadinessSignature()))) {
            state.pendingPreview = null; state.refinementBusy = false; setPrepareBusy(false); updateActionAvailability();
            setStatus("Получено устаревшее превью — правки изменились после запуска подготовки. Запустите обновление ещё раз.", "warn");
            return { stale: true, hasPreview: state.previews.size > 0, hasMask: false };
        }
        const completedRequestKind = state.pendingRequestKind;
        state.pendingPreview = null;
        state.pendingPromptId = null;
        state.pendingRequestKind = "";
        state.refinementBusy = false;
        clearPreviewCaches();
        state.previewStale = false;
        state.resetBlank = false;
        if (nextSource) {
            state.source = nextSource;
            state.frameCount = nextSource.frames;
            if (!state.localEdits) documentState.source = clone(nextSource);
        }
        state.metadata = rawMeta && typeof rawMeta === "object" ? clone(rawMeta) : {};
        state.metadataAnchors = Array.isArray(rawMeta?.anchors) ? explicitAnchors({ anchors: rawMeta.anchors }) : [];
        state.fps = number(rawMeta?.fps, number(widgetValue(node, "fps", 24), 24));
        state.refinementError = typeof rawMeta?.refinement_error === "string" ? refinementMessage(rawMeta.refinement_error) : "";
        if (state.refinementError) state.previewStale = true;
        applySam3Metadata(rawMeta?.sam3, completedRequestKind);
        ingestDescriptors(output, { allowMask: !state.refinementError });
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
            setStatus(state.sam3.frameReady ? "Кадр SAM3 рассчитан. Проверьте маску и нажмите «Отследить видео»." : state.sam3.error || "Кадр SAM3 не готов: задайте рамку и повторите расчёт.", state.sam3.frameReady ? "ready" : "warn");
            if (state.frameItems.size) loadFrame(state.frame, { generation: state.loadGeneration }); else draw();
        } else if (completedRequestKind === "sam3_track") {
            empty.textContent = state.frameItems.size ? "Загрузка кадра…" : "Рассчитайте трекинг SAM3 заново";
            setStatus(state.sam3.trackReady ? `Трекинг SAM3 готов: ${state.sam3.computedFrames.length || "все"} кадров. Можно запускать H3.` : state.sam3.error || "Трекинг SAM3 не готов: повторите расчёт.", state.sam3.trackReady ? "ready" : "warn");
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
        if (state.pendingH3 && output?.prompt_id && String(output.prompt_id) === String(state.pendingH3.promptId)) finishPending("h3");
        return ingestOutput(output);
    }

    function addEvent(event, label = "Изменение") {
        if (state.mismatchBlocked) { setStatus("Источник не совпадает с сохранённой маской. Примите новый клип или восстановите прежний источник.", "warn"); return false; }
        const events = eventsAt(state.frame, true);
        if (events.length >= MAX_EVENTS || totalPoints() + (event.points?.length || 0) > MAX_TOTAL_POINTS) { setStatus("Лимит правок достигнут: отмените или очистите кадр.", "warn"); return false; }
        if (event.points && event.points.length > MAX_POINTS_PER_STROKE) event.points = event.points.slice(0, MAX_POINTS_PER_STROKE);
        const isSam3Seed = sam3Mode() && event?.type === "rect";
        if (isSam3Seed) {
            event = { ...event, refine: true, refine_method: "sam3" };
            // A new seed changes the SAM3 identity. Manual brush/erase events
            // never rotate this token, so retracking can preserve them.
            state.sam3.seedToken = makeSeedToken();
            documentState.sam3 = { ...(documentState.sam3 || {}), action: "frame", frame: state.frame, direction: sam3Direction(directionSelect.value), seed_token: state.sam3.seedToken };
            invalidateSam3({ frame: true, track: true, reason: "Новая рамка SAM3 требует расчёта кадра." });
        }
        events.push(event); if (frameBucket(state.frame, true)?.inherit === undefined) frameBucket(state.frame, true).inherit = shouldInherit(state.frame);
        state.refinementError = "";
        pushHistory(label); commit({ markDirty: true }); updateReadouts(); updateActionAvailability(); draw();
        if (sam3Mode() && !isSam3Seed) invalidateSam3({ track: true, reason: "Покадровая правка требует повторного трекинга." });
        diagnostics.record("mask.edit", { type: event.type, mode: event.mode, refine: Boolean(event.refine), refineMethod: event.refine ? event.refine_method || "grabcut" : undefined, points: event.points?.length || 0 }); return true;
    }

    function finishPolygon() { if (state.polygon.length < 3) { state.polygon = []; draw(); return; } addEvent({ type: "polygon", mode: "add", points: clone(state.polygon) }, "Полигон добавлен"); state.polygon = []; draw(); }

    function finishPointer(event) {
        if (!state.active || state.playing || (event.pointerId !== undefined && event.pointerId !== state.active.pointerId)) return;
        const active = state.active; state.active = null; canvas.releasePointerCapture?.(active.pointerId);
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

    function cancelGesture() { if (state.panning) canvas.releasePointerCapture?.(state.panning.pointerId); state.panning = null; canvas.style.cursor = state.mode === "brush" || state.mode === "erase" ? "crosshair" : "cell"; state.active = null; state.ghost = null; state.polygon = []; draw(); }

    canvas.addEventListener("pointerdown", (event) => {
        if (state.playing || state.mismatchBlocked || event.button !== 0 || event.altKey || event.shiftKey || state.panning) return;
        canvas.focus?.({ preventScroll: true }); const point = pointFromEvent(event); if (!point) return;
        if (state.mode === "polygon") { state.polygon.push([Number(point.x.toFixed(6)), Number(point.y.toFixed(6))]); draw(); return; }
        state.active = { pointerId: event.pointerId, type: state.mode === "rect" ? "rect" : "stroke", mode: state.mode === "erase" ? "erase" : "add", radius: number(brushInput.value, .035), points: [[Number(point.x.toFixed(6)), Number(point.y.toFixed(6))]], start: point };
        const rectangleSeed = sam3Mode() || state.grabcut;
        state.ghost = state.active.type === "stroke" ? state.active : { type: "rect", mode: "add", x: point.x, y: point.y, w: 0, h: 0, refine: rectangleSeed, ...(rectangleSeed ? { refine_method: sam3Mode() ? "sam3" : state.refineMethod } : {}) };
        canvas.setPointerCapture?.(event.pointerId); event.preventDefault?.(); draw();
    });
    canvas.addEventListener("pointermove", (event) => {
        updateBrushCursor(event); if (state.playing || !state.active) return; const point = pointFromEvent(event); if (!point) return;
        if (state.active.type === "stroke") { const previous = state.active.points.at(-1); if ((!previous || Math.hypot(point.x - previous[0], point.y - previous[1]) > .002) && state.active.points.length < MAX_POINTS_PER_STROKE) { state.active.points.push([Number(point.x.toFixed(6)), Number(point.y.toFixed(6))]); state.ghost = state.active; draw(); } }
        else { const x = Math.min(state.active.start.x, point.x), y = Math.min(state.active.start.y, point.y); const rectangleSeed = sam3Mode() || state.grabcut; state.ghost = { type: "rect", mode: "add", x, y, w: Math.abs(point.x - state.active.start.x), h: Math.abs(point.y - state.active.start.y), refine: rectangleSeed, ...(rectangleSeed ? { refine_method: sam3Mode() ? "sam3" : state.refineMethod } : {}) }; draw(); }
    });
    canvas.addEventListener("pointerup", finishPointer); canvas.addEventListener("pointercancel", cancelGesture); canvas.addEventListener("pointerleave", () => { state.lastPointer = null; if (!state.active) brushCursor.style.display = "none"; }); canvas.addEventListener("dblclick", () => { if (state.mode === "polygon") finishPolygon(); });

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
        canvas.style.cursor = state.mode === "brush" || state.mode === "erase" ? "crosshair" : "cell";
        canvas.releasePointerCapture?.(event.pointerId);
    });
    canvas.addEventListener("auxclick", (event) => { if (event.button === 1) { event.preventDefault?.(); event.stopPropagation?.(); } });

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

    tools.brush.addEventListener("click", () => chooseMode("brush")); tools.erase.addEventListener("click", () => chooseMode("erase")); tools.rect.addEventListener("click", () => chooseMode("rect")); tools.polygon.addEventListener("click", () => chooseMode("polygon"));
    function updateRefineControls() {
        const modeIsSam3 = sam3Mode();
        refineMethod.value = state.refineMethod;
        refine.textContent = "Выделять объект";
        refine.setAttribute("aria-pressed", String(state.grabcut));
        refine.title = modeIsSam3 ? "SAM3: прямоугольник задаёт объект для выделения. После расчёта кисть и ластик исправляют маску текущего кадра." : state.grabcut
            ? state.refineMethod === "sam3" ? "SAM3: обведите объект близко к границам и нажмите «Рассчитать маску»." : "GrabCut: обведите объект с небольшим запасом фона и нажмите «Рассчитать маску»."
            : "Выключено: прямоугольник станет обычной прямоугольной маской.";
        refine.style.background = modeIsSam3 || state.grabcut ? COLORS.selected : COLORS.panel2;
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
    opacity.addEventListener("input", () => { opacityReadout.textContent = `${Math.round(number(opacity.value) * 100)}%`; draw(); });
    for (const [key, control] of Object.entries(viewButtons)) control.addEventListener("click", () => { state.view = key; for (const [name, item] of Object.entries(viewButtons)) item.style.background = name === key ? COLORS.selected : COLORS.panel2; draw(); });
    modeSelect.value = safeMode(node, documentState); updateRefineControls(); modeSelect.addEventListener("change", () => {
        if (state.mismatchBlocked) { modeSelect.value = safeMode(node, documentState); setStatus("Источник не совпадает с сохранённой маской. Настройки заблокированы до восстановления.", "warn"); return; }
        const previousMode = safeMode(node, documentState);
        setWidget(node, "tracking_mode", modeSelect.value); documentState.tracking_mode = modeSelect.value;
        if (modeSelect.value === "sam3") {
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
        const applySetting = () => {
            if (state.mismatchBlocked) { syncMaskSettingsInputs(); setStatus("Источник не совпадает с сохранённой маской. Настройки заблокированы до восстановления.", "warn"); return; }
            const current = snapshotMaskSettings()[name];
            const value = String(input.value).trim() === "" ? current : clamp(Math.round(number(input.value, current)), 0, 256);
            input.value = String(value);
            if (input.disabled || value === current || !setWidget(node, name, value)) return;
            pushHistory(`${name === "grow" ? "Grow" : "Feather"}: ${value} px`);
            commit({ markDirty: true }); updateReadouts();
            setStatus(sam3Mode() ? "Настройки маски изменены. Значение применяется к кэшу трекинга; для нового трека нажмите «Отследить видео»." : "Настройки маски изменены. Нажмите «Рассчитать маску», чтобы увидеть результат.", "dirty");
        };
        input.addEventListener("change", applySetting);
        input.addEventListener("blur", applySetting);
        input.addEventListener("keydown", (event) => { if (event.key === "Enter") { applySetting(); event.preventDefault?.(); event.stopPropagation?.(); } });
    }
    slider.addEventListener("input", () => setFrame(slider.value));
    direct.addEventListener("change", () => setFrame(number(direct.value, 1) - 1));
    direct.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        setFrame(number(direct.value, 1) - 1, { source: "direct" });
        event.preventDefault?.(); event.stopPropagation?.();
    });
    first.addEventListener("click", () => setFrame(0, { source: "nav" })); last.addEventListener("click", () => setFrame(state.frameCount - 1, { source: "nav" })); back.addEventListener("click", () => move(-1)); next.addEventListener("click", () => move(1)); back10.addEventListener("click", () => move(-10)); next10.addEventListener("click", () => move(10)); previousKeyButton.addEventListener("click", () => { stopPlayback(); previousKey(-1); }); nextKeyButton.addEventListener("click", () => { stopPlayback(); previousKey(1); }); play.addEventListener("click", togglePlayback); speed.addEventListener("change", () => { state.playbackSpeed = number(speed.value, 1); });
    fit.addEventListener("click", () => { state.zoom = 1; state.panX = 0; state.panY = 0; draw(); }); zoomOut.addEventListener("click", () => zoomBy(.8)); zoomIn.addEventListener("click", () => zoomBy(1.25));
    undo.addEventListener("click", () => { if (state.historyIndex > 0) restoreHistory(state.historyIndex - 1); else setStatus("Отменять больше нечего.", "warn"); }); redo.addEventListener("click", () => { if (state.historyIndex + 1 < state.history.length) restoreHistory(state.historyIndex + 1); else setStatus("Повторять больше нечего.", "warn"); });
    clear.addEventListener("click", () => { if (state.mismatchBlocked) return; const bucket = frameBucket(state.frame, true); bucket.events = []; bucket.inherit = false; delete bucket.cleared; if (sam3Mode() && state.sam3.frame === state.frame) { state.sam3.seedToken = ""; invalidateSam3({ frame: true, track: true, reason: "Выделение опорного кадра очищено — задайте новую рамку SAM3." }); } else if (sam3Mode()) invalidateSam3({ track: true, reason: "Покадровая правка требует повторного трекинга." }); pushHistory(`Кадр ${state.frame + 1} очищен`); commit({ markDirty: true }); updateReadouts(); draw(); setStatus(`Кадр ${state.frame + 1} очищен.`, "dirty"); });
    copyPrevious.addEventListener("click", () => {
        if (state.mismatchBlocked) return;
        const candidates = Object.keys(documentState.frames || {}).map(Number).filter((frame) => frame < state.frame && frameHasDraft(frame)).sort((a, b) => b - a);
        if (!candidates.length) { setStatus("Предыдущего ключа с правкой нет.", "warn"); return; }
        const sourceFrame = candidates[0];
        documentState.frames[String(state.frame)] = clone(documentState.frames[String(sourceFrame)]);
        if (sam3Mode()) { state.sam3.seedToken = makeSeedToken(); documentState.sam3 = { ...(documentState.sam3 || {}), action: "frame", frame: state.frame, seed_token: state.sam3.seedToken }; invalidateSam3({ frame: true, track: true, reason: "Скопирована новая рамка SAM3." }); }
        pushHistory(`Ключ ${sourceFrame + 1} скопирован в ${state.frame + 1}`);
        commit({ markDirty: true }); updateReadouts(); draw();
    });
    reset.addEventListener("click", () => {
        if (state.mismatchBlocked) { setStatus("Источник не совпадает с сохранённой маской. Сначала примите новый клип или восстановите прежнюю маску.", "warn"); return; }
        replaceDocument(documentState, { version: 1, frames: {}, source: state.source || undefined, tracking_mode: safeMode(node, documentState), ...(sam3Mode() ? { sam3: { action: "frame", frame: state.frame, direction: sam3Direction(directionSelect.value), seed_token: "" } } : {}) });
        state.metadataAnchors = []; state.previewStale = true; state.resetBlank = true; state.pendingPreview = null; state.refinementError = "";
        if (sam3Mode()) { state.sam3.frameReady = false; state.sam3.trackReady = false; state.sam3.seedToken = ""; state.sam3.frameSignature = ""; state.sam3.trackSignature = ""; state.sam3.error = ""; }
        state.localEdits = true; state.dirty = true; pushHistory("Маска сброшена"); commit({ markDirty: true }); updateReadouts(); draw();
        setStatus("Маска сброшена. Превью источника сохранено; обновите маску перед рисованием.", "dirty");
    });
    function markRangeDirty() {
        if (state.mismatchBlocked) return;
        state.localEdits = true; state.dirty = true; state.previewStale = true; state.editRevision += 1; updateDirtyIndicator();
    }
    startInput.addEventListener("change", () => { if (state.mismatchBlocked) { updateReadouts(); return; } const value = clamp(Math.round(number(startInput.value, 1)) - 1, 0, state.frameCount - 1); setWidget(node, "start_frame", value); markRangeDirty(); if (sam3Mode()) invalidateSam3({ track: true, reason: "Диапазон изменился — повторите трекинг видео." }); updateReadouts(); updateActionAvailability(); });
    endInput.addEventListener("change", () => { if (state.mismatchBlocked) { updateReadouts(); return; } const value = clamp(Math.round(number(endInput.value, state.frameCount)) - 1, 0, state.frameCount - 1); setWidget(node, "end_frame", value); markRangeDirty(); if (sam3Mode()) invalidateSam3({ track: true, reason: "Диапазон изменился — повторите трекинг видео." }); updateReadouts(); updateActionAvailability(); });

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
        const pending = kind === "h3" ? state.pendingH3 : state.pendingPreview;
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
        if (kind === "h3") {
            state.pendingH3 = null;
            state.pendingPromptId = state.pendingPreview?.promptId || null;
        } else {
            state.pendingPreview = null;
            state.pendingPromptId = state.pendingH3?.promptId || null;
            state.refinementBusy = false;
            state.sam3.pendingAction = "";
            setPrepareBusy(false);
        }
        state.pendingRequestKind = state.pendingH3 ? "h3" : state.pendingPreview ? (state.pendingPreview.kind || "preview") : "";
        updateActionAvailability();
    }
    function runtimeEvent(detail = {}, eventType = "execution_error") {
        if (state.destroyed) return false;
        const pending = state.pendingPreview || state.pendingH3;
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
        const kind = state.pendingPreview ? "preview" : "h3";
        const sam3Action = pending.sam3Action || "";
        finishPending(kind === "h3" ? "h3" : "preview");
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
        if (state.destroyed || state.pendingPreview || state.pendingH3 || (state.mismatchBlocked && kind !== "manual")) { diagnostics.record("preview.blocked", { kind }); return false; }
        stopPlayback();
        if (!state.mismatchBlocked && !commit({ sam3Action })) return false;
        state.pendingPreview = { revision: state.editRevision, snapshot: widgetSnapshot(node), kind, sam3Action, sam3Signature: sam3Action ? sam3ReadinessSignature() : "", sourceKey: currentSourceKey(), promptId: null };
        state.pendingRequestKind = kind;
        state.refinementBusy = kind === "refinement" || Boolean(sam3Action);
        state.sam3.pendingAction = sam3Action || "";
        setPrepareBusy(true);
        updateActionAvailability();
        if (app?.graphToPrompt && queueApi?.queuePrompt) {
            setStatus(kind === "refinement" ? "Уточняем объект…" : sam3Action === "frame" ? "Рассчитываем выбранный кадр SAM3…" : sam3Action === "track" ? "Отслеживаем видео SAM3…" : "Подготовка кадров и обновление маски…");
            try {
                const prompt = await app.graphToPrompt(app.graph || app.graph?.rootGraph);
                const result = await queueApi.queuePrompt(0, previewPrompt(prompt, node.id), { partialExecutionTargets: [String(node.id)] });
                setPendingPromptId("preview", result);
                diagnostics.record("preview.queued", { kind, promptId: promptIdFromResult(result) });
                if (state.pendingPreview) setStatus(kind === "refinement" ? "Уточнение поставлено в очередь. H3 не запускается." : sam3Action === "frame" ? "Расчёт кадра поставлен в очередь. H3 не запускается." : sam3Action === "track" ? "Трекинг поставлен в очередь. H3 не запускается." : "Предпросмотр поставлен в очередь. Этот запрос рассчитывает только маску.");
            } catch (error) {
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
        if (!sam3Mode() || state.destroyed || state.pendingPreview || state.pendingH3 || state.mismatchBlocked) return false;
        const seedFrame = Number.isFinite(Number(state.sam3.frame)) ? Math.max(0, Math.round(Number(state.sam3.frame))) : state.frame;
        const currentHasSeed = hasSam3Seed(state.frame);
        const hasAnySeed = Object.keys(documentState.frames || {}).some((key) => hasSam3Seed(Number(key)));
        if (action === "frame" && !currentHasSeed && hasAnySeed) {
            setStatus(`Рамка SAM3 находится на кадре ${seedFrame + 1}. Выберите его или задайте новую рамку.`, "warn");
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
        const pending = state.pendingPreview || state.pendingH3;
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
    function requestQueue() {
        if (state.destroyed || actionBlocked() || state.pendingH3 || state.pendingPreview || (sam3Mode() && !sam3TrackReady())) {
            if (sam3Mode() && !sam3TrackReady()) setStatus("H3 доступен только после успешного полного трекинга SAM3.", "warn");
            return false;
        }
        stopPlayback();
        if (!commit()) return false;
        if (app?.queuePrompt) {
            state.pendingH3 = { promptId: null, revision: state.editRevision, snapshot: widgetSnapshot(node) };
            state.pendingRequestKind = "h3";
            updateActionAvailability();
            Promise.resolve()
                .then(() => app.queuePrompt(0))
                .then((result) => {
                    setPendingPromptId("h3", result);
                    if (!state.pendingH3?.promptId) finishPending("h3");
                    setStatus("H3 поставлен в очередь.", "ok");
                })
                .catch((error) => { finishPending("h3"); setStatus(`Очередь H3: ${error?.message || error}`, "error"); });
        } else setStatus("Автономный режим: H3 не поставлен, mask_data сохранён.", "warn");
        return true;
    }
    prepare.addEventListener("click", requestPrepare); calculateFrame.addEventListener("click", () => requestSam3("frame")); trackVideo.addEventListener("click", () => requestSam3("track")); cancelRequest.addEventListener("click", cancelOwnPending); queue.addEventListener("click", requestQueue);
    recoverSource.addEventListener("click", () => {
        diagnostics.record("source.accept");
        if (!state.pendingSource) return;
        const pending = state.pendingSource;
        capturePreviousMask(pending.mismatch?.previousSource || state.source);
        replaceDocument(documentState, { version: 1, frames: {}, source: clone(pending.source), tracking_mode: safeMode(node, documentState), ...(safeMode(node, documentState) === "sam3" ? { sam3: { action: "frame", frame: 0, direction: sam3Direction(directionSelect.value), seed_token: "" } } : {}) });
        setWidget(node, "start_frame", 0);
        setWidget(node, "end_frame", -1);
        state.localEdits = false; state.dirty = false; state.pendingSource = null; state.sourceMismatch = null; state.mismatchBlocked = false;
        state.refinementError = ""; state.pendingPreview = null; state.pendingH3 = null; state.pendingPromptId = null; state.pendingRequestKind = "";
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
        if (backup.mask_data !== undefined) setWidget(node, "mask_data", typeof backup.mask_data === "string" ? backup.mask_data : JSON.stringify(backup.mask_data));
        for (const name of ["tracking_mode", "grow", "feather", "start_frame", "end_frame"]) if (backup[name] !== undefined) setWidget(node, name, backup[name]);
        state.source = sourceInfo(documentState) || sourceInfo({ source: backup.source }) || state.source;
        state.frameCount = state.source?.frames || 1; state.frame = 0; state.zoom = 1; state.panX = 0; state.panY = 0;
        state.localEdits = false; state.dirty = false; state.previewStale = true; state.resetBlank = false;
        state.pendingPreview = null; state.pendingH3 = null; state.pendingPromptId = null; state.pendingRequestKind = ""; state.refinementBusy = false; state.refinementError = "";
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

    function openDialog() {
        if (state.dialogOpen) return;
        cancelGesture(); state.lastPointer = null; brushCursor.style.display = "none";
        state.dialogReturnFocus = document.activeElement || expand;
        state.dialogOpen = true;
        dialogInner.append(workbench);
        dialogBar.append(headerMeta, debugDownload, closeDialogButton);
        dialog.showModal?.(); if (!dialog.showModal) dialog.setAttribute("open", "");
        closeDialogButton.focus?.(); draw();
        diagnostics.record("dialog.open", { layout: layoutSnapshot() });
    }
    function closeDialog() {
        if (!state.dialogOpen) return;
        state.dialogOpen = false; stopPlayback(); cancelGesture(); state.lastPointer = null; brushCursor.style.display = "none";
        header.append(headerMeta, headerActions);
        headerActions.append(debugDownload, expand);
        if (workbenchPlaceholder.parentNode && workbenchPlaceholder.after) workbenchPlaceholder.after(workbench); else root.append(workbench);
        dialog.close?.(); dialog.removeAttribute?.("open");
        (state.dialogReturnFocus || expand)?.focus?.(); state.dialogReturnFocus = null; draw();
        diagnostics.record("dialog.close", { layout: layoutSnapshot() });
    }
    expand.addEventListener("click", openDialog); closeDialogButton.addEventListener("click", closeDialog); dialog.addEventListener("cancel", (event) => { event.preventDefault?.(); closeDialog(); }); dialog.addEventListener("close", () => { if (state.dialogOpen) closeDialog(); });

    function inTextInput(target) { const tag = String(target?.tagName || "").toLowerCase(); return tag === "input" || tag === "textarea" || tag === "select" || target?.isContentEditable; }
    function consumeKey(event) { event.preventDefault?.(); event.stopPropagation?.(); }
    root.addEventListener("keydown", (event) => {
        if (event.isComposing || inTextInput(event.target) && event.target !== canvas) return;
        if (event.key === "Escape") { if (state.dialogOpen) closeDialog(); else cancelGesture(); consumeKey(event); return; }
        if (event.key === "Enter" && state.mode === "polygon" && state.polygon.length) { finishPolygon(); consumeKey(event); return; }
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
        replaceDocument(documentState, readDocument(node)); state.source = sourceInfo(documentState); state.frameCount = state.source?.frames || state.frameCount || 1; state.anchors = explicitAnchors(documentState); state.metadataAnchors = []; state.frame = clamp(state.frame, 0, Math.max(0, state.frameCount - 1)); modeSelect.value = safeMode(node, documentState); ensureSam3Document(); state.sam3 = { ...state.sam3, frameReady: false, trackReady: false, frame: documentState.sam3?.frame ?? state.sam3.frame, seedToken: String(documentState.sam3?.seed_token || state.sam3.seedToken || ""), direction: sam3Direction(documentState.sam3?.direction || directionSelect.value), computedFrames: [], error: "", sourceKey: currentSourceKey(), frameSignature: "", trackSignature: "", pendingAction: documentState.sam3?.action || "" }; directionSelect.value = state.sam3.direction; syncConfiguredToolState({ resetLegacy: force || modeChanged || maskChanged }); state.widgetSnapshot = snapshot; state.modeSnapshot = modeSnapshot; state.maskSettingsSnapshot = settingsSnapshot; state.previewSignature = ""; state.previewStale = true; state.resetBlank = false; state.loadGeneration += 1; state.frameItems.clear(); state.previews.clear(); state.imageCache.clear(); state.imagePending.clear(); state.history = [{ document: snapshotDocument(), settings: snapshotMaskSettings(), label: "Исходное состояние" }]; state.historyIndex = 0; updateRefineControls(); updateReadouts(); updateActionAvailability(); draw(); return true;
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
    const addApiListener = (type, handler) => {
        if (typeof queueApi?.addEventListener !== "function") return;
        queueApi.addEventListener(type, handler);
        state.apiListeners.push([type, handler]);
    };
    addApiListener("execution_error", ({ detail } = {}) => runtimeEvent(detail || {}, "execution_error"));
    addApiListener("execution_interrupted", ({ detail } = {}) => runtimeEvent(detail || {}, "execution_interrupted"));
    const runtimeProgress = (detail = {}) => {
        const pending = state.pendingPreview || state.pendingH3;
        if (!pending) return false;
        const eventPromptId = promptIdFromResult(detail);
        if (pending.promptId && eventPromptId && String(pending.promptId) !== String(eventPromptId)) return false;
        if (pending.promptId && !eventPromptId) return false;
        const value = detail?.value ?? detail?.progress ?? detail?.percentage;
        const max = detail?.max ?? detail?.total;
        const suffix = Number.isFinite(Number(value)) && Number.isFinite(Number(max)) && Number(max) > 0 ? ` ${Math.round(Number(value))}/${Math.round(Number(max))}` : "";
        const label = pending.sam3Action === "frame" ? "Расчёт кадра SAM3" : pending.sam3Action === "track" ? "Трекинг SAM3" : pending.kind === "refinement" ? "Уточнение" : "Подготовка превью";
        setStatus(`${label}…${suffix}`, "busy");
        diagnostics.record("preview.progress", { kind: pending.kind, value: diagnosticText(value), max: diagnosticText(max) });
        return true;
    };
    addApiListener("progress", ({ detail } = {}) => runtimeProgress(detail || {}));
    addApiListener("executing", ({ detail } = {}) => runtimeProgress(detail || {}));
    function destroy() {
        if (state.destroyed) return;
        state.destroyed = true;
        for (const [type, handler] of state.apiListeners.splice(0)) queueApi?.removeEventListener?.(type, handler);
        layoutObserver?.disconnect(); stopPlayback(); state.loadGeneration += 1;
        if (layoutTimer) clearTimeout(layoutTimer);
        if (wheelLogTimer) clearTimeout(wheelLogTimer);
        window.removeEventListener?.("error", recordPageError); window.removeEventListener?.("unhandledrejection", recordPageError);
        window.removeEventListener?.("pagehide", diagnostics.flush);
        state.pendingPreview = null; state.pendingH3 = null; state.pendingPromptId = null; state.refinementBusy = false;
        closeDialog(); window.removeEventListener?.("resize", draw); root.remove?.();
        diagnostics.record("editor.destroyed"); diagnostics.dispose();
    }
    window.addEventListener?.("resize", draw);
    node.addDOMWidget?.("universal_mask_editor", "THSUniversalMaskEditor", root, { serialize: false, getMinHeight: () => 740, getMaxHeight: () => 1100 });
    const editor = { root, canvas, status, applyExecuted, handleRuntimeEvent: runtimeEvent, setFrame, commit, syncFromConfiguredWidget, destroy, loadFrame, ingestOutput, debugReport: diagnostics.report, get state() { return state; }, get document() { return documentState; } };
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
    if (sam3Mode()) { state.grabcut = true; state.refineMethod = "sam3"; chooseMode("rect"); }
    else chooseMode("brush");
    viewButtons.overlay.style.background = COLORS.selected; updateReadouts(); updateActionAvailability(); draw();
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
