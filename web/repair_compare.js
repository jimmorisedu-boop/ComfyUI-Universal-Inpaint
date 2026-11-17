/* global document, window, Image */

// Lazy source/result compare for THSUniversalRepairCompare.  The backend emits
// bounded descriptors through Comfy's UI event; this widget loads only the
// selected frame and keeps its frame index aligned with the mask editor.

let comfyApp = null;
let comfyApi = null;
try {
    ({ app: comfyApp } = await import("../../scripts/app.js"));
    ({ api: comfyApi } = await import("../../scripts/api.js"));
} catch (_) {
    // Standalone harnesses intentionally have no Comfy runtime.
}

const NODE_CLASS = "THSUniversalRepairCompare";
const UI_KEYS = Object.freeze({
    compare: "universal_repair_compare_frames",
    frames: "universal_repair_compare_frames",
    meta: "universal_repair_compare_meta",
    status: "universal_repair_compare_status",
});
const COLORS = Object.freeze({
    page: "var(--bg-color, #202020)",
    panel: "var(--comfy-menu-bg, #171718)",
    panel2: "var(--comfy-menu-secondary-bg, #303030)",
    border: "var(--border-color, #4e4e4e)",
    text: "var(--input-text, #ddd)",
    muted: "var(--descrip-text, #999)",
    accent: "var(--primary-background, var(--content-bg, #4e4e4e))",
});

function valueFor(output, key) {
    const value = output?.[key];
    if (Array.isArray(value)) return value.length ? value[0] : null;
    return value == null ? null : value;
}

function rawValueFor(output, key) {
    return output?.[key];
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

function asArray(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") {
        if (Array.isArray(value.frames)) return value.frames;
        if (Array.isArray(value.descriptors)) return value.descriptors;
        if (Array.isArray(value.items)) return value.items;
        return [value];
    }
    return [];
}

function normalizeDescriptors(output = {}) {
    const raw = rawValueFor(output, UI_KEYS.frames)
        ?? rawValueFor(output, UI_KEYS.compare)
        ?? rawValueFor(output, "ths_repair_compare")
        ?? rawValueFor(output, "compare");
    const payload = raw && !Array.isArray(raw) && typeof raw === "object" && !raw.source && !raw.result
        ? (raw.frames || raw.descriptors || raw.items || raw)
        : raw;
    const list = asArray(payload);
    const descriptors = list.map((item, index) => {
        if (!item || typeof item !== "object") return null;
        const frame = Number.isFinite(Number(item.frame ?? item.index)) ? Math.max(0, Math.round(Number(item.frame ?? item.index))) : index;
        const source = item.source ?? item.original ?? item.source_image ?? item.before ?? item.left;
        const result = item.result ?? item.generated ?? item.result_image ?? item.after ?? item.right;
        return { ...item, frame, source, result };
    }).filter((item) => item && (item.source || item.result));
    descriptors.sort((a, b) => a.frame - b.frame);
    const seen = new Set();
    return descriptors.filter((item) => {
        if (seen.has(item.frame)) return false;
        seen.add(item.frame);
        return true;
    });
}

function element(tag, styles = {}) {
    const item = document.createElement(tag);
    Object.assign(item.style, styles);
    return item;
}

function button(label, kind = "neutral") {
    const item = document.createElement("button");
    item.type = "button";
    item.textContent = label;
    item.title = label;
    item.setAttribute?.("aria-label", label);
    Object.assign(item.style, {
        minHeight: "30px", padding: "5px 8px", borderRadius: "5px",
        border: `1px solid ${COLORS.border}`,
        background: kind === "primary" ? COLORS.accent : COLORS.panel2,
        color: COLORS.text, cursor: "pointer", font: "600 11px/1.2 Inter, Arial, sans-serif",
        whiteSpace: "nowrap",
    });
    return item;
}

function graphNodes(node, app) {
    return node?.graph?._nodes || node?.graph?.nodes || app?.graph?._nodes || app?.graph?.nodes || [];
}

function createRepairCompare(node, options = {}) {
    if (!node || node.__thsRepairCompare) return node?.__thsRepairCompare;
    const app = options.app || comfyApp;
    const api = options.api || comfyApi;
    const root = element("section", {
        display: "flex", flexDirection: "column", gap: "7px", padding: "9px",
        width: "100%", boxSizing: "border-box", color: COLORS.text,
        background: COLORS.page, border: `1px solid ${COLORS.border}`,
        borderRadius: "8px", fontFamily: "Inter, Arial, sans-serif",
    });
    root.className = "ths-repair-compare";
    root.setAttribute?.("aria-label", "Сравнение исходника и результата");
    const header = element("div", { display: "flex", alignItems: "center", gap: "8px", minHeight: "26px" });
    const title = element("strong", { font: "600 13px/1.2 Inter, Arial, sans-serif" }); title.textContent = "Сравнение · исходник / результат";
    const status = element("span", { marginLeft: "auto", color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif" }); status.textContent = "Ожидается результат";
    header.append(title, status);
    const stage = element("div", { position: "relative", width: "100%", minHeight: "220px", aspectRatio: "16 / 9", overflow: "hidden", borderRadius: "5px", border: `1px solid ${COLORS.border}`, background: COLORS.panel });
    const source = document.createElement("img");
    const result = document.createElement("img");
    for (const image of [source, result]) Object.assign(image.style, { position: "absolute", inset: "0", width: "100%", height: "100%", objectFit: "contain", display: "block" });
    source.alt = "Исходный кадр"; result.alt = "Результат ремонта";
    result.style.clipPath = "inset(0 0 0 50%)";
    const divider = element("div", { position: "absolute", top: "0", bottom: "0", left: "50%", width: "2px", background: "rgba(255,255,255,.9)", pointerEvents: "none", transform: "translateX(-1px)" });
    const empty = element("div", { position: "absolute", inset: "0", display: "grid", placeItems: "center", color: COLORS.muted, font: "500 11px/1.35 Inter, Arial, sans-serif", textAlign: "center", padding: "20px" }); empty.textContent = "После полного запуска здесь появятся исходник и результат";
    stage.append(source, result, divider, empty);
    const controls = element("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "5px" });
    const first = button("Первый"); const back = button("‹"); const next = button("›"); const last = button("Последний");
    const frame = document.createElement("input"); frame.type = "range"; frame.min = "0"; frame.max = "0"; frame.step = "1"; frame.value = "0"; frame.style.flex = "1 1 140px"; frame.setAttribute?.("aria-label", "Кадр сравнения");
    const frameReadout = element("span", { minWidth: "74px", color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif", fontVariantNumeric: "tabular-nums" }); frameReadout.textContent = "Кадр —";
    const wipe = document.createElement("input"); wipe.type = "range"; wipe.min = "0"; wipe.max = "100"; wipe.value = "50"; wipe.step = "1"; wipe.style.width = "92px"; wipe.setAttribute?.("aria-label", "Разделитель сравнения"); wipe.title = "Положение wipe-разделителя";
    const wipeReadout = element("span", { minWidth: "38px", color: COLORS.muted, fontSize: "11px", fontVariantNumeric: "tabular-nums" }); wipeReadout.textContent = "50%";
    controls.append(first, back, next, last, frame, frameReadout, wipe, wipeReadout);
    root.append(header, stage, controls);

    const state = { descriptors: [], index: 0, fps: 24, generation: 0, pending: new Map(), destroyed: false };
    node.__thsRepairCompareState = state;

    function setStatus(value) { status.textContent = value || "—"; }
    function descriptor() { return state.descriptors[state.index] || null; }
    function syncMaskEditor(index) {
        const editorNode = graphNodes(node, app).find((candidate) => candidate?.__thsUniversalMaskEditor);
        editorNode?.__thsUniversalMaskEditor?.setFrame?.(index, { source: "compare" });
    }
    function load(value, slot, generation) {
        const url = imageUrl(value, api);
        if (!url || typeof Image === "undefined") return Promise.resolve(null);
        const cacheKey = `${generation}:${slot}:${url}`;
        if (state.pending.has(cacheKey)) return state.pending.get(cacheKey);
        const promise = new Promise((resolve) => {
            const image = new Image(); image.onload = () => resolve(url); image.onerror = () => resolve(null); image.src = url;
        });
        state.pending.set(cacheKey, promise);
        return promise;
    }
    async function render() {
        const item = descriptor();
        if (!item) { empty.style.display = "grid"; frameReadout.textContent = "Кадр —"; return; }
        empty.style.display = "none"; frameReadout.textContent = `Кадр ${item.frame + 1}/${state.descriptors.length}`;
        const generation = state.generation; setStatus("Загрузка выбранного кадра…");
        const [sourceUrl, resultUrl] = await Promise.all([load(item.source, "source", generation), load(item.result, "result", generation)]);
        if (generation !== state.generation || state.destroyed) return;
        if (sourceUrl) source.src = sourceUrl;
        if (resultUrl) result.src = resultUrl;
        setStatus(sourceUrl || resultUrl ? `Кадр ${item.frame + 1} · ${state.fps.toFixed(2)} fps` : "Не удалось загрузить дескрипторы кадра");
    }
    function setFrame(index, { sync = true } = {}) {
        if (!state.descriptors.length) return 0;
        state.index = Math.max(0, Math.min(state.descriptors.length - 1, Math.round(Number(index) || 0)));
        frame.value = String(state.index); render(); if (sync) syncMaskEditor(descriptor().frame); return descriptor().frame;
    }
    function applyExecuted(output = {}) {
        state.generation += 1; state.descriptors = normalizeDescriptors(output); state.index = 0;
        const metaRaw = rawValueFor(output, UI_KEYS.meta);
        const meta = Array.isArray(metaRaw) ? metaRaw[0] : metaRaw;
        const rawFps = meta?.fps ?? valueFor(output, "fps") ?? valueFor(output, "repair_compare_fps");
        if (Number.isFinite(Number(rawFps))) state.fps = Number(rawFps);
        frame.max = String(Math.max(0, state.descriptors.length - 1));
        setFrame(0, { sync: false });
        return { frames: state.descriptors.length, fps: state.fps };
    }
    first.addEventListener("click", () => setFrame(0)); back.addEventListener("click", () => setFrame(state.index - 1)); next.addEventListener("click", () => setFrame(state.index + 1)); last.addEventListener("click", () => setFrame(state.descriptors.length - 1));
    frame.addEventListener("input", () => setFrame(frame.value));
    wipe.addEventListener("input", () => { const value = Math.max(0, Math.min(100, Number(wipe.value) || 50)); result.style.clipPath = `inset(0 0 0 ${value}%)`; divider.style.left = `${value}%`; wipeReadout.textContent = `${value}%`; });
    node.addDOMWidget?.("repair_compare", NODE_CLASS, root, { serialize: false, getMinHeight: () => 280, getMaxHeight: () => 540 });
    const compare = { root, stage, source, result, status, frame, wipe, applyExecuted, setFrame, normalizeDescriptors, state };
    node.__thsRepairCompare = compare;
    return compare;
}

if (comfyApp?.registerExtension) {
    comfyApp.registerExtension({
        name: "UniversalInpaint.RepairCompare",
        async nodeCreated(node) { if (node.comfyClass === NODE_CLASS) createRepairCompare(node); },
    });
    comfyApi?.addEventListener?.("executed", ({ detail }) => {
        const node = comfyApp.graph?.getNodeById?.(Number(detail?.node));
        if (!node || node.comfyClass !== NODE_CLASS) return;
        const compare = node.__thsRepairCompare || createRepairCompare(node);
        compare?.applyExecuted(detail?.output || {});
    });
}

export { NODE_CLASS, UI_KEYS, createRepairCompare, imageUrl, normalizeDescriptors };
