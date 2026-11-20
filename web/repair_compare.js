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
    accentText: "var(--primary-foreground, #fff)",
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
    const header = element("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "7px", minHeight: "32px" });
    const views = {
        compare: button("Сравнить"),
        source: button("Исходник"),
        result: button("Результат"),
    };
    const status = element("span", { marginLeft: "auto", color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif" }); status.textContent = "Ожидается результат";
    status.setAttribute("role", "status");
    header.append(...Object.values(views), status);
    const stage = element("div", { position: "relative", width: "100%", minHeight: "220px", height: "clamp(220px, 30vw, 420px)", overflow: "hidden", borderRadius: "6px", border: `1px solid ${COLORS.border}`, background: COLORS.panel });
    stage.className = "ths-compare-stage";
    const picture = element("div", { position: "absolute", left: "50%", top: "50%", transform: "translate(-50%, -50%)", touchAction: "none", userSelect: "none" });
    picture.className = "ths-compare-picture";
    const source = document.createElement("img");
    const result = document.createElement("img");
    for (const image of [source, result]) {
        image.draggable = false;
        Object.assign(image.style, { position: "absolute", inset: "0", width: "100%", height: "100%", objectFit: "contain", pointerEvents: "none" });
    }
    source.alt = "Исходный кадр"; result.alt = "Результат";
    const divider = element("div", { position: "absolute", top: "0", bottom: "0", width: "2px", background: "#fff", pointerEvents: "none", transform: "translateX(-1px)" });
    const handle = element("span", { position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", width: "32px", height: "32px", display: "grid", placeItems: "center", border: "1px solid #fff", borderRadius: "50%", background: COLORS.panel, color: "#fff", boxShadow: "0 1px 4px #0008", fontSize: "17px" });
    handle.textContent = "↔"; handle.setAttribute("aria-hidden", "true"); divider.append(handle);
    const wipe = document.createElement("input");
    wipe.type = "range"; wipe.min = "0"; wipe.max = "100"; wipe.step = "1"; wipe.value = "50";
    wipe.setAttribute("aria-label", "Разделитель сравнения");
    Object.assign(wipe.style, { position: "absolute", inset: "0", width: "100%", height: "100%", margin: "0", opacity: "0", pointerEvents: "none" });
    const empty = element("div", { position: "absolute", inset: "0", display: "grid", placeItems: "center", color: COLORS.muted, font: "500 11px/1.35 Inter, Arial, sans-serif", textAlign: "center", padding: "20px" }); empty.textContent = "После полного запуска здесь появятся исходник и результат";
    picture.append(source, result, divider, wipe);
    stage.append(picture, empty);
    const controls = element("div", { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "6px" });
    const back = button("Предыдущий кадр"); back.textContent = "‹";
    const play = button("Воспроизвести"); play.textContent = "▶";
    const next = button("Следующий кадр"); next.textContent = "›";
    for (const control of [back, play, next]) {
        Object.assign(control.style, { minWidth: "32px", minHeight: "32px", fontSize: "16px" });
    }
    const frame = document.createElement("input"); frame.type = "range"; frame.min = "0"; frame.max = "0"; frame.step = "1"; frame.value = "0"; frame.style.flex = "1 1 140px"; frame.setAttribute?.("aria-label", "Кадр сравнения");
    const frameReadout = element("span", { minWidth: "74px", color: COLORS.muted, font: "500 11px/1.2 Inter, Arial, sans-serif", fontVariantNumeric: "tabular-nums" }); frameReadout.textContent = "Кадр —";
    controls.append(back, play, next, frame, frameReadout);
    root.append(header, stage, controls);

    const state = { descriptors: [], index: 0, fps: 24, frameCount: 0, view: "compare",
        generation: 0, images: new Map(), rendering: null, destroyed: false };
    let dimensions = null;
    let displayed = null;
    let pointerId = null;
    let playback = null;
    node.__thsRepairCompareState = state;

    function setStatus(value) { status.textContent = value; }
    function descriptor() { return state.descriptors[state.index] || null; }
    function syncMaskEditor(index) {
        const editorNode = graphNodes(node, app).find((candidate) => candidate?.__thsUniversalMaskEditor);
        editorNode?.__thsUniversalMaskEditor?.setFrame?.(index, { source: "compare" });
    }
    function fitPicture() {
        if (!dimensions) return;
        const [width, height] = dimensions;
        const scale = Math.min(stage.clientWidth / width, stage.clientHeight / height);
        picture.style.width = `${width * scale}px`;
        picture.style.height = `${height * scale}px`;
    }
    const resize = typeof ResizeObserver === "function" ? new ResizeObserver(fitPicture) : null;
    resize?.observe(stage);
    if (!resize && typeof window !== "undefined") window.addEventListener("resize", fitPicture);

    function updateView() {
        const split = state.view === "compare" && displayed?.source && displayed?.result;
        source.style.display = state.view !== "result" && displayed?.source ? "block" : "none";
        result.style.display = state.view !== "source" && displayed?.result ? "block" : "none";
        divider.style.display = wipe.style.display = split ? "block" : "none";
        picture.style.cursor = split ? "ew-resize" : "default";
        const value = Number(wipe.value);
        result.style.clipPath = split ? `inset(0 0 0 ${value}%)` : "none";
        divider.style.left = `${value}%`;
        wipe.setAttribute("aria-valuetext", `Исходник ${value}%, результат ${100 - value}%`);
        for (const [view, control] of Object.entries(views)) {
            const active = state.view === view;
            control.setAttribute("aria-pressed", String(active));
            control.style.background = active ? COLORS.accent : COLORS.panel2;
            control.style.color = active ? COLORS.accentText : COLORS.text;
            control.disabled = !state.descriptors.length;
        }
    }
    function load(value) {
        const url = imageUrl(value, api);
        if (!url || typeof Image === "undefined") return Promise.resolve(null);
        if (state.images.has(url)) return state.images.get(url);
        const promise = new Promise((resolve) => {
            const image = new Image();
            image.onload = () => resolve({ url, width: image.naturalWidth, height: image.naturalHeight });
            image.onerror = () => resolve(null);
            image.src = url;
        });
        state.images.set(url, promise);
        if (state.images.size > 48) state.images.delete(state.images.keys().next().value);
        return promise;
    }
    async function render() {
        const item = descriptor();
        frame.disabled = play.disabled = state.descriptors.length < 2;
        back.disabled = state.index <= 0;
        next.disabled = state.index >= state.descriptors.length - 1;
        if (!item) {
            displayed = null;
            empty.style.display = "grid";
            frameReadout.textContent = "Кадр —";
            updateView();
            return;
        }
        frameReadout.textContent = `${item.frame + 1} / ${state.frameCount}`;
        const generation = state.generation;
        if (!playback) setStatus("Загрузка кадра…");
        const [before, after] = await Promise.all([load(item.source), load(item.result)]);
        if (generation !== state.generation || state.destroyed || item !== descriptor()) return;
        displayed = { source: before, result: after };
        if (before) source.src = before.url; else source.removeAttribute("src");
        if (after) result.src = after.url; else result.removeAttribute("src");
        const sized = before || after;
        if (sized) {
            dimensions = [sized.width, sized.height];
            fitPicture();
        }
        empty.style.display = sized ? "none" : "grid";
        updateView();
        setStatus(!sized ? "Кадр недоступен" : playback ? "" : `${state.fps.toFixed(2)} кадр/с`);
    }
    function setFrame(index, { sync = true } = {}) {
        if (!state.descriptors.length) return 0;
        state.index = Math.max(0, Math.min(state.descriptors.length - 1, Math.round(Number(index) || 0)));
        frame.value = String(state.index);
        state.rendering = render();
        if (sync) syncMaskEditor(descriptor().frame);
        return descriptor().frame;
    }
    function applyExecuted(output = {}) {
        stopPlayback(false);
        state.generation += 1; state.images.clear(); state.descriptors = normalizeDescriptors(output); state.index = 0;
        displayed = null;
        updateView();
        const metaRaw = rawValueFor(output, UI_KEYS.meta);
        const meta = Array.isArray(metaRaw) ? metaRaw[0] : metaRaw;
        const rawFps = meta?.fps ?? valueFor(output, "fps") ?? valueFor(output, "repair_compare_fps");
        state.fps = Number(rawFps) > 0 && Number.isFinite(Number(rawFps)) ? Number(rawFps) : 24;
        state.frameCount = Math.max(Number(meta?.frame_count) || 0, (state.descriptors.at(-1)?.frame ?? -1) + 1);
        frame.max = String(Math.max(0, state.descriptors.length - 1));
        if (state.descriptors.length) setFrame(0, { sync: false });
        else { render(); setStatus("Ожидается результат"); }
        return { frames: state.descriptors.length, fps: state.fps };
    }
    function updatePlayButton() {
        const label = playback ? "Пауза" : "Воспроизвести";
        play.setAttribute("aria-label", label);
        play.title = label;
        play.textContent = playback ? "Ⅱ" : "▶";
    }
    function stopPlayback(sync = true) {
        if (!playback) return;
        clearTimeout(playback.timer);
        playback = null;
        updatePlayButton();
        if (sync && descriptor()) {
            syncMaskEditor(descriptor().frame);
            setStatus(`${state.fps.toFixed(2)} кадр/с`);
        }
    }
    function togglePlayback() {
        if (playback) { stopPlayback(); return; }
        if (state.descriptors.length < 2) return;
        if (state.index === state.descriptors.length - 1) setFrame(0, { sync: false });
        const playing = playback = { timer: null };
        updatePlayButton();
        setStatus("");
        const interval = 1000 / state.fps;
        async function advance() {
            if (playback !== playing) return;
            if (state.index === state.descriptors.length - 1) { stopPlayback(); return; }
            const started = performance.now();
            setFrame(state.index + 1, { sync: false });
            await state.rendering;
            if (playback === playing) playing.timer = setTimeout(advance, Math.max(0, interval - (performance.now() - started)));
        }
        playing.timer = setTimeout(advance, interval);
    }
    function seek(index) { stopPlayback(false); setFrame(index); }
    for (const [view, control] of Object.entries(views)) {
        control.addEventListener("click", () => { state.view = view; updateView(); });
    }
    back.addEventListener("click", () => seek(state.index - 1));
    next.addEventListener("click", () => seek(state.index + 1));
    play.addEventListener("click", togglePlayback);
    frame.addEventListener("input", () => seek(frame.value));
    wipe.addEventListener("input", updateView);
    const rangeKeys = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -10, PageUp: 10 };
    for (const range of [frame, wipe]) {
        range.addEventListener("keydown", (event) => {
            const step = rangeKeys[event.key];
            if (step === undefined && event.key !== "Home" && event.key !== "End") return;
            event.preventDefault();
            event.stopPropagation();
            const min = Number(range.min), max = Number(range.max);
            const nextValue = event.key === "Home" ? min : event.key === "End" ? max : Number(range.value) + step;
            range.value = String(Math.max(min, Math.min(max, nextValue)));
            if (range === frame) seek(range.value); else updateView();
        });
    }
    wipe.addEventListener("focus", () => { handle.style.outline = `2px solid ${COLORS.accent}`; });
    wipe.addEventListener("blur", () => { handle.style.outline = "none"; });
    function moveDivider(event) {
        const rect = picture.getBoundingClientRect();
        if (!rect.width) return;
        wipe.value = String(Math.round(100 * Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))));
        updateView();
    }
    picture.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || divider.style.display === "none" || pointerId !== null) return;
        pointerId = event.pointerId;
        event.preventDefault();
        event.stopPropagation();
        wipe.focus({ preventScroll: true });
        picture.setPointerCapture(pointerId);
        moveDivider(event);
    });
    picture.addEventListener("pointermove", (event) => { if (event.pointerId === pointerId) moveDivider(event); });
    picture.addEventListener("lostpointercapture", () => { pointerId = null; });
    root.addEventListener("keydown", (event) => {
        if (event.code !== "Space" || event.target !== frame) return;
        event.preventDefault();
        event.stopPropagation();
        togglePlayback();
    });
    const oldRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        state.destroyed = true;
        stopPlayback(false);
        resize?.disconnect();
        if (!resize && typeof window !== "undefined") window.removeEventListener("resize", fitPicture);
        return oldRemoved?.apply(this, args);
    };
    render();
    node.addDOMWidget?.("repair_compare", NODE_CLASS, root, { serialize: false, getMinHeight: () => 330, getMaxHeight: () => 540 });
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
