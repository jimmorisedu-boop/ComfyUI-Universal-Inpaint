/* global document, window */
import { previewPrompt } from "./preview_prompt.js";

// Prompt workbench for THSUniversalAutoPrompt.  The generated prompt is an
// editable draft, not a read-only diagnostic.  Freeze uses a separate
// THSUniversalPromptText node and changes only the prompt-conditioning edge;
// source/mask/model settings remain untouched.

let comfyApp = null;
let comfyApi = null;
try {
    ({ app: comfyApp } = await import("../../scripts/app.js"));
    ({ api: comfyApi } = await import("../../scripts/api.js"));
} catch (_) {
    // Standalone CPU tests intentionally have no Comfy runtime.
}

const NODE_CLASS = "THSUniversalAutoPrompt";
const PROMPT_TEXT_CLASS = "THSUniversalPromptText";
const MUTE_MODE = 2;
const UI_KEYS = Object.freeze({
    text: "auto_prompt_text",
    caption: "auto_prompt_caption",
    status: "auto_prompt_status",
});

const COLORS = Object.freeze({
    page: "var(--bg-color, #202020)",
    panel: "var(--comfy-menu-bg, #171718)",
    panel2: "var(--comfy-menu-secondary-bg, #303030)",
    input: "var(--comfy-input-bg, #222)",
    border: "var(--border-color, #4e4e4e)",
    text: "var(--input-text, #ddd)",
    muted: "var(--descrip-text, #999)",
    accent: "var(--primary-background, var(--content-bg, #4e4e4e))",
    accentText: "var(--primary-foreground, var(--content-fg, #fff))",
    selected: "var(--content-bg, #4e4e4e)",
    warning: "var(--warning-text, #d69b48)",
    ok: "var(--success-text, #69c48a)",
});

function valueFor(output, key) {
    const value = output?.[key];
    if (Array.isArray(value)) return value.length ? String(value[0] ?? "") : "";
    return value == null ? "" : String(value);
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
    item.setAttribute?.("aria-label", label);
    item.title = label;
    Object.assign(item.style, {
        minHeight: "32px", padding: "6px 10px", borderRadius: "6px",
        border: `1px solid ${COLORS.border}`,
        background: kind === "primary" ? COLORS.accent : COLORS.panel2,
        color: kind === "primary" ? COLORS.accentText : COLORS.text,
        cursor: "pointer", font: "600 12px/1.2 Inter, Arial, sans-serif",
        whiteSpace: "nowrap",
    });
    return item;
}

function graphNodes(graph) {
    return graph?._nodes || graph?.nodes || [];
}

function graphFor(node, app = null) {
    return node?.graph || app?.graph || app?.graph?.rootGraph || null;
}

function findNodeById(graph, id) {
    return graphNodes(graph).find((candidate) => Number(candidate?.id) === Number(id)) || null;
}

function findPromptNode(graph) {
    return graphNodes(graph).find((candidate) =>
        candidate?.comfyClass === PROMPT_TEXT_CLASS || candidate?.type === PROMPT_TEXT_CLASS
    ) || null;
}

function nodeWidget(node, name) {
    return (node?.widgets || []).find((widget) => widget?.name === name) || null;
}

function setNodeWidget(node, name, value) {
    const widget = nodeWidget(node, name);
    if (!widget) return false;
    widget.value = value;
    widget.callback?.(value);
    node?.graph?.setDirtyCanvas?.(true, true);
    return true;
}

function linkValue(graph, id) {
    if (id == null) return null;
    if (graph?.links instanceof Map) return graph.links.get(id) || null;
    if (Array.isArray(graph?.links)) return graph.links.find((link) => Number(link?.[0] ?? link?.id) === Number(id)) || null;
    if (graph?.links && typeof graph.links === "object") return graph.links[id] || null;
    return null;
}

function decodeLink(graph, linkOrId) {
    const link = typeof linkOrId === "object" ? linkOrId : linkValue(graph, linkOrId);
    if (!link) return null;
    if (Array.isArray(link)) {
        return { id: link[0], originId: link[1], originSlot: link[2], targetId: link[3], targetSlot: link[4], type: link[5] };
    }
    return {
        id: link.id ?? link.link_id,
        originId: link.origin_id ?? link.originId ?? link.source_id,
        originSlot: link.origin_slot ?? link.originSlot ?? link.source_slot,
        targetId: link.target_id ?? link.targetId,
        targetSlot: link.target_slot ?? link.targetSlot,
        type: link.type,
    };
}

function inputLink(graph, target, slot) {
    const input = target?.inputs?.[slot];
    return decodeLink(graph, input?.link);
}

function findTargetForSource(graph, source, sourceSlot = 0) {
    for (const target of graphNodes(graph)) {
        for (let slot = 0; slot < (target?.inputs || []).length; slot += 1) {
            const link = inputLink(graph, target, slot);
            if (link && Number(link.originId) === Number(source?.id) && Number(link.originSlot) === Number(sourceSlot)) {
                return { target, targetSlot: slot, link };
            }
        }
    }
    return null;
}

function disconnectInput(target, slot) {
    if (!target) return;
    if (typeof target.disconnectInput === "function") {
        target.disconnectInput(slot);
        return;
    }
    const input = target.inputs?.[slot];
    if (input) input.link = null;
}

function connectNodes(source, sourceSlot, target, targetSlot) {
    if (!source || !target) return false;
    disconnectInput(target, targetSlot);
    if (typeof source.connect === "function") {
        source.connect(sourceSlot, target, targetSlot);
        return true;
    }
    // The fallback is deliberately small for standalone harnesses.  Comfy's
    // LiteGraph path above owns link-id allocation and dirty-state updates.
    const input = target.inputs?.[targetSlot];
    if (!input) return false;
    const fallbackId = `ths-prompt-${source.id}-${sourceSlot}-${target.id}-${targetSlot}`;
    input.link = fallbackId;
    source.outputs ||= [];
    source.outputs[sourceSlot] ||= { links: [] };
    source.outputs[sourceSlot].links ||= [];
    if (!source.outputs[sourceSlot].links.includes(fallbackId)) source.outputs[sourceSlot].links.push(fallbackId);
    return true;
}

function setNodeMode(node, mode) {
    if (!node) return;
    node.mode = Number(mode);
    node.graph?.setDirtyCanvas?.(true, true);
    node.graph?.change?.();
}

function markGraphDirty(graph) {
    graph?.setDirtyCanvas?.(true, true);
    graph?.change?.();
    graph?.app?.graph?.setDirtyCanvas?.(true, true);
}

/**
 * Create the persistent freeze/unfreeze controller used by the DOM widget.
 * It intentionally works with both LiteGraph objects and the tiny fake graph
 * used by the browser-less tests.
 */
function createPromptFreezeController(autoNode, options = {}) {
    const app = options.app || comfyApp;
    let graph = options.graph || graphFor(autoNode, app);
    let promptNode = options.promptNode || findPromptNode(graph);
    const controller = {
        autoNode,
        promptNode,
        graph,
        originalMode: Number.isFinite(Number(autoNode?.__thsAutoPromptOriginalMode))
            ? Number(autoNode.__thsAutoPromptOriginalMode) : 0,
        frozen: false,
        target: null,
        lastText: "",
    };
    autoNode.__thsAutoPromptFreeze = controller;

    function targetForPrompt() {
        if (!promptNode) promptNode = findPromptNode(graph);
        const promptTarget = findTargetForSource(graph, promptNode, 0);
        if (promptTarget) return promptTarget;
        const autoTarget = findTargetForSource(graph, autoNode, 0);
        if (autoTarget) return autoTarget;
        // A freshly configured graph can expose serialized inputs before its
        // link map is rebuilt; the conditioner prompt socket is unambiguous.
        const target = graphNodes(graph).find((candidate) =>
            (candidate?.inputs || []).some((input) => input?.name === "prompt")
        );
        if (!target) return null;
        const targetSlot = (target.inputs || []).findIndex((input) => input?.name === "prompt");
        return targetSlot >= 0 ? { target, targetSlot, link: inputLink(graph, target, targetSlot) } : null;
    }

    function refresh() {
        graph = options.graph || graphFor(autoNode, app) || graph;
        if (!promptNode) promptNode = findPromptNode(graph);
        controller.graph = graph;
        controller.promptNode = promptNode;
        controller.target = targetForPrompt();
        const targetInput = controller.target?.target?.inputs?.[controller.target.targetSlot];
        const link = inputLink(graph, controller.target?.target, controller.target?.targetSlot);
        const origin = decodeLink(graph, link || targetInput?.link);
        controller.frozen = Boolean(
            promptNode && Number(autoNode?.mode) === MUTE_MODE &&
            origin && Number(origin.originId) === Number(promptNode.id)
        );
        return controller.frozen;
    }

    controller.refresh = refresh;
    controller.freeze = (text) => {
        refresh();
        if (!promptNode) return { ok: false, reason: "THSUniversalPromptText is missing" };
        const target = targetForPrompt();
        if (!target?.target) return { ok: false, reason: "prompt conditioning target is missing" };
        const exact = String(text ?? "");
        if (!exact.trim()) return { ok: false, reason: "Нельзя зафиксировать пустой промпт" };
        setNodeWidget(promptNode, "text", exact);
        autoNode.__thsAutoPromptOriginalMode = Number.isFinite(Number(autoNode.mode)) && Number(autoNode.mode) !== MUTE_MODE
            ? Number(autoNode.mode) : controller.originalMode;
        controller.originalMode = Number(autoNode.__thsAutoPromptOriginalMode);
        if (!connectNodes(promptNode, 0, target.target, target.targetSlot)) {
            return { ok: false, reason: "could not connect frozen prompt" };
        }
        setNodeMode(promptNode, 0);
        setNodeMode(autoNode, MUTE_MODE);
        controller.lastText = exact;
        controller.target = target;
        controller.frozen = true;
        markGraphDirty(graph);
        return { ok: true, text: exact };
    };
    controller.unfreeze = () => {
        refresh();
        if (!promptNode) return { ok: false, reason: "THSUniversalPromptText is missing" };
        const target = targetForPrompt();
        if (!target?.target) return { ok: false, reason: "prompt conditioning target is missing" };
        if (!connectNodes(autoNode, 0, target.target, target.targetSlot)) {
            return { ok: false, reason: "could not restore auto prompt" };
        }
        setNodeMode(autoNode, controller.originalMode);
        setNodeMode(promptNode, MUTE_MODE);
        controller.target = target;
        controller.frozen = false;
        markGraphDirty(graph);
        return { ok: true };
    };
    controller.getText = () => String(nodeWidget(promptNode, "text")?.value ?? controller.lastText ?? "");
    refresh();
    return controller;
}

function icon(name) {
    const paths = {
        lock: '<rect x="4" y="10" width="16" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
        unlock: '<rect x="4" y="10" width="16" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 7-2"/>',
    };
    return `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || ""}</svg>`;
}

function setButtonIcon(buttonElement, name) {
    if (!buttonElement) return;
    const label = buttonElement.textContent;
    buttonElement.innerHTML = `${icon(name)}<span>${label}</span>`;
    buttonElement.style.display = "inline-flex";
    buttonElement.style.alignItems = "center";
    buttonElement.style.gap = "5px";
}

function createAutoPromptPreview(node, options = {}) {
    if (!node || node.__thsAutoPromptPreview) return node?.__thsAutoPromptPreview;
    const app = options.app || comfyApp;
    const api = options.api || comfyApi;
    const root = element("section", {
        display: "flex", flexDirection: "column", gap: "7px", padding: "9px",
        width: "100%", boxSizing: "border-box", color: COLORS.text,
        background: COLORS.page, border: `1px solid ${COLORS.border}`,
        borderRadius: "8px", fontFamily: "Inter, Arial, sans-serif",
    });
    root.className = "ths-auto-prompt-preview";
    root.setAttribute?.("aria-label", "Промпт ремонта");
    const status = element("div", { minHeight: "27px", padding: "6px 8px", borderRadius: "5px", background: COLORS.panel2, color: COLORS.muted, fontSize: "11px", lineHeight: "1.35" });
    status.setAttribute?.("role", "status");
    status.textContent = "Подготовьте маску, затем сгенерируйте промпт по контексту.";
    const captionLabel = element("div", { color: COLORS.muted, fontSize: "11px", fontWeight: "600" });
    captionLabel.textContent = "Наблюдаемый контекст";
    const caption = element("div", { minHeight: "26px", padding: "6px 8px", borderRadius: "5px", background: COLORS.panel2, color: COLORS.text, fontSize: "11px", lineHeight: "1.4", whiteSpace: "pre-wrap", overflowWrap: "anywhere" });
    caption.textContent = "—";
    const promptLabel = element("div", { color: COLORS.muted, fontSize: "11px", fontWeight: "600" });
    promptLabel.textContent = "Промпт ремонта H3 · можно редактировать";
    const prompt = document.createElement("textarea");
    prompt.rows = 5;
    prompt.value = "";
    prompt.setAttribute?.("aria-label", "Промпт ремонта H3");
    prompt.title = "Изменения сохраняются в текущем черновике; зафиксируйте промпт перед рендером, если нужно исключить автопромпт.";
    Object.assign(prompt.style, { minHeight: "90px", maxHeight: "220px", width: "100%", resize: "vertical", boxSizing: "border-box", overflow: "auto", padding: "7px 8px", borderRadius: "5px", border: `1px solid ${COLORS.border}`, background: COLORS.input, color: COLORS.text, font: "12px/1.45 Inter, Arial, sans-serif", whiteSpace: "pre-wrap" });
    const actions = element("div", { display: "flex", gap: "7px", alignItems: "center", flexWrap: "wrap" });
    const generate = button("Сгенерировать промпт", "primary");
    const freeze = button("Зафиксировать");
    const applyGenerated = button("Применить новый текст");
    applyGenerated.style.display = "none";
    const note = element("span", { color: COLORS.muted, fontSize: "10px", lineHeight: "1.3" });
    note.textContent = "Частичный запуск не загружает H3. Чтобы использовать ручной текст, зафиксируйте его.";
    actions.append(generate, freeze, applyGenerated, note);
    root.append(status, captionLabel, caption, promptLabel, prompt, actions);

    const controller = createPromptFreezeController(node, { app });
    node.properties ||= {};
    const state = {
        controller,
        generated: "",
        applied: "",
        edited: false,
        stale: false,
        generation: 0,
    };
    node.__thsAutoPromptState = state;
    if (controller.refresh() && controller.getText()) {
        prompt.value = controller.getText();
        state.applied = prompt.value;
    } else if (typeof node.properties.__thsAutoPromptDraft === "string") {
        prompt.value = node.properties.__thsAutoPromptDraft;
        state.edited = Boolean(prompt.value);
    }

    function setStatus(value, tone = "normal") {
        status.textContent = value || "—";
        status.style.background = tone === "warn" ? COLORS.panel2 : tone === "ok" ? COLORS.panel2 : COLORS.panel2;
        status.style.color = tone === "warn" ? COLORS.warning : tone === "ok" ? COLORS.ok : COLORS.muted;
    }

    function updateFreezeLabel() {
        const frozen = controller.refresh();
        freeze.textContent = frozen ? "Снять фиксацию" : "Зафиксировать";
        freeze.title = frozen ? "Вернуть связь с автопромптом; сохранённый текст останется в PromptText" : "Скопировать точный текст и переключить conditioning на независимый PromptText";
        setButtonIcon(freeze, frozen ? "unlock" : "lock");
        freeze.setAttribute?.("aria-pressed", String(frozen));
        freeze.style.background = frozen ? COLORS.selected : COLORS.panel2;
        freeze.style.color = frozen ? COLORS.accentText : COLORS.text;
    }

    function applyPromptText(text, { generated = false, force = false } = {}) {
        const incoming = String(text ?? "");
        if (!force && state.edited && !generated) return false;
        prompt.value = incoming;
        if (generated) state.generated = incoming;
        state.applied = incoming;
        if (force) state.edited = false;
        node.properties.__thsAutoPromptDraft = incoming;
        if (controller.refresh() && (force || generated)) {
            setNodeWidget(controller.promptNode, "text", incoming);
            controller.lastText = incoming;
            markGraphDirty(controller.graph);
        }
        state.stale = false;
        applyGenerated.style.display = "none";
        return true;
    }

    function applyExecuted(output = {}) {
        const text = valueFor(output, UI_KEYS.text);
        const observedCaption = valueFor(output, UI_KEYS.caption);
        const diagnostic = valueFor(output, UI_KEYS.status);
        caption.textContent = observedCaption || "—";
        state.generation += 1;
        const frozen = controller.refresh();
        if (frozen) {
            // A muted AutoPrompt may still emit a late event from a prior
            // queue. Never overwrite the independent literal with it.
            if (!prompt.value.trim() && controller.getText()) applyPromptText(controller.getText(), { force: true });
            state.generated = text || state.generated;
            state.stale = Boolean(text && text !== prompt.value);
            if (state.stale) applyGenerated.style.display = "inline-flex";
            setStatus("Промпт зафиксирован; поздний результат автопромпта не применён.", "ok");
        } else if (text && (!state.edited || !prompt.value.trim())) {
            applyPromptText(text, { generated: true });
        } else if (text && text !== state.generated) {
            // A late caption must never erase an explicit edit.  The user can
            // opt in via the visible action button.
            state.generated = text;
            state.stale = true;
            applyGenerated.style.display = "inline-flex";
            setStatus("Новый автопромпт готов; ваши правки сохранены. Нажмите «Применить новый текст», если нужно заменить их.", "warn");
        }
        if (diagnostic && !state.stale && !frozen) setStatus(diagnostic, /fallback|unavailable|failed|error/i.test(diagnostic) ? "warn" : "ok");
        else if (!state.stale && !frozen) setStatus(text ? "Промпт готов — проверьте текст перед запуском H3." : "Промпт пуст; проверьте статус узла.", text ? "ok" : "warn");
        updateFreezeLabel();
        return { prompt: text, caption: observedCaption, status: diagnostic, applied: !state.stale, stale: state.stale };
    }

    async function generatePrompt() {
        if (controller.refresh()) {
            setStatus("Снимите фиксацию, чтобы сгенерировать новый автопромпт.", "warn");
            return;
        }
        if (!app?.graphToPrompt || !api?.queuePrompt) {
            setStatus("API частичного запуска недоступен; используйте обычную очередь Comfy после настройки маски.", "warn");
            return;
        }
        generate.disabled = true;
        try {
            setStatus("Генерируем локальный промпт…");
            const graph = graphFor(node, app);
            const promptGraph = await app.graphToPrompt(graph);
            await api.queuePrompt(0, previewPrompt(promptGraph, node.id), { partialExecutionTargets: [String(node.id)] });
        } catch (error) {
            setStatus(`Ошибка генерации промпта: ${error?.message || error}`, "warn");
        } finally {
            generate.disabled = false;
        }
    }

    prompt.addEventListener("input", () => {
        state.edited = prompt.value !== state.generated;
        node.properties.__thsAutoPromptDraft = prompt.value;
        if (controller.frozen) {
            // While frozen, the independent node is the source of truth.  Save
            // every keystroke so a workflow save/reload cannot lose text.
            setNodeWidget(controller.promptNode, "text", prompt.value);
            controller.lastText = prompt.value;
            markGraphDirty(controller.graph);
        }
        if (state.edited) setStatus("Промпт изменён вручную; автогенерация его не перезапишет.", "warn");
    });
    generate.addEventListener("click", generatePrompt);
    applyGenerated.addEventListener("click", () => {
        applyPromptText(state.generated, { force: true });
        setStatus(controller.refresh() ? "Последний автопромпт записан в независимый PromptText." : "Последний автопромпт применён к редактору.", "ok");
    });
    freeze.addEventListener("click", () => {
        const result = controller.frozen ? controller.unfreeze() : controller.freeze(prompt.value);
        if (!result.ok) { setStatus(result.reason, "warn"); return; }
        state.edited = false;
        node.properties.__thsAutoPromptDraft = prompt.value;
        setStatus(controller.frozen ? "Промпт зафиксирован в независимом узле; автопромпт отключён." : "Автопромпт восстановлен; текст PromptText сохранён.", "ok");
        updateFreezeLabel();
    });
    if (!node.__thsAutoPromptConfigureSync) {
        const previousOnConfigure = node.onConfigure;
        node.onConfigure = function onAutoPromptConfigure(...args) {
            const result = typeof previousOnConfigure === "function" ? previousOnConfigure.apply(this, args) : undefined;
            const sync = () => {
                controller.refresh();
                if (controller.frozen && controller.getText()) {
                    prompt.value = controller.getText();
                    state.applied = prompt.value;
                } else if (typeof node.properties?.__thsAutoPromptDraft === "string" && !state.edited) {
                    prompt.value = node.properties.__thsAutoPromptDraft;
                }
                updateFreezeLabel();
                return result;
            };
            return result && typeof result.then === "function" ? result.then(sync) : sync();
        };
        node.__thsAutoPromptConfigureSync = true;
    }
    updateFreezeLabel();
    node.addDOMWidget?.("auto_prompt_preview", "THSUniversalAutoPrompt", root, {
        serialize: false,
        getMinHeight: () => 300,
        getMaxHeight: () => 520,
    });
    const preview = {
        root, status, caption, prompt, generate, freeze, applyGenerated,
        applyExecuted, generatePrompt, uiKeys: UI_KEYS, controller, state,
        freezePrompt: () => controller.freeze(prompt.value),
        unfreezePrompt: () => controller.unfreeze(),
        get frozen() { return controller.refresh(); },
    };
    node.__thsAutoPromptPreview = preview;
    return preview;
}

if (comfyApp?.registerExtension) {
    comfyApp.registerExtension({
        name: "UniversalInpaint.AutoPromptPreview",
        async nodeCreated(node) {
            if (node.comfyClass === NODE_CLASS) createAutoPromptPreview(node);
        },
    });
    comfyApi?.addEventListener?.("executed", ({ detail }) => {
        const node = comfyApp.graph?.getNodeById?.(Number(detail?.node));
        if (!node || node.comfyClass !== NODE_CLASS) return;
        const preview = node.__thsAutoPromptPreview || createAutoPromptPreview(node);
        preview?.applyExecuted(detail?.output || {});
    });
}

export {
    NODE_CLASS,
    PROMPT_TEXT_CLASS,
    UI_KEYS,
    MUTE_MODE,
    createAutoPromptPreview,
    createPromptFreezeController,
    decodeLink,
    findPromptNode,
    valueFor,
};
