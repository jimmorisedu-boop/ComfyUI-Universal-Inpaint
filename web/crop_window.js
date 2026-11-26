/* global document, window */

function fitCropAxis(lo, hi, limit, multiple = 1) {
    const minSize = Math.min(limit, Math.max(1, Math.min(128, Math.floor(limit / 3))));
    const rawSize = Math.min(limit, Math.max(minSize, hi - lo));
    const mult = Math.max(1, Math.min(512, Math.round(Number(multiple) || 1)));
    let size = rawSize;
    if (mult > 1) {
        const maxMult = Math.floor(limit / mult) * mult;
        if (maxMult >= mult) {
            const rounded = Math.ceil(rawSize / mult) * mult;
            size = Math.min(maxMult, Math.max(mult, rounded));
        } else {
            size = limit;
        }
    }
    const start = Math.max(0, Math.min(limit - size, Math.floor((lo + hi - size) / 2)));
    return [start, start + size];
}

// Mirrors Python plan_crop_window() in mask_core.py using half-open canvas pixel coordinates.
export function cropWindowBox(width, height, extent, opts = {}) {
    const mode = opts?.mode || "full";
    const multiple = Math.max(1, Math.min(512, Math.round(Number(opts?.multiple ?? 1) || 1)));
    if (mode === "full" || !extent?.frames?.length || !Array.isArray(extent.box) || extent.box.length !== 4) {
        return [0, 0, width, height];
    }
    const [x, y, r, b] = extent.box.map((value) => Math.round(Number(value) || 0));
    const margin = Math.max(0, Math.min(4096, Math.round(Number(opts?.margin ?? 64) || 0)));
    let edges = [x - margin, y - margin, r + margin, b + margin];
    if (mode === "manual" && Array.isArray(opts?.box) && opts.box.length === 4 && opts?.size?.[0] === width && opts?.size?.[1] === height) {
        const [u, v, s, t] = opts.box.map((value) => Math.round(Number(value) || 0));
        edges = [Math.min(x, u), Math.min(y, v), Math.max(r, s), Math.max(b, t)];
    }
    let [left, right] = fitCropAxis(Math.max(0, edges[0]), Math.min(width, edges[2]), width, multiple);
    let [top, bottom] = fitCropAxis(Math.max(0, edges[1]), Math.min(height, edges[3]), height, multiple);
    if (mode === "auto" && (right - left) * (bottom - top) >= 0.90 * width * height) {
        if (multiple > 1) {
            const maxW = Math.floor(width / multiple) * multiple || width;
            const maxH = Math.floor(height / multiple) * multiple || height;
            left = Math.max(0, Math.min(width - maxW, Math.floor((x + r - maxW) / 2)));
            right = left + maxW;
            top = Math.max(0, Math.min(height - maxH, Math.floor((y + b - maxH) / 2)));
            bottom = top + maxH;
        } else {
            return [0, 0, width, height];
        }
    }
    return [left, top, right, bottom];
}

export function createCropControls({
    state,
    documentState,
    section,
    labelControl,
    iconButton,
    canvasWrap,
    geometry,
    getExtent = null,
    assignTooltipMetadata,
    setTooltip,
    onCropChange,
}) {
    const infoButton = iconButton("Об автокропе", "info", "Автокроп вырезает окно вокруг маски для выходов cropped_images и cropped_mask, ускоряя диффузию и инпейнт.");
    infoButton.className += " ths-mask-info-btn";
    const panel = section("4. Автокроп", "ths-crop-section", infoButton);

    const mode = document.createElement("select");
    mode.setAttribute("aria-label", "Режим автокропа");
    assignTooltipMetadata(mode, "Весь кадр — без обрезки; Авто по маске — окно вокруг объекта с запасом; Рамка вручную — перетаскивание углов.");
    for (const [value, text] of [
        ["full", "Весь кадр"],
        ["auto", "Авто по маске"],
        ["manual", "Рамка вручную"],
    ]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = text;
        mode.append(option);
    }
    const modeRow = labelControl("Режим", mode);
    modeRow.className += " ths-crop-mode";

    const makeNumericField = (labelText, input) => {
        const field = document.createElement("label");
        field.className = "ths-crop-field";
        const title = document.createElement("span");
        title.textContent = labelText;
        const wrap = document.createElement("div");
        wrap.className = "ths-crop-input-wrap";
        const unit = document.createElement("span");
        unit.className = "ths-crop-unit";
        unit.textContent = "px";
        wrap.append(input, unit);
        field.append(title, wrap);
        return field;
    };

    const margin = document.createElement("input");
    margin.type = "number";
    margin.min = "0";
    margin.max = "4096";
    margin.step = "1";
    margin.value = "64";
    margin.setAttribute("aria-label", "Запас автокропа");
    assignTooltipMetadata(margin, "Отступ вокруг границ маски в пикселях для режима «Авто по маске» (0–4096 px).");

    const multipleInput = document.createElement("input");
    multipleInput.type = "number";
    multipleInput.min = "1";
    multipleInput.max = "512";
    multipleInput.step = "1";
    multipleInput.value = "1";
    multipleInput.setAttribute("aria-label", "Кратность сторон кропа");
    assignTooltipMetadata(multipleInput, "Округлять ширину и высоту окна кропа кратно указанному числу пикселей (например, 8, 16, 32, 64; 1 — без округления).");
    const multipleListId = `ths-crop-multiples-${Math.random().toString(36).slice(2, 8)}`;
    const multipleList = typeof document.createElement === "function" ? document.createElement("datalist") : null;
    if (multipleList) {
        multipleList.id = multipleListId;
        for (const preset of [1, 8, 16, 32, 64, 128]) {
            const opt = document.createElement("option");
            opt.value = String(preset);
            multipleList.append(opt);
        }
        multipleInput.setAttribute?.("list", multipleListId);
    }

    const sizeGrid = document.createElement("div");
    sizeGrid.className = "ths-crop-grid";
    sizeGrid.append(makeNumericField("Запас", margin), makeNumericField("Кратность", multipleInput));

    const presetsRow = document.createElement("div");
    presetsRow.className = "ths-crop-presets";
    presetsRow.setAttribute("role", "group");
    presetsRow.setAttribute("aria-label", "Быстрый выбор кратности");
    const presetButtons = [];

    const info = document.createElement("div");
    info.className = "ths-crop-readout";
    info.setAttribute("aria-live", "polite");
    assignTooltipMetadata(info, "Итоговый размер окна cropped_images и cropped_mask.");

    let plan = null;
    let draft = null;
    const knobs = [];
    const currentExtent = () => (typeof getExtent === "function" ? getExtent() : state.metadata?.crop_extent);

    function read() {
        const raw = documentState.crop;
        const parsedMargin = Number(raw?.margin);
        const parsedMultiple = Number(raw?.multiple);
        return {
            mode: ["full", "auto", "manual"].includes(raw?.mode) ? raw.mode : "full",
            margin: Number.isFinite(parsedMargin) ? Math.max(0, Math.min(4096, Math.round(parsedMargin))) : 64,
            multiple: Number.isFinite(parsedMultiple) && parsedMultiple >= 1
                ? Math.max(1, Math.min(512, Math.round(parsedMultiple)))
                : 1,
            ...(Array.isArray(raw?.box) && raw.box.length === 4 ? { box: raw.box.slice() } : {}),
            ...(Array.isArray(raw?.size) && raw.size.length === 2 ? { size: raw.size.slice() } : {}),
        };
    }

    function write(nextOpts) {
        onCropChange?.(nextOpts);
    }

    for (const preset of [1, 8, 16, 32, 64]) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "ths-mask-button ths-crop-preset";
        chip.textContent = preset === 1 ? "×1" : String(preset);
        chip.setAttribute("aria-label", `Кратность ${preset}`);
        chip.setAttribute("aria-pressed", "false");
        assignTooltipMetadata(chip, preset === 1 ? "Без округления сторон (1 px)" : `Округлять ширину и высоту кропа кратно ${preset} px`);
        chip.addEventListener("click", () => {
            const current = read();
            multipleInput.value = String(preset);
            if (preset === current.multiple && current.mode === mode.value) {
                renderPlan(current);
                return;
            }
            write({ ...current, multiple: preset });
        });
        presetButtons.push([preset, chip]);
        presetsRow.append(chip);
    }

    panel.append(modeRow, sizeGrid, presetsRow, ...(multipleList ? [multipleList] : []), info);

    const outline = document.createElement("div");
    outline.className = "ths-crop-outline";
    const tag = document.createElement("span");
    tag.className = "ths-crop-tag";
    tag.textContent = "Автокроп";
    outline.append(tag);
    canvasWrap.append(outline);

    function place(boxGeometry) {
        const showOutline = Boolean(plan && boxGeometry && mode.value !== "full");
        outline.style.display = showOutline ? "block" : "none";
        if (!showOutline) return;
        const [x, y, r, b] = draft?.box || plan.box;
        const w = Math.max(1, r - x);
        const h = Math.max(1, b - y);
        const mult = Math.max(1, Math.round(Number(plan?.multiple || read().multiple) || 1));
        const multTag = mult > 1 ? ` (×${mult})` : "";
        Object.assign(outline.style, {
            left: `${boxGeometry.offsetX + (x / plan.width) * boxGeometry.drawnWidth}px`,
            top: `${boxGeometry.offsetY + (y / plan.height) * boxGeometry.drawnHeight}px`,
            width: `${(w / plan.width) * boxGeometry.drawnWidth}px`,
            height: `${(h / plan.height) * boxGeometry.drawnHeight}px`,
        });
        tag.textContent = mode.value === "manual" ? `Кроп · ${w}×${h} px${multTag}` : `Автокроп · ${w}×${h} px${multTag}`;
        for (const knob of knobs) {
            knob.style.display = mode.value === "manual" && !mode.disabled ? "inline-flex" : "none";
        }
    }

    function renderPlan(opts) {
        const extent = currentExtent();
        const sam3Ready = !state.metadata?.sam3 || Boolean(state.sam3?.trackReady || state.sam3?.frameReady);
        const valid = Boolean(state.source && extent?.frames?.length && !state.previewStale && !state.mismatchBlocked && sam3Ready);
        for (const [preset, chip] of presetButtons) {
            chip.setAttribute("aria-pressed", String(opts.mode !== "full" && opts.multiple === preset));
        }
        plan = null;
        if (opts.mode === "full" && state.source) {
            const { width, height } = state.source;
            plan = { width, height, box: [0, 0, width, height], multiple: 1 };
            info.textContent = `Весь кадр · ${width}×${height} px`;
        } else if (valid) {
            const { width, height } = state.source;
            const box = cropWindowBox(width, height, extent, opts);
            const cw = box[2] - box[0];
            const ch = box[3] - box[1];
            const areaSaved = Math.max(0, Math.round((1 - (cw * ch) / Math.max(1, width * height)) * 100));
            const multSuffix = opts.multiple > 1 ? ` · ×${opts.multiple}` : "";
            const areaSuffix = areaSaved >= 3 ? ` · −${areaSaved}% площади` : "";
            plan = { width, height, box, multiple: opts.multiple };
            info.textContent = `Кроп ${cw}×${ch} px${multSuffix}${areaSuffix}`;
        } else {
            info.textContent = "Рассчитайте маску, чтобы увидеть окно кропа.";
        }
        setTooltip?.(info, info.textContent);
        place(geometry?.());
    }

    function refresh(busy = Boolean(state.pendingPreview || state.pendingQueue)) {
        const opts = read();
        mode.value = opts.mode;
        if (typeof document === "undefined" || document.activeElement !== margin) {
            margin.value = String(opts.margin);
        }
        if (typeof document === "undefined" || document.activeElement !== multipleInput) {
            multipleInput.value = String(opts.multiple);
        }
        mode.disabled = Boolean(busy || state.mismatchBlocked);
        // Keep margin, multiple & preset controls permanently rendered (0 px layout jump).
        margin.disabled = Boolean(busy || state.mismatchBlocked || opts.mode !== "auto");
        const multDisabled = Boolean(busy || state.mismatchBlocked || opts.mode === "full");
        multipleInput.disabled = multDisabled;
        for (const [, chip] of presetButtons) chip.disabled = multDisabled;
        renderPlan(opts);
    }

    mode.addEventListener("change", () => {
        const current = read();
        const nextMode = ["full", "auto", "manual"].includes(mode.value) ? mode.value : "full";
        const nextOpts = { ...current, mode: nextMode };
        const extent = currentExtent();
        if (nextMode === "manual" && state.source && extent?.frames?.length) {
            const seedBox = plan?.box || cropWindowBox(state.source.width, state.source.height, extent, { ...current, mode: "auto" });
            nextOpts.box = seedBox;
            nextOpts.size = [state.source.width, state.source.height];
        }
        write(nextOpts);
    });

    const previewSizing = () => {
        const current = read();
        const rawMargin = String(margin.value).trim();
        const parsedMargin = rawMargin === "" ? NaN : Number(rawMargin);
        const nextMargin = Number.isFinite(parsedMargin)
            ? Math.max(0, Math.min(4096, Math.round(parsedMargin)))
            : current.margin;
        const rawMult = String(multipleInput.value).trim();
        const parsedMult = rawMult === "" ? NaN : Number(rawMult);
        const nextMult = Number.isFinite(parsedMult) && parsedMult >= 1
            ? Math.max(1, Math.min(512, Math.round(parsedMult)))
            : current.multiple;
        renderPlan({ ...current, margin: nextMargin, multiple: nextMult });
    };

    const applyMargin = () => {
        const current = read();
        const value = Math.max(0, Math.min(4096, Math.round(Number(margin.value) || 0)));
        margin.value = String(value);
        if (value === current.margin && current.mode === mode.value) return;
        write({ ...current, margin: value });
    };
    margin.addEventListener("input", previewSizing);
    margin.addEventListener("change", applyMargin);
    margin.addEventListener("blur", applyMargin);
    margin.addEventListener("keydown", (event) => {
        if (event.isComposing || event.keyCode === 229) return;
        if (event.key === "Enter") {
            applyMargin();
            event.preventDefault?.();
            event.stopPropagation?.();
        }
    });

    const applyMultiple = () => {
        const current = read();
        const raw = String(multipleInput.value).trim();
        const parsed = raw === "" ? 1 : Number(raw);
        const value = Number.isFinite(parsed) && parsed >= 1
            ? Math.max(1, Math.min(512, Math.round(parsed)))
            : 1;
        multipleInput.value = String(value);
        if (value === current.multiple && current.mode === mode.value) {
            renderPlan(current);
            return;
        }
        write({ ...current, multiple: value });
    };
    multipleInput.addEventListener("input", previewSizing);
    multipleInput.addEventListener("change", applyMultiple);
    multipleInput.addEventListener("blur", applyMultiple);
    multipleInput.addEventListener("keydown", (event) => {
        if (event.isComposing || event.keyCode === 229) return;
        if (event.key === "Enter") {
            applyMultiple();
            event.preventDefault?.();
            event.stopPropagation?.();
        }
    });

    function resize(index, original, dx, dy) {
        const current = read();
        const box = original.slice();
        box[index] += dx;
        box[index + 1] += dy;
        return cropWindowBox(plan.width, plan.height, currentExtent(), {
            mode: "manual",
            margin: current.margin,
            multiple: current.multiple,
            box,
            size: [plan.width, plan.height],
        });
    }

    for (const index of [0, 2]) {
        const knob = document.createElement("button");
        knob.type = "button";
        knob.className = index ? "ths-crop-se" : "ths-crop-nw";
        const tip = index
            ? "Правый нижний угол автокропа · тяните или используйте стрелки"
            : "Левый верхний угол автокропа · тяните или используйте стрелки";
        knob.setAttribute("aria-label", index ? "Правый нижний угол автокропа" : "Левый верхний угол автокропа");
        assignTooltipMetadata(knob, tip);
        knobs.push(knob);
        outline.append(knob);

        knob.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 || !plan || draft) return;
            event.preventDefault?.();
            event.stopPropagation?.();
            knob.setPointerCapture?.(event.pointerId);
            draft = { id: event.pointerId, x: event.clientX, y: event.clientY, orig: plan.box, box: plan.box };
        });
        knob.addEventListener("pointermove", (event) => {
            if (!draft || draft.id !== event.pointerId || !plan) return;
            const g = geometry?.();
            if (!g) return;
            const dx = Math.round(((event.clientX - draft.x) * plan.width) / g.drawnWidth / g.clientScaleX);
            const dy = Math.round(((event.clientY - draft.y) * plan.height) / g.drawnHeight / g.clientScaleY);
            draft.box = resize(index, draft.orig, dx, dy);
            const [bx, by, br, bb] = draft.box;
            const mult = read().multiple;
            info.textContent = `Кроп ${br - bx}×${bb - by} px${mult > 1 ? ` · ×${mult}` : ""}`;
            place(g);
        });
        knob.addEventListener("pointerup", (event) => {
            if (!plan || draft?.id !== event.pointerId) return;
            const { box } = draft;
            draft = null;
            write({ ...read(), mode: "manual", box, size: [plan.width, plan.height] });
        });
        knob.addEventListener("lostpointercapture", () => {
            draft = null;
            place(geometry?.());
        });
        knob.addEventListener("keydown", (event) => {
            const vector = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
            if (!vector || !plan) return;
            event.stopPropagation?.();
            event.preventDefault?.();
            const mult = read().multiple;
            const step = event.shiftKey ? Math.max(16, mult * 2) : Math.max(1, mult);
            write({
                ...read(),
                mode: "manual",
                box: resize(index, plan.box, vector[0] * step, vector[1] * step),
                size: [plan.width, plan.height],
            });
        });
    }

    return {
        panel,
        outline,
        refresh,
        place,
        read,
        get plan() { return plan; },
    };
}
