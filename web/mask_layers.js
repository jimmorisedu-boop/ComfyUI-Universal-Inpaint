// Pure state & serialization helpers for non-destructive multi-layer masks.
// Intentionally free of DOM/Comfy globals so Node tests and the browser editor
// share identical layer semantics.

export const MAX_MASK_LAYERS = 16;
export const LAYER_TYPES = ["sam3", "optical_flow", "keyframes", "static"];
export const LAYER_BLENDS = ["add", "subtract", "intersect"];

const TYPE_LABELS = {
    sam3: "SAM3",
    optical_flow: "Optical Flow",
    keyframes: "Ключи",
    static: "Статика",
};

const TYPE_SHORT_LABELS = {
    sam3: "SAM3",
    optical_flow: "Flow",
    keyframes: "Ключи",
    static: "Статика",
};

const BLEND_SYMBOLS = {
    add: "+",
    subtract: "−",
    intersect: "∩",
};

const BLEND_LABELS = {
    add: "Добавить (+)",
    subtract: "Вычесть (−)",
    intersect: "Пересечь (∩)",
};

function cloneValue(value) {
    if (value === null || typeof value !== "object") return value;
    if (Array.isArray(value)) return value.map(cloneValue);
    const copy = {};
    for (const key of Object.keys(value)) {
        copy[key] = cloneValue(value[key]);
    }
    return copy;
}

export function normalizeLayerType(value, fallback = "sam3") {
    const text = String(value || "").trim();
    if (LAYER_TYPES.includes(text)) return text;
    return LAYER_TYPES.includes(fallback) ? fallback : "sam3";
}

export function normalizeBlendMode(value, fallback = "add") {
    const text = String(value || "").trim();
    if (LAYER_BLENDS.includes(text)) return text;
    return LAYER_BLENDS.includes(fallback) ? fallback : "add";
}

export function layerTypeLabel(type) {
    return TYPE_LABELS[normalizeLayerType(type)] || "SAM3";
}

export function layerShortLabel(type) {
    return TYPE_SHORT_LABELS[normalizeLayerType(type)] || "SAM3";
}

export function blendSymbol(blend) {
    return BLEND_SYMBOLS[normalizeBlendMode(blend)] || "+";
}

export function blendLabel(blend) {
    return BLEND_LABELS[normalizeBlendMode(blend)] || "Добавить (+)";
}

export function nextBlendMode(blend) {
    const current = normalizeBlendMode(blend);
    const idx = LAYER_BLENDS.indexOf(current);
    return LAYER_BLENDS[(idx + 1) % LAYER_BLENDS.length];
}

export function isDefaultLayerName(name, type) {
    const text = String(name || "").trim();
    if (!text) return true;
    const base = layerTypeLabel(type);
    return text === base || new RegExp(`^${base.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")} #\\d+$`).test(text);
}

export function nextLayerId(layers = []) {
    const used = new Set(
        (Array.isArray(layers) ? layers : [])
            .map((layer) => String(layer?.id || "").trim())
            .filter(Boolean)
    );
    let n = 1;
    while (used.has(`layer-${n}`)) n += 1;
    return `layer-${n}`;
}

export function nextLayerName(layers = [], type = "sam3") {
    const base = layerTypeLabel(type);
    const usedNames = new Set(
        (Array.isArray(layers) ? layers : [])
            .map((layer) => String(layer?.name || "").trim())
            .filter(Boolean)
    );
    if (!usedNames.has(base)) return base;
    let suffix = 2;
    while (usedNames.has(`${base} #${suffix}`)) suffix += 1;
    return `${base} #${suffix}`;
}

function extractFrameEvents(bucket) {
    if (!bucket || typeof bucket !== "object") return [];
    if (Array.isArray(bucket)) return bucket;
    if (Array.isArray(bucket.events)) return bucket.events;
    let count = 0;
    for (const key of ["strokes", "rects", "rectangles", "polygons", "edits"]) {
        if (Array.isArray(bucket[key])) count += bucket[key].length;
    }
    return count > 0 ? new Array(count) : [];
}

export function layerHasContent(layer) {
    if (!layer || typeof layer !== "object") return false;
    const frames = layer.frames;
    if (frames && typeof frames === "object") {
        for (const bucket of Object.values(frames)) {
            if (extractFrameEvents(bucket).length > 0) return true;
        }
    }
    if (layer.type === "sam3" && layer.sam3 && typeof layer.sam3 === "object") {
        if (Array.isArray(layer.sam3.objects) && layer.sam3.objects.length > 0) return true;
        const prompt = layer.sam3.prompt;
        if (prompt && typeof prompt === "object") {
            if (Array.isArray(prompt.points) && prompt.points.length > 0) return true;
            if (Number(prompt.w) > 0 && Number(prompt.h) > 0) return true;
        }
    }
    return false;
}

function clampFrameStart(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.max(0, Math.round(n)) : Math.max(0, Math.round(Number(fallback) || 0));
}

function clampFrameEnd(value, fallback = -1) {
    const n = Number(value);
    if (!Number.isFinite(n)) return Number.isFinite(Number(fallback)) ? Math.round(Number(fallback)) : -1;
    const r = Math.round(n);
    return r < 0 ? -1 : r;
}

export function normalizeLayer(raw, fallback = {}) {
    const src = raw && typeof raw === "object" ? raw : {};
    const type = normalizeLayerType(src.type ?? fallback.type, "sam3");
    const rawId = typeof src.id === "string" ? src.id.trim().slice(0, 64) : "";
    const rawName = typeof src.name === "string" ? src.name.trim().slice(0, 64) : "";
    const id = rawId || String(fallback.id || "layer-1");
    const name = rawName || String(fallback.name || layerTypeLabel(type));
    const blend = normalizeBlendMode(src.blend ?? fallback.blend, "add");
    const enabled = typeof src.enabled === "boolean" ? src.enabled : (typeof fallback.enabled === "boolean" ? fallback.enabled : true);
    const start_frame = clampFrameStart(src.start_frame, fallback.startFrame ?? fallback.start_frame ?? 0);
    const end_frame = clampFrameEnd(src.end_frame, fallback.endFrame ?? fallback.end_frame ?? -1);
    const frames = src.frames && typeof src.frames === "object" && !Array.isArray(src.frames)
        ? cloneValue(src.frames)
        : {};
    const layer = {
        id,
        name,
        type,
        blend,
        enabled,
        start_frame,
        end_frame,
        frames,
    };
    const rawAnchors = Array.isArray(src.anchors) ? src.anchors : (Array.isArray(fallback.anchors) ? fallback.anchors : []);
    const cleanAnchors = [...new Set(
        rawAnchors
            .map((item) => (item && typeof item === "object" ? item.frame : item))
            .map((val) => Number(val))
            .filter((val) => Number.isFinite(val))
            .map((val) => Math.max(0, Math.round(val)))
    )].sort((a, b) => a - b);
    if (cleanAnchors.length) layer.anchors = cleanAnchors;
    if (type === "sam3" && src.sam3 && typeof src.sam3 === "object") {
        layer.sam3 = cloneValue(src.sam3);
    }
    return layer;
}

export function ensureDocumentLayers(doc, defaults = {}) {
    const target = doc && typeof doc === "object" ? doc : { version: 1, frames: {} };
    const defaultMode = normalizeLayerType(defaults.mode ?? target.tracking_mode, "sam3");
    const defaultStart = clampFrameStart(defaults.startFrame, 0);
    const defaultEnd = clampFrameEnd(defaults.endFrame, -1);

    if (!Array.isArray(target.layers) || target.layers.length === 0) {
        const single = normalizeLayer({
            id: "layer-1",
            name: layerTypeLabel(defaultMode),
            type: defaultMode,
            blend: "add",
            enabled: true,
            start_frame: defaultStart,
            end_frame: defaultEnd,
            frames: target.frames && typeof target.frames === "object" ? target.frames : {},
            anchors: target.anchors || target.keyframes || target.keys,
            ...(defaultMode === "sam3" && target.sam3 && typeof target.sam3 === "object" ? { sam3: target.sam3 } : {}),
        });
        target.layers = [single];
        target.active_layer_id = single.id;
        return target.layers;
    }

    const usedIds = new Set();
    const normalized = [];
    for (const rawLayer of target.layers.slice(0, MAX_MASK_LAYERS)) {
        const provisional = normalizeLayer(rawLayer, {
            id: nextLayerId(normalized),
            type: defaultMode,
            startFrame: defaultStart,
            endFrame: defaultEnd,
        });
        if (usedIds.has(provisional.id)) {
            provisional.id = nextLayerId(normalized);
        }
        usedIds.add(provisional.id);
        normalized.push(provisional);
    }
    target.layers = normalized;
    if (!target.active_layer_id || !usedIds.has(String(target.active_layer_id))) {
        target.active_layer_id = normalized[0].id;
    }
    return target.layers;
}

export function activeLayerOf(doc) {
    if (!doc || !Array.isArray(doc.layers) || doc.layers.length === 0) return null;
    const activeId = String(doc.active_layer_id || "");
    return doc.layers.find((layer) => layer.id === activeId) || doc.layers[0];
}

export function shouldSerializeLayers(layers, globalRange = {}) {
    if (!Array.isArray(layers) || layers.length === 0) return false;
    if (layers.length > 1) return true;
    const only = layers[0];
    if (normalizeBlendMode(only?.blend) !== "add") return true;
    if (only?.enabled === false) return true;
    const gStart = clampFrameStart(globalRange.startFrame, 0);
    const gEnd = clampFrameEnd(globalRange.endFrame, -1);
    const lStart = clampFrameStart(only?.start_frame, 0);
    const lEnd = clampFrameEnd(only?.end_frame, -1);
    return lStart !== gStart || lEnd !== gEnd;
}

export function syncActiveLayerToList(doc, { mode, startFrame, endFrame } = {}) {
    if (!doc || typeof doc !== "object") return null;
    if (!Array.isArray(doc.layers) || doc.layers.length === 0) {
        ensureDocumentLayers(doc, { mode, startFrame, endFrame });
    } else if (!doc.active_layer_id || !doc.layers.some((layer) => layer?.id === String(doc.active_layer_id))) {
        doc.active_layer_id = String(doc.layers[0].id || "layer-1");
    }
    const active = activeLayerOf(doc);
    if (!active) return null;
    if (mode !== undefined) {
        const nextType = normalizeLayerType(mode, active.type);
        if (nextType !== active.type) {
            if (isDefaultLayerName(active.name, active.type)) {
                active.name = layerTypeLabel(nextType);
            }
            active.type = nextType;
        }
    }
    if (startFrame !== undefined) active.start_frame = clampFrameStart(startFrame, active.start_frame);
    if (endFrame !== undefined) active.end_frame = clampFrameEnd(endFrame, active.end_frame);
    active.frames = doc.frames && typeof doc.frames === "object" && !Array.isArray(doc.frames)
        ? cloneValue(doc.frames)
        : {};
    if (Array.isArray(doc.anchors) && doc.anchors.length) {
        active.anchors = cloneValue(doc.anchors);
    } else {
        delete active.anchors;
    }
    if (active.type === "sam3" && doc.sam3 && typeof doc.sam3 === "object") {
        active.sam3 = cloneValue(doc.sam3);
    } else {
        delete active.sam3;
    }
    return active;
}

export function projectLayerToDocument(doc, layer) {
    if (!doc || typeof doc !== "object" || !layer || typeof layer !== "object") return null;
    const normalized = normalizeLayer(layer);
    doc.active_layer_id = normalized.id;
    doc.tracking_mode = normalized.type;
    doc.frames = cloneValue(normalized.frames || {});
    if (Array.isArray(normalized.anchors) && normalized.anchors.length) {
        doc.anchors = cloneValue(normalized.anchors);
    } else {
        delete doc.anchors;
    }
    delete doc.keyframes;
    delete doc.keys;
    if (normalized.type === "sam3") {
        doc.sam3 = normalized.sam3 && typeof normalized.sam3 === "object"
            ? cloneValue(normalized.sam3)
            : { action: "frame", frame: 0, direction: "both", seed_token: "" };
    } else {
        delete doc.sam3;
    }
    return normalized;
}
