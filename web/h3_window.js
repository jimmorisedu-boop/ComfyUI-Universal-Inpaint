import { previewPrompt } from "./preview_prompt.js";

function fitAxis(lo, hi, limit) {
    const size = Math.min(limit, Math.max(256, hi - lo));
    const start = Math.max(0, Math.min(limit - size, Math.floor((lo + hi - size) / 2)));
    return [start, start + size];
}

// Same half-open source-pixel coordinates as plan_window(). No preview pixels are sampled.
export function windowBox(width, height, extent, opts) {
    const [x, y, r, b] = extent.box;
    const margin = opts.margin ?? 64;
    let edges = [x - margin, y - margin, r + margin, b + margin];
    if (opts.mode === "manual" && opts.box?.length === 4 && opts.size?.[0] === width && opts.size?.[1] === height) {
        const [l,t,s,v] = opts.box;
        edges = [Math.min(x,l), Math.min(y,t), Math.max(r,s), Math.max(b,v)];
    }
    const [left,right] = fitAxis(Math.max(0,edges[0]), Math.min(width,edges[2]), width);
    const [top,bottom] = fitAxis(Math.max(0,edges[1]), Math.min(height,edges[3]), height);
    if (opts.mode === "full" || ((opts.mode ?? "auto") === "auto" && (right-left)*(bottom-top) >= .85*width*height)) return [0,0,width,height];
    return [left,top,right,bottom];
}

export function trialSpan(frames, frame) {
    const end = frames.at(-1) + 1;
    const length = Math.min(12, end - frames[0]);
    const start = Math.max(frames[0], Math.min(end - length, frame - Math.floor(length / 2)));
    return [start, start + length];
}

export function trialPrompt(prompt, winId, frame, span, total) {
    const entry = Object.entries(prompt.output).find(([, n]) => n.class_type === "THSUniversalRepairCompare" && String(n.inputs.work_window?.[0]) === String(winId));
    if (!entry) throw new Error("Подключите сравнение V70 к области H3.");
    const [target] = entry;
    const scoped = previewPrompt(prompt, target);
    const nodes = scoped.output;
    const guide = Object.values(nodes).find(n => n.class_type === "THSUniversalCorrectedFrameGuide" && String(n.inputs.work_window?.[0]) === String(winId));
    if (guide) {
        const mode = guide.inputs.mode;
        const value = Array.isArray(mode) ? nodes[mode[0]]?.inputs.mode : mode;
        const anchor = Number(guide.inputs.frame_number) - 1;
        if (value === "corrected_frame" && (anchor < Math.max(0,span[0]-4) || anchor >= Math.min(total,span[1]+4))) {
            throw new Error(`Выберите пробу рядом с исправленным кадром ${anchor+1}.`);
        }
    }
    nodes[winId] = {...nodes[winId], inputs: {...nodes[winId].inputs, probe_frame: frame}};
    return {target, scoped};
}

export function createWindowControls({node, state, section, labelControl, canvasWrap, geometry, button, redraw}) {
    const panel = section("Область H3");
    const mode = document.createElement("select");
    mode.setAttribute("aria-label", "Область H3");
    for (const [value,text] of [["auto","Авто по маске"],["manual","Рамка вручную"],["full","Весь кадр"]]) {
        const opt = document.createElement("option"); opt.value = value; opt.textContent = text; mode.append(opt);
    }
    const margin = document.createElement("input"); margin.type = "number"; margin.min = "0"; margin.max = "4096"; margin.step = "1";
    margin.setAttribute("aria-label","Запас вокруг маски в пикселях");
    const marginRow = labelControl("Запас, px", margin);
    const info = document.createElement("div"); info.className = "ths-mask-section-copy"; info.setAttribute("aria-live","polite");
    panel.append(mode, marginRow, info);
    const trial = button("Проба на фрагменте");
    const outline = document.createElement("div"); outline.className = "ths-h3-outline";
    const tag = document.createElement("span"); tag.textContent = "Что увидит H3"; outline.append(tag);
    canvasWrap.append(outline);
    let plan = null, draft = null;
    const knobs = [];

    function target() {
        return node.getOutputNodes?.(0)?.find(n => n?.comfyClass === "THSUniversalH3Window" || n?.type === "THSUniversalH3Window");
    }
    function widget() { return target()?.widgets?.find(w => w.name === "settings"); }
    function read() { try { return JSON.parse(widget()?.value || "{}"); } catch { return {}; } }
    function write(opts) {
        const w = widget();
        if (!w) return;
        w.value = JSON.stringify(opts); w.callback?.(w.value); node.graph?.setDirtyCanvas?.(true,true);
        node.graph?.change?.();
        refresh(); redraw();
    }
    function refresh(busy = !!(state.pendingPreview || state.pendingH3)) {
        const w = widget();
        panel.style.display = w ? "" : "none"; trial.style.display = w ? "" : "none";
        const extent = state.metadata?.h3_extent;
        const valid = w && extent?.frames?.length && !state.previewStale && !state.mismatchBlocked && (!state.metadata.sam3 || state.metadata.sam3.track_ready);
        const opts = read();
        mode.value = opts.mode || "auto";
        if (document.activeElement !== margin) margin.value = String(opts.margin ?? 64);
        marginRow.style.display = mode.value === "auto" ? "" : "none";
        mode.disabled = margin.disabled = busy;
        plan = null;
        if (valid && state.source) {
            const {width,height} = state.source;
            const box = windowBox(width,height,extent,opts);
            const frames = extent.frames, span = trialSpan(frames,state.frame);
            plan = {width,height,box,span};
            info.textContent = `${box[2]-box[0]}×${box[3]-box[1]} px · H3 получит кадры ${Math.max(0,frames[0]-4)+1}–${Math.min(state.frameCount,frames.at(-1)+5)}. Проба ${span[0]+1}–${span[1]}.`;
            trial.disabled = busy || !frames.some(i => i >= span[0] && i < span[1]);
            trial.title = `Рассчитать H3 для кадров ${span[0]+1}–${span[1]} и открыть результат в сравнении`;
        } else {
            info.textContent = "Рассчитайте маску, чтобы увидеть область и диапазон H3.";
            trial.disabled = true;
        }
        place(geometry());
    }
    function place(g) {
        outline.style.display = plan && g ? "block" : "none";
        if (!plan || !g) return;
        const [x,y,r,b] = draft?.box || plan.box;
        Object.assign(outline.style,{left:`${g.offsetX+x/plan.width*g.drawnWidth}px`, top:`${g.offsetY+y/plan.height*g.drawnHeight}px`,
            width:`${(r-x)/plan.width*g.drawnWidth}px`,height:`${(b-y)/plan.height*g.drawnHeight}px`});
        for (const k of knobs) k.style.display = mode.value === "manual" && !mode.disabled ? "" : "none";
    }
    mode.addEventListener("change", () => {
        const opts = {...read(), mode:mode.value};
        if (opts.mode === "manual" && plan) Object.assign(opts,{box:plan.box,size:[plan.width,plan.height]});
        write(opts);
    });
    margin.addEventListener("change", () => {
        const value = Math.max(0,Math.min(4096,Math.round(Number(margin.value) || 0)));
        margin.value=String(value); write({...read(),margin:value});
    });
    function resize(index, original, dx, dy) {
        const box = original.slice();
        box[index] += dx; box[index+1] += dy;
        return windowBox(plan.width,plan.height,state.metadata.h3_extent,{mode:"manual",box,size:[plan.width,plan.height]});
    }
    for (const index of [0,2]) {
        const knob = document.createElement("button"); knobs.push(knob); knob.type = "button";
        knob.className = index ? "ths-h3-se" : "ths-h3-nw";
        knob.title = (index ? "Правый нижний угол" : "Левый верхний угол") + " · тяните или используйте стрелки";
        knob.setAttribute("aria-label",knob.title);
        outline.append(knob);
        knob.addEventListener("pointerdown", e => {
            if (e.button !== 0 || !plan || draft) return;
            e.preventDefault(); e.stopPropagation(); knob.setPointerCapture(e.pointerId);
            draft = {id:e.pointerId, x:e.clientX,y:e.clientY,orig:plan.box,box:plan.box};
        });
        knob.addEventListener("pointermove", e => {
            if (!draft || draft.id !== e.pointerId || !plan) return;
            const g = geometry(); if (!g) return;
            draft.box = resize(index,draft.orig,Math.round((e.clientX-draft.x)*plan.width/g.drawnWidth/g.clientScaleX),
                Math.round((e.clientY-draft.y)*plan.height/g.drawnHeight/g.clientScaleY));
            place(g);
        });
        knob.addEventListener("pointerup", e => {
            if (!plan || draft?.id !== e.pointerId) return;
            const {box} = draft; draft = null;
            write({...read(),mode:"manual",box,size:[plan.width,plan.height]});
        });
        knob.addEventListener("lostpointercapture", () => { draft = null; place(geometry()); });
        knob.addEventListener("keydown", e => {
            const vector = {ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];
            if (!vector || !plan) return;
            e.stopPropagation(); e.preventDefault();
            const step = e.shiftKey ? 16 : 1;
            write({...read(),mode:"manual",box:resize(index,plan.box,vector[0]*step,vector[1]*step),size:[plan.width,plan.height]});
        });
    }
    return {panel,trial,refresh,place,target,get span(){return plan?.span;}};
}
