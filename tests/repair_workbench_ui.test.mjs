import assert from "node:assert/strict";

class FakeElement {
    constructor(tagName) {
        this.tagName = String(tagName).toUpperCase();
        this.children = [];
        this.style = {};
        this.listeners = new Map();
        this.attributes = {};
        this.textContent = "";
        this.value = "";
        this.dataset = {};
        this.innerHTML = "";
    }
    append(...children) { this.children.push(...children.filter(Boolean)); }
    appendChild(child) { this.append(child); return child; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; delete this[name]; }
    addEventListener(type, handler) { const list = this.listeners.get(type) || []; list.push(handler); this.listeners.set(type, list); }
    dispatchEvent(event) { event.target ||= this; event.currentTarget = this; for (const handler of this.listeners.get(event.type) || []) handler(event); return !event.defaultPrevented; }
    focus() {}
}

globalThis.document = { createElement(tagName) { return new FakeElement(tagName); } };
globalThis.window = { addEventListener() {}, removeEventListener() {} };

const { createAutoPromptPreview, createPromptFreezeController, MUTE_MODE } = await import("../web/auto_prompt_preview.js");
const { createRepairCompare, normalizeDescriptors } = await import("../web/repair_compare.js");

function makeGraph({ frozen = false, promptText = "literal prompt" } = {}) {
    const graph = { _nodes: [], links: [], setDirtyCanvas() {}, change() {} };
    const auto = { id: 22, type: "THSUniversalAutoPrompt", comfyClass: "THSUniversalAutoPrompt", mode: frozen ? MUTE_MODE : 0, inputs: [{ name: "images", link: null }, { name: "mask", link: null }], outputs: [{ name: "prompt", links: [] }], widgets: [] };
    const conditioner = { id: 61, type: "THSUniversalH3RepairConditioning", inputs: [{ name: "clip", link: null }, { name: "vae", link: null }, { name: "prompt", link: frozen ? 90 : 45 }], outputs: [] };
    const promptNode = { id: 103, type: "THSUniversalPromptText", mode: frozen ? 0 : 2, inputs: [{ name: "text", link: null }], outputs: [{ name: "prompt", links: frozen ? [90] : [] }], widgets: [{ name: "text", value: promptText }] };
    graph._nodes.push(auto, conditioner, promptNode);
    graph.links.push(frozen ? [90, 103, 0, 61, 2, "STRING"] : [45, 22, 0, 61, 2, "STRING"]);
    auto.outputs[0].links = frozen ? [] : [45];
    auto.connect = (sourceSlot, target, targetSlot) => {
        for (const link of graph.links.filter((item) => item[3] === target.id && item[4] === targetSlot)) {
            const index = graph.links.indexOf(link);
            if (index >= 0) graph.links.splice(index, 1);
            const source = graph._nodes.find((candidate) => candidate.id === link[1]);
            source?.outputs?.[link[2]]?.links?.splice(source.outputs[link[2]].links.indexOf(link[0]), 1);
        }
        const next = Math.max(0, ...graph.links.map((item) => Number(item[0]))) + 1;
        graph.links.push([next, this?.id ?? auto.id, sourceSlot, target.id, targetSlot, "STRING"]);
        target.inputs[targetSlot].link = next;
        const source = graph._nodes.find((candidate) => candidate.id === (this?.id ?? auto.id));
        source?.outputs?.[sourceSlot]?.links?.push(next);
    };
    // LiteGraph calls source.connect with `this` set to the source node.
    for (const source of [auto, promptNode]) source.connect = function connect(sourceSlot, target, targetSlot) {
        for (const link of [...graph.links]) {
            if (link[3] !== target.id || link[4] !== targetSlot) continue;
            graph.links.splice(graph.links.indexOf(link), 1);
            const previous = graph._nodes.find((candidate) => candidate.id === link[1]);
            const links = previous?.outputs?.[link[2]]?.links || [];
            const index = links.indexOf(link[0]); if (index >= 0) links.splice(index, 1);
        }
        const next = Math.max(0, ...graph.links.map((item) => Number(item[0]))) + 1;
        graph.links.push([next, this.id, sourceSlot, target.id, targetSlot, "STRING"]);
        target.inputs[targetSlot].link = next;
        this.outputs[sourceSlot].links ||= [];
        this.outputs[sourceSlot].links.push(next);
    };
    for (const target of [conditioner]) target.disconnectInput = function disconnectInput(slot) {
        const linkId = this.inputs[slot]?.link;
        if (linkId == null) return;
        const link = graph.links.find((item) => item[0] === linkId);
        if (link) {
            graph.links.splice(graph.links.indexOf(link), 1);
            const source = graph._nodes.find((candidate) => candidate.id === link[1]);
            const links = source?.outputs?.[link[2]]?.links || [];
            const index = links.indexOf(linkId); if (index >= 0) links.splice(index, 1);
        }
        this.inputs[slot].link = null;
    };
    auto.graph = graph; promptNode.graph = graph; conditioner.graph = graph;
    return { graph, auto, conditioner, promptNode };
}

// Freeze round-trip: exact text, independent node, muted AutoPrompt mode, and
// then the original connection/mode restored after a simulated reload.
{
    const initial = makeGraph();
    const controller = createPromptFreezeController(initial.auto, { graph: initial.graph, promptNode: initial.promptNode });
    assert.equal(controller.freeze("line one\nline two").ok, true);
    assert.equal(initial.promptNode.widgets[0].value, "line one\nline two");
    assert.equal(initial.auto.mode, MUTE_MODE);
    const frozenLink = initial.graph.links.find((link) => link[3] === 61 && link[4] === 2);
    assert.equal(frozenLink[1], 103);
    assert.equal(controller.refresh(), true);

    const reloaded = makeGraph({ frozen: true, promptText: "line one\nline two" });
    const reloadedController = createPromptFreezeController(reloaded.auto, { graph: reloaded.graph });
    assert.equal(reloadedController.refresh(), true);
    assert.equal(reloadedController.getText(), "line one\nline two");
    assert.equal(reloadedController.unfreeze().ok, true);
    assert.equal(reloaded.auto.mode, 0);
    assert.equal(reloaded.promptNode.mode, MUTE_MODE);
    const restored = reloaded.graph.links.find((link) => link[3] === 61 && link[4] === 2);
    assert.equal(restored[1], 22);
}

// A manual edit survives a late caption event; the user must explicitly apply
// the newer generated text.  The Comfy theme token is used instead of a fixed
// dark background, and the old five-widget AutoPrompt shape is accepted.
{
    const { graph, auto, promptNode } = makeGraph();
    const node = { ...auto, graph, widgets: [], properties: {}, addDOMWidget(_name, _type, root) { this.root = root; } };
    const preview = createAutoPromptPreview(node, { app: { graph }, api: null });
    assert.equal(preview.root.style.background, "var(--bg-color, #202020)");
    preview.applyExecuted({ auto_prompt_text: ["generated one"], auto_prompt_caption: ["caption"], auto_prompt_status: ["ready"] });
    assert.equal(preview.prompt.value, "generated one");
    preview.prompt.value = "manual edit";
    preview.prompt.dispatchEvent({ type: "input" });
    const late = preview.applyExecuted({ auto_prompt_text: ["generated two"], auto_prompt_caption: ["new caption"], auto_prompt_status: ["ready"] });
    assert.equal(preview.prompt.value, "manual edit");
    assert.equal(late.stale, true);
    preview.applyGenerated.click?.();
    // `click` is not needed in the tiny fake; the state remains explicit and
    // the preview exposes the generated text for a real button click.
    assert.equal(preview.state.generated, "generated two");
}

// Typing a new task in a saved manual workflow restores the live conditioner
// link and makes its text available to queued H3 runs without freezing.
{
    const { graph, auto, promptNode } = makeGraph({ frozen: true });
    auto.widgets = [{name: "mode", value:"manual"}, {name:"user_hint", value:""}, {name:"manual_prompt", value:"previous prompt"}];
    const preview = createAutoPromptPreview(auto, { app: { graph } });
    preview.task.value = "Сделай волосы седыми";
    preview.task.dispatchEvent({ type:"input" });
    assert.equal(auto.widgets[0].value, "instruction");
    assert.equal(auto.widgets[1].value, preview.task.value);
    assert.equal(auto.widgets[2].value, "previous prompt");
    assert.equal(graph.links.find(link => link[3] === 61)[1], 22);
    assert.equal(promptNode.mode, MUTE_MODE);
    assert.equal(preview.freeze.style.display, "none");
    assert.equal(preview.prompt.readOnly, true);
    assert.equal(preview.applyExecuted({auto_prompt_text:["old result"], auto_prompt_instruction:["другая задача"]}).stale, true);
    assert.equal(preview.prompt.value, "");
    assert.equal(preview.applyExecuted({auto_prompt_text:["Make the hair gray"], auto_prompt_instruction:[preview.task.value]}).applied, true);
    assert.equal(preview.prompt.value, "Make the hair gray");
}

// A text-only preview bypasses an unfinished video mask while the regular
// H3 prompt graph still uses the original connections.
{
    const { graph, auto } = makeGraph();
    const request = "измени дверь на стену ";
    auto.widgets = [{name:"mode", value:"instruction"}, {name:"user_hint", value:request}];
    const output = {
        "1": { class_type:"LoadVideo", inputs:{} },
        "20": { class_type:"THSUniversalMaskEditor", inputs:{images:["1",0]} },
        "22": { class_type:"THSUniversalAutoPrompt", inputs:{images:["1",0], mask:["20",0], mode:"instruction", user_hint:request, repair_mode:["23",1]} },
        "23": { class_type:"THSUniversalReferenceImage", inputs:{mode:"with_reference", reference_image:"test.png"} },
        "61": { class_type:"THSUniversalH3RepairConditioning", inputs:{prompt:["22",0]} },
    };
    const before = JSON.stringify(output);
    let submitted;
    let history = { status:{status_str:"success"}, outputs:{22:{auto_prompt_instruction:[request], auto_prompt_text:["Replace the door with a wall."]}} };
    const api = {
        async queuePrompt(_number, compiled) { submitted = compiled; return {prompt_id:"test-preview"}; },
        async fetchApi(path) {
            assert.equal(path,"/history/test-preview");
            return {ok:true, async json() { return {"test-preview":history}; }};
        },
    };
    const preview = createAutoPromptPreview(auto, {app:{graph, async graphToPrompt() { return {output}; }}, api});
    await preview.generatePrompt();
    assert.deepEqual(Object.values(submitted.output).map(node=>node.class_type).sort(), ["EmptyImage","SolidMask","THSUniversalAutoPrompt"]);
    assert.equal(submitted.output[22].inputs.user_hint,request);
    assert.equal(submitted.output[22].inputs.preview_enabled,false);
    assert.equal(submitted.output[22].inputs.repair_mode,undefined);
    assert.equal(JSON.stringify(output),before);
    assert.equal(preview.prompt.value,"Replace the door with a wall.");
    assert.equal(preview.generate.disabled,false);

    auto.widgets[0].value = "auto";
    output[22].inputs.mode = "auto";
    history = {status:{status_str:"success"}, outputs:{20:{universal_mask_meta:[{sam3:{track_ready:false}}]}}};
    await preview.generatePrompt();
    assert.equal(submitted.output[20].class_type,"THSUniversalMaskEditor");
    assert.equal(submitted.output[23],undefined);
    assert.match(preview.status.textContent,/Отследить видео/);
    assert.equal(preview.generate.disabled,false);
    history = {status:{status_str:"error", messages:[["execution_error", {exception_message:"Missing local model"}]]}};
    await preview.generatePrompt();
    assert.match(preview.status.textContent,/Missing local model/);
    assert.equal(preview.generate.disabled,false);
}

// Launch from the prompt node uses the connected mask editor's readiness gate
// and can start directly from the current instruction.
{
    const { graph, auto } = makeGraph();
    auto.widgets = [{name:"mode", value:"instruction"}, {name:"user_hint", value:""}];
    const preview = createAutoPromptPreview(auto, {app:{graph}});
    assert.equal(preview.launch.style.display, "inline-flex");
    await preview.launchH3();
    assert.match(preview.status.textContent, /напишите, что изменить/);

    preview.task.value = "измени галстук на бабочку";
    preview.task.dispatchEvent({type:"input"});
    await preview.launchH3();
    assert.match(preview.status.textContent, /Подключите выход edit_mask/);

    let queued = false;
    let calls = 0;
    const editor = {
        status: {textContent:""},
        async requestQueue() {
            calls++;
            this.status.textContent = queued ? "H3 поставлен в очередь." : "В редакторе маски нажмите «Отследить видео».";
            return queued;
        },
    };
    graph._nodes.push({id:20, __thsUniversalMaskEditor:editor});
    graph.links.push([39, 20, 0, 22, 1, "MASK"]);
    auto.inputs[1].link = 39;
    await preview.launchH3();
    assert.match(preview.status.textContent, /Отследить видео/);
    assert.equal(calls, 1);

    queued = true;
    await Promise.all([preview.launchH3(), preview.launchH3()]);
    assert.equal(calls, 2);
    assert.match(preview.status.textContent, /H3 поставлен в очередь/);
    assert.equal(preview.prompt.value, "");
    assert.equal(preview.launch.disabled, false);
    preview.task.value = "";
    preview.task.dispatchEvent({type:"input"});
    const repairMode = {name:"mode", value:"with_reference"};
    graph._nodes.push({id:23, widgets:[repairMode]});
    graph.links.push([40, 23, 1, 22, 2, "COMBO"]);
    auto.inputs.push({name:"repair_mode", link:40});
    await preview.launchH3();
    assert.equal(calls, 3);
    repairMode.value = "corrected_frame";
    await preview.launchH3();
    assert.equal(calls, 4);
    repairMode.value = "without_reference";
    await preview.launchH3();
    assert.equal(calls, 4);
    assert.match(preview.status.textContent, /напишите, что изменить/);
}

// Compare descriptors are bounded, zero-based and lazy-compatible: only
// filename descriptors are retained, with duplicate frames removed.
{
    const descriptors = normalizeDescriptors({
        universal_repair_compare_frames: [
            { frame: 4, original: { filename: "four.png" }, result: { filename: "four-r.png" } },
            { frame: 1, original: { filename: "one.png" }, result: { filename: "one-r.png" } },
            { frame: 1, original: { filename: "duplicate.png" }, result: { filename: "duplicate-r.png" } },
        ],
        universal_repair_compare_meta: [{ fps: 24, frame_count: 5 }],
    });
    assert.deepEqual(descriptors.map((item) => item.frame), [1, 4]);
    assert.equal(descriptors[0].source.filename, "one.png");
    assert.equal(descriptors[1].result.filename, "four-r.png");
}

// Fast scrubbing can finish downloads out of order. Keep both image layers on
// the latest requested frame and allow the divider to reach either edge.
{
    const loading = new Map();
    const oldImage = globalThis.Image;
    globalThis.Image = class {
        naturalWidth = 544;
        naturalHeight = 576;
        set src(url) { loading.set(url, () => this.onload()); }
    };
    try {
        const compare = createRepairCompare({});
        compare.stage.clientWidth = 800;
        compare.stage.clientHeight = 400;
        compare.applyExecuted({
            universal_repair_compare_frames: [
                {frame: 0, source: "original-0", result: "result-0"},
                {frame: 1, source: "original-1", result: "result-1"},
            ],
        });
        compare.setFrame(1);
        loading.get("original-1")();
        loading.get("result-1")();
        await compare.state.rendering;
        loading.get("original-0")();
        loading.get("result-0")();
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(compare.source.src, "original-1");
        assert.equal(compare.result.src, "result-1");
        compare.wipe.value = "0";
        compare.wipe.dispatchEvent({ type: "input" });
        assert.equal(compare.result.style.clipPath, "inset(0 0 0 0%)");
        compare.wipe.value = "100";
        compare.wipe.dispatchEvent({ type: "input" });
        assert.equal(compare.result.style.clipPath, "inset(0 0 0 100%)");
    } finally {
        globalThis.Image = oldImage;
    }
}

console.log("repair_workbench_ui: PASS");
