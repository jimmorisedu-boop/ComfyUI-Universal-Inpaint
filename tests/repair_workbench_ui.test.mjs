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
    addEventListener(type, handler) { const list = this.listeners.get(type) || []; list.push(handler); this.listeners.set(type, list); }
    dispatchEvent(event) { event.target ||= this; event.currentTarget = this; for (const handler of this.listeners.get(event.type) || []) handler(event); return !event.defaultPrevented; }
    focus() {}
}

globalThis.document = { createElement(tagName) { return new FakeElement(tagName); } };
globalThis.window = { addEventListener() {}, removeEventListener() {} };

const { createAutoPromptPreview, createPromptFreezeController, MUTE_MODE } = await import("../web/auto_prompt_preview.js");
const { normalizeDescriptors } = await import("../web/repair_compare.js");

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

console.log("repair_workbench_ui: PASS");
