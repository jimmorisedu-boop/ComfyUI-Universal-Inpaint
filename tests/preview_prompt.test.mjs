import assert from "node:assert/strict";
import { previewPrompt } from "../web/preview_prompt.js";
const prompt = { workflow: { id: "original-workflow" }, output: {
    "10": { class_type: "LoadVideo", inputs: { file: "clip.mp4" } },
    "11": { class_type: "GetVideoComponents", inputs: { video: ["10", 0] } },
    "20": { class_type: "THSUniversalMaskEditor", inputs: { images: ["11", 0], mask_data: "{}" } },
    "22": { class_type: "THSUniversalAutoPrompt", inputs: { images: ["11", 0], mask: ["20", 0] } },
    "50": { class_type: "UNETLoader", inputs: {} },
    "70": { class_type: "KSampler", inputs: { model: ["50", 0], prompt: ["22", 0] } },
    "101": { class_type: "SaveVideo", inputs: { samples: ["70", 0] } },
} };
const before = JSON.stringify(prompt);
const mask = previewPrompt(prompt, 20);
assert.deepEqual(Object.keys(mask.output), ["10", "11", "20"]);
assert.deepEqual(Object.keys(previewPrompt(prompt, "22").output), ["10", "11", "20", "22"]);
assert.equal(mask.workflow, prompt.workflow);
assert.equal(JSON.stringify(prompt), before);
assert.throws(() => previewPrompt(prompt, "missing"), /отсутствует/);
assert.throws(() => previewPrompt({ output: { "20": { inputs: { images: ["missing", 0] } } } }, "20"), /Не найден/);
// An older queue wrapper ignoring the third argument still receives no H3 nodes.
let submitted;
const legacyQueue = (_number, graph) => { submitted = graph.output; };
legacyQueue(0, mask, { partialExecutionTargets: ["20"] });
assert.ok(Object.values(submitted).every((node) => !["KSampler", "UNETLoader", "SaveVideo"].includes(node.class_type)));
console.log("preview_prompt: PASS");
