from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "workflows" / "V69_Universal_Masked_Repair.json"
MASK_ONLY = ROOT / "workflows" / "V69_Universal_Masked_Repair_MASK_ONLY.json"
AUTO = ROOT / "workflows" / "V69_Universal_AutoPrompt_Repair.json"
REFERENCE = ROOT / "workflows" / "V69_Universal_Reference_Repair.json"
WORKBENCH = ROOT / "workflows" / "V70_Universal_Repair_Workbench.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def by_type(workflow):
    return {node["type"]: node for node in workflow["nodes"]}


def nodes_of_type(workflow, node_type):
    return [node for node in workflow["nodes"] if node["type"] == node_type]


def edge(workflow, source, target):
    return [link for link in workflow["links"] if link[1] == source and link[3] == target]


def assert_no_collisions(workflow, minimum_gap=48):
    nodes = workflow["nodes"]
    for index, left in enumerate(nodes):
        lx, ly = left["pos"]
        lw, lh = left["size"]
        left_right, left_bottom = lx + lw, ly + lh
        for right in nodes[index + 1 :]:
            rx, ry = right["pos"]
            rw, rh = right["size"]
            right_right, right_bottom = rx + rw, ry + rh
            separated = (
                left_right + minimum_gap <= rx
                or right_right + minimum_gap <= lx
                or left_bottom + minimum_gap <= ry
                or right_bottom + minimum_gap <= ly
            )
            assert separated, f"nodes {left['id']} and {right['id']} are too close"


def assert_groups_fit(workflow, membership):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    groups = {group["id"]: group for group in workflow["groups"]}
    for group_id, node_ids in membership.items():
        gx, gy, gw, gh = groups[group_id]["bounding"]
        right, bottom = gx + gw, gy + gh
        for node_id in node_ids:
            node = nodes[node_id]
            nx, ny = node["pos"]
            nw, nh = node["size"]
            assert gx <= nx and gy <= ny
            assert right >= nx + nw and bottom >= ny + nh
        # The builder uses a fixed 64px inset; reject the old huge empty
        # right-hand panels while allowing a little future breathing room.
        max_right = max(nodes[node_id]["pos"][0] + nodes[node_id]["size"][0] for node_id in node_ids)
        assert right - max_right <= 96


def test_full_workflow_has_source_latent_masked_repair_chain():
    workflow = load(FULL)
    nodes = by_type(workflow)
    required = {
        "LoadVideo", "GetVideoComponents", "THSUniversalMaskEditor", "THSUniversalH3PadSource",
        "MVEx_MaskToLatentSpace", "VAEEncode", "SetLatentNoiseMask", "LTXVConcatAVLatent",
        "THSUniversalH3RepairConditioning", "KSampler", "VAEDecode", "THSUniversalTrimFrames",
        "THSUniversalMaskedComposite", "CreateVideo", "SaveVideo",
    }
    assert required <= set(nodes)
    forbidden = {"Sapiens2Loader", "Sapiens2Seg", "THSSAM3VideoSegmentation", "THSH3ReferenceBalance", "THSMiniMaxH3FixedFourReferencesWithMotion", "PathchSageAttentionKJ"}
    assert not forbidden.intersection(nodes)
    serialised = json.dumps(workflow, ensure_ascii=False).lower()
    assert "motion_driver" not in serialised
    assert "sapiens" not in serialised
    assert "referencebalance" not in serialised

    # The latent mask comes from the padded user mask, and only the AV latent
    # reaches KSampler's latent_image socket.
    assert edge(workflow, nodes["THSUniversalH3PadSource"]["id"], nodes["MVEx_MaskToLatentSpace"]["id"])
    assert any(
        edge(workflow, nodes["MVEx_MaskToLatentSpace"]["id"], latent_mask_node["id"])
        for latent_mask_node in nodes_of_type(workflow, "SetLatentNoiseMask")
    )
    assert edge(workflow, nodes["LTXVConcatAVLatent"]["id"], nodes["KSampler"]["id"])
    assert edge(workflow, nodes["THSUniversalMaskedComposite"]["id"], nodes["CreateVideo"]["id"])

    # FPS/audio are direct source components, not generated H3 output.
    source = nodes["GetVideoComponents"]["id"]
    create = nodes["CreateVideo"]["id"]
    assert any(link[3:] == [create, 2, "AUDIO"] for link in workflow["links"] if link[1] == source)
    assert any(link[3:] == [create, 1, "FLOAT"] for link in workflow["links"] if link[1] == source)
    assert nodes["THSUniversalH3RepairConditioning"]["widgets_values"][2:5] == [768, 768, 22]
    assert nodes["CreateVideo"]["widgets_values"] == [24, 8, "sRGB", "none"]
    assert [item["name"] for item in nodes["THSUniversalMaskEditor"]["inputs"]] == ["images", "fps"]
    assert [item["name"] for item in nodes["THSUniversalH3PadSource"]["inputs"]] == ["images", "mask", "audio"]
    assert [item["name"] for item in nodes["THSUniversalH3RepairConditioning"]["inputs"]] == ["clip", "vae", "width", "height", "length", "source_images"]
    assert [item["name"] for item in nodes["MVEx_MaskToLatentSpace"]["inputs"]] == ["masks", "vae"]
    assert [item["name"] for item in nodes["CreateVideo"]["inputs"]] == ["images", "fps", "audio"]

    # No input is wired twice; this catches a common UI JSON corruption mode.
    targets = Counter((link[3], link[4]) for link in workflow["links"])
    assert all(count == 1 for count in targets.values())


def test_mask_only_graph_has_no_model_loads_and_same_editor_id():
    full = by_type(load(FULL))
    mask = by_type(load(MASK_ONLY))
    assert mask["THSUniversalMaskEditor"]["id"] == full["THSUniversalMaskEditor"]["id"]
    assert {"UNETLoader", "CLIPLoader", "VAELoader", "KSampler"}.isdisjoint(mask)
    assert "SaveImage" in mask and "PreviewImage" in mask


def test_auto_prompt_variant_wires_real_images_mask_and_prompt_widget():
    workflow = load(AUTO)
    nodes = by_type(workflow)
    assert "THSUniversalAutoPrompt" in nodes
    auto = nodes["THSUniversalAutoPrompt"]
    conditioner = nodes["THSUniversalH3RepairConditioning"]
    assert [item["name"] for item in auto["inputs"]] == ["images", "mask"]
    assert [item["name"] for item in auto["outputs"]] == ["prompt", "caption", "preview"]
    assert [item["name"] for item in conditioner["inputs"]] == [
        "clip", "vae", "prompt", "width", "height", "length", "source_images"
    ]
    # Both sockets come from the actual source/mask nodes and only prompt slot
    # 0 is converted to the existing conditioner's prompt widget.
    assert edge(workflow, nodes["GetVideoComponents"]["id"], auto["id"])
    assert edge(workflow, nodes["THSUniversalMaskEditor"]["id"], auto["id"])
    prompt_edges = [link for link in workflow["links"] if link[1] == auto["id"] and link[2] == 0]
    assert prompt_edges and prompt_edges[0][3:] == [conditioner["id"], 2, "STRING"]
    assert conditioner["inputs"][2]["widget"] == {"name": "prompt"}
    assert conditioner["inputs"][2]["link"] == prompt_edges[0][0]
    assert auto["widgets_values"] == ["instruction", "", "", 3, 0.6, False]
    assert workflow["extra"]["ths"]["auto_prompt"]["ui_keys"] == [
        "auto_prompt_text", "auto_prompt_caption", "auto_prompt_status"
    ]


def test_auto_prompt_variant_layout_has_no_collisions_and_manual_graph_stays_separate():
    workflow = load(AUTO)
    assert_no_collisions(workflow)
    nodes = by_type(workflow)
    assert nodes["THSUniversalAutoPrompt"]["pos"][0] >= nodes["THSUniversalMaskEditor"]["pos"][0]
    manual = load(FULL)
    assert "THSUniversalAutoPrompt" not in by_type(manual)
    assert [item["name"] for item in by_type(manual)["THSUniversalH3RepairConditioning"]["inputs"]] == [
        "clip", "vae", "width", "height", "length", "source_images"
    ]


def test_reference_variant_has_one_authoritative_mode_and_native_reference_path():
    workflow = load(REFERENCE)
    nodes = by_type(workflow)
    assert {
        "THSUniversalReferenceImage",
        "THSUniversalH3ReferenceConditioning",
        "THSUniversalCorrectedFrameGuide",
        "PreviewImage",
        "THSUniversalMaskedComposite",
    } <= set(nodes)
    # The reference graph is based on the AutoPrompt variant so OFF preserves
    # node 22's source-context prompt while ON/corrected modes use their own
    # conditioning prompts.
    assert "THSUniversalAutoPrompt" in nodes
    assert nodes["THSUniversalReferenceImage"]["widgets_values"] == ["without_reference", "__none__"]
    assert [output["name"] for output in nodes["THSUniversalReferenceImage"]["outputs"]] == ["reference_image", "mode"]
    assert nodes["THSUniversalReferenceImage"]["outputs"][0]["links"]
    assert nodes["THSUniversalReferenceImage"]["outputs"][1]["links"]
    conditioner = nodes["THSUniversalH3ReferenceConditioning"]
    # Mode and reference image are both linked from the same upload-aware node;
    # there is no second independent toggle that can silently disagree.
    assert [item["name"] for item in conditioner["inputs"]] == [
        "clip", "vae", "prompt", "width", "height", "length", "mode", "reference_image", "source_images"
    ]
    assert edge(workflow, nodes["THSUniversalReferenceImage"]["id"], conditioner["id"])
    mode_edges = [link for link in workflow["links"] if link[1] == nodes["THSUniversalReferenceImage"]["id"] and link[2] == 1]
    assert mode_edges and mode_edges[0][3:] == [conditioner["id"], 6, "COMBO"]
    image_edges = [link for link in workflow["links"] if link[1] == nodes["THSUniversalReferenceImage"]["id"] and link[2] == 0]
    assert any(link[3:] == [conditioner["id"], 7, "IMAGE"] for link in image_edges)
    guide = nodes["THSUniversalCorrectedFrameGuide"]
    assert [item["name"] for item in guide["inputs"]] == [
        "positive", "latent", "vae", "source_images", "mask", "mode", "frame_number", "corrected_image"
    ]
    assert guide["widgets_values"] == ["without_reference", 1]
    assert edge(workflow, conditioner["id"], guide["id"])
    assert edge(workflow, guide["id"], nodes["KSampler"]["id"])
    assert edge(workflow, guide["id"], nodes["PreviewImage"]["id"])
    assert any(link[1] == nodes["GetVideoComponents"]["id"] and link[3:] == [guide["id"], 3, "IMAGE"] for link in workflow["links"])
    assert any(link[1] == nodes["THSUniversalMaskEditor"]["id"] and link[3:] == [guide["id"], 4, "MASK"] for link in workflow["links"])
    assert edge(workflow, nodes["THSUniversalMaskedComposite"]["id"], nodes["CreateVideo"]["id"])
    assert nodes["SaveVideo"]["widgets_values"][0] == "video/universal_inpaint_reference_output"
    assert workflow["extra"]["ths"]["reference_mode"]["none_sentinel"] == "__none__"
    assert workflow["extra"]["ths"]["reference_mode"]["native_prompt_label"] == "<Picture 1>"
    assert workflow["extra"]["ths"]["reference_mode"]["mode_values"] == [
        "without_reference", "with_reference", "corrected_frame"
    ]


def test_reference_variant_layout_is_readable_and_manual_graph_is_unchanged():
    workflow = load(REFERENCE)
    assert_no_collisions(workflow)
    assert_groups_fit(workflow, {
        4600: [10, 11, 20, 21, 22, 23],
            4601: [30, 40, 41, 42, 43, 50, 51, 52, 53, 54, 55, 56, 60, 61, 80, 24, 25],
        4602: [90, 91, 92, 100, 101],
    })
    manual = load(FULL)
    assert by_type(manual)["THSUniversalH3RepairConditioning"]["id"] == 61
    assert "THSUniversalH3ReferenceConditioning" not in by_type(manual)


def test_workbench_runs_three_h3_modes_through_one_mask_and_one_sampler():
    workflow = load(WORKBENCH)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    def origin(target, name):
        link_id = next(item["link"] for item in nodes[target]["inputs"] if item["name"] == name)
        return links[link_id][1:3]

    assert nodes[23]["type"] == "THSUniversalReferenceImage"
    assert nodes[23]["widgets_values"] == ["without_reference", "__none__"]
    assert nodes[22]["widgets_values"][0] == "instruction"
    assert origin(22, "repair_mode") == origin(61, "mode") == origin(24, "mode") == [23, 1]
    assert origin(61, "reference_image") == origin(24, "corrected_image") == [23, 0]
    assert origin(61, "prompt") == [22, 0]
    assert origin(24, "positive") == [61, 0]
    assert origin(24, "latent") == [61, 2]
    assert origin(24, "source_images") == origin(92, "source") == [11, 0]
    assert origin(24, "mask") == origin(105, "mask") == origin(92, "repair_mask") == [20, 0]
    assert origin(80, "positive") == [24, 0]
    assert origin(80, "latent_image") == [56, 0]
    assert origin(100, "audio") == [11, 1]
    assert_groups_fit(workflow, {4702: [22, 23, 24, 103]})


def test_workflow_layout_is_readable_and_editor_has_room():
    full = load(FULL)
    mask = load(MASK_ONLY)
    for workflow in (full, mask):
        assert_no_collisions(workflow)
        editor = next(node for node in workflow["nodes"] if node["type"] == "THSUniversalMaskEditor")
        assert editor["size"][0] >= 900
        assert editor["size"][1] >= 760
    assert all(node["size"][0] >= 400 for node in full["nodes"] if node["type"] in {"VAELoader", "UNETLoader", "CLIPLoader"})
    assert_groups_fit(full, {
        4600: [10, 11, 20, 21],
        4601: [30, 40, 41, 42, 43, 50, 51, 52, 53, 54, 55, 56, 60, 61, 80],
        4602: [90, 91, 92, 100, 101],
    })
    assert_groups_fit(mask, {4600: [10, 11, 20, 21, 22, 23]})


def test_public_workflows_use_portable_placeholders_and_no_source_fingerprints():
    for path in (FULL, MASK_ONLY, AUTO, REFERENCE, ROOT / "workflows" / "V70_Universal_Repair_Workbench.json"):
        workflow = load(path)
        serialised = json.dumps(workflow, ensure_ascii=False)
        assert "REPLACE_WITH_INPUT_VIDEO.mp4" in serialised
        assert "source_workflow_sha256" not in serialised
        assert "F:\\" not in serialised and "F:/" not in serialised
        assert "HeadSwap" not in serialised
        assert "universal_repair_input.mp4" not in serialised
        assert "123456" not in serialised


def test_full_graph_has_no_dangling_links_after_default_attention_reroute():
    for path in (FULL, AUTO, REFERENCE, ROOT / "workflows" / "V70_Universal_Repair_Workbench.json"):
        workflow = load(path)
        nodes = {int(node["id"]): node for node in workflow["nodes"]}
        output_links = {
            (node_id, slot): set(int(link_id) for link_id in output.get("links", []))
            for node_id, node in nodes.items()
            for slot, output in enumerate(node.get("outputs", []))
        }
        target_links = set()
        for link_id, source, source_slot, target, target_slot, _kind in workflow["links"]:
            assert int(source) in nodes and int(target) in nodes
            assert int(source_slot) < len(nodes[int(source)].get("outputs", []))
            assert int(target_slot) < len(nodes[int(target)].get("inputs", []))
            assert int(link_id) in output_links[(int(source), int(source_slot))]
            assert nodes[int(target)]["inputs"][int(target_slot)].get("link") == int(link_id)
            target_links.add(int(link_id))
        assert target_links == {int(link[0]) for link in workflow["links"]}
        assert not any(node["type"] == "PathchSageAttentionKJ" for node in workflow["nodes"])


def test_model_selectors_match_installer_paths_on_windows_and_linux():
    models = load(ROOT / "config" / "models.json")["models"]
    destinations = {item["destination"] for item in models if "h3" in item["profiles"]}
    categories = {"VAELoader": "vae", "UNETLoader": "diffusion_models", "CLIPLoader": "text_encoders"}
    for path in ROOT.joinpath("workflows").glob("*.json"):
        for node in load(path)["nodes"]:
            if node["type"] in categories:
                selected = node["widgets_values"][0]
                assert "/" not in selected and "\\" not in selected
                assert f"models/{categories[node['type']]}/{selected}" in destinations
