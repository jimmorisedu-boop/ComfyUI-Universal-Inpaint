from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MASK_WORKFLOW = ROOT / "workflows" / "Universal_Video_Mask_Editor.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def by_type(workflow: dict) -> dict[str, dict]:
    return {node["type"]: node for node in workflow["nodes"]}


def edge(workflow: dict, source: int, target: int) -> list:
    return [link for link in workflow["links"] if link[1] == source and link[3] == target]


def assert_no_collisions(workflow: dict, minimum_gap: int = 48) -> None:
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


def assert_groups_fit(workflow: dict, membership: dict[int, list[int]]) -> None:
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
        max_right = max(nodes[node_id]["pos"][0] + nodes[node_id]["size"][0] for node_id in node_ids)
        assert right - max_right <= 96


def test_mask_workflow_wiring_and_no_model_or_h3_references():
    workflows = list(ROOT.joinpath("workflows").glob("*.json"))
    assert workflows == [MASK_WORKFLOW]

    workflow = load(MASK_WORKFLOW)
    nodes = by_type(workflow)
    assert {"LoadVideo", "GetVideoComponents", "THSUniversalMaskEditor", "MaskToImage", "SaveImage", "PreviewImage"} <= set(nodes)
    assert {"UNETLoader", "CLIPLoader", "VAELoader", "KSampler", "THSUniversalH3Window"}.isdisjoint(nodes)

    editor = nodes["THSUniversalMaskEditor"]
    assert [item["name"] for item in editor["inputs"]] == ["images", "fps"]
    assert [output["name"] for output in editor["outputs"]] == ["edit_mask", "preview", "cropped_images", "cropped_mask", "crop_info", "original_images"]
    assert editor["widgets_values"][1] == "sam3"
    assert editor["size"][0] >= 900 and editor["size"][1] >= 760

    assert edge(workflow, nodes["LoadVideo"]["id"], nodes["GetVideoComponents"]["id"])
    assert edge(workflow, nodes["GetVideoComponents"]["id"], editor["id"])
    assert edge(workflow, editor["id"], nodes["MaskToImage"]["id"])
    assert edge(workflow, nodes["MaskToImage"]["id"], nodes["SaveImage"]["id"])
    assert edge(workflow, editor["id"], nodes["PreviewImage"]["id"])

    targets = Counter((link[3], link[4]) for link in workflow["links"])
    assert all(count == 1 for count in targets.values())


def test_mask_workflow_layout_and_portable_placeholders():
    workflow = load(MASK_WORKFLOW)
    assert_no_collisions(workflow)
    assert_groups_fit(workflow, {4600: [10, 11, 20, 21, 22, 23]})

    serialised = json.dumps(workflow, ensure_ascii=False)
    assert "REPLACE_WITH_INPUT_VIDEO.mp4" in serialised
    assert "h3" not in serialised.lower()
    assert "v69" not in serialised.lower()
    assert "v70" not in serialised.lower()
    assert "F:\\" not in serialised and "F:/" not in serialised


def test_package_manifests_contain_only_mask_editor_and_sam3():
    deps = load(ROOT / "config" / "dependencies.json")
    models = load(ROOT / "config" / "models.json")
    assert set(deps["profiles"]) == {"editor", "sam3"}
    assert set(deps["custom_nodes"]) == {"comfyui-sam3"}
    assert [item["id"] for item in models["models"]] == ["sam3-native"]
    assert "h3" not in json.dumps(deps, ensure_ascii=False).lower()
    assert "h3" not in json.dumps(models, ensure_ascii=False).lower()
