# ComfyUI Universal Inpaint

[Русская инструкция](README_RU.md) · [Installation](docs/INSTALL.md) · [User guide (RU)](docs/USER_GUIDE_RU.md)

Universal Inpaint is a portable ComfyUI custom-node repository for repairing a
user-selected region of a short video while keeping the unmasked source path
explicit. It is an alpha, composable workbench rather than a hosted service or
a claim of universal video support.

The repository is designed to be cloned directly under a ComfyUI checkout's
`custom_nodes/` directory. The root `__init__.py` is the node entry point and
`WEB_DIRECTORY = "./web"` serves the browser editor.

## What is included

| Path | Purpose | H3 required? |
| --- | --- | --- |
| `workflows/V70_Universal_Repair_Workbench.json` | Main workbench with instruction, image reference and corrected-frame modes sharing one mask and export path | Yes for generation |
| `workflows/V69_Universal_Masked_Repair_MASK_ONLY.json` | Mask calculation/preview and image export | No |
| `workflows/V69_Universal_Masked_Repair.json` | Manual mask → H3 repair baseline | Yes for generation |
| `workflows/V69_Universal_AutoPrompt_Repair.json` | Optional local caption-assisted prompt branch | Yes for generation; caption model optional |
| `workflows/V69_Universal_Reference_Repair.json` | Reference/corrected-frame branch with one authoritative mode selector | Yes for generation |
| `web/` | Comfy node editor, AutoPrompt preview and compare panels | No |

Mask editing has distinct tracking modes. **SAM3** is a two-step flow: calculate
one seed frame, then explicitly track the requested range. **Optical Flow** is a
separate legacy/CPU mode and does not silently substitute for a failed SAM3
request. The mask-only graph is the honest way to inspect or export masks on a
machine without an H3 runtime.

## Install

1. Install a compatible ComfyUI checkout and start it once.
2. Clone this repository directly into `ComfyUI/custom_nodes/`:

   ```powershell
   cd custom_nodes
   git clone https://github.com/jimmorisedu-boop/ComfyUI-Universal-Inpaint.git
   cd ComfyUI-Universal-Inpaint
   ```

3. Before restarting, install the editor profile into the same ComfyUI Python
   environment (the installer is read-only unless `--apply` is supplied):

   ```powershell
   python scripts/setup.py install --profile editor --comfy-root "<COMFYUI_ROOT>" --apply
   ```

   Or install the small CPU extras with that interpreter using
   `python -m pip install -e ".[cpu]"`.
4. Restart ComfyUI and start with `workflows/V69_Universal_Masked_Repair_MASK_ONLY.json`.
   For CPU use, select **Область · Optical Flow**, choose your video and load the frames.
   For H3 generation, install the optional
   H3 profile and models first.

The maintained installer and model checklist are being kept alongside this
checkout at [docs/INSTALL.md](docs/INSTALL.md), [docs/INSTALL_RU.md](docs/INSTALL_RU.md),
[docs/MODELS.md](docs/MODELS.md), and `scripts/setup.py`. The scripts are
deliberately separate from the node package so a manual clone remains useful.

## Runtime tiers

- **CPU:** the mask editor, static edits, Optical Flow where OpenCV is
  available, mask previews, exact outside-mask compositing, workflow checks and
  the mask-only graph are the supported starting point.
- **Optional GPU:** full H3 generation needs a compatible ComfyUI build, the
  upstream H3/Video nodes and the model files selected in the workflow. Those
  weights are not bundled here and no download is attempted by this addon.
- **Optional AutoPrompt:** `THSUniversalAutoPrompt` can use an already-installed
  Florence-2/PromptGen extension and local checkpoint. Configure its model path
  with ComfyUI's `folder_paths` registry or
  `THS_UNIVERSAL_INPAINT_LLM_PATH`; the node never downloads a model.
- **Optional SAM3:** the two-step native SAM3 path needs a compatible local
  `comfyui-sam3` adapter, tokenizer and checkpoint. SAM3 model and tokenizer
  licences remain those of their upstream projects; this repository does not
  redistribute or relicense them.

## Limitations and evidence boundary

The current editor acceptance covered one 119-frame, 1080×1920, 30 FPS portrait
clip, including native SAM3 frame calculation/tracking and Optical Flow edits.
That is bounded integration evidence, not a universal support or quality claim.
No new H3 render was performed for this public checkout. H3 quality, speed,
memory use, audio compatibility and model availability depend on the user's
ComfyUI installation and are not guaranteed by the mask editor.

Do not commit personal video, masks, prompts, model weights, browser debug
exports, host paths or acceptance captures. The checked-in workflows contain
obvious placeholders such as `REPLACE_WITH_INPUT_VIDEO.mp4` and
`REPLACE_WITH_REPAIR_PROMPT` for users to replace locally.

## Development checks

```powershell
python -m pytest -q
node --test tests/*.test.mjs
```

The Python suite is CPU/unit-contract coverage; the JavaScript suites exercise
the editor without a browser or ComfyUI server. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
for the direct-loader import smoke test and contribution boundaries.

## Project

- Product: **Universal Inpaint**
- Maintainer: **jimmorisedu-boop**
- Repository: [jimmorisedu-boop/ComfyUI-Universal-Inpaint](https://github.com/jimmorisedu-boop/ComfyUI-Universal-Inpaint)
- Version: **0.1.2**
- License: **GPL-3.0-only** (`LICENSE`)
