# Install

This repository is a ComfyUI custom-node package focused exclusively on the interactive video mask editor (`THSUniversalMaskEditor`). The CPU baseline is the `editor` profile; GPU object tracking with `sam3` is optional and never replaces the Python, PyTorch, CUDA, torchvision, or torchaudio stack owned by ComfyUI.

## 1. Choose the runtime

Clone inside `ComfyUI/custom_nodes/`:

```powershell
cd custom_nodes
git clone https://github.com/jimmorisedu-boop/ComfyUI-Universal-Inpaint.git
cd ComfyUI-Universal-Inpaint
```

Inspect compatibility before installing (`plan` and `check` are read-only):

```powershell
python scripts/setup.py check --comfy-root "<ComfyUI-root>" --json
python scripts/setup.py plan --comfy-root "<ComfyUI-root>" --profile editor --json
```

## 2. Install the CPU mask editor (`editor` profile)

```powershell
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile editor --apply
```

The installer runs a `pip --dry-run --report` guard first and aborts if any `torch`, `torchvision`, `torchaudio`, `xformers`, `triton`, or `nvidia-*` package would be modified. Restart ComfyUI and open `workflows/Universal_Video_Mask_Editor.json`.

## 3. Optional SAM3 tracking (`sam3` profile)

```powershell
python scripts/setup.py plan --comfy-root "<ComfyUI-root>" --profile editor --profile sam3
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile sam3 --apply
```

This clones only the pinned `ComfyUI-SAM3` wrapper revision without invoking its optional `comfy-env`/pixi helper.

SAM3 weights on Hugging Face (`facebook/sam3`) are gated. Visit the repository URL printed by `plan`, accept the license yourself, authenticate with `hf auth login` or `HF_TOKEN`, and opt in explicitly:

```powershell
python scripts/setup.py download --comfy-root "<ComfyUI-root>" --profile sam3 --allow-gated --apply
```

## 4. Verify without mutation

```powershell
python scripts/setup.py check --comfy-root "<ComfyUI-root>" --profile editor --profile sam3 --json
python -m pytest --import-mode=importlib -q tests/test_setup.py tests/test_model.py
```
