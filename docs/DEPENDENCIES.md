# Dependencies

Pinned sources are defined in `config/dependencies.json`.

| Component | Repository / revision | Purpose |
| --- | --- | --- |
| ComfyUI core | `Comfy-Org/ComfyUI@master` | Host runtime, `LoadVideo`, `GetVideoComponents`, `ExecutionBlocker` |
| ComfyUI-SAM3 | `PozzettiAndrea/ComfyUI-SAM3@de0ff5d2…` | Optional native SAM3 model patcher and video predictor |

Requirements files deliberately omit `torch`, `torchvision`, `torchaudio`, `xformers`, and CUDA wheels:
- `requirements-editor.txt`: `numpy`, `Pillow`, and `opencv-python-headless` for CPU mask rasterization and Optical Flow.
- `requirements-sam3.txt`: non-torch dependencies required by `ComfyUI-SAM3`.

## Upstream references

- [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3)
- [Meta SAM3](https://github.com/facebookresearch/sam3)
