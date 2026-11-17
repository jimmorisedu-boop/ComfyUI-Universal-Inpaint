# Dependency provenance and compatibility

Pinned public sources are recorded in `config/dependencies.json`.

| Component | Repository/revision | Why it is used |
| --- | --- | --- |
| ComfyUI core | `Comfy-Org/ComfyUI@master` (current research snapshot) | Native `comfy_extras.nodes_minimax_h3` and video nodes |
| MaskVidExperiments | `drozbay/MaskVidExperiments@c32ed8c1…` | `MVEx_MaskToLatentSpace` used by H3 workflows |
| ComfyUI-Florence2 | `kijai/ComfyUI-Florence2@9ece3de9…` | `Florence2ModelLoader`/`Florence2Run` AutoPrompt branch |
| ComfyUI-SAM3 | `PozzettiAndrea/ComfyUI-SAM3@de0ff5d2…` | Native SAM3 model patcher and video tracker adapter |

The installer verifies an existing checkout's normalized `origin` URL and exact
`HEAD` before treating it as installed. It never pulls an unpinned branch into
an existing directory. New checkouts are cloned into a temporary sibling,
fetched at the requested revision and detached before being moved into place.

`requirements-*.txt` intentionally omit torch, torchvision, torchaudio,
xformers and CUDA wheels. `scripts/setup.py` first runs pip with
`--dry-run --report`, rejects any proposed protected-package change, and only
then invokes the normal resolver (without `--upgrade`). A working ComfyUI
runtime is not silently replaced while transitive CPU dependencies are still
installed.
If a profile reports an incompatible/missing torch or native package, install a
matching wheel manually in the environment you selected and rerun `check`.

* `requirements-editor.txt`: numpy, Pillow and OpenCV headless for CPU mask
  calculations. It uses one OpenCV distribution only.
* `requirements-sam3.txt`: public SAM3 wrapper's non-torch dependencies. Its
  upstream optional `comfy-env`/pixi installer is deliberately not executed.
* `requirements-autoprompt.txt`: transformers, timm, Pillow, PEFT, accelerate,
  safetensors and matplotlib at the pinned Florence2-compatible ranges.
* `requirements-h3.txt`: `av` only; H3 model/runtime packages come from the
  selected ComfyUI build and are never upgraded here.

Compatibility is capability-based rather than a deployment claim. The local
runtime used for research contained ComfyUI H3 code and SAM3 assets, but this
public release does not bundle weights and does not claim a clean H3 render,
visual fidelity, or long-run GPU stability on arbitrary hardware.

## Upstream references

* [ComfyUI H3 node](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
* [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3)
* [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2)
* [MaskVidExperiments](https://github.com/drozbay/MaskVidExperiments)
* [Meta SAM3 code](https://github.com/facebookresearch/sam3)
