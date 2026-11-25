# Model registry

`config/models.json` lists optional model weights needed outside CPU editing. The registry contains no bundled weights or tokens.

| Profile | Hub revision | Destination | Access |
| --- | --- | --- | --- |
| `sam3` | `facebook/sam3@3c879f39826c…` | `models/sam3/sam3.safetensors` | Gated; requires manual license acceptance |

CPU modes (`static`, `keyframes`, `optical_flow`) and manual rectangle refinement (`grabcut`) do not require downloading any Hugging Face models.

SAM3 download is strictly opt-in: `scripts/setup.py download --profile sam3 --allow-gated --apply` requires accepting the Meta SAM 3 terms manually and providing an authenticated `HF_TOKEN` or `hf auth login` session. Downloads resume via `.part` files inside `ComfyUI/models/.universal-inpaint-cache/` and never overwrite existing matching files.
