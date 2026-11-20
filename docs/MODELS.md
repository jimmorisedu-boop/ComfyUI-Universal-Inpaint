# Model registry

`config/models.json` is the source of truth. Every entry pins a Hugging Face
repository and immutable revision, a destination relative to the selected
ComfyUI root, and a byte size/checksum where the Hub exposed one. The manifest
contains no weights, user paths or tokens.

| Profile | Hub revision | Destination | Access |
| --- | --- | --- | --- |
| `instruction` | `Helsinki-NLP/opus-mt-ru-en@fbd6dc73284f…` | `models/LLM/opus-mt-ru-en/` | public, CC-BY-4.0 |
| `autoprompt` | `MiaoshouAI/Florence-2-base-PromptGen-v2.0@59b6e4bf…` | `models/LLM/Florence-2-base-PromptGen-v2.0/` | public, MIT model card |
| `sam3` | `facebook/sam3@3c879f39826c…` | `models/sam3/sam3.safetensors` | gated; manual licence/access |
| `h3` | `Comfy-Org/MiniMax-H3@4b53a351…` | `models/vae/`, `models/diffusion_models/`, `models/text_encoders/` | public Hub repo; review its `other` licence |

The H3 files selected for the shipped workflows are:

* `vae/minimax_h3_video_vae_fp16.safetensors` → `models/vae/minimax_h3_video_vae_fp16.safetensors` (5,207,808,496 bytes)
* `vae/minimax_h3_audio_vae_fp32.safetensors` → `models/vae/minimax_h3_audio_vae_fp32.safetensors` (605,254,808 bytes)
* `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` → `models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` (20,970,379,616 bytes)
* `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` → `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (15,687,142,551 bytes)

The Florence bundle includes the model and tokenizer/configuration files needed
by `Florence2ModelLoader`. Its pinned `model.safetensors` is 1,083,916,964
bytes at the manifest revision. Auxiliary files have pinned sizes and SHA-256 values, checked after download.

SAM3 access is intentionally two-step. `scripts/setup.py download --profile
sam3` refuses to download until `--allow-gated` is provided and an
`HF_TOKEN`/authenticated Hugging Face CLI session is present. The script never
accepts gated terms on a user's behalf. A 401/403 is reported as an access
blocker, not silently replaced with another model.

All downloads are resumable from `.part` files, use a cache under the selected
ComfyUI `models/` directory, check free disk space before applying, and refuse
to overwrite a destination whose bytes do not match the manifest.
