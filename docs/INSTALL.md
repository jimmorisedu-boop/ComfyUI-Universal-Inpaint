# Universal Inpaint 0.1.2 — portable installation

This repository is a ComfyUI custom-node package. The supported CPU starting
point is the editor profile; generation profiles are opt-in and never replace
the Python, PyTorch, CUDA, torchvision, torchaudio or xformers stack already
owned by ComfyUI.

## 1. Choose the runtime

Run these commands from your ComfyUI directory:

```powershell
cd custom_nodes
git clone https://github.com/jimmorisedu-boop/ComfyUI-Universal-Inpaint.git
cd ComfyUI-Universal-Inpaint
```



Point the commands at the directory containing `main.py` (or the `comfy/`
package). The installer detects a sibling Windows portable interpreter at
`python_embeded/python.exe`, a `.venv`, or accepts `--python` explicitly. It
does not scan arbitrary drives and does not use project-specific host paths.

```powershell
python scripts/setup.py check --comfy-root "<ComfyUI-root>" --json
python scripts/setup.py plan --comfy-root "<ComfyUI-root>" --profile editor --json
```

`plan` is read-only and is the default when no subcommand is supplied. Review
the JSON before allowing a mutation. Existing checkouts with a different Git
remote/revision, workflows, models, or add-on files are reported as conflicts;
they are never deleted or overwritten.

If the historical `custom_nodes/ths_universal_inpaint/` directory is present,
the plan reports a legacy duplicate and stops. Migrate it manually only after
taking your own ComfyUI backup; the installer never deletes it or silently runs
two copies of the same node IDs.

## 2. Install the editor

```powershell
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile editor --apply
```

The command copies this package to
`ComfyUI/custom_nodes/ComfyUI-Universal-Inpaint/` and installs the profile's
requirements only after a pip `--dry-run --report` guard proves that no
torch/CUDA/vision package would change. A package that is already identical is
a no-op. Restart ComfyUI yourself after reviewing the result.

The baseline can also be installed manually with the selected ComfyUI Python:

```powershell
python -m pip install -r requirements-editor.txt
```

## 3. Add optional integrations

Profiles are additive and can be combined by repeating `--profile`:

```powershell
python scripts/setup.py plan --comfy-root "<ComfyUI-root>" --profile editor --profile sam3
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile sam3 --apply
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile autoprompt --apply
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile h3 --apply
```

The installer clones only the pinned public custom-node revisions in
`config/dependencies.json`, using a temporary checkout and detached revision.
The SAM3 upstream repository has an optional `comfy-env`/pixi helper; it is not
run by this tool. The H3 profile uses native `comfy_extras.nodes_minimax_h3.py`
from a compatible ComfyUI build and does not install a private attention patch.

## 4. Download models (explicit action)

Model downloads are separate from `install` and are never implicit:

```powershell
python scripts/setup.py download --comfy-root "<ComfyUI-root>" --profile h3
python scripts/setup.py download --comfy-root "<ComfyUI-root>" --profile h3 --apply
python scripts/setup.py download --comfy-root "<ComfyUI-root>" --model florence2-promptgen-v2 --apply
```

The first command prints a disk-space and cache plan without network access.
The second resumes `.part` files in
`ComfyUI/models/.universal-inpaint-cache/`, verifies the pinned size/hash when
known, and atomically copies only to a new model destination. A complete
verified cache is reused. Existing model files are checked and never replaced.

For SAM3, first visit the model-card URL shown by the command and accept its
terms yourself. Then authenticate with `hf auth login` or an `HF_TOKEN`
environment variable and opt in explicitly:

```powershell
python scripts/setup.py download --comfy-root "<ComfyUI-root>" --profile sam3 --allow-gated --apply
```

The tool never accepts a licence, stores a token in the repository, or claims
that a gated download succeeded without an authenticated response. If access
is unavailable, use the editor/Optical Flow path or place a compatible local
SAM3 safetensors checkpoint manually at the destination reported by `plan`.

## 5. Verify without mutation

```powershell
python scripts/setup.py check --comfy-root "<ComfyUI-root>" --profile editor --profile sam3 --json
python -m pytest --import-mode=importlib -q tests/test_setup.py tests/test_model.py
```

`check` reports interpreter discovery, native H3 module presence and profile
metadata. It is a capability report, not visual or GPU acceptance. No clean H3
end-to-end render is claimed by this public 0.1.2 release.

For Windows portable without a `python` command, replace it with
`& "../../../python_embeded/python.exe"` from the checkout directory.
Install `editor` first, then add the optional profiles you need. After the pip
report passes validation, the complete resolved transaction (including transitive
dependencies) is installed from SHA-256-pinned archives. If SAM3 requires a
different Triton version, configure a compatible ComfyUI runtime separately.
