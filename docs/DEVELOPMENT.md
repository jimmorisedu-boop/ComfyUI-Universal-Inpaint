# Development

This repository is intentionally a direct ComfyUI custom-node checkout. Keep
`__init__.py`, the Python node modules, and `web/` at the repository root.
Moving them under a second `custom_nodes/` directory makes a clone silently
fail to load in ComfyUI.

## Local checks

Use a normal Python environment for the CPU contract suite:

```powershell
python -m pytest -q
node --test tests/*.test.mjs
```

The Python tests do not require a running ComfyUI server or GPU. One optional
test may skip when torch is not installed. The JavaScript tests use small DOM
fakes and do not submit jobs.

The direct-loader smoke test mirrors ComfyUI's custom-node import contract:

```powershell
python -c "import importlib.util,sys; root='.'; spec=importlib.util.spec_from_file_location('universal_inpaint_node','__init__.py',submodule_search_locations=[root]); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); assert module.NODE_CLASS_MAPPINGS and module.WEB_DIRECTORY == './web'; print('node import: PASS')"
```

## Graph and UI contracts

- Keep the `THS*` node class IDs stable; existing saved graphs depend on them.
- Keep `WEB_DIRECTORY = "./web"` and serve the current editor from the root
  `web/` directory.
- Preserve source/mask geometry and exact outside-mask compositing contracts.
- SAM3 remains two-step (frame seed, then range tracking). Do not add an
  implicit Optical Flow fallback to a native SAM3 error.
- Point selection and positive/negative corrections automatically request only
  a frame preview. Range tracking and H3 must still require separate clicks.
  Preserve `sam3.prompt` through serialization/history; changing a point
  invalidates the old frame proof. Brush corrections must not change the prompt.
- `editor_request: sam3_frame_v1` belongs only to the scoped queue request.
  Its response patches the selected mask and edit base over loaded source
  frames. Track and the final graph use normal node outputs and verify the
  seed against the canonical digest of the complete clip.
- Keep mask-only workflows free of H3 model nodes so they work without H3.
- New workflows must use placeholders, not personal media names, prompt text,
  mask history, seed tokens, source fingerprints, host IDs or local paths.

## Dependency boundaries

ComfyUI core, H3/Video nodes, Florence-2, SAM3 and their checkpoints are
external runtime components. Do not vendor model files or write downloaders in
the node package. Model setup belongs in the separately maintained installer
and model documentation (`scripts/setup.py`, `docs/INSTALL*.md`,
`docs/MODELS.md`).

## Pull requests

Add focused tests for changes in the same pull request. Do not include source
video, raw masks, prompts, debug logs containing media, model weights, machine
paths or private acceptance evidence. See `CONTRIBUTING.md` for the review
checklist.
