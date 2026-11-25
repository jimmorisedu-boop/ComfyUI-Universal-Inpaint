# Development

This repository is a direct ComfyUI custom-node package for `THSUniversalMaskEditor`. Keep `__init__.py`, `mask_*.py`, and `web/` at the repository root so ComfyUI loads the extension directly without a nested folder wrapper.

## Local checks

```powershell
python -m pytest -q
node --test tests/*.test.mjs
npx -y modern-web-guidance@latest legacy-scan web
```

The direct-loader smoke test mirrors ComfyUI's custom-node import contract:

```powershell
python -c "import importlib.util,sys; root='.'; spec=importlib.util.spec_from_file_location('ths_universal_inpaint','__init__.py',submodule_search_locations=[root]); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); assert 'THSUniversalMaskEditor' in module.NODE_CLASS_MAPPINGS and module.WEB_DIRECTORY == './web'; print('node import: PASS')"
```

## Contracts

- Keep the `THSUniversalMaskEditor` class ID, `RETURN_TYPES = ("MASK", "IMAGE")`, and `WEB_DIRECTORY = "./web"` stable.
- SAM3 stays strictly two-step (`action: "frame"` seed proof followed by explicit `action: "track"`). Never silently substitute Optical Flow on a native SAM3 error.
- `editor_request: "sam3_frame_v1"` is transient to scoped single-frame preview prompts and must never be persisted into workflow `mask_data`.
- Workflow files in `workflows/` must contain portable placeholders (`REPLACE_WITH_INPUT_VIDEO.mp4`), not local paths, source fingerprints, or private media.
