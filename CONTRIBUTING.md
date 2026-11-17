# Contributing

Thanks for helping improve Universal Inpaint. Keep changes small, testable and
portable across ComfyUI installations.

Before opening a pull request:

1. Run `python -m pytest -q` and `node --test tests/*.test.mjs`.
2. Run the direct-loader import smoke command in `docs/DEVELOPMENT.md`.
3. Check that workflows contain placeholders rather than personal media,
   prompts, seed tokens, source fingerprints, host IDs or local paths.
4. Do not add model weights, acceptance media, browser logs containing source
   pixels, or private deployment files.

Preserve `THS*` node IDs, the root `WEB_DIRECTORY = "./web"` contract, exact
outside-mask compositing and the mask-only no-H3 graph. Changes to H3, Florence2
or SAM3 integrations must document the external dependency and its licence
boundary; this repository does not redistribute those models.

Please describe the ComfyUI version, Python version, operating system and
whether the issue reproduces on CPU. Redact tokens, personal paths, source
media and prompts from logs before attaching them.
