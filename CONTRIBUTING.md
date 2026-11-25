# Contributing

Keep changes focused, testable, and portable across ComfyUI installations.

Before opening a pull request:

1. Run `python -m pytest -q` and `node --test tests/*.test.mjs`.
2. Run `npx -y modern-web-guidance@latest legacy-scan web` when changing frontend code.
3. Run the direct-loader import smoke command in `docs/DEVELOPMENT.md`.
4. Ensure workflows contain placeholders rather than personal media, seed tokens, source fingerprints, host IDs, or local paths.
5. Do not commit model weights, acceptance media, source pixels, or private deployment files.

Preserve the `THSUniversalMaskEditor` node class ID and the root `WEB_DIRECTORY = "./web"` contract. Changes touching SAM3 integration must keep external dependencies and licensing boundaries clear.
