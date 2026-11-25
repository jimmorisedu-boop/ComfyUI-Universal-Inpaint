# Live verification notes

Live verification history for the `THSUniversalMaskEditor` (`Video Mask Editor`) node across ComfyUI and automated harnesses:

## 0.2.0 — Standalone mask editor, Pen tool & hover-only UI

- **Node-only scope**: Verified `/object_info/THSUniversalMaskEditor` exposes `THSUniversalMaskEditor` (`CATEGORY = "Universal Mask Editor"`, `RETURN_TYPES = ("MASK", "IMAGE")`) with zero legacy modules or labels.
- **Hover-only tooltips & zero layout shift**: Measured `getBoundingClientRect()` across `.ths-canvas-wrap`, `.ths-mask-tools`, `.ths-mask-viewgroup`, `.ths-mask-inspector`, and `.ths-mask-export` in live ComfyUI at 81% graph zoom while hovering `Перо`, disabled `Запустить граф`, `?`, and `ⓘ`, and switching view tabs (`Наложение` / `Источник` / `Маска`) and tools — observed `0px` layout shift (`dx = 0, dy = 0, dw = 0, dh = 0`) and zero runtime JS errors.
- **Bezier Pen & regression coverage**: Verified straight corner contours, cubic Bezier handle dragging, snap-to-close on first anchor, `Backspace` anchor pop, `Escape` cancellation, tooltip timings (`650ms` initial / `180ms` warm switch), `modern-web-guidance legacy-scan web` (`0 findings`), `pytest -q`, and `node --test tests/*.test.mjs`.

## Core tracking & interaction milestones

- **SAM3 multi-object & right-click removal**: Verified independent object tracking on a 119-frame 1080×1920 clip with `+` / `−` point refinements, right-click point/object deletion, and manual stroke retention across all tracked frames.
- **Two-step SAM3 caching & warm clicks**: Verified single-frame SAM3 preview updates over cached frames in ~2.0–2.5s warm without triggering full-video tracking or downstream graph execution.
- **Fullscreen `<dialog>` modal & coordinate accuracy**: Verified cursor precision across ComfyUI graph zoom levels and seamless state retention when toggling inline vs fullscreen views.
