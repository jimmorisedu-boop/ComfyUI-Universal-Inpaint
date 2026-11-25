# Changelog

## 0.2.0 — Standalone Video Mask Editor & Hover-Only Zero-Shift UI

- Scoped the repository and ComfyUI package exclusively to `THSUniversalMaskEditor` (`Video Mask Editor`).
- Added the **Bezier Pen tool (`Перо`)**: sharp corner clicks, symmetric cubic Bezier handle dragging, live rubber-band curve and anchor/handle preview, snap-to-close on anchor 0, `Backspace`/`Delete` undo-anchor, `Enter`/double-click completion, and lossless serialization into normalized mask polygons.
- Upgraded UI interactions to **hover-only Top-Layer tooltips** (`popover="hint"`, `650ms` initial hover delay / `180ms` warm transition) with `@supports (position-area: block-start)` placement, `@supports (container-type: anchored)` directional arrows, `.ths-tip-wrap` coverage for disabled action buttons, and header/section info triggers (`?` and `ⓘ`).
- Removed permanent inline hint banners, mode helper paragraphs, queue captions, and `<details>` accordions from normal layout flow and stabilized segmented controls, tool rails, inspector scrollbars (`scrollbar-gutter: stable`), and numeric readouts (`tabular-nums`) for `0px` layout shift across tool/tab switches and hovers.
- Upgraded core web architecture (`web/universal_mask_editor.js`) with CSS Anchor Positioning pill/rail indicators, IME-safe `Enter` handling, native `<dialog>` top-layer transitions, ARIA toolbars (`focusgroup` + roving tabindex), native range custom properties, `@container` responsive stacking, and `content-visibility: auto` offscreen playback pausing.
- Shipped clean standalone mask workflow [`workflows/Universal_Video_Mask_Editor.json`](workflows/Universal_Video_Mask_Editor.json).

### SAM3 multi-object point & box segmentation

- Explicit two-step execution: click objects on a seed frame (up to 8 independent objects), refine with positive (`+`) / negative (`−`) points or right-click point/object removal, then run explicit range/direction tracking (`forward`, `backward`, `both`).
- Persistent manual brush/eraser/pen/rect overrides cleanly layered on top of SAM3, Optical Flow, Keyframes, or Static masks.

## 0.1.2 — SAM3 undo/redo readiness

- Fixed undo/redo of brush corrections unnecessarily discarding the approved SAM3 seed and disabling `Track video`.
- Prevented historical client tokens from replacing the verified backend seed token.

## 0.1.1 — Mask editor layout

- Grouped selection, mask settings, tracking range, and frame edits in a side inspector, with calculation actions pinned below the scrollable settings area.
- Shared one live workbench DOM tree between inline and fullscreen modal views.
