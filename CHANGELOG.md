# Changelog

## Unreleased — V70 repair workbench

- Route text, photo reference and corrected-frame edits through one V70 graph.
- Offer context and neutral-gray preparation for supported edits. Preserve source shading for direct recolor commands.
- Document all three modes and the fill selector in the illustrated Russian guide.

### SAM3 point selection

- Click an object for a single-frame SAM3 preview, with positive and negative
  point tools for refinement. Video tracking and H3 remain explicit actions.
- Reuse the loaded source frames on each SAM3 preview. The scoped queue job
  returns the selected frame's mask and brush baseline, and ComfyUI manages
  reuse and eviction of the SAM3 model between clicks.
- Keep point prompts through saving and history, show their markers on the
  frame, and retain brush corrections when selecting again or tracking.
- Copying manual corrections retains the original verified point seed.
- Added point interaction regression coverage and real-video acceptance notes.

## 0.1.2 — SAM3 undo/redo readiness

- Fixed undo/redo of brush corrections unnecessarily discarding the approved
  SAM3 seed and disabling Track video. A verified seed is retained only when
  the source, seed frame and object prompt still match.
- Historical client tokens no longer replace the live verified backend token.
  Changing or clearing the object prompt still requires a new frame calculation.
- Added regression coverage for correction history, historical token promotion
  and refusing to resurrect revoked seed readiness.

## 0.1.1 — mask editor layout

- Grouped selection, mask settings, tracking range and frame edits in a side
  inspector, with calculation actions fixed below its scrollable settings.
- Placed drawing tools beside the frame, view controls above it, and video
  navigation below it. Inline and fullscreen editing share the same layout.
- Kept ComfyUI theme colors, all mask modes, Grow/Feather and the separate
  SAM3 frame-calculation and video-tracking steps.
- Added component-width responsive layout and a visible next calculation action.
- Verified brush/cursor alignment after fullscreen at scaled node size, wheel
  zoom, middle-button panning, navigation and edit persistence in Chrome using
  real local video frames. No new SAM3 inference or H3 render is claimed.
- Updated the Russian user guide and documented the layout decisions.

## 0.1.0 — initial public checkout

- Published the direct ComfyUI custom-node layout for Universal Inpaint.
- Included the current mask editor with distinct Optical Flow and two-step
  native SAM3 paths.
- Included mask-only, manual H3, AutoPrompt, reference/corrected-frame and
  Repair Workbench workflows with portable placeholders.
- Added CPU Python contracts, frontend MJS suites, direct-loader smoke guidance
  and CI configuration.
- Excluded private media, acceptance captures, host paths, model weights and
  deployment/installer internals.
