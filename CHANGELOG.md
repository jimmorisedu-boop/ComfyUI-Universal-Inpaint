# Changelog

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
