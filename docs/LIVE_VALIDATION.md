# Live ComfyUI verification — 2026-11-18

Version 0.1.2, editor build `20261118-r11b-sam3-history`.

The installed extension was exercised in a separate Chrome session running the
actual ComfyUI frontend and local backend. Calculations were submitted through
the editor's buttons. No mocked responses or standalone preview harness were
used for this acceptance run. The test workflow contained video loading,
component extraction and the mask editor; H3 generation was not submitted.

The source was one real portrait video: 119 frames, 1080×1920, 30 FPS. Private
media, masks, screenshots, source fingerprints and prompt IDs are not published.

## Results

- **SAM3:** drawing the object rectangle did not submit inference. Calculate
  frame produced one non-empty mask; Track video then produced masks for all
  119 frames. H3 stayed disabled until tracking completed. A new native track
  with a different direction key confirmed `cache_hit=false` and
  `native_tracker_add_new_mask`, without an Optical Flow fallback.
- **SAM3 corrections:** after adding two brush marks and erasing an area on
  frame 60, fresh native tracking retained the edits. The other 118 preview
  masks were byte-identical. The corrected frame gained 1,463 selected pixels
  and lost 312; event geometry and inheritance matched the submitted document.
- **Optical Flow:** three manually painted strokes propagated through all 119
  frames. After adding and erasing on frame 60, another stroke and recalculation
  preserved the previous 731 added pixels and 315 erased pixels. The original
  anchor stayed identical; the corrected frame changed only around the new
  stroke. Flow corrections can affect subsequent frames as tracking anchors.
- **Canvas:** wheel zoom, middle-button panning, fullscreen brush placement,
  and brush/eraser placement after returning to the real graph at 75% zoom
  passed. Cursor centres were within 1.5 CSS pixels of the requested pointer
  positions. Three fullscreen cycles retained every edit.
- **Controls:** direct frame entry, playback/pause, last-frame navigation,
  source/mask/overlay views, mode switching, reset and Grow/Feather persistence
  worked. No JavaScript errors were recorded in the verification session.

Pixel comparisons use the actual 360×640 preview masks for every frame; visual
inspection sampled frames 1, 30, 60, 90 and 119. This is evidence for the tested
clip and interaction paths, not a claim about tracking quality for all footage,
occlusions, scene cuts or H3 generation.

## Bug found and fixed

Undo/redo of a brush correction previously discarded the approved SAM3 seed
proof and disabled Track video. Version 0.1.2 retains the live verified seed
when the source, seed frame and rectangle prompts still match. It also replaces
historical client placeholder tokens with the still-valid backend token.

The full track remains invalid until recalculated. A changed/cleared seed or
source must still be recalculated; undo cannot resurrect a revoked proof.
Regression tests reproduced the old failure and passed after the fix, and the
same undo/redo-plus-retracking sequence then passed in ComfyUI.
