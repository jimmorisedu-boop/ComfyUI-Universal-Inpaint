# Live ComfyUI verification

## V70 fill selector — 2026-11-21 (Europe/Moscow)

The ComfyUI browser serialized all nine combinations of the three edit modes
and the three fill options. Old graphs loaded with their original pad-node
setting, including saves that serialized the upload button as a widget value.
The live preparation checks confirmed that no fill leaves source pixels intact
and that the photo remains connected to native H3 conditioning.

On a 119-frame, 1080×1920 source clip, static, keyframe, Optical Flow and
GrabCut mask jobs completed. Native SAM3 produced a box preview, an updated
point preview with positive and negative points, and tracked the clip across
all 119 frames. Direct recolor exported 119 frames at 30 fps with audio.

H3 ran 12-frame trials for text removal and reference replacement with each
fill option, a detail edit, a text replacement, and a corrected frame.
Context fill removed the original tie. The photo replacement yielded a flower
on the shirt both with context fill and with no fill at denoise 0.85. Gray fill
left a visible tie silhouette in these trials. A red tie on corrected frame 3
carried through all 12 trial frames at denoise 1.0. At 0.65 the color faded away
after the anchor. A text edit produced silver stripes. The bow-tie replacement
retained the original shape with the short instruction. A mask widened at the
collar and a descriptive prompt at denoise 1.0 produced a shorter bow shape,
with the wrong orientation. For an exact silhouette, use an image guide.

## V70 model window and short H3 trials

The editor's trial and full-run buttons were exercised in a local ComfyUI browser on a 56-frame, 544×576, 24 fps clip. The marked interval was frames 17–40. An automatic 256×256 window covered the mask. The trial showed frames 21–32 in the comparison node, and the frame numbers matched their positions in the original timeline. The instruction route then saved a 56-frame H.264 file at the source dimensions and frame rate. All 32 outside-interval frame pairs matched in the comparison cache. Separate reference-image and corrected-frame trials also completed for the same interval, with the corrected guide on original frame 27. Unit tests cover the cropped audio sample offset; this particular source clip had no audio track.

## Russian edit instructions — 2026-11-19 (Europe/Moscow)

The V70 workbench was loaded in the actual local ComfyUI frontend with a
previously frozen manual prompt. Entering a Russian task restored the link
from the AutoPrompt node to the H3 conditioner's prompt input. A partial job
translated the task with local OPUS-MT and returned an English edit prompt
in the browser with no JavaScript errors. The partial job used small native
source/mask nodes and verified prompt generation and wiring without running
video generation. The Python and frontend suites passed.

## SAM3 groups and right-click removal — 2026-11-19 (Europe/Moscow)

The local 119-frame video was tested with hair and shirt as separate SAM3
objects. The second click added 10,197 preview pixels while retaining all
3,493 hair pixels. A negative point changed 671 shirt mask pixels and left
the hair mask identical. Native tracking returned masks across all 119
frames. Both objects were visible in sampled frames 1, 30, 60, 90 and 119,
and a manual brush mark stayed on frame 1.

In build `20261119-r14b-sam3-context`, right-clicking the negative marker
returned the exact two-object mask from before that marker was added.
Right-clicking the shirt away from its marker returned the exact hair-only
mask even with hair selected in the object list. Right-clicking the last
remaining positive marker cleared the final object and disabled Track video.

## Faster SAM3 preview — 2026-11-19 (Europe/Moscow)

Editor build `20261119-r13-sam3-preview` was tested in the actual local
ComfyUI frontend on the same 119-frame, 1080×1920 clip. With source frames
loaded, click-to-overlay time was 6.02 seconds with initial model loading.
Warm positive, negative and replacement clicks took 2.07, 2.48 and 2.03
seconds respectively. The earlier jobs took 34.45–36.43 seconds.

The three shirt masks were pixel-identical to the earlier previews. The
hair region differed by three preview pixels (99.91% intersection over union).
The separate Track video action produced 119 nonempty native SAM3 masks.
The seed frame retained a brush mark, and the mark was absent on frame 60.
The preview jobs contained only the three source and mask nodes, and the
browser test recorded zero JavaScript errors. All 114 Python tests and nine
frontend suites passed.

## SAM3 point selection — 2026-11-19 (Europe/Moscow)

Editor build `20261119-r12-sam3-points` was installed with served-file hash
verification. The same actual ComfyUI frontend/backend and 119-frame,
1080×1920, 30 FPS video were used for these additional checks:

- Clicking the shirt produced an overlaid object mask on frame 1 only.
  Positive and negative points submitted new single-frame previews; their
  labels and coordinates reached the native predictor. Tracking stayed off.
- Selecting hair with a new click replaced the point prompt. A deliberate
  brush mark remained in the document and rendered preview.
- Explicit Track video produced 119 nonempty masks using native SAM3
  (`cache_hit=false`, `native_tracker_add_new_mask`), in 80.2 seconds.
  The seed frame retained its brush mark; the mark was not a tracking prompt.
- After leaving fullscreen at 75% graph zoom, brush/cursor centres agreed
  within 0.01 CSS pixels. Point selection agreed with the requested source
  coordinate within 0.00001. Undo/redo preserved the brush event.
- No JavaScript errors were recorded. These jobs submitted only LoadVideo,
  GetVideoComponents and the mask editor; H3 was not run.

Single-frame previews took 34–37 seconds end-to-end on this setup, including
ComfyUI's current video/output preparation. This is a calculated preview,
not instant hover segmentation. Model quality still needs visual inspection;
adding a point can alter the whole predicted object boundary.

The first source-preview attempt exhausted available system commit memory.
With the queue idle, ComfyUI's model/execution cache was freed through its
normal API. The full-resolution retry and all subsequent jobs succeeded;
the video was not downscaled for this acceptance run. This is not a fix for
the existing full-batch preview memory cost.

Nine frontend regression suites and 37 focused Python mask/SAM3/node-contract
tests passed. The additional manual-key copy fix was regression-tested and
installed with HTTP hash verification; it does not change SAM3 inference.

## Earlier rectangle and layout acceptance

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
