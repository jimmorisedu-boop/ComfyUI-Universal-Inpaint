# Mask editor layout

The editor uses one live DOM tree in both the Comfy node and the native dialog.
Only the workbench wrapper moves. The canvas, timeline, settings and calculation
actions retain their relationships and event handlers across fullscreen cycles.

The canvas and timeline occupy the main column. Drawing tools sit beside the
canvas. The inspector groups selection, mask boundaries, brush controls and the
tracking range. Its scrollable settings area is separate from the fixed action
area, so calculation buttons remain visible on desktop. Below 760px of available
component width, the inspector moves below the viewer and the dialog body scrolls.

## Design guidance used

- **stageshow**, `references/craft.md`: “Simplify before polishing.” The design
  removes the full-width strips of unrelated controls, uses consistent spacing
  and keeps the actual video as the focal content. This is a functional editor,
  so illustration-only skeletons, staged cursor animation and fake UI components
  are not part of the implementation.
- **modern-web-guidance / responsive-design**: “Use `@container` queries”. The
  inline node and fullscreen dialog establish their own query containers. A node
  resized inside a large Comfy viewport therefore still adapts to its own width.
- **modern-web-guidance / dialog**: “Use the native `<dialog>` element”. The
  existing modal retains `showModal()`, receives an accessible name and scopes
  display styling to its open state. Scroll containment is on the modal/backdrop;
  there is no global body overflow override to shift the graph on opening.
- **modern-web-guidance / css**: “No global resets”. Layout and control rules
  stay under the editor root and reuse ComfyUI theme variables. Rebuilt controls
  use CSS classes instead of fixed inline geometry.

## Verification

- Existing eight frontend test suites pass, including SAM3 actions, mask
  persistence, source recovery, navigation and diagnostics.
- A real Chrome rendering check used frames from a local 119-frame portrait
  video at 1920×900, 1366×768, 900×600 and 390×844, with no horizontal dialog
  overflow or JavaScript errors.
- Browser interaction checks covered wheel zoom, middle-button panning, exact
  frame navigation, three fullscreen cycles retaining edits, and brush/cursor
  alignment after returning to an inline node transformed to 75% scale.
- These are editor acceptance checks. No new SAM3 inference or H3 render was
  required for this layout change. Private frames and captures are not shipped.
