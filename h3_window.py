"""Plan a single native-resolution model window on the original timeline."""
import json
import math
from dataclasses import dataclass

import numpy as np


def mask_extent(mask):
    """Half-open coordinates. Include even the faint outermost feather pixels."""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    active = []
    left, top = mask.shape[2], mask.shape[1]
    right = bottom = 0
    for index, frame in enumerate(mask):
        pixels = frame > 0
        rows = np.flatnonzero(pixels.any(axis=1))
        if not rows.size:
            continue
        cols = np.flatnonzero(pixels.any(axis=0))
        active.append(index)
        left, top = min(left, int(cols[0])), min(top, int(rows[0]))
        right, bottom = max(right, int(cols[-1]) + 1), max(bottom, int(rows[-1]) + 1)
    return {"box": [left, top, right, bottom], "frames": active} if active else None


def _axis(lo, hi, limit):
    size = min(limit, max(256, hi - lo))
    start = max(0, min(limit - size, (lo + hi - size) // 2))
    return start, start + size


@dataclass(frozen=True)
class WorkWindow:
    shape: tuple
    box: tuple
    first: int
    stop: int
    edit_first: int
    edit_stop: int
    trial: bool

    def crop(self, batch):
        x, y, right, bottom = self.box
        return batch[:, y:bottom, x:right]


def plan_window(shape, extent, settings="{}", probe_frame=-1):
    count, height, width = map(int, shape)
    if not extent:
        raise ValueError("Для расчёта H3 нужна непустая маска.")
    opts = json.loads(settings)
    mode = opts.get("mode", "auto")
    margin = int(opts.get("margin", 64))
    if mode not in ("auto", "full", "manual") or not 0 <= margin <= 4096:
        raise ValueError("Некорректные настройки области H3.")
    x, y, r, b = extent["box"]
    left, top, right, bottom = x-margin, y-margin, r+margin, b+margin
    if mode == "manual":
        box = opts.get("box")
        if isinstance(box, list) and len(box) == 4 and opts.get("size") == [width, height]:
            u, v, s, t = map(int, box)
            left, top, right, bottom = min(x,u), min(y,v), max(r,s), max(b,t)
    left, right = _axis(max(0,left), min(width,right), width)
    top, bottom = _axis(max(0,top), min(height,bottom), height)
    if mode == "full" or (mode == "auto" and (right-left)*(bottom-top) >= .85*width*height):
        left, top, right, bottom = 0, 0, width, height
    frames = extent["frames"]
    start, end = frames[0], frames[-1] + 1
    trial = probe_frame >= 0
    if trial:
        length = min(12, end - start)
        start = max(start, min(end - length, int(probe_frame) - length // 2))
        end = start + length
        if not any(start <= frame < end for frame in frames):
            raise ValueError("В пробном фрагменте нет маски. Выберите кадр с выделением.")
    return WorkWindow((count,height,width), (left,top,right,bottom), max(0,start-4),
                      min(count,end+4), start, end, trial)


class THSUniversalH3Window:
    CATEGORY = "Universal Inpaint/Frames"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "THS_H3_WINDOW")
    RETURN_NAMES = ("images", "mask", "audio", "work_window")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE",), "mask": ("MASK",),
                         "settings": ("STRING", {"default": "{}"})},
            "optional": {"audio": ("AUDIO",), "fps": ("FLOAT", {"forceInput": True}),
                         "probe_frame": ("INT", {"forceInput": True, "min": -1, "default": -1})},
        }

    def run(self, images, mask, settings="{}", audio=None, fps=24., probe_frame=-1):
        if tuple(images.shape[:3]) != tuple(mask.shape):
            raise ValueError("H3 window requires the source and mask on the same canvas and timeline.")
        window = plan_window(mask.shape, mask_extent(mask), settings, probe_frame)
        span = slice(window.first, window.stop)
        if audio is not None:
            rate = int(audio["sample_rate"])
            if rate <= 0 or not math.isfinite(fps) or fps <= 0:
                raise ValueError("Invalid source audio sample rate or FPS.")
            start, stop = (round(frame * rate / fps) for frame in (window.first, window.stop))
            audio = {**audio, "waveform": audio["waveform"][..., start:stop]}
        return window.crop(images[span]), window.crop(mask[span]), audio, window
