"""Motion overlays for VLM video frames.

Bakes motion trail (ghosting) into the frames that a VLM will sample,
so the model can perceive inter-frame movement that would otherwise be
invisible at low FPS.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Motion trail constants
# ---------------------------------------------------------------------------
TRAIL_BASE_ALPHA: float = 0.40
"""Peak blending opacity for the nearest sub-frame."""

TRAIL_LOOKBACK: int = 40
"""Number of source frames before each VLM frame to blend."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_motion_overlays(
    frames: list[np.ndarray],
    vlm_indices: list[int],
    *,
    motion_trail: bool = False,
) -> list[np.ndarray]:
    """Apply motion overlays to VLM-sampled frames.

    Parameters
    ----------
    frames:
        All source video frames (H, W, 3) at the original FPS.
    vlm_indices:
        Indices into *frames* that the VLM will see (center-of-interval).
    motion_trail:
        Blend preceding sub-frames onto each VLM frame (ghosting effect).

    Returns
    -------
    list[np.ndarray]
        Processed frames — same length as *vlm_indices*.
    """
    if not motion_trail:
        return [frames[i].copy() for i in vlm_indices]

    result: list[np.ndarray] = []
    for k, vlm_idx in enumerate(vlm_indices):
        frame = frames[vlm_idx].copy()

        if motion_trail:
            start_idx = max(0, vlm_idx - TRAIL_LOOKBACK)
            sub_indices = list(range(start_idx, vlm_idx))
            if sub_indices:
                sub_frames = [frames[si] for si in sub_indices]
                frame = _blend_trail(frame, sub_frames)

        result.append(frame)
    return result


# ---------------------------------------------------------------------------
# Motion trail
# ---------------------------------------------------------------------------


def _blend_trail(
    base: np.ndarray,
    sub_frames: list[np.ndarray],
) -> np.ndarray:
    """Alpha-blend *sub_frames* onto *base* to create a motion ghost.

    ``sub_frames[0]`` is the oldest (faintest), ``sub_frames[-1]`` is nearest
    to *base* (brightest).  Blending is masked per-pixel so that only regions
    with actual content change show ghosting.
    """
    if not sub_frames:
        return base.copy()

    result = base.astype(np.float32)
    n = len(sub_frames)
    for j, sf in enumerate(sub_frames):
        alpha = TRAIL_BASE_ALPHA * ((j + 1) / n)
        sf_f = sf.astype(np.float32)
        diff = np.mean(np.abs(sf_f - result), axis=2, keepdims=True)
        pixel_alpha = np.clip(diff / 30.0, 0, 1) * alpha
        result = result * (1 - pixel_alpha) + sf_f * pixel_alpha
    return np.clip(result, 0, 255).astype(np.uint8)
