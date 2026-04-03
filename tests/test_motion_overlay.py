"""Tests for p2p.inference.motion_overlay."""

from __future__ import annotations

import numpy as np

from p2p.inference.motion_overlay import apply_motion_overlays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 64, w: int = 64, value: int = 128) -> np.ndarray:
    """Solid-color (H, W, 3) uint8 frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _make_moving_frames(n: int = 20, h: int = 64, w: int = 64) -> list[np.ndarray]:
    """Generate frames with a white square moving rightward."""
    frames = []
    sq = 10
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        x = (i * 3) % (w - sq)
        f[20 : 20 + sq, x : x + sq] = 255
        frames.append(f)
    return frames


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoOverlay:
    def test_returns_unmodified_frames(self):
        frames = [_make_frame(value=v) for v in (100, 110, 120, 130, 140)]
        indices = [1, 3]
        result = apply_motion_overlays(frames, indices, motion_trail=False)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], frames[1])
        np.testing.assert_array_equal(result[1], frames[3])

    def test_returns_copies(self):
        """Returned frames must be copies, not views of the originals."""
        frames = [_make_frame()]
        result = apply_motion_overlays(frames, [0], motion_trail=False)
        result[0][0, 0, 0] = 0
        assert frames[0][0, 0, 0] != 0


class TestMotionTrail:
    def test_output_shape_preserved(self):
        frames = _make_moving_frames(20, h=64, w=64)
        indices = [2, 5, 8, 11, 14, 17]
        result = apply_motion_overlays(frames, indices, motion_trail=True)
        assert len(result) == len(indices)
        for r in result:
            assert r.shape == (64, 64, 3)
            assert r.dtype == np.uint8

    def test_blending_changes_moving_pixels(self):
        """Trail should modify pixels that differ from the base frame."""
        frames = _make_moving_frames(20)
        indices = [5, 10, 15]
        original = [frames[i].copy() for i in indices]
        result = apply_motion_overlays(frames, indices, motion_trail=True)
        any_diff = False
        for k in range(1, len(result)):
            if not np.array_equal(result[k], original[k]):
                any_diff = True
                break
        assert any_diff, "Trail should produce visible blending on moving content"

    def test_static_scene_unchanged(self):
        """If all frames are identical, trail should leave them untouched."""
        frame = _make_frame(value=100)
        frames = [frame.copy() for _ in range(10)]
        indices = [2, 5, 8]
        result = apply_motion_overlays(frames, indices, motion_trail=True)
        for r in result:
            np.testing.assert_array_equal(r, frame)
