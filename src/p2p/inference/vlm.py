"""VLM utilities for behavior judging — multi-provider dispatch.

Providers:
  - vLLM: Qwen3.5-27B — native video_url input
  - Anthropic: Claude — image-only via base64
  - Google: Gemini — native video (inline MP4 bytes) or image
  - Ollama (local): image-only via composite (legacy fallback)

Utilities:
  - create_composite(): START / PEAK ACTION / END labeled composite image
  - find_peak_frame(): Peak action frame via center-crop consecutive diffs
  - compute_video_motion(): Motion score for video ranking
  - extract_json(): Robust JSON extraction from LLM responses
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

from p2p.settings import LLM_MODEL, VLM_BASE_URL, VLM_MODEL

logger = logging.getLogger(__name__)

MAX_VLM_TOKENS = 4096
"""Default max output tokens for VLM calls (all providers)."""

VLM_FPS = 10
"""Target frame rate for VLM video sampling (Gemini/vLLM)."""


def _extract_first_frame_b64(video_path: Path) -> str | None:
    """Extract the first frame from a video as base64 JPEG."""
    try:
        import io

        import imageio.v3 as iio
        from PIL import Image

        frame = iio.imread(video_path, index=0)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _create_padded_video(video_path: Path, out_path: Path, target_fps: int = VLM_FPS) -> Path:
    """Prepend duplicate first frames so Gemini's center-of-interval
    sampling always includes the original first frame."""
    import math

    import imageio

    reader = imageio.get_reader(str(video_path))
    source_fps = reader.get_meta_data().get("fps", 30.0)
    frames = [f for f in reader]
    reader.close()

    if not frames:
        return video_path

    interval_size = source_fps / target_fps
    pad_count = math.floor((interval_size - 1) / 2)

    if pad_count <= 0:
        return video_path

    padded = [frames[0]] * pad_count + frames
    writer = imageio.get_writer(
        str(out_path), fps=source_fps, output_params=["-movflags", "+faststart"]
    )
    for f in padded:
        writer.append_data(f)
    writer.close()
    return out_path


def _emit_vlm_call(model: str, response: Any, *, duration_ms: int | None = None) -> None:
    """Emit a vlm.call event with token usage from a Gemini response."""
    from p2p.event_log import emit

    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    emit(
        "vlm.call",
        data={
            "model": model,
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "thinking_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
        },
        duration_ms=duration_ms,
    )


# Gemini thinking mappings — effort level to provider-specific config.
# Gemini 3.x uses thinking_level (qualitative); 2.5 uses thinking_budget (tokens).
# Gemini has no "max" level — map to "high" (its highest).
_EFFORT_TO_LEVEL = {
    "max": "high",
    "xhigh": "high",
    "high": "high",
    "medium": "medium",
    "low": "low",
}
_EFFORT_TO_BUDGET = {"max": 65536, "xhigh": 32768, "high": 32768, "medium": 16384, "low": 4096}


class VLMError(Exception):
    """Raised when a VLM API call fails."""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response text.

    Handles raw JSON, ```json blocks, and embedded JSON objects.
    """
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Strip markdown code fences and try the inner content
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Handle truncated ```json block (opening fence but no closing fence)
    fence_match = re.search(r"```(?:json)?\s*\n?(.*)", text, re.DOTALL)
    if fence_match:
        inner = fence_match.group(1).rstrip("`").strip()
        first_b = inner.find("{")
        last_b = inner.rfind("}")
        if first_b != -1 and last_b > first_b:
            try:
                return json.loads(inner[first_b : last_b + 1])
            except json.JSONDecodeError:
                pass

    # Find the first '{' and last '}' — works for arbitrarily nested JSON
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            pass

    msg = f"Could not extract JSON from text: {text[:200]}..."
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Video frame utilities
# ---------------------------------------------------------------------------


def _read_video_frames(video_path: Path) -> list[np.ndarray]:
    """Read all frames from a video file as numpy arrays (H, W, 3)."""
    try:
        import imageio.v3 as iio

        return list(iio.imread(str(video_path), plugin="pyav"))
    except Exception:
        try:
            import imageio

            reader = imageio.get_reader(str(video_path))
            frames = list(reader)
            reader.close()
            return frames
        except Exception:
            return []


def find_peak_frame(frames_raw: list[np.ndarray]) -> int:
    """Find the frame with peak action using consecutive center-crop diffs.

    Camera tracking causes uniform background shift, so we use center-crop
    diffs to isolate actual body motion.
    """
    if len(frames_raw) < 3:
        return len(frames_raw) // 2

    h, w = frames_raw[0].shape[:2]
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4

    diffs = []
    for i in range(1, len(frames_raw)):
        d = np.mean(
            np.abs(
                frames_raw[i][y1:y2, x1:x2].astype(float)
                - frames_raw[i - 1][y1:y2, x1:x2].astype(float)
            )
        )
        diffs.append(d)

    window = min(30, len(diffs) // 3)
    if window < 1:
        return max(range(len(diffs)), key=lambda i: diffs[i]) + 1

    max_sum = 0.0
    peak_center = len(diffs) // 2
    for i in range(len(diffs) - window):
        s = sum(diffs[i : i + window])
        if s > max_sum:
            max_sum = s
            peak_center = i + window // 2

    search_start = max(0, peak_center - window)
    search_end = min(len(diffs), peak_center + window)
    peak_idx = max(range(search_start, search_end), key=lambda i: diffs[i]) + 1
    return peak_idx


def create_composite(video_path: Path) -> str | None:
    """Create a labeled START / PEAK ACTION / END composite from a video.

    Returns base64-encoded JPEG string, or None if video can't be read.
    """
    from PIL import Image, ImageDraw

    frames_raw = _read_video_frames(video_path)
    if not frames_raw:
        return None

    peak_idx = find_peak_frame(frames_raw)
    img_start = Image.fromarray(frames_raw[0])
    img_peak = Image.fromarray(frames_raw[peak_idx])
    img_end = Image.fromarray(frames_raw[-1])

    w, h = img_start.size
    label_h = 35
    gap = 8
    canvas = Image.new("RGB", (w * 3 + gap * 2, h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    panels = [
        (img_start, "START"),
        (img_peak, "PEAK ACTION"),
        (img_end, "END"),
    ]
    for i, (img, label) in enumerate(panels):
        x = i * (w + gap)
        canvas.paste(img, (x, label_h))
        draw.rectangle([x, 0, x + w, label_h - 1], fill=(0, 0, 0))
        text_x = x + w // 2 - len(label) * 4
        draw.text((text_x, 8), label, fill=(255, 255, 255))

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def compute_video_motion(video_path: Path) -> float:
    """Compute a motion score for a video (higher = more dynamic).

    Used to select the most interesting eval episode for VLM judging.
    """
    frames_raw = _read_video_frames(video_path)
    if len(frames_raw) < 2:
        return 0.0

    h, w = frames_raw[0].shape[:2]
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4

    step = max(1, len(frames_raw) // 50)
    total_diff = 0.0
    count = 0
    for i in range(step, len(frames_raw), step):
        total_diff += np.mean(
            np.abs(
                frames_raw[i][y1:y2, x1:x2].astype(float)
                - frames_raw[i - step][y1:y2, x1:x2].astype(float)
            )
        )
        count += 1

    return total_diff / max(count, 1)


REASK_MAX_FPS: int = 30
"""Hard ceiling on FPS for reask_vlm tool calls."""

REASK_MAX_FRAMES: int = 100
"""Hard ceiling on total frames for reask_vlm tool calls."""


def _center_of_interval_indices(n_frames: int, source_fps: float, target_fps: float) -> list[int]:
    """Compute center-of-interval frame indices for downsampling.

    Divides the timeline into equal intervals and picks the center frame
    of each.  For source_fps=30, target_fps=5 (interval_size=6), selected
    indices are 2, 8, 14, 20, ... (0-based center of each 6-frame interval).
    """
    duration = n_frames / source_fps
    n_samples = max(1, int(duration * target_fps))
    interval_size = source_fps / target_fps
    indices: list[int] = []
    for i in range(n_samples):
        idx = int(i * interval_size + (interval_size - 1) / 2)
        indices.append(min(idx, n_frames - 1))
    return indices


def _write_sampled_video(
    frames: list[np.ndarray],
    indices: list[int],
    out_path: Path,
    fps: int,
) -> Path | None:
    """Write selected frames to a video file. Returns *out_path* or None on error."""
    import imageio

    try:
        with imageio.get_writer(
            str(out_path), fps=fps, output_params=["-movflags", "+faststart"]
        ) as writer:
            for idx in indices:
                writer.append_data(frames[idx])
        return out_path
    except Exception:
        logger.warning("Failed to write video %s", out_path, exc_info=True)
        return None


def _read_video_meta(video_path: Path) -> tuple[float, int] | None:
    """Return ``(fps, n_frames)`` from video metadata, or ``None`` on failure.

    Opens the container once and guarantees the reader is closed even on error.
    """
    try:
        import imageio

        reader = imageio.get_reader(str(video_path))
        try:
            meta = reader.get_meta_data()
            fps = float(meta.get("fps", 0) or 0)
            n_frames = meta.get("nframes")
            if n_frames is None or n_frames == float("inf"):
                n_frames = reader.count_frames()
            n_frames = int(n_frames)
        finally:
            reader.close()
        if fps > 0 and n_frames > 0:
            return fps, n_frames
    except Exception:
        logger.debug("Could not read video meta for %s", video_path, exc_info=True)
    return None


def _get_video_fps(video_path: Path) -> float:
    """Read the FPS from a video file's metadata."""
    result = _read_video_meta(video_path)
    return result[0] if result else 30.0


def get_video_duration(video_path: Path) -> float | None:
    """Return video duration in seconds from container metadata.

    Usually metadata-only; falls back to ``count_frames()`` when the
    container header omits a frame count.
    """
    result = _read_video_meta(video_path)
    return result[1] / result[0] if result else None


def extract_video_segment(
    video_path: Path,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    target_fps: int | None = None,
) -> Path | None:
    """Extract a time-windowed video segment resampled at *target_fps*.

    If no parameters differ from the defaults (full video at ``VLM_FPS``),
    returns ``None`` so the caller can use the original video as-is.

    Guardrails:
    - *target_fps* is clamped to ``REASK_MAX_FPS`` and the source FPS.
    - Total frame count is capped at ``REASK_MAX_FRAMES``; if exceeded,
      *target_fps* is reduced and a warning is logged.
    - Time values are clamped to [0, video_duration]; swapped if inverted.
    """
    has_time_range = start_time is not None or end_time is not None
    has_custom_fps = target_fps is not None and target_fps != VLM_FPS

    if not has_time_range and not has_custom_fps:
        return None  # no change from e2e defaults

    frames = _read_video_frames(video_path)
    if not frames:
        return None

    source_fps = _get_video_fps(video_path)
    duration = len(frames) / source_fps

    # Clamp and normalize time range
    t_start = max(0.0, start_time) if start_time is not None else 0.0
    t_end = min(duration, end_time) if end_time is not None else duration
    if t_end < t_start:
        t_start, t_end = t_end, t_start

    # Frame index range
    frame_start = int(t_start * source_fps)
    frame_end = min(int(t_end * source_fps), len(frames))
    segment_frames = frames[frame_start:frame_end]
    if not segment_frames:
        return None

    # Clamp FPS: min(requested, source, hard cap)
    effective_fps = target_fps if target_fps is not None else VLM_FPS
    effective_fps = max(1, min(effective_fps, int(source_fps), REASK_MAX_FPS))

    # Enforce max frame count by reducing FPS if needed
    seg_duration = len(segment_frames) / source_fps
    n_samples = max(1, int(seg_duration * effective_fps))
    if n_samples > REASK_MAX_FRAMES:
        effective_fps = max(1, int(REASK_MAX_FRAMES / seg_duration))
        logger.warning(
            "reask_vlm: clamped FPS to %d (max %d frames) for %.1fs segment",
            effective_fps,
            REASK_MAX_FRAMES,
            seg_duration,
        )

    indices = _center_of_interval_indices(len(segment_frames), source_fps, effective_fps)

    suffix = f"_reask_{t_start:.1f}_{t_end:.1f}_{effective_fps}fps"
    out_path = video_path.with_stem(video_path.stem + suffix)
    return _write_sampled_video(segment_frames, indices, out_path, effective_fps)


def _render_overlay_preview(
    video_path: Path,
    *,
    suffix: str,
    motion_trail: bool,
    target_fps: int = VLM_FPS,
) -> Path | None:
    """Read video, apply overlays, write to ``{stem}{suffix}.mp4``.

    Shared core for :func:`save_vlm_preview`.
    Returns the output path, or ``None`` on failure.
    """
    from p2p.inference.motion_overlay import apply_motion_overlays

    frames = _read_video_frames(video_path)
    if not frames:
        return None

    source_fps = _get_video_fps(video_path)
    indices = _center_of_interval_indices(len(frames), source_fps, target_fps)

    processed = apply_motion_overlays(
        frames,
        indices,
        motion_trail=motion_trail,
    )
    out_path = video_path.with_stem(video_path.stem + suffix)
    return _write_sampled_video(processed, list(range(len(processed))), out_path, target_fps)


def save_vlm_preview(video_path: Path, *, target_fps: int = VLM_FPS) -> Path | None:
    """Save a VLM-perspective preview video using center-of-interval sampling.

    Returns the output path, or None if no downsampling is needed or on failure.
    """
    frames = _read_video_frames(video_path)
    if not frames:
        return None

    source_fps = _get_video_fps(video_path)
    if source_fps <= target_fps:
        return None

    indices = _center_of_interval_indices(len(frames), source_fps, target_fps)

    preview_path = video_path.with_stem(video_path.stem + "_vlm")
    return _write_sampled_video(frames, indices, preview_path, target_fps)


# ---------------------------------------------------------------------------
# Motion overlay helpers
# ---------------------------------------------------------------------------


def _load_seg_masks(video_path: Path) -> list[np.ndarray] | None:
    """Load segmentation masks saved alongside a video, or ``None``."""
    seg_path = video_path.with_suffix(".seg.npz")
    if not seg_path.exists():
        return None
    try:
        data = np.load(str(seg_path))
        masks_3d = data["masks"]  # (N, H, W) uint8
        return [masks_3d[i] for i in range(masks_3d.shape[0])]
    except Exception:
        logger.debug("Could not load seg masks from %s", seg_path, exc_info=True)
        return None


def _dilate_masks(masks: list[np.ndarray], dilate_px: int = 75) -> list[np.ndarray]:
    """Dilate binary masks to cover motion halo around robot limbs."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    return [(cv2.dilate(m * 255, kernel, iterations=2) > 0).astype(np.uint8) for m in masks]


def _apply_motion_overlay_to_video(
    video_path: Path,
    *,
    motion_trail: bool = False,
) -> tuple[Path, Path | None]:
    """Apply motion overlays and return ``(effective_video, overlay_path)``.

    Reads *video_path*, applies overlays to the center-of-interval sampled
    frames, and writes a new video at ``VLM_FPS``.  Returns the overlay path
    as second element so the caller can clean it up.

    If overlay creation fails for any reason the original *video_path* is
    returned unchanged and *overlay_path* is ``None``.
    """
    try:
        from p2p.inference.motion_overlay import apply_motion_overlays

        all_frames = _read_video_frames(video_path)
        if not all_frames:
            return video_path, None

        source_fps = _get_video_fps(video_path)
        vlm_indices = _center_of_interval_indices(len(all_frames), source_fps, VLM_FPS)

        processed = apply_motion_overlays(
            all_frames,
            vlm_indices,
            motion_trail=motion_trail,
        )
        overlay_path = video_path.with_stem(video_path.stem + "_motion")
        written = _write_sampled_video(
            processed, list(range(len(processed))), overlay_path, VLM_FPS
        )
        if written:
            return overlay_path, overlay_path
    except Exception:
        logger.warning(
            "Motion overlay failed for %s, using original video",
            video_path,
            exc_info=True,
        )
    return video_path, None


# ---------------------------------------------------------------------------
# VLM API
# ---------------------------------------------------------------------------


def call_vlm(
    prompt: str,
    images_b64: list[str] | None = None,
    *,
    model: str = VLM_MODEL,
    base_url: str = VLM_BASE_URL,
    max_tokens: int = MAX_VLM_TOKENS,
) -> str:
    """Call local VLM via Ollama native API.

    Uses think=false because Qwen3.5's chain-of-thought exhausts the token
    budget before producing the final answer.

    Returns response text.

    Raises:
        VLMError: If the API call fails.
    """
    import requests

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }
    if images_b64:
        payload["messages"][0]["images"] = images_b64

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        logger.warning("VLM call failed: %s", e)
        raise VLMError(str(e)) from e


def _is_remote_host(host: str) -> bool:
    """Return True if the host is not the local machine."""
    return host not in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


_VLLM_EXTRA_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "mm_processor_kwargs": {"fps": VLM_FPS, "do_sample_frames": True},
}
"""Shared vLLM extra_body: thinking disabled + video sampling config."""


def _build_vllm_media_content(
    video_path: Path | None,
    images_b64: list[str] | None,
    remote: bool,
) -> list[dict]:
    """Build vLLM media content blocks for video or image input.

    Shared by single-turn and two-turn vLLM functions.
    """
    content: list[dict] = []
    if video_path and video_path.exists():
        if remote:
            video_bytes = video_path.read_bytes()
            video_b64 = base64.b64encode(video_bytes).decode()
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                }
            )
        else:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": f"file://{video_path.resolve()}"},
                }
            )
    elif images_b64:
        for img in images_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                }
            )
    return content


def call_vlm_vllm(
    prompt: str,
    video_path: Path | None = None,
    images_b64: list[str] | None = None,
    *,
    model: str = "Qwen/Qwen3.5-27B",
    host: str = "0.0.0.0",
    port: int = 8100,
    max_tokens: int = MAX_VLM_TOKENS,
) -> str:
    """Call vLLM's OpenAI-compatible API with native video or image input.

    When *video_path* is provided:
      - Local server: sends ``file://`` video_url (vLLM reads from disk).
      - Remote server: reads video bytes, sends as ``data:video/mp4;base64,...``
        so the remote server doesn't need filesystem access.

    Falls back to base64 image_url if no video.

    Returns response text.

    Raises:
        VLMError: If the API call fails.
    """
    from openai import OpenAI

    connect_host = "localhost" if host == "0.0.0.0" else host
    remote = _is_remote_host(host)
    try:
        client = OpenAI(base_url=f"http://{connect_host}:{port}/v1", api_key="unused")

        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend(_build_vllm_media_content(video_path, images_b64, remote))

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0.0,
            extra_body=_VLLM_EXTRA_BODY,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("VLM call failed: %s", e)
        raise VLMError(str(e)) from e


# ---------------------------------------------------------------------------
# Multi-provider VLM dispatch
# ---------------------------------------------------------------------------


def _anthropic_image_block(image_b64: str) -> dict:
    """Build an Anthropic image content block from base64 JPEG data."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_b64,
        },
    }


def call_vlm_anthropic(
    prompt: str,
    image_b64: str,
    *,
    model: str = LLM_MODEL,
) -> str:
    """Call Anthropic Claude Vision API with an image.

    Returns response text.

    Raises:
        VLMError: If the API call fails.
    """
    from p2p.inference.llm_client import create_message, extract_response_text, get_client

    client = get_client()
    try:
        response = create_message(
            client,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        _anthropic_image_block(image_b64),
                    ],
                }
            ],
        )
        return extract_response_text(response)
    except Exception as e:
        logger.warning("VLM call failed: %s", e)
        raise VLMError(str(e)) from e


def _build_gemini_thinking_config(model: str) -> Any:
    """Build Gemini thinking config based on THINKING_EFFORT setting.

    Returns a ThinkingConfig or None if thinking is disabled.
    Shared by single-turn and two-turn Gemini functions.
    """
    from google.genai import types

    from p2p.settings import THINKING_EFFORT

    if not THINKING_EFFORT or THINKING_EFFORT not in _EFFORT_TO_LEVEL:
        return None

    if model.startswith("gemini-3"):
        return types.ThinkingConfig(
            thinking_level=_EFFORT_TO_LEVEL[THINKING_EFFORT],
        )
    return types.ThinkingConfig(
        thinking_budget=_EFFORT_TO_BUDGET[THINKING_EFFORT],
    )


def _build_gemini_config(model: str) -> Any:
    """Build Gemini GenerateContentConfig with thinking + high media resolution.

    Gemini 3.x defaults to 70 tokens/frame; HIGH bumps to 280 tokens/frame.
    The media_resolution field is ignored on text-only requests.
    """
    from google.genai import types

    return types.GenerateContentConfig(
        thinking_config=_build_gemini_thinking_config(model),
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    )


def _build_gemini_media_part(
    video_path: Path | None,
    image_b64: str,
) -> Any:
    """Build a Gemini media Part from video (preferred) or base64 image.

    Shared by single-turn and two-turn Gemini functions.
    """
    from google.genai import types

    if video_path and video_path.exists():
        video_bytes = video_path.read_bytes()
        return types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=VLM_FPS),
        )
    image_bytes = base64.b64decode(image_b64)
    return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")


def call_vlm_gemini(
    prompt: str,
    image_b64: str = "",
    *,
    video_path: Path | None = None,
    model: str = "vllm-Qwen/Qwen3.5-27B",
) -> str:
    """Call Google Gemini API with video, image, or text-only.

    When *video_path* is provided, sends the MP4 inline as bytes.
    Otherwise falls back to base64 image.  If neither is provided,
    performs a text-only call (used by criteria review).

    Returns response text.

    Raises:
        VLMError: If the API call fails.
    """
    has_media = (video_path and video_path.exists()) or bool(image_b64)

    try:
        from google import genai

        from p2p.settings import GEMINI_API_KEY

        client = genai.Client(api_key=GEMINI_API_KEY)
        config = _build_gemini_config(model)

        if has_media:
            media_part = _build_gemini_media_part(video_path, image_b64)
            contents = [prompt, media_part]
        else:
            contents = [prompt]

        t0 = _time.monotonic()
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        _emit_vlm_call(model, response, duration_ms=int((_time.monotonic() - t0) * 1000))
        return response.text or ""
    except Exception as e:
        logger.warning("VLM call failed: %s", e)
        raise VLMError(str(e)) from e


def call_vlm_auto(
    prompt: str,
    images_b64: list[str],
    *,
    vlm_model: str = "vllm-Qwen/Qwen3.5-27B",
    max_tokens: int = MAX_VLM_TOKENS,
    video_path: Path | None = None,
) -> str:
    """Route VLM call to the appropriate provider based on model ID.

    Routing:
      - vllm-*  → vLLM (local, native video via video_url)
      - gemini*  → Google (native video via inline MP4 bytes)
      - claude*  → Anthropic (image only)
      - default  → Ollama (image only)

    The ``vllm-`` prefix is stripped to get the actual model name
    (e.g. ``vllm-Qwen/Qwen3.5-27B`` → ``Qwen/Qwen3.5-27B``).

    Args:
        prompt: Text prompt for the VLM.
        images_b64: List of base64-encoded JPEG images (fallback when no video).
        vlm_model: Model ID with optional provider prefix.
        max_tokens: Maximum response tokens.
        video_path: Path to MP4 video file (used by vLLM and Gemini providers).

    Returns:
        Response text string.
    """
    model_lower = vlm_model.lower()

    if model_lower.startswith("vllm-"):
        actual_model = vlm_model[5:]  # strip "vllm-" prefix
        from p2p.settings import VLLM_HOST, VLLM_PORT

        return call_vlm_vllm(
            prompt,
            video_path=video_path,
            images_b64=images_b64,
            model=actual_model,
            host=VLLM_HOST,
            port=VLLM_PORT,
            max_tokens=max_tokens,
        )

    if model_lower.startswith("claude"):
        image_b64 = images_b64[0] if images_b64 else ""
        return call_vlm_anthropic(prompt, image_b64, model=vlm_model)

    if model_lower.startswith("gemini"):
        image_b64 = images_b64[0] if images_b64 else ""
        return call_vlm_gemini(
            prompt,
            image_b64,
            video_path=video_path,
            model=vlm_model,
        )

    # Default: Ollama (qwen, llama, etc.)
    return call_vlm(prompt, images_b64, model=vlm_model, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Two-turn VLM dispatch (agreement bias mitigation)
# ---------------------------------------------------------------------------


_TURN1_MAX_TOKENS = 1024
"""Turn 1 asks for 3-5 visual criteria — a short list, ~200-400 tokens."""


def call_vlm_two_turn(
    turn1_prompt: str,
    turn2_prompt: str,
    images_b64: list[str],
    *,
    vlm_model: str = "vllm-Qwen/Qwen3.5-27B",
    max_tokens: int = MAX_VLM_TOKENS,
    video_path: Path | None = None,
    refined_initial_frame: bool = False,
    cached_criteria: str | None = None,
) -> tuple[str, str]:
    """Two-turn VLM call to mitigate agreement bias.

    Turn 1 (text-only): VLM pre-commits visual success criteria.
    Turn 2 (with media): VLM scores against its own criteria.

    When *cached_criteria* is provided, Turn 1 is skipped and the cached
    criteria text is injected into the conversation history for Turn 2.

    Returns (criteria_text, scoring_response).
    """
    model_lower = vlm_model.lower()

    if model_lower.startswith("vllm-"):
        actual_model = vlm_model[5:]
        from p2p.settings import VLLM_HOST, VLLM_PORT

        return _two_turn_vllm(
            turn1_prompt,
            turn2_prompt,
            video_path=video_path,
            images_b64=images_b64,
            model=actual_model,
            host=VLLM_HOST,
            port=VLLM_PORT,
            max_tokens=max_tokens,
            cached_criteria=cached_criteria,
        )

    if model_lower.startswith("claude"):
        image_b64 = images_b64[0] if images_b64 else ""
        return _two_turn_anthropic(
            turn1_prompt,
            turn2_prompt,
            image_b64,
            model=vlm_model,
            cached_criteria=cached_criteria,
        )

    if model_lower.startswith("gemini"):
        image_b64 = images_b64[0] if images_b64 else ""
        return _two_turn_gemini(
            turn1_prompt,
            turn2_prompt,
            image_b64,
            video_path=video_path,
            model=vlm_model,
            refined_initial_frame=refined_initial_frame,
            cached_criteria=cached_criteria,
        )

    # Default: Ollama
    return _two_turn_ollama(
        turn1_prompt,
        turn2_prompt,
        images_b64,
        model=vlm_model,
        max_tokens=max_tokens,
        cached_criteria=cached_criteria,
    )


def _two_turn_vllm(
    turn1_prompt: str,
    turn2_prompt: str,
    *,
    video_path: Path | None = None,
    images_b64: list[str] | None = None,
    model: str = "Qwen/Qwen3.5-27B",
    host: str = "0.0.0.0",
    port: int = 8100,
    max_tokens: int = MAX_VLM_TOKENS,
    cached_criteria: str | None = None,
) -> tuple[str, str]:
    """Two-turn vLLM call via OpenAI-compatible API."""
    from openai import OpenAI

    connect_host = "localhost" if host == "0.0.0.0" else host
    remote = _is_remote_host(host)

    try:
        client = OpenAI(base_url=f"http://{connect_host}:{port}/v1", api_key="unused")

        # Turn 1: text-only (skip if cached)
        if cached_criteria is not None:
            criteria = cached_criteria
        else:
            resp1 = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": turn1_prompt}],
                max_tokens=_TURN1_MAX_TOKENS,
                temperature=0.0,
                extra_body=_VLLM_EXTRA_BODY,
            )
            criteria = resp1.choices[0].message.content or ""

        # Turn 2: with media + conversation history
        turn2_content: list[dict] = [{"type": "text", "text": turn2_prompt}]
        turn2_content.extend(_build_vllm_media_content(video_path, images_b64, remote))

        resp2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": turn1_prompt},
                {"role": "assistant", "content": criteria},
                {"role": "user", "content": turn2_content},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            extra_body=_VLLM_EXTRA_BODY,
        )
        scoring = resp2.choices[0].message.content or ""
        return criteria, scoring
    except Exception as e:
        logger.warning("Two-turn vLLM call failed: %s", e)
        raise VLMError(str(e)) from e


def _two_turn_gemini(
    turn1_prompt: str,
    turn2_prompt: str,
    image_b64: str = "",
    *,
    video_path: Path | None = None,
    model: str = "gemini-3.1-pro-preview",
    refined_initial_frame: bool = False,
    cached_criteria: str | None = None,
) -> tuple[str, str]:
    """Two-turn Gemini call with thinking support.

    When refined_initial_frame=True and a video is provided:
    - Turn 1: first frame JPEG appended so VLM grounds criteria in actual scene
    - Turn 2: padded video (first frame duplicated to fix center-of-interval sampling)
    """
    if not (video_path and video_path.exists()) and not image_b64:
        raise VLMError("no video or image provided for Gemini two-turn")

    padded_path: Path | None = None
    overlay_path: Path | None = None
    try:
        from google import genai
        from google.genai import types

        from p2p.settings import GEMINI_API_KEY

        client = genai.Client(api_key=GEMINI_API_KEY)
        config = _build_gemini_config(model)

        # Turn 1: text + optional first frame JPEG
        turn1_parts: list = [types.Part.from_text(text=turn1_prompt)]
        actual_video = video_path
        if refined_initial_frame and video_path and video_path.exists():
            # Add first frame to Turn 1 so VLM grounds criteria in actual scene
            frame_b64 = _extract_first_frame_b64(video_path)
            if frame_b64:
                turn1_parts.append(
                    types.Part.from_bytes(data=base64.b64decode(frame_b64), mime_type="image/jpeg")
                )
            # Pad video for Turn 2 (fixes center-of-interval first-frame skip)
            padded_path = video_path.with_stem(video_path.stem + "_padded_vlm")
            if padded_path.exists():
                actual_video = padded_path
                padded_path = None  # don't clean up pre-created file
            else:
                import uuid as _uuid

                padded_path = video_path.with_stem(
                    video_path.stem + f"_padded_vlm_{_uuid.uuid4().hex[:8]}"
                )
                actual_video = _create_padded_video(video_path, padded_path)

        # Motion overlays: bake motion trail into the VLM video
        from p2p.settings import VLM_MOTION_TRAIL_DUAL

        trail_video_path: Path | None = None
        if VLM_MOTION_TRAIL_DUAL and actual_video and actual_video.exists():
            # Dual mode: keep actual_video clean, create separate trail video
            trail_video_path_obj, _ = _apply_motion_overlay_to_video(
                actual_video,
                motion_trail=True,
            )
            if trail_video_path_obj != actual_video:
                trail_video_path = trail_video_path_obj

        # Turn 1: generate criteria (skip if cached)
        if cached_criteria is not None:
            criteria = cached_criteria
        else:
            t0 = _time.monotonic()
            resp1 = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=turn1_parts)],
                config=config,
            )
            _emit_vlm_call(model, resp1, duration_ms=int((_time.monotonic() - t0) * 1000))
            criteria = resp1.text or ""

        # Turn 2: build parts depending on dual mode
        if VLM_MOTION_TRAIL_DUAL and trail_video_path:
            normal_part = _build_gemini_media_part(actual_video, "")
            trail_part = _build_gemini_media_part(trail_video_path, "")
            turn2_parts = [
                types.Part.from_text(text="Video 1: Standard 10fps recording of the rollout."),
                normal_part,
                types.Part.from_text(
                    text="Video 2: Motion trail version — translucent ghosts "
                    "of previous frames show movement continuity. The camera "
                    "tracks the agent, so background moves opposite to agent motion."
                ),
                trail_part,
                types.Part.from_text(text=turn2_prompt),
            ]
        else:
            media_part = _build_gemini_media_part(actual_video, image_b64)
            turn2_parts = [
                types.Part.from_text(text=turn2_prompt),
                media_part,
            ]

        t0 = _time.monotonic()
        resp2 = client.models.generate_content(
            model=model,
            contents=[
                types.Content(role="user", parts=turn1_parts),
                types.Content(role="model", parts=[types.Part.from_text(text=criteria)]),
                types.Content(role="user", parts=turn2_parts),
            ],
            config=config,
        )
        _emit_vlm_call(model, resp2, duration_ms=int((_time.monotonic() - t0) * 1000))
        scoring = resp2.text or ""
        return criteria, scoring
    except Exception as e:
        logger.warning("Two-turn Gemini call failed: %s", e)
        raise VLMError(str(e)) from e
    finally:
        if overlay_path and overlay_path.exists():
            overlay_path.unlink(missing_ok=True)
        # trail_video_path is kept for web app viewing
        if padded_path and padded_path.exists():
            padded_path.unlink(missing_ok=True)


def _two_turn_anthropic(
    turn1_prompt: str,
    turn2_prompt: str,
    image_b64: str,
    *,
    model: str = LLM_MODEL,
    cached_criteria: str | None = None,
) -> tuple[str, str]:
    """Two-turn Anthropic Claude call with thinking support."""
    from p2p.inference.llm_client import create_message, extract_response_text, get_client

    client = get_client()
    try:
        # Turn 1: text-only (skip if cached)
        if cached_criteria is not None:
            criteria = cached_criteria
        else:
            resp1 = create_message(
                client,
                model=model,
                messages=[{"role": "user", "content": turn1_prompt}],
            )
            criteria = extract_response_text(resp1)

        # Turn 2: with image + conversation history
        turn2_content: list[dict] = [{"type": "text", "text": turn2_prompt}]
        if image_b64:
            turn2_content.append(_anthropic_image_block(image_b64))

        resp2 = create_message(
            client,
            model=model,
            messages=[
                {"role": "user", "content": turn1_prompt},
                {"role": "assistant", "content": criteria},
                {"role": "user", "content": turn2_content},
            ],
        )
        scoring = extract_response_text(resp2)
        return criteria, scoring
    except Exception as e:
        logger.warning("Two-turn Anthropic call failed: %s", e)
        raise VLMError(str(e)) from e


def _two_turn_ollama(
    turn1_prompt: str,
    turn2_prompt: str,
    images_b64: list[str] | None = None,
    *,
    model: str = VLM_MODEL,
    max_tokens: int = MAX_VLM_TOKENS,
    cached_criteria: str | None = None,
) -> tuple[str, str]:
    """Two-turn Ollama call with thinking disabled."""
    import requests

    base_opts: dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": False,
    }
    try:
        # Turn 1: text-only (skip if cached)
        if cached_criteria is not None:
            criteria = cached_criteria
        else:
            payload1 = {
                **base_opts,
                "options": {"num_predict": _TURN1_MAX_TOKENS, "temperature": 0.0},
                "messages": [{"role": "user", "content": turn1_prompt}],
            }
            resp1 = requests.post(f"{VLM_BASE_URL}/api/chat", json=payload1, timeout=300)
            resp1.raise_for_status()
            criteria = resp1.json().get("message", {}).get("content", "")

        # Turn 2: with images + conversation history
        turn2_msg: dict[str, Any] = {"role": "user", "content": turn2_prompt}
        if images_b64:
            turn2_msg["images"] = images_b64

        payload2 = {
            **base_opts,
            "options": {"num_predict": max_tokens, "temperature": 0.0},
            "messages": [
                {"role": "user", "content": turn1_prompt},
                {"role": "assistant", "content": criteria},
                turn2_msg,
            ],
        }
        resp2 = requests.post(f"{VLM_BASE_URL}/api/chat", json=payload2, timeout=300)
        resp2.raise_for_status()
        scoring = resp2.json().get("message", {}).get("content", "")
        return criteria, scoring
    except Exception as e:
        logger.warning("Two-turn Ollama call failed: %s", e)
        raise VLMError(str(e)) from e
