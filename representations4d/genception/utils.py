# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for the GenCeption model.

Provides helpers for checkpoint I/O, video loading/saving, and
video pre/post-processing used by the inference pipeline.
"""

from typing import Any, Dict, Optional, Tuple
import imageio
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
from PIL import Image as _PILImage

# BlockSizes is used as a type hint for flash-attention block sizes.
# It is a dict mapping block-size names to integer values, or None.
BlockSizes = Optional[Dict[str, int]]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_transformer_checkpoint(path: str) -> Dict[str, jnp.ndarray]:
  """Load a GenCeption transformer checkpoint from an ``.npz`` file.

  Args:
    path: Path to a ``.npz`` file containing transformer weights.

  Returns:
    A flat dictionary mapping ``'key1/key2/...'`` strings to NumPy arrays.
  """
  data = jnp.load(path, allow_pickle=False)
  params = {k: data[k] for k in data.files}
  return params

# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


def load_video(
    path: str,
    num_frames: Optional[int] = None,
) -> jnp.ndarray:
  """Load a video file into a NumPy array.

  Requires *mediapy* (``pip install mediapy``) or *imageio*.

  Args:
    path: Path to a video file (mp4, webm, etc.).
    num_frames: If given, uniformly sub-sample to this many frames.

  Returns:
    ``uint8`` array of shape ``[T, H, W, 3]`` with values in ``[0, 255]``.
  """
  try:
    video = media.read_video(path)
  except ImportError:
    try:
      video = jnp.asarray(imageio.imread(path, plugin='pyav'))
    except ImportError as exc:
      raise ImportError(
          'Either `mediapy` or `imageio[pyav]` is required to load videos. '
          'Install one with: pip install mediapy'
      ) from exc

  if num_frames is not None and num_frames < len(video):
    indices = jnp.linspace(0, len(video) - 1, num_frames, dtype=int)
    video = video[indices]

  return jnp.asarray(video, dtype=np.uint8)


def save_video(
    video: jnp.ndarray,
    path: str,
    fps: int = 24,
) -> None:
  """Save a NumPy video array as a video file.

  Requires *mediapy* (``pip install mediapy``).

  Args:
    video: Array of shape ``[T, H, W, 3]``, ``uint8`` or ``float`` in
      ``[0, 1]``.
    path: Destination file path (e.g. ``'output.mp4'``).
    fps: Frames per second.
  """
  if video.dtype != jnp.uint8:
    video = jnp.clip(video * 255.0, 0, 255).astype(np.uint8)

  try:
    media.write_video(path, video, fps=fps)
  except ImportError:
    try:
      writer = imageio.get_writer(path, fps=fps)
      for frame in video:
        writer.append_data(frame)
      writer.close()
    except ImportError as exc:
      raise ImportError(
          'Either `mediapy` or `imageio` is required to save videos. '
          'Install one with: pip install mediapy'
      ) from exc


# ---------------------------------------------------------------------------
# Video preprocessing / postprocessing
# ---------------------------------------------------------------------------


def _resize_frames(
    frames: jnp.ndarray,
    height: int,
    width: int,
) -> jnp.ndarray:
  """Resize video frames using bilinear interpolation via JAX.

  Args:
    frames: ``[T, H_in, W_in, 3]`` array (``uint8`` or ``float``).
    height: Target height.
    width: Target width.

  Returns:
    Resized array of shape ``[T, height, width, 3]``.
  """
  if frames.shape[1] == height and frames.shape[2] == width:
    return frames

  # Use JAX image resize for quality (Lanczos-like via linear on float).
  was_uint8 = frames.dtype == jnp.uint8
  if was_uint8:
    frames = frames.astype(np.float32) / 255.0

  resized = jax.image.resize(
      jnp.asarray(frames),
      shape=(frames.shape[0], height, width, 3),
      method='bilinear',
  )
  resized = jnp.asarray(resized)

  if was_uint8:
    resized = jnp.clip(resized * 255.0, 0, 255).astype(np.uint8)
  return resized


def preprocess_video(
    video: jnp.ndarray,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
) -> jnp.ndarray:
  """Resize, temporally sample, and normalize a video for VAE encoding.

  Args:
    video: Input video, shape ``[T, H, W, 3]``.  Can be ``uint8``
      (values 0–255) or ``float`` (values 0.0–1.0).
    height: Target spatial height (default 480).
    width: Target spatial width (default 832).
    num_frames: Target number of frames (default 81).

  Returns:
    ``float32`` array of shape ``[T', H', W', 3]`` with values in
    ``[-1, 1]``, ready for VAE encoding.
  """
  # 1. Convert to float [0, 1] if needed.
  if video.dtype == jnp.uint8:
    video = video.astype(np.float32) / 255.0

  # 2. Temporal sub-sampling.
  if video.shape[0] > num_frames:
    indices = jnp.linspace(0, video.shape[0] - 1, num_frames, dtype=int)
    video = video[indices]
  elif video.shape[0] < num_frames:
    # Pad by repeating the last frame.
    pad_count = num_frames - video.shape[0]
    padding = jnp.repeat(video[-1:], pad_count, axis=0)
    video = jnp.concatenate([video, padding], axis=0)

  # 3. Spatial resize.
  video = _resize_frames(video, height, width)

  # 4. Normalize to [-1, 1] (the range expected by the WAN VAE).
  video = 2.0 * video - 1.0

  return video.astype(np.float32)


def postprocess_video(video: jnp.ndarray) -> jnp.ndarray:
  """Denormalize and convert a decoded video back to ``uint8``.

  Reverses the normalization applied by :func:`preprocess_video` and
  the VAE decode output (which is in ``[-1, 1]``).

  Args:
    video: Array of shape ``[T, H, W, 3]`` (or ``[B, T, H, W, 3]``)
      with values in ``[-1, 1]``.

  Returns:
    ``uint8`` array in ``[0, 255]`` with the same shape.
  """
  # Denormalize: [-1, 1] → [0, 1].
  video = (video + 1.0) / 2.0
  video = jnp.clip(video, 0.0, 1.0)
  # Convert to uint8.
  video = (video * 255.0).astype(np.uint8)
  return video


# ---------------------------------------------------------------------------
# Depth coloring / uncoloring (banana colormap)
# ---------------------------------------------------------------------------

# The 8 vertices of the RGB cube traversed by the "banana" / "vision banana"
# colormap.  The path Black → Red → Yellow → Green → Cyan → Blue → Magenta →
# White defines 7 piecewise-linear segments along the edges of the unit cube.
_BANANA_COLORS = np.array([
    [0.0, 0.0, 0.0],  # 0: Black
    [1.0, 0.0, 0.0],  # 1: Red
    [1.0, 1.0, 0.0],  # 2: Yellow
    [0.0, 1.0, 0.0],  # 3: Green
    [0.0, 1.0, 1.0],  # 4: Cyan
    [0.0, 0.0, 1.0],  # 5: Blue
    [1.0, 0.0, 1.0],  # 6: Magenta
    [1.0, 1.0, 1.0],  # 7: White
], dtype=np.float32)

_NUM_SEGMENTS = len(_BANANA_COLORS) - 1  # 7


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
  """Map scalar depth values in ``[0, 1]`` to RGB via the banana colormap.

  This is the **forward** coloring operation applied during training.
  A depth value of 0 maps to black (near) and 1 maps to white (far), with
  intermediate values interpolated along the 7 edges of the RGB cube.

  Args:
    depth: Array of any shape with values in ``[0, 1]``.

  Returns:
    Array of shape ``(*depth.shape, 3)`` with RGB values in ``[0, 1]``.
  """
  depth = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)

  # Map depth to [0, 7] and find the segment.
  scaled = depth * _NUM_SEGMENTS
  idx0 = np.clip(np.floor(scaled).astype(np.int32), 0, _NUM_SEGMENTS - 1)
  idx1 = idx0 + 1
  t = scaled - idx0

  # Add channel dim for broadcasting: (...,) → (..., 1).
  if t.ndim > 0:
    t = t[..., np.newaxis]

  rgb = (1.0 - t) * _BANANA_COLORS[idx0] + t * _BANANA_COLORS[idx1]
  return rgb.astype(np.float32)


def rgb_to_depth_robust(rgb: np.ndarray) -> np.ndarray:
  """Recover scalar depth from (noisy) banana-colormapped RGB values.

  Each RGB pixel is orthogonally projected onto each of the 7 line segments
  of the banana colormap and the closest segment is selected.  This is
  robust to the color noise introduced by VAE decoding.

  Args:
    rgb: Array of shape ``(..., 3)`` with RGB values in ``[0, 1]`` (float)
      or ``[0, 255]`` (``uint8``).

  Returns:
    Array of shape ``(...)`` with ``float32`` depth values in ``[0, 1]``.
  """
  rgb = np.asarray(rgb)
  if rgb.dtype == np.uint8:
    rgb = rgb.astype(np.float32) / 255.0
  else:
    rgb = rgb.astype(np.float32)

  seg_starts = _BANANA_COLORS[:-1]   # (7, 3), float32
  seg_ends = _BANANA_COLORS[1:]      # (7, 3), float32
  seg_dirs = seg_ends - seg_starts   # (7, 3), float32

  # Expand rgb for broadcasting: (..., 3) → (..., 1, 3).
  rgb_expanded = rgb[..., np.newaxis, :]

  # Vector from segment start to the query point.
  to_query = rgb_expanded - seg_starts  # (..., 7, 3)

  # Projection coefficient t onto each segment.
  # All seg_dirs vectors have unit L2 norm, so ||seg_dirs||² = 1 — no
  # denominator needed.
  t = np.sum(to_query * seg_dirs, axis=-1)   # (..., 7)
  t_clamped = np.clip(t, 0.0, 1.0)           # (..., 7)

  # Closest point on each segment.
  proj_pts = seg_starts + t_clamped[..., np.newaxis] * seg_dirs  # (..., 7, 3)

  # Squared distance from the query to each projection.
  dist_sq = np.sum((rgb_expanded - proj_pts) ** 2, axis=-1)  # (..., 7)

  # Pick the nearest segment.
  best_seg = np.argmin(dist_sq, axis=-1)  # (...)
  best_t = np.take_along_axis(
      t_clamped, best_seg[..., np.newaxis], axis=-1
  )[..., 0]  # (...)

  depth = (best_seg + best_t) / float(_NUM_SEGMENTS)
  return depth.astype(np.float32)


def colored_depth_to_relative_depth(
    rgb_video: np.ndarray,
    depth_scaling_factor: float = 0.4,
) -> np.ndarray:
  """Convert a colored, log-scaled depth video back to relative depth.

  The GenCeption model predicts depth as a colored RGB video where:

  1. Relative depth is log-scaled: ``scaled = clip(c * ln(d + 1), 0, 1)``
  2. The scalar is mapped to RGB via the banana colormap.

  This function **inverts both operations**:

  1. :func:`rgb_to_depth_robust` recovers the log-scaled scalar.
  2. The inverse log transform recovers relative depth:
     ``d = exp(scaled / c) - 1``.

  Args:
    rgb_video: Array of shape ``[T, H, W, 3]`` with RGB values in
      ``[0, 1]`` (float) or ``[0, 255]`` (``uint8``).
    depth_scaling_factor: The log-scale factor ``c`` used during training.
      Default ``0.4`` matches the current GenCeption model.

  Returns:
    Float32 array of shape ``[T, H, W]`` containing relative depth values.
  """
  rgb_video = np.asarray(rgb_video)
  if rgb_video.dtype == np.uint8:
    rgb_video = rgb_video.astype(np.float32) / 255.0
  else:
    rgb_video = rgb_video.astype(np.float32)

  depth_unscaling = lambda x: np.exp(x / depth_scaling_factor) - 1.0

  num_frames = rgb_video.shape[0]
  depth_frames = []
  for t in range(num_frames):
    scaled_depth = rgb_to_depth_robust(rgb_video[t])
    relative_depth = depth_unscaling(scaled_depth)
    depth_frames.append(relative_depth)

  return np.stack(depth_frames, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Point cloud visualization (NumPy point splatting)
# ---------------------------------------------------------------------------


def unproject_depth_to_points(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    min_depth: float = 0.01,
    max_depth_percentile: float = 99.9,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
  """Unproject a 2D depth map (and optional RGB frame) to 3D camera coordinates.

  Uses standard pinhole camera projection:
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

  Args:
    depth: 2D array of shape ``[H, W]`` with depth values.
    rgb: Optional array of shape ``[H, W, 3]`` with RGB colors in ``[0, 1]``.
    fx: Focal length along x. Defaults to ``0.75 * W``.
    fy: Focal length along y. Defaults to ``0.75 * W``.
    cx: Principal point x. Defaults to ``W / 2.0``.
    cy: Principal point y. Defaults to ``H / 2.0``.
    min_depth: Minimum depth threshold to filter near/invalid points.
    max_depth_percentile: Filter out points beyond this depth percentile.

  Returns:
    A tuple of ``(points, colors)`` where ``points`` is an ``[N, 3]`` float32
    array and ``colors`` is an ``[N, 3]`` float32 array (or None if rgb is
    None).
  """
  depth = np.asarray(depth, dtype=np.float32)
  h, w = depth.shape
  if fx is None:
    fx = float(w) * 0.75
  if fy is None:
    fy = float(w) * 0.75
  if cx is None:
    cx = w / 2.0
  if cy is None:
    cy = h / 2.0

  uu, vv = np.meshgrid(
      np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
  )
  pts = np.stack(
      [(uu - cx) * depth / fx, (vv - cy) * depth / fy, depth], axis=-1
  ).reshape(-1, 3)

  max_d = (
      float(np.percentile(pts[:, 2], max_depth_percentile))
      if len(pts) > 0
      else np.inf
  )
  valid = (
      np.isfinite(pts[:, 2]) & (pts[:, 2] > min_depth) & (pts[:, 2] <= max_d)
  )

  pts_valid = pts[valid]
  if rgb is not None:
    cols = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    cols_valid = cols[valid]
    return pts_valid, cols_valid
  return pts_valid, None


def compute_pointcloud_framing(
    pts: np.ndarray,
    yaw_deg: float = 0.0,
    pitch_deg: float = -5.0,
    pull_back: float = 0.22,
    out_w: int = 1080,
    out_h: int = 750,
) -> Dict[str, Any]:
  """Computes locked camera framing from 3D points.

  Orbits around the scene median, rotates by (yaw, pitch), pulls the camera
  back along Z, and computes 2D perspective scale and offset to center the
  point cloud in the output canvas.

  Args:
    pts: Array of shape ``[N, 3]`` containing 3D points.
    yaw_deg: Camera yaw angle in degrees.
    pitch_deg: Camera pitch angle in degrees.
    pull_back: Distance multiplier behind the frustum apex.
    out_w: Target output width in pixels.
    out_h: Target output height in pixels.

  Returns:
    Dictionary containing camera parameters: ``'pivot'``, ``'z_offset'``,
    ``'R'``, ``'ux_mid'``, ``'uy_mid'``, and ``'scale'``.
  """
  if len(pts) == 0:
    return {
        'pivot': np.zeros(3, dtype=np.float32),
        'z_offset': 1.0,
        'R': np.eye(3, dtype=np.float32),
        'ux_mid': 0.0,
        'uy_mid': 0.0,
        'scale': 1.0,
    }

  pivot = np.median(pts, axis=0)
  z_span = float(np.percentile(pts[:, 2], 90) - np.percentile(pts[:, 2], 5))
  yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)

  ry = np.array(
      [
          [np.cos(yaw), 0.0, np.sin(yaw)],
          [0.0, 1.0, 0.0],
          [-np.sin(yaw), 0.0, np.cos(yaw)],
      ],
      dtype=np.float32,
  )
  rx = np.array(
      [
          [1.0, 0.0, 0.0],
          [0.0, np.cos(pitch), -np.sin(pitch)],
          [0.0, np.sin(pitch), np.cos(pitch)],
      ],
      dtype=np.float32,
  )
  r = rx @ ry

  p = (pts - pivot) @ r.T
  z_offset = float(pivot[2] + pull_back * z_span)
  p[:, 2] += z_offset

  front = p[:, 2] > 0.05
  p = p[front] if np.any(front) else p

  ux = p[:, 0] / np.maximum(p[:, 2], 1e-4)
  uy = p[:, 1] / np.maximum(p[:, 2], 1e-4)
  ux_lo, ux_hi = np.percentile(ux, [0.5, 99.5])
  uy_lo, uy_hi = np.percentile(uy, [0.5, 99.5])

  scale = 0.85 * min(
      out_w / max(float(ux_hi - ux_lo), 1e-5),
      out_h / max(float(uy_hi - uy_lo), 1e-5),
  )

  return {
      'pivot': pivot,
      'z_offset': z_offset,
      'R': r,
      'ux_mid': float(0.5 * (ux_lo + ux_hi)),
      'uy_mid': float(0.5 * (uy_lo + uy_hi)),
      'scale': float(scale),
  }


def render_pointcloud_frame(
    pts: np.ndarray,
    cols: np.ndarray,
    framing: Optional[Dict[str, Any]] = None,
    yaw_deg: float = 0.0,
    pitch_deg: float = -5.0,
    pull_back: float = 0.22,
    out_w: int = 1080,
    out_h: int = 750,
    radius: int = 1,
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
  """Renders a single point cloud frame using painter's algorithm and splatting.

  Args:
    pts: Array of shape ``[N, 3]`` containing 3D points.
    cols: Array of shape ``[N, 3]`` containing RGB colors in ``[0, 1]``.
    framing: Optional precomputed camera framing dict. If None, computed from
      pts.
    yaw_deg: Camera yaw angle in degrees (used when framing is None).
    pitch_deg: Camera pitch angle in degrees (used when framing is None).
    pull_back: Pull back factor (used when framing is None).
    out_w: Canvas width in pixels.
    out_h: Canvas height in pixels.
    radius: Radius of point splat in pixels.
    bg_color: Background color tuple in ``[0, 1]`` (default white).

  Returns:
    ``uint8`` array of shape ``[out_h, out_w, 3]`` with values in ``[0, 255]``.
  """
  if framing is None:
    framing = compute_pointcloud_framing(
        pts,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        pull_back=pull_back,
        out_w=out_w,
        out_h=out_h,
    )

  if len(pts) == 0:
    return (
        np.full((out_h, out_w, 3), bg_color, dtype=np.float32) * 255.0
    ).astype(np.uint8)

  p = (pts - framing['pivot']) @ framing['R'].T
  p[:, 2] += framing['z_offset']

  front = p[:, 2] > 0.05
  p, c = p[front], cols[front]

  if len(p) == 0:
    return (
        np.full((out_h, out_w, 3), bg_color, dtype=np.float32) * 255.0
    ).astype(np.uint8)

  z_safe = np.maximum(p[:, 2], 1e-4)
  px = (
      (p[:, 0] / z_safe - framing['ux_mid']) * framing['scale'] + out_w / 2.0
  ).astype(np.int32)
  py = (
      (p[:, 1] / z_safe - framing['uy_mid']) * framing['scale'] + out_h / 2.0
  ).astype(np.int32)

  # Painter's algorithm: sort by distance (far -> near)
  order = np.argsort(-p[:, 2])
  px, py, c = px[order], py[order], c[order]

  img = np.full((out_h, out_w, 3), bg_color, dtype=np.float32)
  for dy in range(-radius, radius + 1):
    for dx in range(-radius, radius + 1):
      if dx * dx + dy * dy <= radius * radius + 1:
        xi, yi = px + dx, py + dy
        ok = (xi >= 0) & (xi < out_w) & (yi >= 0) & (yi < out_h)
        img[yi[ok], xi[ok]] = c[ok]

  return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_pointcloud_video(
    relative_depth: np.ndarray,
    video: np.ndarray,
    out_w: int = 1080,
    out_h: int = 750,
    yaw_deg: float = 0.0,
    pitch_deg: float = -5.0,
    pull_back: float = 0.22,
    radius: int = 1,
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    show_progress: bool = True,
) -> np.ndarray:
  """Render a point cloud video from relative depth and RGB video via splatting.

  Locks camera framing from frame 0 so the virtual camera remains stable and
  jitter-free throughout the video sequence.

  Args:
    relative_depth: Array of shape ``[T, H, W]`` containing relative depth.
    video: Array of shape ``[T, H_in, W_in, 3]`` containing RGB frames. Can be
      uint8 ``[0, 255]``, float ``[0, 1]``, or float ``[-1, 1]``.
    out_w: Output video width in pixels.
    out_h: Output video height in pixels.
    yaw_deg: Camera yaw angle in degrees.
    pitch_deg: Camera pitch angle in degrees.
    pull_back: Camera pull-back factor.
    radius: Point splat radius in pixels.
    bg_color: Canvas background color in ``[0, 1]`` (default white).
    show_progress: Whether to print progress during rendering.

  Returns:
    ``uint8`` array of shape ``[T, out_h, out_w, 3]`` with values in ``[0,
    255]``.
  """
  relative_depth = np.asarray(relative_depth, dtype=np.float32)
  t_len, h, w = relative_depth.shape

  # Normalize RGB frames to [0, 1].
  rgb_all = np.asarray(video, dtype=np.float32)
  if rgb_all.min() < -0.01:
    rgb_all = (rgb_all + 1.0) / 2.0
  elif rgb_all.max() > 1.5:
    rgb_all = rgb_all / 255.0
  rgb_all = np.clip(rgb_all, 0.0, 1.0)

  # Resize RGB if needed to match depth resolution (h, w).
  if rgb_all.shape[1:3] != (h, w):
    resized = []
    for t in range(len(rgb_all)):
      u8 = (rgb_all[t] * 255.0).astype(np.uint8)
      resized.append(
          np.asarray(
              _PILImage.fromarray(u8).resize((w, h), _PILImage.BILINEAR),
              dtype=np.float32,
          )
          / 255.0
      )
    rgb_all = np.stack(resized)

  # Lock framing from frame 0.
  pts0, _ = unproject_depth_to_points(relative_depth[0])
  framing = compute_pointcloud_framing(
      pts0,
      yaw_deg=yaw_deg,
      pitch_deg=pitch_deg,
      pull_back=pull_back,
      out_w=out_w,
      out_h=out_h,
  )

  if show_progress:
    print(f'Rendering point cloud video: {t_len} frames at {out_w}x{out_h}...')

  rendered_frames = []
  for t in range(t_len):
    c_frame = rgb_all[min(t, len(rgb_all) - 1)]
    pts, cols = unproject_depth_to_points(relative_depth[t], rgb=c_frame)
    frame = render_pointcloud_frame(
        pts,
        cols,
        framing=framing,
        out_w=out_w,
        out_h=out_h,
        radius=radius,
        bg_color=bg_color,
    )
    rendered_frames.append(frame)
    if show_progress and ((t + 1) % 20 == 0 or t == t_len - 1):
      print(f'  {t + 1}/{t_len} frames rendered')

  pointcloud_video = np.stack(rendered_frames, axis=0)
  return pointcloud_video
