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

from typing import Dict, Optional
import imageio
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np

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
