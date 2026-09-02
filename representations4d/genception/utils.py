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
