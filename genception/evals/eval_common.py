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

"""Shared utilities for GenCeption evaluation scripts.

Runs the unmodified GenCeptionPipeline from representations4d/genception
by staging small checkpoint files locally, streaming the transformer .npz,
and sharding parameters across JAX devices.
"""

from __future__ import annotations

import concurrent.futures
import html
import io
import json
import os
import re
import shutil
import tempfile
from typing import Any, Dict
import zipfile

from absl import logging
import cv2
from flax import nnx
import jax
import jax.numpy as jnp
import mediapy as media
import ml_dtypes
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import UMT5EncoderModel

from representations4d.genception import attention as genception_attention
from representations4d.genception import pipeline as genception_pipeline
from representations4d.genception.pipeline import GenCeptionPipeline
from representations4d.genception.utils import postprocess_video

# Override these module-level constants to point at your local data directories.
RELEASE_DATA_DIR = ''
DEFAULT_CHECKPOINT_DIR = ''
EMBEDDINGS_DIR = ''
WAN_DIFFUSERS_DIR = ''

TASK_PROMPTS = {
    'depth': 'Input: RGB. Output: Depth.',
    'normals': 'Input: RGB. Output: Normals.',
    'raymap': 'Input: RGB. Output: Raymap.',
    'negative': '',
}


def _clean_prompt(text: str) -> str:
  text = html.unescape(html.unescape(text))
  return re.sub(r'\s+', ' ', text).strip()


def _copy_dir_to_local(src_dir: str, dst_dir: str) -> None:
  os.makedirs(dst_dir, exist_ok=True)
  for entry in os.listdir(src_dir):
    src_path = os.path.join(src_dir, entry)
    dst_path = os.path.join(dst_dir, entry)
    if os.path.isdir(src_path):
      _copy_dir_to_local(src_path, dst_path)
    else:
      shutil.copy2(src_path, dst_path)


def get_or_compute_prompt_embeds(
    task: str,
    custom_path: str = '',
) -> np.ndarray:
  """Load precomputed prompt embeddings or compute & cache them via UMT5."""
  candidates = []
  if custom_path:
    candidates.append(custom_path)
  if EMBEDDINGS_DIR:
    candidates.append(os.path.join(EMBEDDINGS_DIR, f'{task}.npy'))

  for path in candidates:
    if os.path.exists(path):
      logging.info('Loading prompt embeddings for %s from %s', task, path)
      with open(path, 'rb') as f:
        return np.load(f)

  if task not in TASK_PROMPTS:
    raise FileNotFoundError(
        f'Prompt embeddings not found for {task} in {candidates}'
    )

  prompt_text = _clean_prompt(TASK_PROMPTS[task])
  logging.info(
      'Prompt embedding for %s not found on disk; computing with UMT5 for %r',
      task,
      prompt_text,
  )
  tmp_dir = tempfile.mkdtemp(prefix='wan_umt5_')
  try:
    tok_local = os.path.join(tmp_dir, 'tokenizer')
    te_local = os.path.join(tmp_dir, 'text_encoder')
    _copy_dir_to_local(
        os.path.join(WAN_DIFFUSERS_DIR, 'tokenizer'), tok_local
    )
    _copy_dir_to_local(
        os.path.join(WAN_DIFFUSERS_DIR, 'text_encoder'), te_local
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_local)
    text_encoder = UMT5EncoderModel.from_pretrained(
        te_local, torch_dtype=torch.bfloat16
    )
    text_encoder.eval()
    text_inputs = tokenizer(
        [prompt_text],
        padding='max_length',
        max_length=226,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    with torch.no_grad():
      embeds = text_encoder(
          text_inputs.input_ids,
          attention_mask=text_inputs.attention_mask,
      ).last_hidden_state
    mask = (
        text_inputs.attention_mask.unsqueeze(-1).expand(embeds.size()).float()
    )
    embeds = (embeds.float() * mask).cpu().numpy().astype(np.float32)
    del text_encoder
    del tokenizer
  finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

  if EMBEDDINGS_DIR:
    cache_path = os.path.join(EMBEDDINGS_DIR, f'{task}.npy')
    try:
      os.makedirs(os.path.dirname(cache_path), exist_ok=True)
      with open(cache_path, 'wb') as f:
        np.save(f, embeds)
      logging.info('Saved computed prompt embeddings to %s', cache_path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning(
          'Could not cache prompt embeddings to %s: %s', cache_path, exc
      )

  return embeds


def _install_tpu_chunked_attention_if_needed() -> None:
  """Avoid 85GB O(S^2) attention matrix on TPU without modifying attention.py."""
  if jax.local_devices()[0].platform == 'gpu':
    return

  orig_fn = genception_attention._apply_dot_product_attention  # pylint: disable=protected-access

  def _chunked_dot_product_attention(
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      heads: int,
      dim_head: int,
      scale: float,
      dtype: jnp.dtype,
  ) -> jax.Array:
    b, h, s_q, d = query.shape
    s_k = key.shape[2]
    chunk_size = 1024
    if s_q > 2048 and s_k > 2048:
      pad_len = (chunk_size - (s_q % chunk_size)) % chunk_size
      query_padded = (
          jnp.pad(query, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
          if pad_len > 0
          else query
      )
      num_chunks = query_padded.shape[2] // chunk_size
      q_chunks = jnp.transpose(
          query_padded.reshape(b, h, num_chunks, chunk_size, d),
          (2, 0, 1, 3, 4),
      )

      @jax.checkpoint
      def _attend_chunk(q_c: jax.Array) -> jax.Array:
        scores = jnp.einsum('bhsd,bhtd->bhst', q_c, key) * scale
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
            dtype
        )
        return jnp.einsum('bhst,bhtd->bhsd', probs, value)

      out_chunks = jax.lax.map(_attend_chunk, q_chunks)
      hidden_states = jnp.transpose(out_chunks, (1, 2, 0, 3, 4)).reshape(
          b, h, num_chunks * chunk_size, d
      )[:, :, :s_q, :]
      return genception_attention._flatten_heads(hidden_states)  # pylint: disable=protected-access
    return orig_fn(query, key, value, heads, dim_head, scale, dtype)

  genception_attention._apply_dot_product_attention = (  # pylint: disable=protected-access
      _chunked_dot_product_attention
  )


def _stage_checkpoint_metadata_and_vae(
    remote_checkpoint_dir: str,
    model_size: str,
) -> str:
  """Stage small config and VAE files locally so unmodified pipeline.py can open them."""
  local_dir = tempfile.mkdtemp(prefix='genception_ckpt_')
  os.makedirs(os.path.join(local_dir, 'vae'), exist_ok=True)

  files_to_copy = [
      f'genception_{model_size}_config.json',
      'vae/config.json',
      'vae/diffusion_pytorch_model.safetensors',
  ]
  for rel_path in files_to_copy:
    src = os.path.join(remote_checkpoint_dir, rel_path)
    dst = os.path.join(local_dir, rel_path)
    if os.path.exists(src):
      logging.info('Staging %s -> %s', src, dst)
      shutil.copy2(src, dst)

  # Create empty marker file for transformer .npz so os.path.exists() is True;
  # load_transformer_checkpoint is hooked to stream the real .npz.
  marker_npz = os.path.join(
      local_dir, f'genception_{model_size}_transformer.npz'
  )
  with open(marker_npz, 'wb') as f:
    f.write(b'')

  return local_dir


def load_pipeline(
    checkpoint_dir: str = '',
    model_size: str = '14b',
    diffusion_single_step: int = 0,
) -> GenCeptionPipeline:
  """Load the GenCeptionPipeline with FSDP sharding across devices."""
  _install_tpu_chunked_attention_if_needed()
  model_size = model_size.lower()
  ckpt_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
  local_ckpt_dir = _stage_checkpoint_metadata_and_vae(ckpt_dir, model_size)
  real_npz_path = os.path.join(
      ckpt_dir, f'genception_{model_size}_transformer.npz'
  )

  # Set up 1D FSDP mesh across all available JAX devices up front so we can
  # stream tensors directly into sharded TPU HBM without accumulating CPU RAM.
  devices = jax.devices()
  mesh = jax.sharding.Mesh(np.array(devices), ('fsdp',))
  replicated_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec()
  )
  num_devices = len(devices)

  def _shard_leaf(x: Any) -> Any:
    if isinstance(x, np.ndarray):
      if x.dtype.char == 'V' and x.dtype.itemsize == 2:
        x = x.view(ml_dtypes.bfloat16)
      elif x.dtype == np.float32:
        x = x.astype(ml_dtypes.bfloat16)
    # Shard along the last feature axis (e.g. 5120 or 13824) so axis 0 (the
    # 40-layer nnx.scan dimension) remains unsharded (None) on all devices.
    if (
        isinstance(x, (jax.Array, np.ndarray))
        and x.ndim >= 2
        and x.shape[-1] % num_devices == 0
    ):
      spec = jax.sharding.PartitionSpec(
          *(None for _ in range(x.ndim - 1)), 'fsdp'
      )
      return jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))
    if isinstance(x, (jax.Array, np.ndarray)):
      return jax.device_put(x, replicated_sharding)
    return x

  def _load_transformer_checkpoint(unused_path: str) -> Dict[str, Any]:
    file_size = os.stat(real_npz_path).st_size
    block_size = 64 * 1024 * 1024  # 64 MB blocks
    num_blocks = (file_size + block_size - 1) // block_size
    logging.info(
        'Streaming transformer checkpoint from %s'
        ' (%.2f GB, %d blocks)',
        real_npz_path,
        file_size / 1e9,
        num_blocks,
    )

    class _ParallelBufferedFile:
      """Seekable buffered reader backed by parallel 64MB block reads."""

      def __init__(self, path: str, size: int, blk_size: int):
        self._path = path
        self._size = size
        self._blk_size = blk_size
        self._num_blks = (size + blk_size - 1) // blk_size
        self._pos = 0
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        self._futures: Dict[int, concurrent.futures.Future[bytes]] = {}
        self._prefetch_window = 16

      def _read_block(self, blk_idx: int) -> bytes:
        offset = blk_idx * self._blk_size
        read_len = min(self._blk_size, self._size - offset)
        with open(self._path, 'rb') as f:
          f.seek(offset)
          return f.read(read_len)

      def ensure_block(
          self, blk_idx: int, prefetch_ahead: bool = True
      ) -> bytes:
        """Fetch block blk_idx and optionally prefetch subsequent blocks."""
        if blk_idx not in self._futures:
          self._futures[blk_idx] = self._pool.submit(self._read_block, blk_idx)
        if prefetch_ahead:
          upper = min(self._num_blks, blk_idx + 1 + self._prefetch_window)
          for nxt in range(blk_idx + 1, upper):
            if nxt not in self._futures:
              self._futures[nxt] = self._pool.submit(self._read_block, nxt)
          # Evict already-passed blocks to keep memory bounded (~1 GB).
          for old_idx in [k for k in self._futures if k < blk_idx - 1]:
            fut = self._futures.pop(old_idx, None)
            if fut is not None:
              fut.cancel()
        return self._futures[blk_idx].result()

      def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
          self._pos = offset
        elif whence == 1:
          self._pos += offset
        elif whence == 2:
          self._pos = self._size + offset
        self._pos = max(0, min(self._size, self._pos))
        return self._pos

      def tell(self) -> int:
        return self._pos

      def seekable(self) -> bool:
        return True

      def readable(self) -> bool:
        return True

      def read(self, n: int = -1) -> bytes:
        """Read up to n bytes from the current stream position."""
        if n is None or n < 0:
          n = self._size - self._pos
        n = min(n, self._size - self._pos)
        if n <= 0:
          return b''
        start_blk = self._pos // self._blk_size
        end_blk = (self._pos + n - 1) // self._blk_size
        prefetch = start_blk < self._num_blks - 2
        if start_blk == end_blk:
          blk = self.ensure_block(start_blk, prefetch_ahead=prefetch)
          off = self._pos - start_blk * self._blk_size
          self._pos += n
          return blk[off : off + n]
        chunks = []
        rem = n
        while rem > 0:
          cur_blk = self._pos // self._blk_size
          blk = self.ensure_block(cur_blk, prefetch_ahead=prefetch)
          off = self._pos - cur_blk * self._blk_size
          take = min(rem, len(blk) - off)
          chunks.append(blk[off : off + take])
          self._pos += take
          rem -= take
        return b''.join(chunks)

      def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._futures.clear()

    reader = _ParallelBufferedFile(real_npz_path, file_size, block_size)
    try:
      with zipfile.ZipFile(reader, 'r') as zf:
        infos = sorted(zf.infolist(), key=lambda info: info.header_offset)
        reader.ensure_block(0, prefetch_ahead=True)
        tensors: Dict[str, Any] = {}
        for idx, info in enumerate(infos):
          key = (
              info.filename[:-4]
              if info.filename.endswith('.npy')
              else info.filename
          )
          with zf.open(info, 'r') as zf_entry:
            raw_bytes = zf_entry.read()
          arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
          del raw_bytes
          # Immediately shard onto TPU HBM in bfloat16 and free CPU array.
          tensors[key] = _shard_leaf(arr)
          del arr
          if (idx + 1) % 200 == 0 or (idx + 1) == len(infos):
            logging.info(
                'Loaded & sharded %d / %d transformer tensors (%.2f / %.2f GB)',
                idx + 1,
                len(infos),
                reader.tell() / 1e9,
                file_size / 1e9,
            )
        return tensors
    finally:
      reader.close()

  loaded_state_holder: Dict[str, Any] = {}

  def _sharded_load_params_into_model(
      model: Any,
      checkpoint_params: Dict[str, Any],
      dtype: jnp.dtype,
  ) -> None:
    graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
    flat_state = dict(nnx.to_flat_state(state))
    loaded_count = 0
    for path_str, value in checkpoint_params.items():
      parts = path_str.split('.')
      if parts[-1] == 'value':
        parts = parts[:-1]
      key = tuple(int(k) if k.isdigit() else k for k in parts)
      if key in flat_state:
        expected_shape = flat_state[key].value.shape
        if value.shape == expected_shape:
          if isinstance(value, jax.Array):
            val_ary = (
                value.astype(dtype)
                if (dtype is not None and value.dtype != dtype)
                else value
            )
          else:
            val_ary = _shard_leaf(value)
          flat_state[key].value = val_ary
          loaded_count += 1
    state = nnx.from_flat_state(flat_state)
    loaded_state_holder['params'] = state
    loaded_state_holder['rest'] = jax.tree.map(
        lambda x: jax.device_put(x, replicated_sharding)
        if isinstance(x, (jax.Array, np.ndarray))
        else x,
        rest_of_state,
    )
    merged = nnx.merge(graphdef, state, rest_of_state)
    _, new_state, new_rest = nnx.split(merged, nnx.Param, ...)
    nnx.update(model, new_state)
    nnx.update(model, new_rest)
    logging.info(
        'Applied %d / %d sharded checkpoint parameters into WanModel'
        ' (model has %d params).',
        loaded_count,
        len(checkpoint_params),
        len(flat_state),
    )

  genception_pipeline.load_transformer_checkpoint = (
      _load_transformer_checkpoint
  )
  genception_pipeline._load_params_into_model = _sharded_load_params_into_model  # pylint: disable=protected-access

  # Initialize WanModel on CPU first to avoid OOM on device 0, while streaming
  # checkpoint weights directly into sharded TPU HBM.
  cpu_device = jax.local_devices(backend='cpu')[0]
  with jax.default_device(cpu_device):
    pipe = GenCeptionPipeline.from_pretrained(
        checkpoint_dir=local_ckpt_dir,
        model_size=model_size,
        dtype=jnp.bfloat16,
    )

  pipe.config['diffusion_single_step'] = diffusion_single_step
  logging.info(
      'Configured GenCeptionPipeline with diffusion_single_step=%d'
      ' (timestep t=%d)',
      diffusion_single_step,
      diffusion_single_step,
  )

  # Move VAE weights onto accelerator device 0.
  dev0 = jax.local_devices()[0]
  _, vae_state = nnx.split(pipe.vae)
  vae_state_dev0 = jax.tree.map(lambda x: jax.device_put(x, dev0), vae_state)
  nnx.update(pipe.vae, vae_state_dev0)

  if 'params' in loaded_state_holder:
    pipe._params = jax.tree.map(_shard_leaf, loaded_state_holder['params'])  # pylint: disable=protected-access
    pipe._rest = loaded_state_holder['rest']  # pylint: disable=protected-access
  else:
    pipe._params = jax.tree.map(_shard_leaf, pipe._params)  # pylint: disable=protected-access
    pipe._rest = jax.tree.map(  # pylint: disable=protected-access
        lambda x: jax.device_put(x, replicated_sharding)
        if isinstance(x, (jax.Array, np.ndarray))
        else x,
        pipe._rest,  # pylint: disable=protected-access
    )
  logging.info(
      'Sharded transformer parameters across %d JAX devices.', num_devices
  )

  orig_forward = genception_pipeline._transformer_forward  # pylint: disable=protected-access

  def _sharded_forward(
      graphdef: Any,
      params: Any,
      rest_of_state: Any,
      latents: jnp.ndarray,
      timestep: jnp.ndarray,
      prompt_embeds: jnp.ndarray,
  ):
    latents_r, timestep_r, prompt_r = jax.tree.map(
        lambda x: jax.device_put(x, replicated_sharding),
        (latents, timestep, prompt_embeds),
    )
    pred1, pred2, pred3 = orig_forward(
        graphdef, params, rest_of_state, latents_r, timestep_r, prompt_r
    )
    return (
        jax.device_put(pred1, dev0),
        jax.device_put(pred2, dev0),
        jax.device_put(pred3, dev0),
    )

  genception_pipeline._transformer_forward = _sharded_forward  # pylint: disable=protected-access

  # Capture unquantized float32 decoded video alongside uint8 video.
  def _decode_video_with_float_cache(latents: jnp.ndarray) -> np.ndarray:
    dtype = pipe.config.get('weights_dtype', jnp.bfloat16)
    latents = jax.device_put(latents.astype(dtype), dev0)
    if pipe.config.get('normalize_latents', False):
      latents_mean = jnp.array(pipe._LATENTS_MEAN).reshape(1, 16, 1, 1, 1)  # pylint: disable=protected-access
      latents_std = 1.0 / jnp.array(pipe._LATENTS_STD).reshape(1, 16, 1, 1, 1)  # pylint: disable=protected-access
      latents = (latents / latents_std + latents_mean).astype(dtype)
    pipe.vae_cache.clear_cache()
    decoded = pipe.vae.decode(latents, pipe.vae_cache)
    video = decoded.sample if hasattr(decoded, 'sample') else decoded[0]
    video = jnp.asarray(video, dtype=np.float32)
    video_float = np.asarray(
        jnp.clip((video + 1.0) / 2.0, 0.0, 1.0), dtype=np.float32
    )
    if video_float.shape[0] == 1:
      pipe._last_video_float = video_float[0]  # pylint: disable=protected-access
    else:
      pipe._last_video_float = video_float  # pylint: disable=protected-access
    return postprocess_video(video)

  pipe.decode_video = _decode_video_with_float_cache
  return pipe


def get_last_video_float(
    pipe: GenCeptionPipeline,
    result: Dict[str, Any],
) -> np.ndarray:
  """Retrieve the unquantized float32 [0, 1] decoded video from the pipeline run."""
  if hasattr(pipe, '_last_video_float'):
    return np.asarray(pipe._last_video_float, dtype=np.float32)  # pylint: disable=protected-access
  return np.asarray(result['video'], dtype=np.float32) / 255.0


def read_video(video_path: str) -> np.ndarray:
  """Read an RGB video into float32 [0, 1] [T, H, W, 3]."""
  video = media.read_video(video_path)
  video_np = np.asarray(video, dtype=np.float32) / 255.0
  return video_np


def resize_exact_process_frame(
    frame: np.ndarray,
    target_width: int = 832,
    target_height: int = 480,
    is_depth: bool = False,
) -> np.ndarray:
  """Resize frame directly to (target_width, target_height) ignoring aspect ratio."""
  interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
  return cv2.resize(
      frame, (target_width, target_height), interpolation=interpolation
  )


def resize_with_aspect_ratio(
    frame: np.ndarray,
    target_height: int = 480,
    target_width: int = 832,
    is_depth: bool = True,
) -> np.ndarray:
  """Resize frame preserving aspect ratio to fit within (target_width, target_height)."""
  height, width = frame.shape[:2]
  aspect_ratio = width / height
  target_aspect_ratio = target_width / target_height
  if aspect_ratio > target_aspect_ratio:
    target_height = int(target_width / aspect_ratio)
  else:
    target_width = int(target_height * aspect_ratio)

  if height == 0 or width == 0:
    num_channels = frame.shape[2] if len(frame.shape) > 2 else 1
    shape = (
        (target_height, target_width, num_channels)
        if num_channels > 1
        else (target_height, target_width)
    )
    return np.zeros(shape, dtype=frame.dtype)

  inter_method = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
  return cv2.resize(
      frame, (target_width, target_height), interpolation=inter_method
  )


def spatial_process_frame(
    frame: np.ndarray,
    target_width: int = 832,
    target_height: int = 480,
) -> np.ndarray:
  """Resize frame while maintaining aspect ratio, then zero-pad (Resize -> Pad)."""
  height, width = frame.shape[:2]
  if height == target_height and width == target_width:
    return frame
  if height == 0 or width == 0:
    if len(frame.shape) == 3:
      return np.zeros(
          (target_height, target_width, frame.shape[2]), dtype=frame.dtype
      )
    return np.zeros((target_height, target_width), dtype=frame.dtype)

  scale = min(target_width / width, target_height / height)
  new_width = max(1, int(round(width * scale)))
  new_height = max(1, int(round(height * scale)))

  resized_frame = cv2.resize(
      frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
  )

  delta_w = target_width - new_width
  delta_h = target_height - new_height
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)

  value = 0 if len(frame.shape) == 2 else [0, 0, 0]
  return cv2.copyMakeBorder(
      resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value
  )


def spatial_process_frame_old(
    frame: np.ndarray,
    target_width: int = 832,
    target_height: int = 480,
) -> np.ndarray:
  """Zero-pad frame to target aspect ratio, then resize (Pad -> Resize)."""
  height, width = frame.shape[:2]
  if height == target_height and width == target_width:
    return frame
  if height == 0 or width == 0:
    num_channels = frame.shape[2] if len(frame.shape) > 2 else 1
    shape = (
        (target_height, target_width, num_channels)
        if num_channels > 1
        else (target_height, target_width)
    )
    return np.zeros(shape, dtype=frame.dtype)

  target_aspect_ratio = target_width / target_height
  aspect_ratio = width / height
  pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

  if abs(aspect_ratio - target_aspect_ratio) < 1e-4:
    pass
  elif aspect_ratio < target_aspect_ratio:
    new_width = int(round(height * target_aspect_ratio))
    pad_left = (new_width - width) // 2
    pad_right = new_width - width - pad_left
  else:
    new_height = int(round(width / target_aspect_ratio))
    pad_top = (new_height - height) // 2
    pad_bottom = new_height - height - pad_top

  value = 0 if len(frame.shape) == 2 else [0, 0, 0]
  padded_frame = cv2.copyMakeBorder(
      frame,
      pad_top,
      pad_bottom,
      pad_left,
      pad_right,
      cv2.BORDER_CONSTANT,
      value=value,
  )
  return cv2.resize(
      padded_frame,
      (target_width, target_height),
      interpolation=cv2.INTER_LINEAR,
  )


def pad_or_truncate_to_81(
    video: np.ndarray, target_len: int = 81
) -> np.ndarray:
  """Pad video by repeating last frame or slice to 81 frames."""
  if video.shape[0] < target_len:
    last_frame = video[-1:]
    repeat_frames = np.repeat(last_frame, target_len - video.shape[0], axis=0)
    return np.concatenate([video, repeat_frames], axis=0)
  return video[:target_len]


def run_single_81_video(
    pipe: GenCeptionPipeline,
    video_81: np.ndarray,
    prompt_embeds: np.ndarray,
) -> np.ndarray:
  """Run GenCeptionPipeline on an 81-frame [81, 480, 832, 3] float32 video."""
  result = pipe(
      prompt_embeds=prompt_embeds,
      input_video=video_81,
  )
  return get_last_video_float(pipe, result)


def run_chunked_81_video(
    pipe: GenCeptionPipeline,
    video: np.ndarray,
    prompt_embeds: np.ndarray,
    chunk_size: int = 81,
) -> np.ndarray:
  """Run GenCeptionPipeline on contiguous 81-frame chunks and concatenate."""
  total_frames = video.shape[0]
  outputs = []
  for start_idx in range(0, total_frames, chunk_size):
    chunk = video[start_idx : start_idx + chunk_size]
    chunk_81 = pad_or_truncate_to_81(chunk, target_len=chunk_size)
    pred_chunk = run_single_81_video(pipe, chunk_81, prompt_embeds)
    outputs.append(pred_chunk)
  return np.concatenate(outputs, axis=0)


def save_metrics_json(
    output_dir: str, filename: str, data: Dict[str, Any]
) -> None:
  """Write evaluation results dictionary to a JSON file in output_dir."""
  if not output_dir:
    return
  os.makedirs(output_dir, exist_ok=True)
  out_path = os.path.join(output_dir, filename)
  with open(out_path, 'w') as f:
    json.dump(data, f, indent=2)
  logging.info('Saved evaluation metrics to %s', out_path)
