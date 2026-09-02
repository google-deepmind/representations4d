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

"""GenCeption inference pipeline.

This module provides :class:`GenCeptionPipeline`, a high-level interface for
running the GenCeption model. GenCeption is a WAN 2.1 diffusion transformer
fine-tuned for multi-modal video tasks. Inference uses a single forward pass:
the model predicts the clean output latent directly from the VAE-encoded input.

Text embeddings are precomputed (e.g. using the 22 GB UMT5-XXL text encoder) and
loaded directly during inference as NumPy/JAX arrays or ``.npy`` files, removing
any runtime dependency on HuggingFace transformers or PyTorch.

Typical usage::

    from genception import GenCeptionPipeline
    from genception.utils import load_video, save_video
    import numpy as np

    pipe = GenCeptionPipeline.from_pretrained(
        '/path/to/checkpoint', model_size='1.3b'
    )
    video = load_video('input.mp4')
    depth_embeds = np.load('depth.npy')

    result = pipe(
        prompt_embeds=depth_embeds,
        input_video=video,
    )
    save_video(result['video'], 'depth.mp4')
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

from flax import nnx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

# pylint: disable=g-importing-member,g-bare-generic
from representations4d.genception.model import WanModel
from representations4d.genception.utils import load_transformer_checkpoint
from representations4d.genception.utils import postprocess_video
from representations4d.genception.utils import preprocess_video
from representations4d.genception.vae import AutoencoderKLWan
from representations4d.genception.vae import AutoencoderKLWanCache
from representations4d.genception.vae import load_weights_from_pytorch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JIT-compiled transformer forward pass
# ---------------------------------------------------------------------------


@jax.jit
def _transformer_forward(
    graphdef: Any,
    params: Any,
    rest_of_state: Any,
    latents: jnp.ndarray,
    timestep: jnp.ndarray,
    prompt_embeds: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Execute a single forward pass through the transformer.

  Args:
      graphdef: NNX graph definition of the WanModel.
      params: NNX parameter state (``nnx.Param``).
      rest_of_state: Remaining NNX state variables.
      latents: VAE-encoded input video latents, shape ``[B, C, T', H', W']``.
      timestep: Integer diffusion timestep array of shape ``[B]``.
      prompt_embeds: Text encoder embeddings, shape ``[B, seq_len, text_dim]``.

  Returns:
      Tuple of ``(pred_latents1, pred_latents2, pred_latents3)``.
  """
  model = nnx.merge(graphdef, params, rest_of_state)
  pred_latents1, pred_latents2, pred_latents3 = model(
      hidden_states=latents,
      timestep=timestep,
      encoder_hidden_states=prompt_embeds,
  )
  return pred_latents1, pred_latents2, pred_latents3


# ---------------------------------------------------------------------------
# Default model configurations for 14B and 1.3B variants
# ---------------------------------------------------------------------------

# These are the default transformer configurations extracted from the training
# config YAML files. They can be overridden by a ``config.json`` file
# shipped alongside the checkpoint.
_DEFAULT_CONFIGS = {
    '14b': {
        'num_attention_heads': 40,
        'attention_head_dim': 128,
        'in_channels': 16,
        'out_channels': 16,
        'text_dim': 4096,
        'freq_dim': 256,
        'ffn_dim': 13824,
        'num_layers': 40,
        'cross_attn_norm': True,
        'qk_norm': 'rms_norm_across_heads',
        'eps': 1e-6,
        'rope_max_seq_len': 1024,
        'token_att_num_layers': 10,
        'token1_num_frames_rope': 1,
        'token1_num_per_frame': 1,
        'token1_proj_dim': 13,
        'token1_output_dim': 1,
        'token2_num_frames_rope': 1,
        'token2_num_per_frame': 1,
        'token2_proj_dim': 99,
        'token2_output_dim': 3,
        'total_token_num_lst': '_32760_81_81',
        'learnable_token_init_0': True,
        'height': 480,
        'width': 832,
        'normalize_latents': False,
        'diffusion_single_step': -1,
    },
    '1.3b': {
        'num_attention_heads': 12,
        'attention_head_dim': 128,
        'in_channels': 16,
        'out_channels': 16,
        'text_dim': 4096,
        'freq_dim': 256,
        'ffn_dim': 8960,
        'num_layers': 30,
        'cross_attn_norm': True,
        'qk_norm': 'rms_norm_across_heads',
        'eps': 1e-6,
        'rope_max_seq_len': 1024,
        'token_att_num_layers': 10,
        'token1_num_frames_rope': 1,
        'token1_num_per_frame': 1,
        'token1_proj_dim': 13,
        'token1_output_dim': 1,
        'token2_num_frames_rope': 1,
        'token2_num_per_frame': 1,
        'token2_proj_dim': 99,
        'token2_output_dim': 3,
        'total_token_num_lst': '_32760_81_81',
        'learnable_token_init_0': True,
        'height': 480,
        'width': 832,
        'normalize_latents': False,
        'diffusion_single_step': -1,
    },
}


# ---------------------------------------------------------------------------
# GenCeptionPipeline
# ---------------------------------------------------------------------------


class GenCeptionPipeline:
  """End-to-end inference pipeline for the GenCeption model.

  GenCeption is a WAN 2.1 diffusion transformer (14B or 1.3B) fine-tuned
  for multi-modal video understanding. The model is a **single-step
  predictor**: given a VAE-encoded input video at a fixed timestep, it
  predicts the clean output latent in one forward pass (no iterative
  denoising).

  The pipeline wraps the following components:

  * **Transformer** -- the WAN diffusion transformer (``WanModel``) running
    in JAX / Flax NNX.
  * **VAE** -- the WAN video autoencoder (``AutoencoderKLWan``) for
    encoding / decoding videos to / from a latent space.

  Prompt embeddings are fed directly to :meth:`__call__` as precomputed arrays
  or `.npy` filepaths.

  Attributes:
      transformer: The WAN diffusion transformer model.
      vae: The WAN video autoencoder.
      vae_cache: Causal-convolution cache for the VAE.
      config: Model configuration dictionary.
  """

  # VAE scale factors.
  # Derived from ``temperal_downsample = [False, True, True]``.
  _VAE_SCALE_FACTOR_TEMPORAL: int = 4
  _VAE_SCALE_FACTOR_SPATIAL: int = 8

  # Default VAE latent normalization constants (16-dim latent space).
  _LATENTS_MEAN: Tuple[float, ...] = (
      -0.7571,
      -0.7089,
      -0.9113,
      0.1075,
      -0.1745,
      0.9653,
      -0.1517,
      1.5508,
      0.4134,
      -0.0715,
      0.5517,
      -0.3632,
      -0.1922,
      -0.9497,
      0.2503,
      -0.2921,
  )

  _LATENTS_STD: Tuple[float, ...] = (
      2.8184,
      1.4541,
      2.3275,
      2.6558,
      1.2196,
      1.7708,
      2.6052,
      2.0743,
      3.2687,
      2.1526,
      2.8652,
      1.5579,
      1.6382,
      1.1253,
      2.8251,
      1.9160,
  )

  def __init__(
      self,
      transformer: WanModel,
      vae: AutoencoderKLWan,
      vae_cache: AutoencoderKLWanCache,
      config: Dict[str, Any],
  ):
    """Initialize the pipeline from pre-loaded components.

    Prefer :meth:`from_pretrained` for loading from disk.

    Args:
        transformer: WanModel instance (Flax NNX).
        vae: AutoencoderKLWan instance (Flax NNX).
        vae_cache: AutoencoderKLWanCache for the VAE decoder.
        config: Model configuration dictionary.
    """
    self.transformer = transformer
    self.vae = vae
    self.vae_cache = vae_cache
    self.config = config

    # Pre-split the transformer into graphdef + state for JIT-compiled
    # forward passes.
    self._graphdef, self._params, self._rest = nnx.split(
        self.transformer, nnx.Param, ...
    )

  # -----------------------------------------------------------------------
  # Factory
  # -----------------------------------------------------------------------

  @classmethod
  def from_pretrained(
      cls,
      checkpoint_dir: str,
      model_size: str = '1.3b',
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> 'GenCeptionPipeline':
    """Load all components from a checkpoint directory.

    Expected directory structure::

        checkpoint_dir/
            genception_{model_size}_transformer.npz   # transformer weights
            genception_{model_size}_config.json       # optional config
            vae/
                config.json
                diffusion_pytorch_model.safetensors

    Args:
        checkpoint_dir: Root directory containing the model artifacts.
        model_size: ``'1.3b'`` or ``'14b'``.
        dtype: Weight dtype (default ``bfloat16``).

    Returns:
        A fully initialized :class:`GenCeptionPipeline`.
    """
    logger.info('Loading GenCeption pipeline from %s', checkpoint_dir)
    model_size = model_size.lower()
    if model_size not in _DEFAULT_CONFIGS:
      raise ValueError(
          f"Unknown model_size='{model_size}'. Choose from"
          f' {list(_DEFAULT_CONFIGS.keys())}.'
      )

    # -- 1. Load configuration -----------------------------------------------
    config = dict(_DEFAULT_CONFIGS[model_size])  # copy defaults
    config_path = os.path.join(
        checkpoint_dir, f'genception_{model_size}_config.json'
    )
    if os.path.exists(config_path):
      with open(config_path, 'r') as f:
        overrides = json.load(f)
      config.update(overrides)
      logger.info('Loaded config overrides from %s', config_path)

    config['dtype'] = dtype
    config['weights_dtype'] = dtype

    # -- 2. Load VAE ---------------------------------------------------------
    vae_path = os.path.join(checkpoint_dir, 'vae')
    logger.info('Loading VAE from %s', vae_path)
    rngs = nnx.Rngs(jax.random.key(0))
    vae_config_path = os.path.join(vae_path, 'config.json')
    if os.path.exists(vae_config_path):
      with open(vae_config_path, 'r') as f:
        vae_config = json.load(f)
    else:
      vae_config = {}

    vae = AutoencoderKLWan.from_config(
        vae_config,
        rngs=rngs,
        dtype=dtype,
        weights_dtype=dtype,
    )
    vae_safetensors_path = os.path.join(
        vae_path, 'diffusion_pytorch_model.safetensors'
    )
    if os.path.exists(vae_safetensors_path):
      logger.info('Loading VAE weights from %s', vae_safetensors_path)
      vae_state_dict = load_weights_from_pytorch(vae, vae_safetensors_path)
      target_device = jax.local_devices()[0]
      vae_state = jax.tree.map(
          lambda x: jax.device_put(x, target_device), nnx.State(vae_state_dict)
      )
      nnx.update(vae, vae_state)
    else:
      logger.warning(
          'VAE weights not found at %s. VAE is randomly initialized!',
          vae_safetensors_path,
      )

    vae_cache = AutoencoderKLWanCache(vae)

    # -- 3. Load transformer -------------------------------------------------
    transformer_path = os.path.join(
        checkpoint_dir, f'genception_{model_size}_transformer.npz'
    )
    logger.info('Loading transformer from %s', transformer_path)

    # Build model config for WanModel constructor.
    # Filter out pipeline-specific config keys so WanModel doesn't choke on them
    pipeline_keys = (
        'dtype',
        'weights_dtype',
        'normalize_latents',
        'diffusion_single_step',
    )
    wan_config = {
        k: config[k]
        for k in config
        if k not in pipeline_keys and not k.startswith('_')
    }
    wan_config['dtype'] = dtype
    wan_config['weights_dtype'] = dtype

    # Flash attention (cuDNN SDPA) only has a GPU lowering rule.
    # Fall back to standard dot-product attention on CPU and TPU.
    default_device = jax.local_devices()[0]
    if default_device.platform == 'gpu':
      wan_config['attention'] = 'flash'
    else:
      logger.info(
          '%s runtime detected. Using dot_product attention.',
          default_device.platform.upper(),
      )
      wan_config['attention'] = 'dot_product'

    wan_config['precision'] = None
    wan_config['flash_block_sizes'] = None
    wan_config['flash_min_seq_length'] = 4096

    logger.info('Constructing WanModel with config: %s', wan_config)
    transformer = WanModel(**wan_config, rngs=rngs)
    logger.info('WanModel constructed successfully.')

    # Load checkpoint weights.
    if os.path.exists(transformer_path):
      logger.info('Loading checkpoint (this may take a while)...')
      checkpoint_params = load_transformer_checkpoint(transformer_path)
      logger.info('Checkpoint loaded, applying weights...')
      _load_params_into_model(transformer, checkpoint_params, dtype)
      logger.info('Loaded transformer weights from %s', transformer_path)
    else:
      logger.warning(
          'No transformer checkpoint found at %s -- using random weights.',
          transformer_path,
      )

    return cls(
        transformer=transformer,
        vae=vae,
        vae_cache=vae_cache,
        config=config,
    )

  # -----------------------------------------------------------------------
  # Prompt embedding validation & normalization
  # -----------------------------------------------------------------------

  def _prepare_prompt_embeds(
      self,
      embeds: Union[np.ndarray, jnp.ndarray, str, os.PathLike],
      name: str = 'prompt_embeds',
  ) -> jnp.ndarray:
    """Validate, load if path, and format prompt embeddings.

    Args:
        embeds: A numpy/JAX array of embeddings, or a filepath / CNS path to a
          ``.npy`` file. Expected shape: ``[seq_len, 4096]`` or ``[B, seq_len,
          4096]``.
        name: Argument name for informative error messages.

    Returns:
        JAX array of shape ``[B, seq_len, 4096]`` in the pipeline's weight
        dtype.

    Raises:
        ValueError: If the shape or dimensionality is invalid.
        FileNotFoundError: If the filepath does not exist.
    """
    if isinstance(embeds, (str, os.PathLike)):
      path_str = os.fspath(embeds)
      if not os.path.exists(path_str):
        raise FileNotFoundError(f"Embedding file not found: '{path_str}'")
      with open(path_str, 'rb') as f:
        embeds = np.load(f)

    if not isinstance(embeds, (np.ndarray, jnp.ndarray)):
      raise TypeError(
          f'Expected {name} to be a numpy/JAX array or a filepath string, '
          f'got {type(embeds).__name__}'
      )

    expected_dim = self.config.get('text_dim', 4096)

    if embeds.ndim == 2:
      # [seq_len, text_dim] -> [1, seq_len, text_dim]
      embeds = embeds[np.newaxis, ...]
    elif embeds.ndim != 3:
      raise ValueError(
          f'Invalid {name} shape: {embeds.shape}. Expected 2D [seq_len,'
          f' {expected_dim}] or 3D [B, seq_len, {expected_dim}].'
      )

    if embeds.shape[-1] != expected_dim:
      raise ValueError(
          f'Invalid embedding dimension for {name}: {embeds.shape[-1]}. Model'
          f' expects text_dim={expected_dim} (shape: [B, seq_len,'
          f' {expected_dim}]).'
      )

    target_dtype = self.config.get('weights_dtype', jnp.bfloat16)
    return jnp.array(embeds, dtype=target_dtype)

  # -----------------------------------------------------------------------
  # VAE encoding / decoding
  # -----------------------------------------------------------------------

  def encode_video(self, video: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Encode video frames to the VAE latent space.

    Args:
        video: Input video array of shape ``[T, H, W, 3]`` with values in ``[0,
          1]`` (float) or ``[0, 255]`` (uint8).

    Returns:
        JAX array of latents with shape ``[1, C, T', H', W']`` where
        ``C=16`` and the spatial/temporal dims are down-sampled by the
        VAE factors.
    """
    # Preprocess: resize, normalize to [-1, 1].
    height = self.config.get('height', 480)
    width = self.config.get('width', 832)
    num_frames = 81  # Default frame count

    processed = preprocess_video(
        video, height=height, width=width, num_frames=num_frames
    )

    # Add batch dim: [T, H, W, C] -> [1, T, H, W, C]
    video_input = jnp.array(
        processed[np.newaxis],
        dtype=self.config.get('weights_dtype', jnp.bfloat16),
    )

    # Encode via VAE.
    self.vae_cache.clear_cache()
    encoder_output = self.vae.encode(video_input, self.vae_cache)

    # Use the mode (mean) of the posterior -- no sampling noise.
    latents = encoder_output.latent_dist.mode()

    # Rearrange from [B, T', H', W', C] -> [B, C, T', H', W']
    latents = jnp.transpose(latents, (0, 4, 1, 2, 3))

    # Apply latent normalization if required by config
    if self.config.get('normalize_latents', False):
      latents_mean = jnp.array(self._LATENTS_MEAN).reshape(1, 16, 1, 1, 1)
      latents_std = 1.0 / jnp.array(self._LATENTS_STD).reshape(1, 16, 1, 1, 1)
      latents = (latents - latents_mean) * latents_std

    return latents.astype(self.config.get('weights_dtype', jnp.bfloat16))

  def decode_video(self, latents: jnp.ndarray) -> np.ndarray:
    """Decode VAE latents back to a video.

    Args:
        latents: Latent array of shape ``[B, C, T', H', W']`` — the model's
          predicted clean latents.

    Returns:
        ``uint8`` NumPy array of shape ``[B, T, H, W, 3]`` with values
        in ``[0, 255]``.
    """
    dtype = self.config.get('weights_dtype', jnp.bfloat16)
    latents = latents.astype(dtype)

    # Re-apply latent standardisation if required by config
    if self.config.get('normalize_latents', False):
      latents_mean = jnp.array(self._LATENTS_MEAN).reshape(1, 16, 1, 1, 1)
      latents_std = 1.0 / jnp.array(self._LATENTS_STD).reshape(1, 16, 1, 1, 1)
      latents = latents / latents_std + latents_mean
      latents = latents.astype(dtype)

    # Decode via VAE: accepts [B, C, T', H', W'] (auto-transposes internally).
    self.vae_cache.clear_cache()
    decoded = self.vae.decode(latents, self.vae_cache)

    if hasattr(decoded, 'sample'):
      video = decoded.sample
    else:
      video = decoded[0]

    # Convert to float32 numpy and denormalize.
    video = jnp.asarray(video, dtype=np.float32)
    video = postprocess_video(video)  # [-1, 1] -> uint8 [0, 255]
    return video

  # -----------------------------------------------------------------------
  # Full inference pipeline
  # -----------------------------------------------------------------------

  def __call__(
      self,
      prompt_embeds: Union[np.ndarray, jnp.ndarray, str, os.PathLike],
      input_video: Optional[Union[np.ndarray, jnp.ndarray]] = None,
      latents: Optional[jnp.ndarray] = None,
      num_inference_steps: int = 50,
      return_tokens: bool = False,
  ) -> Dict[str, Any]:
    """Run the full GenCeption inference pipeline.

    The model is a single-step predictor: it encodes the input video via the
    VAE, runs **one** transformer forward pass at timestep
    ``num_train_timesteps - 1``, and decodes the predicted clean latent
    back to pixel space.

    Steps:

    1. Encode ``input_video`` to VAE latent space (or use provided ``latents``).
    2. Validate and prepare ``prompt_embeds``.
    3. Run a single transformer forward pass at ``t = num_train_timesteps - 1``.
    4. Decode the predicted latents via the VAE.

    Args:
        prompt_embeds: Precomputed text prompt embedding array of shape
          ``[seq_len, 4096]`` or ``[B, seq_len, 4096]``, or filepath / CNS path
          to a ``.npy`` file.
        input_video: Input video array of shape ``[T, H, W, 3]`` with values in
          ``[0, 1]`` (float) or ``[0, 255]`` (uint8). Can be omitted if
          ``latents`` are provided directly.
        latents: Optional pre-encoded VAE latents. If provided, skips the
          ``encode_video`` step.
        num_inference_steps: Number of inference steps (default 50).
        return_tokens: If ``True``, also return the token predictions (e.g.
          keypoint regressions).

    Returns:
        Dictionary with:

        - ``'video'``: Output video, ``uint8`` shape ``[T, H, W, 3]``.
        - ``'latents'``: Predicted latents (before VAE decode).
        - ``'token1_pred'``: (only if ``return_tokens=True``) Token-1
          predictions.
        - ``'token2_pred'``: (only if ``return_tokens=True``) Token-2
          predictions.
    """
    # 1. VAE-encode input video (raw latents, no normalization).
    if latents is None:
      if input_video is None:
        raise ValueError(
            'Either `input_video` or pre-computed `latents` must be provided.'
        )
      latents = self.encode_video(input_video)  # [1, 16, T', H', W']
      logger.info('Encoded video to latents: %s', latents.shape)
    else:
      logger.info('Using provided latents: %s', latents.shape)

    # 2. Validate and prepare prompt embeddings.
    prompt_embeds = self._prepare_prompt_embeds(
        prompt_embeds, name='prompt_embeds'
    )
    logger.info('Prepared prompt embeddings: %s', prompt_embeds.shape)

    # 3. Single forward pass.
    # DeepSense timestep resolution logic
    bsz = latents.shape[0]
    single_step = self.config.get('diffusion_single_step', -1)
    if single_step != -1:
      timestep_val = single_step
    else:
      timestep_val = num_inference_steps - 1

    timestep_array = jnp.full((bsz,), timestep_val, dtype=jnp.int32)

    target_dev = jax.local_devices()[0]
    latents, timestep_array, prompt_embeds = jax.tree.map(
        lambda x: jax.device_put(x, target_dev),
        (latents, timestep_array, prompt_embeds),
    )

    pred_latents1, pred_latents2, pred_latents3 = _transformer_forward(
        self._graphdef,
        self._params,
        self._rest,
        latents,
        timestep_array,
        prompt_embeds,
    )

    logger.info(
        'Single-step forward pass complete. Predicted latents: %s',
        pred_latents1.shape,
    )

    # 4. Decode the predicted latents via the VAE.
    # The model directly predicts the clean output latents (the WAN
    # architecture's ``return -video_latents`` already accounts for the
    # flow matching sign convention). No ODE resolve step is needed.
    output_video = self.decode_video(pred_latents1)

    # Remove batch dim if single sample.
    if output_video.shape[0] == 1:
      output_video = output_video[0]

    result: Dict[str, Any] = {
        'video': np.array(output_video),
        'latents': np.array(pred_latents1),
    }

    if return_tokens:
      result['token1_pred'] = np.array(pred_latents2)
      result['token2_pred'] = np.array(pred_latents3)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_params_into_model(
    model: Any,
    checkpoint_params: Dict[str, jnp.ndarray],
    dtype: jnp.dtype,
) -> None:
  """Load flat checkpoint params into a live NNX model (in-place).

  The arrays in the .npz checkpoint are exactly the flattened NNX state
  exported via `nnx.to_flat_state()`, with keys separated by `.` and ending
  in `.value` (e.g., `blocks.attn1.key.kernel.value`).

  Args:
      model: An NNX model instance.
      checkpoint_params: Flat ``{path_str: ndarray}`` dict from
        :func:`load_transformer_checkpoint`.
      dtype: Target dtype for the parameters.
  """
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
  flat_state = dict(nnx.to_flat_state(state))

  loaded_count = 0
  for path_str, value in checkpoint_params.items():
    # Convert path string 'a.b.c.value' to tuple key ('a', 'b', 'c').
    parts = path_str.split('.')
    if parts[-1] == 'value':
      parts = parts[:-1]

    # Try int conversion for block indices etc.
    key = tuple(int(k) if k.isdigit() else k for k in parts)

    if key in flat_state:
      expected_shape = flat_state[key].value.shape
      if value.shape == expected_shape:
        if value.dtype.char == 'V' and value.dtype.itemsize == 2:
          # Reinterpret opaque numpy type as bfloat16
          value = value.view(ml_dtypes.bfloat16)

        # Fix cast error for np arrays
        val_ary = jnp.array(value)
        if dtype is not None and val_ary.dtype != dtype:
          val_ary = val_ary.astype(dtype)

        flat_state[key].value = val_ary
        loaded_count += 1
      else:
        logger.warning(
            'Shape mismatch for %s: checkpoint=%s, model=%s. Skipping.',
            path_str,
            value.shape,
            expected_shape,
        )
    else:
      logger.debug('Key %s not found in model state. Skipping.', path_str)

  state = nnx.from_flat_state(flat_state)
  merged = nnx.merge(graphdef, state, rest_of_state)

  # Copy state back into the original model object.
  _, new_state, new_rest = nnx.split(merged, nnx.Param, ...)
  nnx.update(model, new_state)
  nnx.update(model, new_rest)

  logger.info(
      'Loaded %d / %d checkpoint parameters.',
      loaded_count,
      len(checkpoint_params),
  )
