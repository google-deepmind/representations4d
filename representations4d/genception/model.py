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

"""WAN Diffusion Transformer model for GenCeption.

This module implements the WAN 2.1 diffusion transformer architecture adapted
for multi-modal video understanding. The model performs a SINGLE forward pass
(not iterative diffusion) to produce:
  1. Predicted video latents (denoised output).
  2. Token1 predictions: 2D keypoint coordinates per frame.
  3. Token2 predictions: 3D keypoint coordinates per frame.

The architecture is based on the WAN transformer with:
  - 3D patch embedding for video inputs (temporal + spatial).
  - Rotary Position Embeddings (RoPE) for video and learnable tokens.
  - Self-attention and cross-attention blocks with optional flash attention.
  - Learnable token heads that are appended to the video sequence and
    processed jointly through the transformer blocks.

Forked from: maxdiffusion/models/wan/transformers/transformer_wan.py
"""

import math
from typing import Any, Dict, Optional, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# pylint: disable=g-importing-member
from representations4d.genception.attention import FlaxWanAttention
from representations4d.genception.embeddings import get_1d_float_rotary_pos_embed
from representations4d.genception.embeddings import get_1d_rotary_pos_embed
from representations4d.genception.embeddings import NNXFlaxTimesteps
from representations4d.genception.embeddings import NNXPixArtAlphaTextProjection
from representations4d.genception.embeddings import NNXTimestepEmbedding
from representations4d.genception.normalization import FP32LayerNorm


class SimpleConfig:
  """Simple attribute-based configuration container.

  Replaces ConfigMixin from the original codebase. Stores all constructor
  keyword arguments as attributes for easy access during the forward pass.
  """

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  def __getattr__(self, name: str) -> Any:
    raise AttributeError(f'SimpleConfig has no attribute {name!r}')


def get_activation(name: str):
  """Return a JAX activation function by name.

  Args:
    name: Activation function name. One of 'swish', 'silu', 'relu',
      'gelu', 'gelu_tanh', 'mish'.

  Returns:
    The corresponding JAX activation function.

  Raises:
    ValueError: If the activation name is not recognized.
  """
  activations = {
      'swish': jax.nn.silu,
      'silu': jax.nn.silu,
      'relu': jax.nn.relu,
      'gelu': jax.nn.gelu,
      'gelu_tanh': jax.nn.gelu,
      'mish': jax.nn.mish,
  }
  func = activations.get(name)
  if func is None:
    raise ValueError(f'Unknown activation function: {name}')
  return func


def get_frequencies(
    max_seq_len: int, theta: int, attention_head_dim: int
) -> list[jax.Array]:
  """Compute RoPE frequency components for video (temporal + height + width).

  Splits the attention head dimension into three parts for temporal, height,
  and width dimensions, then computes rotary position embeddings for each.

  Args:
    max_seq_len: Maximum sequence length for position embeddings.
    theta: Base frequency for rotary embeddings.
    attention_head_dim: Dimension of each attention head (e.g. 128).

  Returns:
    List of three complex-valued arrays [freqs_t, freqs_h, freqs_w], where
    each has shape [max_seq_len, dim_i // 2].
  """
  h_dim = w_dim = 2 * (attention_head_dim // 6)
  t_dim = attention_head_dim - h_dim - w_dim
  freqs = []
  for dim in [t_dim, h_dim, w_dim]:
    freq = get_1d_rotary_pos_embed(
        dim, max_seq_len, theta, freqs_dtype=jnp.float64, use_real=False
    )
    freqs.append(freq)
  freqs = jnp.concatenate(freqs, axis=1)

  t_size = attention_head_dim // 2 - 2 * (attention_head_dim // 6)
  hw_size = attention_head_dim // 6
  dims = [t_size, hw_size, hw_size]

  cumulative_sizes = np.cumsum(dims)
  split_indices = cumulative_sizes[:-1].tolist()
  freqs_split = jnp.split(freqs, split_indices, axis=1)
  return freqs_split


class FixedTemporalLearnableSpatialRotaryPosEmbed(nnx.Module):
  """RoPE with fixed temporal positions and learnable spatial positions.

  Used for the learnable token heads (token1, token2). The temporal RoPE is
  computed from fixed positions based on the temporal ratio between token
  frames and video frames. The spatial (height/width) RoPE uses learnable
  parameters that are optimized during training.

  Attributes:
    attention_head_dim: Dimension of each attention head.
    patch_size: 3D patch size (t, h, w) for video patching.
    max_seq_len: Maximum sequence length for RoPE.
    theta: Base frequency for rotary embeddings.
  """

  def __init__(
      self,
      attention_head_dim: int,
      patch_size: Tuple[int, int, int],
      max_seq_len: int,
      theta: float = 10000.0,
  ):
    self.attention_head_dim = attention_head_dim
    self.patch_size = patch_size
    self.max_seq_len = max_seq_len
    self.theta = theta

  def __call__(
      self,
      latent_frames: int,
      latent_height: int,
      latent_width: int,
      learnable_tokens: jax.Array,
      learnable_hw_rope: jax.Array,
  ) -> jax.Array:
    """Compute rotary embeddings for learnable tokens.

    Args:
      latent_frames: Number of frames in the video latent.
      latent_height: Height of the video latent.
      latent_width: Width of the video latent.
      learnable_tokens: Learnable token tensor of shape [1, num_token_frames,
        num_tokens, dim].
      learnable_hw_rope: Learnable spatial position parameters of shape [1,
        num_frames_rope, num_tokens, 2].

    Returns:
      Complex-valued RoPE tensor of shape [1, 1, num_token_frames * num_tokens,
      head_dim // 2].
    """
    _, num_token_frames, num_tokens, _ = learnable_tokens.shape
    _, num_token_frames_hw, num_tokens_hw, _ = learnable_hw_rope.shape

    p_t, p_h, p_w = self.patch_size  # pylint: disable=unused-variable
    ppf = latent_frames // p_t
    temporal_ratio = (num_token_frames - 1) // (ppf - 1)

    h_dim = w_dim = 2 * (self.attention_head_dim // 6)
    t_dim = self.attention_head_dim - h_dim - w_dim

    # Temporal RoPE: fixed positions based on temporal ratio.
    token_timestep = jnp.array(
        [0]
        + [
            1
            - ((temporal_ratio + 1) / 2 - 1) / temporal_ratio
            + (i - 1) / temporal_ratio
            for i in range(1, num_token_frames)
        ]
    )
    freqs_f = get_1d_float_rotary_pos_embed(
        t_dim, token_timestep, theta=self.theta,
        freqs_dtype=jnp.float64, use_real=False,
    )
    freqs_f = jnp.expand_dims(freqs_f, axis=1)
    freqs_f = jnp.broadcast_to(
        freqs_f, (num_token_frames, num_tokens, freqs_f.shape[-1])
    )

    # Spatial RoPE: from learnable height/width positions.
    token_hw = learnable_hw_rope.reshape(-1,)
    freqs_hw = get_1d_float_rotary_pos_embed(
        h_dim, token_hw, theta=self.theta,
        freqs_dtype=jnp.float64, use_real=False,
    )
    freqs_hw = jnp.reshape(
        freqs_hw, (num_token_frames_hw, num_tokens_hw, 2, -1)
    )
    if num_token_frames_hw == 1:
      broadcast_shape = (num_token_frames, num_tokens, 2, freqs_hw.shape[-1])
      freqs_hw = jnp.broadcast_to(freqs_hw, broadcast_shape)

    # Concatenate temporal + height + width RoPE.
    freqs_concat = jnp.concatenate(
        [freqs_f, freqs_hw[:, :, 0], freqs_hw[:, :, 1]], axis=-1
    )
    freqs_final = jnp.reshape(
        freqs_concat, (1, 1, -1, freqs_concat.shape[-1])
    )
    return freqs_final


class WanRotaryPosEmbed(nnx.Module):
  """Standard 3D RoPE for video latent sequences.

  Computes rotary position embeddings for video tokens by decomposing
  positions into temporal, height, and width components.

  Attributes:
    attention_head_dim: Dimension of each attention head.
    patch_size: 3D patch size (t, h, w).
    max_seq_len: Maximum sequence length for RoPE.
    theta: Base frequency for rotary embeddings.
  """

  def __init__(
      self,
      attention_head_dim: int,
      patch_size: Tuple[int, int, int],
      max_seq_len: int,
      theta: float = 10000.0,
  ):
    self.attention_head_dim = attention_head_dim
    self.patch_size = patch_size
    self.max_seq_len = max_seq_len
    self.theta = theta

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    """Compute 3D RoPE for video latent tokens.

    Args:
      hidden_states: Video latent tensor of shape [B, T, H, W, C].

    Returns:
      Complex-valued RoPE tensor of shape [1, 1, T'*H'*W', head_dim // 2],
      where T', H', W' are the post-patch dimensions.
    """
    _, num_frames, height, width, _ = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    freqs_split = get_frequencies(
        self.max_seq_len, self.theta, self.attention_head_dim
    )

    freqs_f = jnp.expand_dims(
        jnp.expand_dims(freqs_split[0][:ppf], axis=1), axis=1
    )
    freqs_f = jnp.broadcast_to(
        freqs_f, (ppf, pph, ppw, freqs_split[0].shape[-1])
    )

    freqs_h = jnp.expand_dims(
        jnp.expand_dims(freqs_split[1][:pph], axis=0), axis=2
    )
    freqs_h = jnp.broadcast_to(
        freqs_h, (ppf, pph, ppw, freqs_split[1].shape[-1])
    )

    freqs_w = jnp.expand_dims(
        jnp.expand_dims(freqs_split[2][:ppw], axis=0), axis=1
    )
    freqs_w = jnp.broadcast_to(
        freqs_w, (ppf, pph, ppw, freqs_split[2].shape[-1])
    )

    freqs_concat = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)
    freqs_final = jnp.reshape(freqs_concat, (1, 1, ppf * pph * ppw, -1))
    return freqs_final


class WanTimeTextImageEmbedding(nnx.Module):
  """Condition embedding module for timestep, text, and optional image.

  Embeds the diffusion timestep via sinusoidal + MLP, projects text
  embeddings, and optionally handles image conditioning.

  Attributes:
    rngs: Random number generators for parameter initialization.
    dim: Inner model dimension (e.g. 5120).
    time_freq_dim: Dimension for sinusoidal timestep embedding (e.g. 256).
    time_proj_dim: Output dimension for timestep projection (e.g. 30720).
    text_embed_dim: Dimension of input text embeddings (e.g. 4096).
    image_embed_dim: Optional dimension for image embeddings.
    pos_embed_seq_len: Optional positional embedding sequence length.
    dtype: Computation dtype.
    weights_dtype: Parameter storage dtype.
    precision: JAX matmul precision.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      time_freq_dim: int,
      time_proj_dim: int,
      text_embed_dim: int,
      image_embed_dim: Optional[int] = None,
      pos_embed_seq_len: Optional[int] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.timesteps_proj = NNXFlaxTimesteps(
        dim=time_freq_dim, flip_sin_to_cos=True, freq_shift=0
    )
    self.time_embedder = NNXTimestepEmbedding(
        rngs=rngs,
        in_channels=time_freq_dim,
        time_embed_dim=dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.act_fn = get_activation('silu')
    self.time_proj = nnx.Linear(
        rngs=rngs,
        in_features=dim,
        out_features=time_proj_dim,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.zeros,
    )
    self.text_embedder = NNXPixArtAlphaTextProjection(
        rngs=rngs,
        in_features=text_embed_dim,
        hidden_size=dim,
        act_fn='gelu_tanh',
    )

  def __call__(
      self,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
  ) -> Tuple[jax.Array, jax.Array, jax.Array, Optional[jax.Array]]:
    """Embed conditioning signals.

    Args:
      timestep: Diffusion timestep, shape [B].
      encoder_hidden_states: Text embeddings, shape [B, seq_len, text_dim].
      encoder_hidden_states_image: Optional image embeddings.

    Returns:
      Tuple of (temb, timestep_proj, encoder_hidden_states,
      encoder_hidden_states_image):
        - temb: Timestep embedding, shape [B, dim].
        - timestep_proj: Projected timestep, shape [B, time_proj_dim].
        - encoder_hidden_states: Projected text embeddings, shape [B, seq_len,
        dim].
        - encoder_hidden_states_image: Projected image embeddings or None.
    """
    timestep = self.timesteps_proj(timestep)
    temb = self.time_embedder(timestep)
    timestep_proj = self.time_proj(self.act_fn(temb))
    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    if encoder_hidden_states_image is not None:
      raise NotImplementedError('Image-to-video conditioning is not supported.')
    return (
        temb,
        timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
    )


class ApproximateGELU(nnx.Module):
  """Approximate GELU activation with a learned linear projection.

  Applies a linear projection followed by the approximate GELU activation.
  See: https://arxiv.org/abs/1606.08415

  Attributes:
    rngs: Random number generators for parameter initialization.
    dim_in: Input dimension.
    dim_out: Output dimension.
    bias: Whether to use bias in the linear projection.
    dtype: Computation dtype.
    weights_dtype: Parameter storage dtype.
    precision: JAX matmul precision.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim_in: int,
      dim_out: int,
      bias: bool,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.proj = nnx.Linear(
        rngs=rngs,
        in_features=dim_in,
        out_features=dim_out,
        use_bias=bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.zeros,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Apply linear projection followed by GELU activation.

    Args:
      x: Input tensor.

    Returns:
      Activated tensor of the same shape (with potentially different last dim).
    """
    with jax.named_scope('gelu'):
      x = self.proj(x)
    return nnx.gelu(x)


class WanFeedForward(nnx.Module):
  """Feed-forward network block used in each transformer layer.

  Consists of an activation function (ApproximateGELU) followed by a
  linear output projection.

  Attributes:
    rngs: Random number generators for parameter initialization.
    dim: Input and output dimension.
    dim_out: Optional output dimension override.
    mult: FFN expansion multiplier (unused when inner_dim is provided).
    dropout: Dropout rate (unused in inference).
    activation_fn: Activation function name (must be 'gelu-approximate').
    final_dropout: Whether to apply final dropout (unused in inference).
    inner_dim: Inner FFN dimension (e.g. 13824 for 14B model).
    bias: Whether to use bias in linear layers.
    dtype: Computation dtype.
    weights_dtype: Parameter storage dtype.
    precision: JAX matmul precision.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      dim_out: Optional[int] = None,
      mult: int = 4,
      dropout: float = 0.0,  # pylint: disable=unused-argument
      activation_fn: str = 'geglu',
      final_dropout: bool = False,  # pylint: disable=unused-argument
      inner_dim: int = None,
      bias: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    if inner_dim is None:
      inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim

    self.act_fn = nnx.data(None)
    if activation_fn == 'gelu-approximate':
      self.act_fn = ApproximateGELU(
          rngs=rngs,
          dim_in=dim,
          dim_out=inner_dim,
          bias=bias,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    else:
      raise NotImplementedError(f'{activation_fn} is not implemented.')

    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=dim_out,
        use_bias=bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
    )

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    """Apply feed-forward network.

    Args:
      hidden_states: Input tensor of shape [..., dim].

    Returns:
      Output tensor of shape [..., dim_out].
    """
    hidden_states = self.act_fn(hidden_states)
    with jax.named_scope('proj_out'):
      return self.proj_out(hidden_states)


class WanTransformerBlock(nnx.Module):
  """Single transformer block with self-attention, cross-attention, and FFN.

  Each block applies:
    1. AdaLN-modulated self-attention with RoPE.
    2. Cross-attention to text/condition embeddings.
    3. AdaLN-modulated feed-forward network.

  The adaptive layer norm (AdaLN) parameters are derived from the timestep
  embedding, allowing the model to condition on the diffusion timestep.

  Attributes:
    rngs: Random number generators for parameter initialization.
    dim: Model dimension (inner_dim = num_heads * head_dim).
    ffn_dim: Feed-forward network inner dimension.
    num_heads: Number of attention heads.
    qk_norm: Query-key normalization strategy.
    cross_attn_norm: Whether to use layer norm before cross-attention.
    eps: Layer norm epsilon.
    flash_min_seq_length: Minimum sequence length to use flash attention.
    flash_block_sizes: Block sizes for flash attention.
    dtype: Computation dtype.
    weights_dtype: Parameter storage dtype.
    precision: JAX matmul precision.
    attention: Attention kernel type ('flash' or 'dot_product').
    total_token_num_lst: Comma-separated list of token counts per segment for
      attention masking.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      ffn_dim: int,
      num_heads: int,
      qk_norm: str = 'rms_norm_across_heads',
      cross_attn_norm: bool = False,
      eps: float = 1e-6,
      flash_min_seq_length: int = 4096,
      flash_block_sizes=None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = 'dot_product',
      total_token_num_lst: str = '32760',
  ):
    # 1. Self-attention.
    self.norm1 = FP32LayerNorm(
        rngs=rngs, dim=dim, eps=eps, elementwise_affine=False
    )
    self.attn1 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        total_token_num_lst=total_token_num_lst,
    )

    # 2. Cross-attention.
    self.attn2 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
    )
    assert cross_attn_norm
    self.norm2 = FP32LayerNorm(
        rngs=rngs, dim=dim, eps=eps, elementwise_affine=True
    )

    # 3. Feed-forward.
    self.ffn = WanFeedForward(
        rngs=rngs,
        dim=dim,
        inner_dim=ffn_dim,
        activation_fn='gelu-approximate',
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.norm3 = FP32LayerNorm(
        rngs=rngs, dim=dim, eps=eps, elementwise_affine=False
    )

    key = rngs.params()
    self.adaln_scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 6, dim)) / dim**0.5
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      temb: jax.Array,
      rotary_emb: jax.Array,
  ) -> jax.Array:
    """Apply transformer block.

    Args:
      hidden_states: Input tensor, shape [B, seq_len, dim].
      encoder_hidden_states: Text embeddings, shape [B, text_len, dim].
      temb: Timestep embedding (reshaped to [B, 6, dim]).
      rotary_emb: Rotary position embeddings for self-attention.

    Returns:
      Output tensor of same shape as hidden_states.
    """
    # Compute AdaLN modulation parameters from timestep.
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        jnp.split((self.adaln_scale_shift_table + temb), 6, axis=1)
    )

    # 1. Self-attention with RoPE.
    with jax.named_scope('attn1'):
      norm_hidden_states = (
          self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
      ).astype(hidden_states.dtype)
      attn_output = self.attn1(
          hidden_states=norm_hidden_states,
          encoder_hidden_states=norm_hidden_states,
          rotary_emb=rotary_emb,
      )
      hidden_states = (hidden_states + attn_output * gate_msa).astype(
          hidden_states.dtype
      )

    # 2. Cross-attention to text embeddings.
    with jax.named_scope('attn2'):
      norm_hidden_states = self.norm2(hidden_states)
      attn_output = self.attn2(
          hidden_states=norm_hidden_states,
          encoder_hidden_states=encoder_hidden_states,
      )
      hidden_states = hidden_states + attn_output

    # 3. Feed-forward network.
    with jax.named_scope('ffn'):
      norm_hidden_states = (
          self.norm3(hidden_states) * (1 + c_scale_msa) + c_shift_msa
      ).astype(hidden_states.dtype)
      ff_output = self.ffn(norm_hidden_states)
      hidden_states = (hidden_states + ff_output * c_gate_msa).astype(
          hidden_states.dtype
      )
    return hidden_states


class WanModel(nnx.Module):
  """WAN Diffusion Transformer for multi-modal video understanding.

  This is the main model class that implements the GenCeption architecture.
  It processes VAE-encoded video latents together with text embeddings in a
  SINGLE forward pass to produce:
    1. Denoised video latents.
    2. 2D keypoint predictions (token1).
    3. 3D keypoint predictions (token2).

  The model appends learnable token sequences to the video token sequence,
  processes them jointly through shared transformer blocks, then extracts
  and projects each output separately.

  Architecture:
    - 3D patch embedding converts [B, C, T, H, W] video to a token sequence.
    - Learnable tokens (token1, token2) are appended with their own RoPE.
    - All tokens pass through N transformer blocks (self-attn + cross-attn +
    FFN).
    - Video tokens are unpatched back to latent space.
    - Token heads are projected to keypoint predictions.

  Attributes:
    rngs: Random number generators for parameter initialization.
    lazy_init: If True, skip block initialization (for lazy loading).
    model_type: Model variant ('t2v' for text-to-video).
    patch_size: 3D patch size for video (temporal, height, width).
    num_attention_heads: Number of attention heads per layer.
    attention_head_dim: Dimension per attention head.
    in_channels: Number of input VAE latent channels.
    out_channels: Number of output latent channels.
    text_dim: Dimension of text encoder output.
    freq_dim: Dimension for sinusoidal timestep embedding.
    ffn_dim: Feed-forward network inner dimension.
    num_layers: Number of transformer blocks.
    cross_attn_norm: Whether to normalize before cross-attention.
    qk_norm: Query-key normalization strategy.
    eps: Layer norm epsilon.
    image_dim: Optional image embedding dimension (for I2V).
    added_kv_proj_dim: Optional added KV projection dimension.
    rope_max_seq_len: Maximum sequence length for RoPE.
    pos_embed_seq_len: Optional positional embedding sequence length.
    flash_min_seq_length: Minimum sequence length for flash attention.
    flash_block_sizes: Block sizes for flash attention kernels.
    dtype: Computation dtype.
    weights_dtype: Parameter storage dtype.
    precision: JAX matmul precision.
    attention: Attention kernel type ('flash' or 'dot_product').
    additional_token: Whether to use learnable token heads.
    token_att_num_layers: Number of layers for token attention.
    token1_num_frames: Number of temporal frames for token1.
    token1_num_per_frame: Number of token1 tokens per frame.
    token1_num_frames_rope: Number of RoPE frames for token1 spatial.
    token1_proj_dim: Projection dimension for token1 output.
    token1_output_dim: Final output dimension per keypoint for token1 (2 for
      2D).
    token2_num_frames: Number of temporal frames for token2.
    token2_num_per_frame: Number of token2 tokens per frame.
    token2_num_frames_rope: Number of RoPE frames for token2 spatial.
    token2_proj_dim: Projection dimension for token2 output.
    token2_output_dim: Final output dimension per keypoint for token2 (3 for
      3D).
    learnable_token_init_0: If True, initialize spatial RoPE to center position.
    height: Default video height (for spatial RoPE initialization).
    width: Default video width (for spatial RoPE initialization).
    total_token_num_lst: Comma-separated token counts for attention masking.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      lazy_init: bool = False,
      model_type: str = 't2v',  # pylint: disable=unused-argument
      patch_size: Tuple[int, ...] = (1, 2, 2),
      num_attention_heads: int = 40,
      attention_head_dim: int = 128,
      in_channels: int = 16,
      out_channels: int = 16,
      text_dim: int = 4096,
      freq_dim: int = 256,
      ffn_dim: int = 13824,
      num_layers: int = 40,
      cross_attn_norm: bool = True,
      qk_norm: Optional[str] = 'rms_norm_across_heads',
      eps: float = 1e-6,
      image_dim: Optional[int] = None,
      added_kv_proj_dim: Optional[int] = None,  # pylint: disable=unused-argument
      rope_max_seq_len: int = 1024,
      pos_embed_seq_len: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes=None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = 'dot_product',
      additional_token: bool = False,  # pylint: disable=unused-argument
      token_att_num_layers: int = 10,  # pylint: disable=unused-argument
      token1_num_frames: int = 81,
      token1_num_per_frame: int = 1,
      token1_num_frames_rope: int = 1,
      token1_proj_dim: int = 66,
      token1_output_dim: int = 2,
      token2_num_frames: int = 81,
      token2_num_per_frame: int = 1,
      token2_num_frames_rope: int = 1,
      token2_proj_dim: int = 99,
      token2_output_dim: int = 3,
      learnable_token_init_0: bool = True,
      height: int = 480,
      width: int = 832,
      total_token_num_lst: str = '32760',
  ):
    # Store configuration for forward pass access.
    self.config = SimpleConfig(
        patch_size=patch_size,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        text_dim=text_dim,
        freq_dim=freq_dim,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        cross_attn_norm=cross_attn_norm,
        qk_norm=qk_norm,
        eps=eps,
        image_dim=image_dim,
        rope_max_seq_len=rope_max_seq_len,
        attention=attention,
        token1_num_frames=token1_num_frames,
        token1_num_per_frame=token1_num_per_frame,
        token1_num_frames_rope=token1_num_frames_rope,
        token1_proj_dim=token1_proj_dim,
        token1_output_dim=token1_output_dim,
        token2_num_frames=token2_num_frames,
        token2_num_per_frame=token2_num_per_frame,
        token2_num_frames_rope=token2_num_frames_rope,
        token2_proj_dim=token2_proj_dim,
        token2_output_dim=token2_output_dim,
        height=height,
        width=width,
    )

    inner_dim = num_attention_heads * attention_head_dim
    out_channels = out_channels or in_channels
    self.num_layers = num_layers

    # 1. Patch embedding and position embedding.
    self.rope = WanRotaryPosEmbed(
        attention_head_dim, patch_size, rope_max_seq_len
    )
    self.patch_embedding = nnx.Conv(
        in_channels,
        inner_dim,
        rngs=rngs,
        kernel_size=patch_size,
        strides=patch_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
    )

    # 2. Condition embeddings (timestep + text).
    self.condition_embedder = WanTimeTextImageEmbedding(
        rngs=rngs,
        dim=inner_dim,
        time_freq_dim=freq_dim,
        time_proj_dim=inner_dim * 6,
        text_embed_dim=text_dim,
        image_embed_dim=image_dim,
        pos_embed_seq_len=pos_embed_seq_len,
    )

    # 3. Transformer blocks (initialized via vmap for parameter stacking).
    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(in_axes=0, out_axes=0)
    def init_block(rngs):
      return WanTransformerBlock(
          rngs=rngs,
          dim=inner_dim,
          ffn_dim=ffn_dim,
          num_heads=num_attention_heads,
          qk_norm=qk_norm,
          cross_attn_norm=cross_attn_norm,
          eps=eps,
          flash_min_seq_length=flash_min_seq_length,
          flash_block_sizes=flash_block_sizes,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          attention=attention,
          total_token_num_lst=total_token_num_lst,
      )

    if not lazy_init:
      self.blocks = init_block(rngs)

    # 4. Output projection for video latents.
    self.norm_out = FP32LayerNorm(
        rngs=rngs, dim=inner_dim, eps=eps, elementwise_affine=False
    )
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=out_channels * math.prod(patch_size),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
    )
    key = rngs.params()
    self.scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 2, inner_dim)) / inner_dim**0.5,
    )

    # ---- Token1 head: 2D keypoint predictions ----
    self.token1_rope = FixedTemporalLearnableSpatialRotaryPosEmbed(
        attention_head_dim, patch_size, rope_max_seq_len
    )
    self.token1_shape = (1, token1_num_frames, token1_num_per_frame, inner_dim)
    self.learnable_token1_proj = nnx.Linear(
        rngs=rngs,
        in_features=self.token1_shape[-1],
        out_features=self.token1_shape[-1],
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
    )

    if token1_num_frames_rope == 0 or token1_num_frames_rope == -1:
      self.learnable_hw_rope1 = jnp.array(
          [[[[height / 8 / patch_size[1] / 2, width / 8 / patch_size[2] / 2]]]],
          dtype=jnp.float32,
      )
    else:
      token1_rope_shape = (1, token1_num_frames_rope, token1_num_per_frame, 2)
      self.learnable_hw_rope1 = nnx.Param(
          jax.random.normal(key, token1_rope_shape),
      )
      if learnable_token_init_0:
        self.learnable_hw_rope1[:, :, :, 0] = height / 8 / patch_size[1] / 2
        self.learnable_hw_rope1[:, :, :, 1] = width / 8 / patch_size[2] / 2

    self.token1_proj_out = nnx.Linear(
        rngs=rngs,
        in_features=self.token1_shape[-1],
        out_features=token1_proj_dim,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
    )
    self.token1_scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 2, self.token1_shape[-1]))
        / self.token1_shape[-1] ** 0.5,
    )
    self.token1_norm_out = FP32LayerNorm(
        rngs=rngs, dim=self.token1_shape[-1], eps=eps, elementwise_affine=False
    )

    # ---- Token2 head: 3D keypoint predictions ----
    self.token2_rope = FixedTemporalLearnableSpatialRotaryPosEmbed(
        attention_head_dim, patch_size, rope_max_seq_len
    )
    self.token2_shape = (1, token2_num_frames, token2_num_per_frame, inner_dim)
    self.learnable_token2_proj = nnx.Linear(
        rngs=rngs,
        in_features=self.token2_shape[-1],
        out_features=self.token2_shape[-1],
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
    )

    if token2_num_frames_rope == 0 or token2_num_frames_rope == -1:
      self.learnable_hw_rope2 = jnp.array(
          [[[[height / 8 / patch_size[1] / 2, width / 8 / patch_size[2] / 2]]]],
          dtype=jnp.float32,
      )
    else:
      token2_rope_shape = (1, token2_num_frames_rope, token2_num_per_frame, 2)
      self.learnable_hw_rope2 = nnx.Param(
          jax.random.normal(key, token2_rope_shape),
      )
      if learnable_token_init_0:
        self.learnable_hw_rope2[:, :, :, 0] = height / 8 / patch_size[1] / 2
        self.learnable_hw_rope2[:, :, :, 1] = width / 8 / patch_size[2] / 2

    self.token2_proj_out = nnx.Linear(
        rngs=rngs,
        in_features=self.token2_shape[-1],
        out_features=token2_proj_dim,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
    )
    self.token2_scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 2, self.token2_shape[-1]))
        / self.token2_shape[-1] ** 0.5,
    )
    self.token2_norm_out = FP32LayerNorm(
        rngs=rngs, dim=self.token2_shape[-1], eps=eps, elementwise_affine=False
    )

  @jax.named_scope('WanModel')
  def __call__(
      self,
      hidden_states: jax.Array,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      return_dict: bool = True,
      attention_kwargs: Optional[Dict[str, Any]] = None,
  ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Run a single forward pass of the GenCeption model.

    Takes VAE-encoded video latents and text embeddings, processes them
    through the transformer, and produces denoised video latents along
    with 2D and 3D keypoint predictions.

    Args:
      hidden_states: VAE-encoded video latents, shape [B, C, T, H, W].
      timestep: Diffusion timestep, shape [B].
      encoder_hidden_states: Text embeddings from the text encoder,
        shape [B, text_seq_len, text_dim].
      encoder_hidden_states_image: Optional image embeddings (not supported).
      return_dict: Unused (kept for API compatibility).
      attention_kwargs: Optional additional attention arguments.

    Returns:
      Tuple of (video_latents, token1_pred, token2_pred):
        - video_latents: Predicted (negated) video latents,
          shape [B, C, T, H, W].
        - token1_pred: 2D keypoint predictions,
          shape [B, num_frames, num_keypoints, 2].
        - token2_pred: 3D keypoint predictions,
          shape [B, num_frames, num_keypoints, 3].
    """
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    # Transpose from [B, C, T, H, W] to [B, T, H, W, C] for patch embedding.
    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    rotary_emb = self.rope(hidden_states)

    # Apply 3D patch embedding.
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = jax.lax.collapse(hidden_states, 1, -1)
    video_token_num = hidden_states.shape[1]

    # ---- Prepare learnable tokens ----
    # Token1: 2D keypoints.
    learnable_token1 = self.learnable_token1_proj(
        jnp.zeros(self.token1_shape, dtype=jnp.float32)
    )
    if self.config.token1_num_frames_rope == 0:
      learnable_hw_rope1 = jnp.array(
          [[[[post_patch_height / 2, post_patch_width / 2]]]],
          dtype=jnp.float32,
      )
      token_rope_emb1 = self.token1_rope(
          latent_frames=num_frames,
          latent_height=height,
          latent_width=width,
          learnable_tokens=learnable_token1,
          learnable_hw_rope=learnable_hw_rope1,
      )
    else:
      token_rope_emb1 = self.token1_rope(
          latent_frames=num_frames,
          latent_height=height,
          latent_width=width,
          learnable_tokens=learnable_token1,
          learnable_hw_rope=self.learnable_hw_rope1,
      )

    # Token2: 3D keypoints.
    learnable_token2 = self.learnable_token2_proj(
        jnp.zeros(self.token2_shape, dtype=jnp.float32)
    )
    if self.config.token2_num_frames_rope == 0:
      learnable_hw_rope2 = jnp.array(
          [[[[post_patch_height / 2, post_patch_width / 2]]]],
          dtype=jnp.float32,
      )
      token_rope_emb2 = self.token2_rope(
          latent_frames=num_frames,
          latent_height=height,
          latent_width=width,
          learnable_tokens=learnable_token2,
          learnable_hw_rope=learnable_hw_rope2,
      )
    else:
      token_rope_emb2 = self.token2_rope(
          latent_frames=num_frames,
          latent_height=height,
          latent_width=width,
          learnable_tokens=learnable_token2,
          learnable_hw_rope=self.learnable_hw_rope2,
      )

    # Concatenate RoPE for video + token1 + token2.
    rotary_emb = jnp.concatenate(
        [rotary_emb, token_rope_emb1, token_rope_emb2], axis=2
    )

    # Flatten and repeat learnable tokens for the batch.
    learnable_token1 = jax.lax.collapse(learnable_token1, 1, -1)
    learnable_token1 = jnp.repeat(
        learnable_token1, repeats=batch_size, axis=0
    )
    token1_num = learnable_token1.shape[1]

    learnable_token2 = jax.lax.collapse(learnable_token2, 1, -1)
    learnable_token2 = jnp.repeat(
        learnable_token2, repeats=batch_size, axis=0
    )
    token2_num = learnable_token2.shape[1]

    # Concatenate video tokens + learnable tokens.
    hidden_states = jnp.concatenate(
        [hidden_states, learnable_token1, learnable_token2], axis=1
    )

    # ---- Condition embeddings ----
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
        self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
    )
    timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)

    if encoder_hidden_states_image is not None:
      raise NotImplementedError('Image-to-video is not yet implemented.')

    # ---- Transformer blocks via nnx.scan ----
    def scan_fn(carry, block):
      hidden_states, encoder_hidden_states, timestep_proj, rotary_emb = carry
      hidden_states = block(
          hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
      )
      return (hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    initial_carry = (
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        rotary_emb,
    )
    final_carry = nnx.scan(
        scan_fn,
        length=self.num_layers,
        in_axes=(nnx.Carry, 0),
        out_axes=nnx.Carry,
    )(initial_carry, self.blocks)

    hidden_states = final_carry[0]

    # ---- Token1 output head: 2D keypoints ----
    learnable_token1 = hidden_states[
        :, video_token_num : video_token_num + token1_num, :
    ]
    token1_shift, token1_scale = jnp.split(
        self.token1_scale_shift_table, 2, axis=1
    )
    learnable_token1 = (
        self.token1_norm_out(learnable_token1) * (1 + token1_scale)
        + token1_shift
    ).astype(learnable_token1.dtype)

    with jax.named_scope('proj_out_token1'):
      token1_pred = self.token1_proj_out(learnable_token1)
    token1_pred = token1_pred.reshape(
        batch_size,
        self.config.token1_num_frames,
        self.config.token1_num_per_frame,
        -1,
        self.config.token1_output_dim,
    )
    if self.config.token1_num_per_frame == 1:
      token1_pred = token1_pred[:, :, 0, :, :]

    # ---- Token2 output head: 3D keypoints ----
    token2_start = video_token_num + token1_num
    token2_end = video_token_num + token1_num + token2_num
    learnable_token2 = hidden_states[:, token2_start:token2_end, :]
    token2_shift, token2_scale = jnp.split(
        self.token2_scale_shift_table, 2, axis=1
    )
    learnable_token2 = (
        self.token2_norm_out(learnable_token2) * (1 + token2_scale)
        + token2_shift
    ).astype(learnable_token2.dtype)

    with jax.named_scope('proj_out_token2'):
      token2_pred = self.token2_proj_out(learnable_token2)
    token2_pred = token2_pred.reshape(
        batch_size,
        self.config.token2_num_frames,
        self.config.token2_num_per_frame,
        -1,
        self.config.token2_output_dim,
    )
    if self.config.token2_num_per_frame == 1:
      token2_pred = token2_pred[:, :, 0, :, :]

    # ---- Video output head ----
    video_latents = hidden_states[:, :video_token_num, :]
    shift, scale = jnp.split(
        self.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1
    )
    video_latents = (
        self.norm_out(video_latents) * (1 + scale) + shift
    ).astype(video_latents.dtype)

    with jax.named_scope('proj_out'):
      video_latents = self.proj_out(video_latents)

    # Unpatchify: reshape back to [B, C, T, H, W].
    video_latents = video_latents.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    video_latents = jnp.transpose(video_latents, (0, 7, 1, 4, 2, 5, 3, 6))
    video_latents = jax.lax.collapse(video_latents, 6, None)
    video_latents = jax.lax.collapse(video_latents, 4, 6)
    video_latents = jax.lax.collapse(video_latents, 2, 4)

    return -video_latents, token1_pred, token2_pred
