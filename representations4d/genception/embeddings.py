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

"""Embedding modules for the GenCeption model.

Provides sinusoidal timestep embeddings, timestep MLP projections,
text projection MLPs, and rotary position embedding computation
for the WAN 2.1 diffusion transformer.
"""

import math
from typing import Optional, Union

from flax import nnx
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Activation function registry
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "swish": jax.nn.silu,
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "gelu_tanh": jax.nn.gelu,
    "mish": jax.nn.mish,
}


def get_activation(name: str):
  """Look up an activation function by name.

  Args:
    name: Name of the activation function.

  Returns:
    The corresponding JAX activation function.

  Raises:
    ValueError: If the activation name is not recognized.
  """
  func = _ACTIVATIONS.get(name)
  if func is None:
    raise ValueError(f"Unknown activation function: {name}")
  return func


# ---------------------------------------------------------------------------
# Sinusoidal positional embeddings
# ---------------------------------------------------------------------------


def get_sinusoidal_embeddings(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    freq_shift: float = 1,
    min_timescale: float = 1,
    max_timescale: float = 1.0e4,
    flip_sin_to_cos: bool = False,
    scale: float = 1.0,
) -> jnp.ndarray:
  """Compute sinusoidal positional encoding (Tensor2Tensor style).

  Args:
    timesteps: A 1-D array of N indices, one per batch element.
      These may be fractional.
    embedding_dim: The number of output channels (must be even).
    freq_shift: Frequency shift parameter.
    min_timescale: The smallest time unit.
    max_timescale: The largest time unit.
    flip_sin_to_cos: If True, concatenate [cos, sin] instead of [sin, cos].
    scale: Scaling factor for the time values.

  Returns:
    A float32 array of shape [N, embedding_dim] with timing signals.
  """
  assert timesteps.ndim == 1, "Timesteps should be a 1d-array"
  assert embedding_dim % 2 == 0, (
      f"Embedding dimension {embedding_dim} should be even"
  )
  num_timescales = float(embedding_dim // 2)
  log_timescale_increment = math.log(max_timescale / min_timescale) / (
      num_timescales - freq_shift
  )
  inv_timescales = min_timescale * jnp.exp(
      jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
  )
  emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)

  scaled_time = scale * emb

  if flip_sin_to_cos:
    signal = jnp.concatenate(
        [jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=1
    )
  else:
    signal = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1
    )
  signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])
  return signal


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[jnp.ndarray, int],
    theta: float = 10000.0,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    freqs_dtype: jnp.dtype = jnp.float32,
    use_real: bool = True,
) -> jnp.ndarray:
  """Compute 1D rotary position embeddings (RoPE).

  Precomputes the frequency tensor for complex exponentials (cis) with
  given dimensions.

  Args:
    dim: Embedding dimension (must be even).
    pos: Position indices (1-D array) or an integer length.
    theta: Base frequency for the rotary embeddings.
    linear_factor: Linear interpolation factor for position scaling.
    ntk_factor: NTK-aware scaling factor for the base frequency.
    freqs_dtype: Data type for frequency computation.
    use_real: If True, return real-valued [cos, -sin, sin, cos] stack (Flux).
      If False, return complex exponentials (WAN 2.1).

  Returns:
    Rotary embedding tensor. Shape depends on `use_real`:
      - use_real=True: [seq_len, dim//2, 4]
      - use_real=False: [seq_len, dim//2] (complex64)
  """
  assert dim % 2 == 0

  if isinstance(pos, int):
    pos = jnp.arange(pos)

  theta = theta * ntk_factor
  freqs = (
      1.0
      / (
          theta
          ** (jnp.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim)
      )
      / linear_factor
  )
  freqs = jnp.outer(pos, freqs)
  if use_real:
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    out = jnp.stack([freqs_cos, -freqs_sin, freqs_sin, freqs_cos], axis=-1)
  else:
    # WAN 2.1 uses complex exponentials
    out = jnp.exp(1j * freqs)
  return out


def get_1d_float_rotary_pos_embed(
    dim: int,
    pos: jnp.ndarray,
    theta: float = 10000.0,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    freqs_dtype: jnp.dtype = jnp.float32,
    use_real: bool = True,
) -> jnp.ndarray:
  """Compute 1D rotary position embeddings for float-valued positions.

  Same as `get_1d_rotary_pos_embed` but expects `pos` to always be a
  float-valued array (no integer shorthand).

  Args:
    dim: Embedding dimension (must be even).
    pos: Float-valued position array.
    theta: Base frequency for the rotary embeddings.
    linear_factor: Linear interpolation factor for position scaling.
    ntk_factor: NTK-aware scaling factor for the base frequency.
    freqs_dtype: Data type for frequency computation.
    use_real: If True, return real-valued stack; if False, complex exp.

  Returns:
    Rotary embedding tensor (same shape conventions as
    `get_1d_rotary_pos_embed`).
  """
  assert dim % 2 == 0

  theta = theta * ntk_factor
  freqs = (
      1.0
      / (
          theta
          ** (jnp.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim)
      )
      / linear_factor
  )
  freqs = jnp.outer(pos, freqs)
  if use_real:
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    out = jnp.stack([freqs_cos, -freqs_sin, freqs_sin, freqs_cos], axis=-1)
  else:
    # WAN 2.1 uses complex exponentials
    out = jnp.exp(1j * freqs)
  return out


# ---------------------------------------------------------------------------
# Timestep and text projection modules
# ---------------------------------------------------------------------------


class NNXFlaxTimesteps(nnx.Module):
  """Sinusoidal timestep embedding module.

  Wraps `get_sinusoidal_embeddings` to produce fixed (non-learned)
  sinusoidal embeddings from scalar timesteps.

  Attributes:
    dim: Embedding dimension.
    flip_sin_to_cos: Whether to flip sin/cos ordering.
    freq_shift: Frequency shift parameter.
    scale: Scaling factor for the time values.
  """

  def __init__(
      self,
      dim: int = 32,
      flip_sin_to_cos: bool = False,
      freq_shift: float = 1.0,
      scale: int = 1,
  ):
    self.dim = dim
    self.flip_sin_to_cos = flip_sin_to_cos
    self.freq_shift = freq_shift
    self.scale = scale

  def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
    """Compute sinusoidal embeddings for the given timesteps.

    Args:
      timesteps: 1-D array of timestep values.

    Returns:
      Sinusoidal embeddings of shape [len(timesteps), dim].
    """
    return get_sinusoidal_embeddings(
        timesteps,
        embedding_dim=self.dim,
        flip_sin_to_cos=self.flip_sin_to_cos,
        freq_shift=self.freq_shift,
    )


class NNXTimestepEmbedding(nnx.Module):
  """MLP for projecting timestep embeddings.

  A two-layer MLP with an activation function that projects sinusoidal
  timestep embeddings into a higher-dimensional learned representation.

  Attributes:
    rngs: Random number generators for parameter initialization.
    in_channels: Input feature dimension.
    time_embed_dim: Hidden and output dimension (unless out_dim is set).
    act_fn: Activation function name (e.g., "silu", "gelu").
    out_dim: Output dimension (defaults to time_embed_dim).
    post_act_fn: Optional post-activation function name.
    cond_proj_dim: Optional conditional projection dimension.
    sample_proj_bias: Whether to use bias in linear layers.
    dtype: Computation dtype.
    weights_dtype: Parameter storage dtype.
    precision: JAX matmul precision.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_channels: int,
      time_embed_dim: int = 32,
      act_fn: str = "silu",
      out_dim: Optional[int] = None,
      post_act_fn: Optional[str] = None,
      cond_proj_dim: Optional[int] = None,
      sample_proj_bias: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    self.linear_1 = nnx.Linear(
        rngs=rngs,
        in_features=in_channels,
        out_features=time_embed_dim,
        use_bias=sample_proj_bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.zeros,
    )

    if cond_proj_dim is not None:
      self.cond_proj = nnx.Linear(
          rngs=rngs,
          in_features=cond_proj_dim,
          out_features=in_channels,
          dtype=dtype,
          param_dtype=weights_dtype,
          precision=precision,
          kernel_init=nnx.initializers.xavier_uniform(),
          bias_init=nnx.initializers.zeros,
      )
    else:
      self.cond_proj = None

    self.act = get_activation(act_fn)

    if out_dim is not None:
      time_embed_dim_out = out_dim
    else:
      time_embed_dim_out = time_embed_dim

    self.linear_2 = nnx.Linear(
        rngs=rngs,
        in_features=time_embed_dim,
        out_features=time_embed_dim_out,
        use_bias=sample_proj_bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.zeros,
    )

    if post_act_fn is None:
      self.post_act = None
    else:
      self.post_act = get_activation(post_act_fn)

  def __call__(
      self,
      sample: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Project timestep embeddings through the MLP.

    Args:
      sample: Input embeddings.
      condition: Optional conditioning signal (added after cond_proj).

    Returns:
      Projected embeddings.
    """
    if condition is not None:
      sample = sample + self.cond_proj(condition)
    sample = self.linear_1(sample)

    if self.act is not None:
      sample = self.act(sample)
    sample = self.linear_2(sample)

    if self.post_act is not None:
      sample = self.post_act(sample)
    return sample


class NNXPixArtAlphaTextProjection(nnx.Module):
  """Text projection MLP for caption embeddings.

  Projects caption embeddings through a two-layer MLP with an activation
  function. Used by the WAN transformer to process text encoder outputs.

  Attributes:
    rngs: Random number generators for parameter initialization.
    in_features: Input feature dimension (from text encoder).
    hidden_size: Hidden dimension and default output dimension.
    out_features: Output dimension (defaults to hidden_size).
    act_fn: Activation function name (e.g., "gelu_tanh", "silu").
    dtype: Computation dtype.
    weights_dtype: Parameter storage dtype.
    precision: JAX matmul precision.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_features: int,
      hidden_size: int,
      out_features: Optional[int] = None,
      act_fn: str = "gelu_tanh",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    if out_features is None:
      out_features = hidden_size

    self.linear_1 = nnx.Linear(
        rngs=rngs,
        in_features=in_features,
        out_features=hidden_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.zeros,
    )
    self.act_1 = get_activation(act_fn)

    self.linear_2 = nnx.Linear(
        rngs=rngs,
        in_features=hidden_size,
        out_features=out_features,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.zeros,
    )

  def __call__(self, caption: jnp.ndarray) -> jnp.ndarray:
    """Project caption embeddings through the MLP.

    Args:
      caption: Input caption embeddings from the text encoder.

    Returns:
      Projected caption embeddings.
    """
    hidden_states = self.linear_1(caption)
    hidden_states = self.act_1(hidden_states)
    hidden_states = self.linear_2(hidden_states)
    return hidden_states
