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

"""Attention mechanisms for the GenCeption WAN 2.1 transformer.

Provides the WanAttention module with support for:
- Flash attention via jax.nn.dot_product_attention (default)
- Standard dot-product attention
- Rotary position embeddings (RoPE)
- QK normalization (RMS norm across heads)
- Learnable token attention masking via total_token_num_lst
"""

from typing import Optional, Tuple

from flax import nnx
import jax
import jax.numpy as jnp

# Type alias for flash attention block size configuration.
# This is typically `splash_attention_kernel.BlockSizes` on TPU,
# but we keep it as an optional type for flexibility.
BlockSizes = Optional[object]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _unflatten_heads(
    tensor: jax.Array, heads: int
) -> jax.Array:
  """Reshape from [B, S, H*D] to [B, H, S, D].

  Args:
    tensor: Input tensor of shape [batch, seq_len, heads * dim_head].
    heads: Number of attention heads.

  Returns:
    Tensor of shape [batch, heads, seq_len, dim_head].
  """
  batch, seq, heads_and_dim_head = tensor.shape
  tensor = tensor.reshape(batch, seq, heads, heads_and_dim_head // heads)
  tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  return tensor


def _flatten_heads(tensor: jax.Array) -> jax.Array:
  """Reshape from [B, H, S, D] to [B, S, H*D].

  Args:
    tensor: Input tensor of shape [batch, heads, seq_len, dim_head].

  Returns:
    Tensor of shape [batch, seq_len, heads * dim_head].
  """
  b, h, s, d = tensor.shape
  tensor = jnp.transpose(tensor, axes=[0, 2, 1, 3])
  return jnp.reshape(tensor, (b, s, h * d))


# ---------------------------------------------------------------------------
# Attention implementations
# ---------------------------------------------------------------------------


def _apply_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    scale: float,
) -> jax.Array:
  """Apply flash attention using JAX's built-in dot_product_attention.

  Uses `jax.nn.dot_product_attention` which automatically selects the
  best backend (e.g., cuDNN flash attention on GPU, or efficient
  implementations on TPU).

  Inputs are in [B, H, S, D] format (from ``_unflatten_heads``), but
  ``jax.nn.dot_product_attention`` expects [B, S, H, D], so we transpose.

  Args:
    query: Query tensor [batch, heads, seq_len, dim_head].
    key: Key tensor [batch, heads, seq_len, dim_head].
    value: Value tensor [batch, heads, seq_len, dim_head].
    scale: Attention scale factor (typically dim_head ** -0.5).

  Returns:
    Attention output tensor [batch, seq_len, heads * dim_head].
  """
  # Transpose from [B, H, S, D] to [B, S, H, D] for jax.nn API.
  query = jnp.transpose(query, (0, 2, 1, 3))
  key = jnp.transpose(key, (0, 2, 1, 3))
  value = jnp.transpose(value, (0, 2, 1, 3))

  attn_output = jax.nn.dot_product_attention(
      query,
      key,
      value,
      scale=scale,
      is_causal=False,
      implementation="cudnn",
  )
  # attn_output is [B, S, H, D]; transpose back to [B, H, S, D] for
  # _flatten_heads which expects that layout.
  attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
  return _flatten_heads(attn_output)


def _apply_dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,  # pylint: disable=unused-argument
    dim_head: int,  # pylint: disable=unused-argument
    scale: float,
    dtype: jnp.dtype,
) -> jax.Array:
  """Apply standard dot-product attention.

  Computes attention weights via einsum, applies softmax, and aggregates
  values. Operates on 4D tensors in [B, H, S, D] format.

  Args:
    query: Query tensor [batch, heads, seq_len, dim_head].
    key: Key tensor [batch, heads, seq_len, dim_head].
    value: Value tensor [batch, heads, seq_len, dim_head].
    heads: Number of attention heads.
    dim_head: Dimension per attention head.
    scale: Attention scale factor.
    dtype: Output dtype for attention probabilities.

  Returns:
    Attention output tensor [batch, seq_len, heads * dim_head].
  """
  b = query.shape[0]  # pylint: disable=unused-variable
  # query/key/value: [B, H, S, D]
  attention_scores = jnp.einsum("bhsd,bhtd->bhst", query, key)
  attention_scores = attention_scores * scale
  attention_probs = jax.nn.softmax(attention_scores, axis=-1)
  attention_probs = attention_probs.astype(dtype)

  hidden_states = jnp.einsum("bhst,bhtd->bhsd", attention_probs, value)
  return _flatten_heads(hidden_states)


def _check_attention_inputs(
    query: jax.Array, key: jax.Array, value: jax.Array
) -> None:
  """Validate attention input shapes."""
  assert key.ndim == value.ndim, "k, v must have same rank."
  assert (
      query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), "q, k, v batch dims must match."
  assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
  assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
  assert query.shape[-1] == key.shape[-1], "q, k depths must match."


# ---------------------------------------------------------------------------
# WAN Attention module
# ---------------------------------------------------------------------------


class WanAttention(nnx.Module):
  """Multi-head attention for the WAN 2.1 diffusion transformer.

  Supports both flash attention (via jax.nn.dot_product_attention) and
  standard dot-product attention. Includes optional QK normalization
  and rotary position embeddings (RoPE).

  The `total_token_num_lst` parameter controls attention masking for
  learnable tokens: when multiple token groups are specified, each
  non-video group attends to video tokens plus its own tokens, rather
  than the full sequence.

  Attributes:
    rngs: Random number generators for parameter initialization.
    query_dim: Input/output feature dimension.
    heads: Number of attention heads.
    dim_head: Dimension per attention head.
    eps: Epsilon for RMS normalization.
    qk_norm: Type of QK normalization (e.g., "rms_norm_across_heads"). Set to
      None to disable.
    flash_min_seq_length: Minimum sequence length to use flash attention.
    flash_block_sizes: Block sizes for TPU splash attention (unused on GPU).
    dtype: Computation dtype.
    weights_dtype: Parameter storage dtype.
    precision: JAX matmul precision.
    attention_kernel: Attention implementation ("flash" or "dot_product").
    total_token_num_lst: Underscore-separated token counts per group, prefixed
      with underscore (e.g., "_32760" or "_32760_81_81").
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      query_dim: int,  # pylint: disable=unused-argument
      heads: int = 8,
      dim_head: int = 64,
      eps: float = 1e-6,
      qk_norm: Optional[str] = "rms_norm_across_heads",
      flash_min_seq_length: int = 0,
      flash_block_sizes: BlockSizes = None,  # pylint: disable=unused-argument
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
      attention_kernel: str = "flash",
      total_token_num_lst: str = "_32760",
  ):
    self.dim_head = dim_head
    self.heads = heads
    self.inner_dim = dim_head * heads
    self.scale = dim_head ** -0.5
    self.qk_norm = qk_norm
    self.attention_kernel = attention_kernel
    self.flash_min_seq_length = flash_min_seq_length
    self.total_token_num_lst = total_token_num_lst
    self.dtype = dtype

    # Q/K/V projections
    qkv_init_kernel = nnx.initializers.lecun_normal()

    self.query = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.initializers.zeros,
    )

    self.key = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.initializers.zeros,
    )

    self.value = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.initializers.zeros,
    )

    # Output projection
    self.proj_attn = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=nnx.initializers.lecun_normal(),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

    # Optional QK normalization
    if qk_norm is not None:
      self.norm_q = nnx.RMSNorm(
          num_features=self.inner_dim,
          rngs=rngs,
          epsilon=eps,
          dtype=dtype,
          scale_init=nnx.initializers.ones,
          param_dtype=weights_dtype,
      )
      self.norm_k = nnx.RMSNorm(
          num_features=self.inner_dim,
          rngs=rngs,
          epsilon=eps,
          dtype=dtype,
          scale_init=nnx.initializers.ones,
          param_dtype=weights_dtype,
      )
    else:
      self.norm_q = None
      self.norm_k = None

  def _apply_rope(
      self,
      xq: jax.Array,
      xk: jax.Array,
      freqs_cis: jax.Array,
  ) -> Tuple[jax.Array, jax.Array]:
    """Apply rotary position embeddings to query and key tensors.

    Uses complex multiplication for efficient RoPE application.

    Args:
      xq: Query tensor [batch, heads, seq_len, dim_head].
      xk: Key tensor [batch, heads, seq_len, dim_head].
      freqs_cis: Complex frequency tensor [1, 1, seq_len, dim_head//2].

    Returns:
      Tuple of (rotated_query, rotated_key) with same shapes as inputs.
    """
    dtype = xq.dtype
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    xq_out_complex = xq_ * freqs_cis
    xk_out_complex = xk_ * freqs_cis

    xq_out = (
        jnp.stack(
            [jnp.real(xq_out_complex), jnp.imag(xq_out_complex)], axis=-1
        )
        .reshape(xq.shape)
        .astype(dtype)
    )
    xk_out = (
        jnp.stack(
            [jnp.real(xk_out_complex), jnp.imag(xk_out_complex)], axis=-1
        )
        .reshape(xk.shape)
        .astype(dtype)
    )

    return xq_out, xk_out

  def _run_attention(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
  ) -> jax.Array:
    """Dispatch to the selected attention implementation.

    Args:
      query: Query tensor [batch, heads, seq_len, dim_head].
      key: Key tensor [batch, heads, seq_len, dim_head].
      value: Value tensor [batch, heads, seq_len, dim_head].

    Returns:
      Attention output [batch, seq_len, heads * dim_head].
    """
    _check_attention_inputs(query, key, value)

    if self.attention_kernel == "dot_product":
      return _apply_dot_product_attention(
          query, key, value,
          heads=self.heads,
          dim_head=self.dim_head,
          scale=self.scale,
          dtype=self.dtype,
      )
    else:
      # Default: flash attention
      return _apply_flash_attention(query, key, value, scale=self.scale)

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: Optional[jax.Array] = None,
      rotary_emb: Optional[jax.Array] = None,
  ) -> jax.Array:
    """Apply multi-head attention with optional RoPE and token grouping.

    Args:
      hidden_states: Input tensor [batch, seq_len, dim].
      encoder_hidden_states: Optional cross-attention input. If None,
        self-attention is performed.
      rotary_emb: Optional rotary position embedding tensor.

    Returns:
      Attention output tensor [batch, seq_len, dim].
    """
    dtype = hidden_states.dtype
    if encoder_hidden_states is None:
      encoder_hidden_states = hidden_states

    query_proj = self.query(hidden_states)
    key_proj = self.key(encoder_hidden_states)
    value_proj = self.value(encoder_hidden_states)

    if self.qk_norm:
      query_proj = self.norm_q(query_proj)
      key_proj = self.norm_k(key_proj)

    # Always unflatten to [batch, heads, seq_len, dim_head] for attention
    query_proj = _unflatten_heads(query_proj, self.heads)
    key_proj = _unflatten_heads(key_proj, self.heads)
    value_proj = _unflatten_heads(value_proj, self.heads)

    if rotary_emb is not None:
      query_proj, key_proj = self._apply_rope(
          query_proj, key_proj, rotary_emb
      )

    # Parse token group counts for learnable token attention masking
    total_token_num_lst = [
        int(i) for i in self.total_token_num_lst[1:].split("_")
    ]

    if not total_token_num_lst or len(total_token_num_lst) == 1:
      # Standard full-sequence attention
      attn_output = self._run_attention(query_proj, key_proj, value_proj)
    else:
      # Grouped attention: each non-video group attends to
      # video tokens + its own tokens
      attn_output_lst = []
      video_token_num = total_token_num_lst[0]

      # Video tokens attend only to themselves
      attn_output_lst.append(
          self._run_attention(
              query_proj[:, :, :video_token_num],
              key_proj[:, :, :video_token_num],
              value_proj[:, :, :video_token_num],
          )
      )

      # Each subsequent group attends to video + its own tokens
      end_idx = total_token_num_lst[0]
      for token_num in total_token_num_lst[1:]:
        start_idx = end_idx
        end_idx = end_idx + token_num
        attn_output_lst.append(
            self._run_attention(
                query_proj[:, :, start_idx:end_idx],
                jnp.concatenate(
                    (
                        key_proj[:, :, :video_token_num],
                        key_proj[:, :, start_idx:end_idx],
                    ),
                    axis=2,
                ),
                jnp.concatenate(
                    (
                        value_proj[:, :, :video_token_num],
                        value_proj[:, :, start_idx:end_idx],
                    ),
                    axis=2,
                ),
            )
        )
      attn_output = jnp.concatenate(attn_output_lst, axis=1)

    attn_output = attn_output.astype(dtype=dtype)
    hidden_states = self.proj_attn(attn_output)
    return hidden_states


# Alias for compatibility with model.py imports.
FlaxWanAttention = WanAttention
