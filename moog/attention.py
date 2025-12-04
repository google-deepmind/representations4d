# Copyright 2025 Google LLC
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

"""Improved Transformer attention modules."""

from __future__ import annotations

from typing import Callable

from flax import linen as nn
import jax
import jax.numpy as jnp
from kauldron.typing import Axes, Bool, Float, typechecked  # pylint: disable=g-importing-member,g-multiple-import
from kauldron.utils import interms_property
import numpy as np


@typechecked
def _flatten_token_axes(
    tokens: Float['... d'], num_token_axes: int
) -> tuple[Float['... d'], tuple[int, ...], tuple[int, ...]]:
  """Flatten all token axes (e.g. width/height). Useful for Transformers."""
  first_token_axis = -1 - num_token_axes
  batch_shape = tokens.shape[:first_token_axis]
  token_shape = tokens.shape[first_token_axis:-1]
  if num_token_axes < 1:
    raise ValueError(f'num_token_axes must be >= 1, but is {num_token_axes}')
  if num_token_axes > 1:
    tokens = jnp.reshape(
        tokens, batch_shape + (np.prod(token_shape), tokens.shape[-1])
    )
  return tokens, batch_shape, token_shape


@typechecked
def _unflatten_token_axes(
    tokens: Float['... d'],
    batch_shape: tuple[int, ...],
    token_shape: tuple[int, ...],
) -> Float['... d']:
  """Undo flatten_token_axes."""
  if len(token_shape) > 1:
    tokens = jnp.reshape(
        tokens, batch_shape + token_shape + (tokens.shape[-1],)
    )
  return tokens


@typechecked
def dot_product_attention_weights(
    query: Float['*b q h d'],
    key: Float['*b k h d'],
    softmax_axes: Axes = -1,
    bias: Float['*b #h #q #k'] | None = None,
    mask: Bool['*b #h #q #k'] | None = None,
) -> Float['*b h q k']:
  """Computes dot-product attention weights given query and key.

  q: number of queries, k: number of keys, h: number of heads
  d: dimension of keys/queries

  Args:
    query: Queries for calculating attention
    key: Keys for calculating attention.
    softmax_axes: The axes over which the softmax is taken. defaults to -1 which
      is the keys axis. For Slot-Attention set to -2 (queries).
    bias: Bias for the attention weights. This should be broadcastable to the
      shape `[*b h q k]`. This can be used for incorporating causal masks,
      padding masks, proximity bias, etc.
    mask: Mask for the attention weights. This should be broadcastable to the
      shape `[*b h q k]`. This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value is
      `False`.

  Returns:
    Attention weights of shape `[*b h q k]`.
  """
  query = query / jnp.sqrt(query.shape[-1])
  attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)

  if bias is not None:
    attn_weights = attn_weights + bias

  if mask is not None:
    big_neg = jnp.finfo(query.dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  attn_weights = jax.nn.softmax(attn_weights, axis=softmax_axes)

  return attn_weights


class ImprovedMHDPAttention(nn.Module):
  """Multi-head dot-product attention.

  Simplified nn.MultiheadDotProductAttention with a few modifications:
    - include a softmax axis
    - include normalization of keys and queries
    - dropped out support for dropout

  Attributes:
    num_heads: Number of attention heads.
    qk_size: Total dimension of the keys and queries.
    v_size: Total dimension of the values. Defaults to qk_size.
    softmax_axis: The axis over which the softmax is taken. defaults to -1 which
      is the keys axis. For Slot-Attention set to -2 (queries).
  """

  num_heads: int
  qk_size: int
  v_size: int | None = None
  softmax_axis: Axes = -1

  interms = interms_property.interms_property()

  @typechecked
  @nn.compact
  def __call__(
      self,
      inputs_q: Float['*b q d1'],
      inputs_kv: Float['*b k d2'],
      bias: Float['*b #h #q #k'] | None = None,
      mask: Bool['*b #h #q #k'] | None = None,
  ) -> Float['*b q d1']:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: Input tokens from which queries are computed.
      inputs_kv: Input tokens from which the keys and queries are computed.
      bias: Bias for the attention weights. This can be used for incorporating
        causal masks, padding masks, proximity bias, etc.
      mask: Attention mask, where attention weights are masked out if their mask
        value is `False`.

    Returns:
      output tokens (linear projection of an attention weighted average of value
      tokens per query).
    """
    v_size = self.qk_size if self.v_size is None else self.v_size

    if self.qk_size % self.num_heads:
      raise ValueError(f'{self.num_heads=} must divide {self.qk_size=}.')
    if v_size % self.num_heads:
      raise ValueError(f'{v_size=} must divide {self.num_heads=}.')

    # Project inputs_q to multi-headed queries and keys.
    # dimensions are then [B..., Q, H, qk_size]
    query = nn.DenseGeneral(
        features=(self.num_heads, self.qk_size // self.num_heads),
        use_bias=False,
        dtype=inputs_q.dtype,
        name='dense_query',
    )(inputs_q)
    key = nn.DenseGeneral(
        features=(self.num_heads, self.qk_size // self.num_heads),
        use_bias=False,
        dtype=inputs_kv.dtype,
        name='dense_key',
    )(inputs_kv)

    # Normalize keys and queries before attention.
    # see Gilmer et al. 2023
    # Intriguing Properties of Transformer Training Instabilities
    query = nn.RMSNorm(dtype=query.dtype, name='norm_query')(query)
    key = nn.RMSNorm(dtype=query.dtype, name='norm_key')(key)

    value = nn.DenseGeneral(
        features=(self.num_heads, v_size // self.num_heads),
        use_bias=False,
        dtype=inputs_kv.dtype,
        name='dense_value',
    )(inputs_kv)

    # Compute attention weights.
    attn_weights = dot_product_attention_weights(
        query, key, self.softmax_axis, bias=bias, mask=mask
    )
    # accessible as `interms.[path.to.this.module].attention[0]`
    self.interms['attention'] = attn_weights

    # Return weighted sum over values for each query position.
    x = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)

    # Back to the original input dimensions.
    out = nn.DenseGeneral(
        features=inputs_q.shape[-1],
        axis=(-2, -1),
        use_bias=True,
        dtype=x.dtype,
        name='dense_out',
    )(x)

    return out


class ImprovedTransformer(nn.Module):
  """Improved Transformer using tricks from ViT-22B (w/ cross-attention).

  1) Normalize keys/queries w/ LayerNorm.
  2) Do some ops in parallel (here: cross + self-attention, but not MLP).
  3) Allow for flattening/unflattening of token axes.

  Point (2) is similar to Block-Recurrent Transformers and performed better for
  shallower Transformer models.
  """

  qkv_size: int
  num_heads: int
  mlp_size: int
  num_layers: int
  hidden_size: int | None = None
  output_size: int | None = None
  cross_attn_only: bool = False
  attention_fn: Callable[..., Float] | None = None

  @typechecked
  @nn.compact
  def __call__(
      self,
      queries: Float['... d1'],  # [*b *n d1]
      inputs_kv: Float['*b N D'] | None = None,
      self_attn_bias: Float['*b #h #n #N'] | None = None,
      cross_attn_bias: Float['*b #h #n #N'] | None = None,
      self_attn_mask: Bool['*b #h #n #N'] | None = None,
      cross_attn_mask: Bool['*b #h #n #N'] | None = None,
      num_token_axes: int = 1,
  ) -> Float['... d2']:  # [*b *n d2]
    # Flatten inputs in case we have multiple token axes (e.g. for images).
    queries, batch_shape, token_shape = _flatten_token_axes(
        queries, num_token_axes
    )

    query_size = queries.shape[-1]

    if self.output_size is not None:
      output_size = self.output_size
    elif self.hidden_size is not None and self.hidden_size != query_size:
      # If we don't explicitly set an output_size, but do choose a hidden_size
      # that deviates from the query_size, we map back to query_size.
      output_size = query_size
    else:
      output_size = None

    if self.hidden_size is not None:
      queries = nn.Dense(features=self.hidden_size, dtype=queries.dtype)(
          queries
      )

    for i in range(self.num_layers):
      queries = ImprovedTransformerBlock(
          qkv_size=self.qkv_size,
          num_heads=self.num_heads,
          mlp_size=self.mlp_size,
          cross_attn_only=self.cross_attn_only,
          name=f'layer_{i}',
          attention_fn=self.attention_fn,
      )(
          queries,
          inputs_kv=inputs_kv,
          self_attn_bias=self_attn_bias,
          cross_attn_bias=cross_attn_bias,
          self_attn_mask=self_attn_mask,
          cross_attn_mask=cross_attn_mask,
      )

    # NOTE: This final Layernorm is missing in obvis.
    queries = nn.LayerNorm(
        use_bias=False, use_scale=True, dtype=queries.dtype, name='norm_encoder'
    )(queries)

    if output_size is not None:
      queries = nn.Dense(features=output_size, dtype=queries.dtype)(queries)

    # Undo flattening of token axis.
    queries = _unflatten_token_axes(queries, batch_shape, token_shape)

    return queries


class ImprovedTransformerBlock(nn.Module):
  """Improved Transformer block using tricks from ViT-22B (w/ cross-attention).

  1) RMSNorm instead of LayerNorm.
  2) Normalize keys/queries w/ RMSNorm.
  3) Do some ops in parallel (here: cross + self-attention, but not MLP).

  Point (3) is similar to Block-Recurrent Transformers and performed better for
  shallower Transformer models.

  Adapted from third_party/py/kauldron/modules/attention.py
  """

  mlp_size: int
  num_heads: int
  qkv_size: int
  cross_attn_only: bool = False
  zero_init: bool = False
  attention_fn: Callable[..., Float] | None = None

  @nn.compact
  def __call__(
      self,
      queries: Float['*b n d'],
      inputs_kv: Float['*b N D'] | None = None,
      self_attn_bias: Float['*b #h #n #N'] | None = None,
      cross_attn_bias: Float['*b #h #n #N'] | None = None,
      self_attn_mask: Bool['*b #h #n #N'] | None = None,
      cross_attn_mask: Bool['*b #h #n #N'] | None = None,
  ) -> Float['*b n d']:
    width = queries.shape[-1]
    normed_queries = nn.LayerNorm(
        use_bias=False, use_scale=True, dtype=queries.dtype, name='norm_q'
    )(queries)
    attn_out = queries

    if self.cross_attn_only and inputs_kv is None:
      raise ValueError('Option cross_attn_only requires providing inputs_kv.')

    # Self-attention.
    if not self.cross_attn_only:
      self_attn_out = ImprovedMHDPAttention(
          num_heads=self.num_heads, qk_size=self.qkv_size, name='self_att'
      )(
          inputs_q=normed_queries,
          inputs_kv=normed_queries,
          bias=self_attn_bias,
          mask=self_attn_mask,
      )
      attn_out += self_attn_out

    # Cross-attention.
    if inputs_kv is not None:
      cross_attn_out = ImprovedMHDPAttention(
          num_heads=self.num_heads, qk_size=self.qkv_size, name='cross_att'
      )(
          inputs_q=normed_queries,
          inputs_kv=inputs_kv,
          bias=cross_attn_bias,
          mask=cross_attn_mask,
      )
      attn_out += cross_attn_out

    # MLP.
    normed_attn_out = nn.LayerNorm(
        use_bias=False, use_scale=True, dtype=attn_out.dtype, name='norm_attn'
    )(attn_out)
    h = nn.gelu(
        nn.Dense(self.mlp_size, dtype=normed_attn_out.dtype, name='MLP_in')(
            normed_attn_out
        )
    )
    dense_kwargs = (
        {'kernel_init': nn.initializers.zeros} if self.zero_init else {}
    )
    mlp_out = nn.Dense(width, dtype=h.dtype, name='MLP_out', **dense_kwargs)(h)
    return attn_out + mlp_out
