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

"""Readout module related components."""

from __future__ import annotations

import functools
from typing import Optional

import einops
from flax import linen as nn
from jax import numpy as jnp
from kauldron import kd
from kauldron import typing as kd_typing

check_type = kd_typing.check_type
Float = kd_typing.Float
typechecked = kd_typing.typechecked


class MLP(nn.Module):
  """A simple MLP."""

  hidden_size: int
  output_size: Optional[int] = None
  num_hidden_layers: int = 1

  @typechecked
  @nn.compact
  def __call__(self, inputs: Float['*B c']) -> Float['*B d']:
    output_size = self.output_size or inputs.shape[-1]

    x = inputs
    for _ in range(self.num_hidden_layers):
      x = nn.Dense(self.hidden_size, dtype=x.dtype)(x)
      x = nn.gelu(x)
    x = nn.Dense(output_size, dtype=x.dtype)(x)

    return x


class AttentionReadout(nn.Module):
  """Cross-attention readout with learnable latent queries."""

  num_classes: int
  num_params: int
  num_heads: int
  num_queries: int = 1
  match_vjepa_implementation: bool = True
  add_temporal_posenc: bool = True
  num_test_clips: int = 1
  is_training = kd.nn.train_property()
  dropout_rate: float = 0.0

  output_shape: tuple[int, int, int] | tuple[int, int, int, int] | None = None
  decoding_patch_size: tuple[int, int, int] | None = None

  @nn.compact
  @typechecked
  def __call__(
      self, inputs: Float['B T N C'], queries: Optional[Float['B Q D']] = None
  ):
    num_params_per_head = self.num_params // self.num_heads
    if num_params_per_head * self.num_heads != self.num_params:
      raise ValueError(
          f'num_params ({self.num_params}) must be a multiple of num_heads'
          f' ({self.num_heads}).'
      )
    # Cross-attend from a (different) learned token into each of the given sets.
    feats = inputs

    if self.match_vjepa_implementation:
      # Normalize the input features first
      feats = nn.LayerNorm(dtype=feats.dtype)(feats)
      use_bias = True
    else:
      use_bias = False

    # Optionally add learned posenc to the representation.
    if self.add_temporal_posenc:
      check_type(feats, Float['B T N C'])
      posenc = kd.nn.LearnedEmbedding(name='temporal_posenc')(
          feats.shape, axis=-3
      )
      posenc = posenc.astype(feats.dtype)
      feats += posenc
    feats = einops.rearrange(feats, '... T N C -> ... (T N) C')

    if queries is None:
      # Initialize Learnable queries.
      num_queries = self.num_queries
      query = self.param(
          'query',
          nn.initializers.normal(stddev=0.02),
          [num_queries, self.num_heads, num_params_per_head],
          feats.dtype,
      )
      query = jnp.broadcast_to(query, (feats.shape[0],) + query.shape)
    else:
      num_queries = queries.shape[-2]
      query = nn.Dense(self.num_heads * num_params_per_head, dtype=feats.dtype)(
          queries
      )
      query = einops.rearrange(
          query, '... Q (h n) -> ... Q h n', h=self.num_heads
      )

    # Cross-attention
    key_val_dense = functools.partial(
        nn.DenseGeneral,
        features=(self.num_heads, num_params_per_head),
        axis=-1,
        dtype=feats.dtype,
        use_bias=use_bias,
    )
    key = key_val_dense(name='key_embedding')(feats)
    value = key_val_dense(name='value_embedding')(feats)

    token = nn.dot_product_attention(
        query=query,
        key=key,
        value=value,
    )

    token = nn.Dropout(rate=self.dropout_rate)(
        token, deterministic=not self.is_training
    )
    token = einops.rearrange(token, '...  Q N c -> ... Q (N c)')

    if self.match_vjepa_implementation:
      # Extra MLP layer with residual connection.
      query = einops.rearrange(query, '... Q N c -> ... Q (N c)')

      token = query + nn.Dense(self.num_params, dtype=token.dtype)(token)
      residual = token
      token = nn.LayerNorm(dtype=token.dtype)(token)
      token = MLP(
          hidden_size=self.num_params * 4,
          num_hidden_layers=1,
      )(token)
      token = token + residual

    if num_queries == 1:
      # Squeeze the num_queries dimension.
      token = jnp.squeeze(token, axis=-2)

    out = nn.Dense(self.num_classes, dtype=token.dtype)(token)
    if self.output_shape is not None and self.decoding_patch_size is not None:
      channel_dim = self.output_shape[-1] if len(self.output_shape) == 4 else 1
      # Rearrange the output tensor to match the desired output shape, by
      # reshaping the pixels and patches dimensions.
      out = einops.rearrange(
          out,
          'B (n_pixels_patch0 n_pixels_patch1 n_pixels_patch2) (patch_size0'
          ' patch_size1 patch_size2 c) -> B (n_pixels_patch0 patch_size0)'
          ' (n_pixels_patch1 patch_size1) (n_pixels_patch2 patch_size2) c',
          patch_size0=self.decoding_patch_size[0],
          patch_size1=self.decoding_patch_size[1],
          patch_size2=self.decoding_patch_size[2],
          n_pixels_patch0=self.output_shape[0] // self.decoding_patch_size[0],
          n_pixels_patch1=self.output_shape[1] // self.decoding_patch_size[1],
          n_pixels_patch2=self.output_shape[2] // self.decoding_patch_size[2],
          c=channel_dim,
      )

    # note: these options only make sense for classification-type tasks
    if self.num_test_clips > 1 and not self.is_training:  # multi-clip eval
      out = nn.softmax(out, axis=-1)
      out = einops.reduce(
          out,
          '(b n) ...-> b ...',
          'mean',
          n=self.num_test_clips,
      )
    return out
