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

"""Vision Transformer model definition (Self-Contained, CLIP-Compatible)."""

import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

# --- Scenic Components Inlined to Avoid Dependency ---

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]
Shape = Sequence[int]


class IdentityLayer(nn.Module):
  """Identity layer."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x


class StochasticDepth(nn.Module):
  """Stochastic Depth layer."""

  rate: float

  @nn.compact
  def __call__(
      self, x: jnp.ndarray, deterministic: bool = False
  ) -> jnp.ndarray:
    if not deterministic and self.rate > 0.0:
      keep_prob = 1.0 - self.rate
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      keep = jax.random.bernoulli(self.make_rng('dropout'), keep_prob, shape)
      return jax.lax.select(keep, x / keep_prob, jnp.zeros_like(x))
    return x


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  use_bias: bool = True
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  precision: Optional[jax.lax.Precision] = None
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, deterministic: bool):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
    )(inputs)
    x = IdentityLayer(name='mlp1')(self.activation_fn(x))
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
    )(x)
    output = IdentityLayer(name='mlp2')(output)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic
    )
    return output


def get_fixed_sincos_position_embedding(
    x_shape: Shape, temperature: float = 10_000, dtype: jnp.dtype = jnp.float32
):
  """Provides a fixed position encoding for 2D and 3D coordinates."""
  assert len(x_shape) in (4, 5), f'Unsupported input shape: {x_shape}'
  num_parts = 4 if len(x_shape) == 4 else 6
  channels = x_shape[-1]
  assert channels % num_parts == 0, f'Channels must be multiple of {num_parts}'
  omega = jnp.arange(
      channels // num_parts, dtype=jnp.float32
  ) / (channels / num_parts)
  omega = 1.0 / (temperature**omega)

  if len(x_shape) == 4:  # 2D input.
    _, h, w, _ = x_shape
    y, x = jnp.mgrid[:h, :w]
    y = jnp.einsum('m,d->md', y.flatten(), omega)
    x = jnp.einsum('m,d->md', x.flatten(), omega)
    p = [jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)]
    shape = (1, h, w, channels)
  elif len(x_shape) == 5:  # 3D input.
    _, t, h, w, _ = x_shape
    z, y, x = jnp.mgrid[:t, :h, :w]
    z = jnp.einsum('m,d->md', z.flatten(), omega)
    y = jnp.einsum('m,d->md', y.flatten(), omega)
    x = jnp.einsum('m,d->md', x.flatten(), omega)
    p = [
        jnp.sin(z),
        jnp.cos(z),
        jnp.sin(x),
        jnp.cos(x),
        jnp.sin(y),
        jnp.cos(y),
    ]
    shape = (1, t, h, w, channels)
  else:
    raise ValueError(f'Unsupported input shape: {x_shape}')

  assert (shape[0] == 1) and (shape[1:] == x_shape[1:])
  pe = jnp.concatenate(p, axis=1)
  return jnp.asarray(pe, dtype).reshape(*shape)


class AddFixedSinCosPositionEmbedding(nn.Module):
  """Provides a fixed position encoding for 2D and 3D coordinates."""

  temperature: float = 10_000
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    return inputs + get_fixed_sincos_position_embedding(
        inputs.shape, self.temperature, self.dtype
    )


# --- End of Scenic Copies ---


def quick_gelu(x: jnp.ndarray) -> jnp.ndarray:
  """Quick GELU activation used in CLIP models."""
  return x * jax.nn.sigmoid(1.702 * x)


class ToTokenSequence(nn.Module):
  """Transform a batch of views into a sequence of tokens."""

  patches: ml_collections.ConfigDict
  pos_interpolation_method: str
  hidden_size: int
  num_heads: int | None = None
  num_register_tokens: int = 0
  pooling: str = 'tok'
  posembs: Tuple[int, int] = (14, 14)
  positional_embedding: str = 'learned'
  pos_interpolation_antialias: bool = True
  pos_interpolation_offset: float = 0.0
  rope_config: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict
  )

  def add_positional_encodings(self, x: jnp.ndarray) -> jnp.ndarray:
    """Add positional encodings to the input patch sequence."""
    _, h, w, c = x.shape
    if self.positional_embedding == 'learned':
      posemb = self.param(
          'posembed_input',
          nn.initializers.normal(stddev=1 / np.sqrt(c)),
          (1, self.posembs[0], self.posembs[1], c),
          x.dtype,
      )
      # Optionally resize the positional encodings.
      if (h, w) != self.posembs:
        if self.pos_interpolation_offset:
          posemb = jax.image.scale_and_translate(
              posemb,
              (1, h, w, c),
              spatial_dims=(1, 2),
              scale=(jnp.array([h, w]) + self.pos_interpolation_offset)
              / jnp.array(self.posembs),
              translation=jnp.zeros([2]),
              method=self.pos_interpolation_method,
              antialias=self.pos_interpolation_antialias,
          )
        else:
          posemb = jax.image.resize(
              posemb,
              (1, h, w, c),
              self.pos_interpolation_method,
              antialias=self.pos_interpolation_antialias,
          )
      x = x + posemb
    elif self.positional_embedding == 'sinusoidal_2d':
      x = AddFixedSinCosPositionEmbedding()(x)

    return x

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, seqlen: int = -1, train: bool = False):

    fh, fw = self.patches.size
    x = nn.Conv(
        self.hidden_size,
        (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding',
    )(x)

    b, h, w, c = x.shape
    if self.rope_config.get('use_rope', False):
      raise NotImplementedError(
          'Rope position embedding is not implemented.'
      )
    else:
      x = self.add_positional_encodings(x)

    x = jnp.reshape(x, [b, h * w, c])
    n, l, c = x.shape

    # Possibly drop some tokens.
    if seqlen > 0 and seqlen < l:
      droptok_rng = self.make_rng('droptok')
      i_keep = jax.random.permutation(droptok_rng, l)[:seqlen]
      x = jnp.take(x, i_keep, axis=1)

    # Add register tokens.
    if self.num_register_tokens > 0:
      register_tok = self.param(
          'register',
          nn.initializers.zeros,
          (1, self.num_register_tokens, c),
          x.dtype,
      )
      register_tok = jnp.tile(register_tok, [n, 1, 1])
      x = jnp.concatenate([register_tok, x], axis=1)

    # Add a "class" token.
    if self.pooling == 'tok':
      cls_tok = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls_tok = jnp.tile(cls_tok, [n, 1, 1])
      x = jnp.concatenate([cls_tok, x], axis=1)
    return x


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer."""

  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.0
  activation_fn_str: str = 'gelu'

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    x = nn.LayerNorm()(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
    )(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs

    y = nn.LayerNorm()(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu
        if self.activation_fn_str == 'gelu'
        else quick_gelu,
    )(y, deterministic=deterministic)
    y = StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


class ViTBackbone(nn.Module):
  """Vision Transformer model with CLIP compatibility options."""

  tokenize: bool
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int

  # CLIP-specific options
  activation_fn_str: str = 'gelu'
  use_pre_norm: bool = False

  head_hidden_dim: Optional[int] = None
  head_bottleneck_dim: Optional[int] = None
  head_output_dim: Optional[int] = None
  normalize_last_layer: Optional[bool] = None
  use_bn: Optional[bool] = None
  stop_gradient_at_layer: Optional[int] = None
  positional_embedding: str = 'learned'
  posembs: Tuple[int, int] = (14, 14)
  pos_interpolation_method: Optional[str] = None
  pooling: str = 'tok'
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      *,
      train: bool,
      apply_norm: bool = True,
      pooling: Optional[str] = None,
      modality_embedding: Optional[jnp.ndarray] = None,
  ) -> Dict[str, jnp.ndarray]:
    if self.tokenize:
      x = ToTokenSequence(
          patches=self.patches,
          hidden_size=self.hidden_size,
          pooling=self.pooling,
          posembs=self.posembs,
          positional_embedding=self.positional_embedding,
          pos_interpolation_method=self.pos_interpolation_method,
      )(x)

    if self.use_pre_norm:
      x = nn.LayerNorm(name='pre_ln')(x)

    for lyr in range(self.num_layers):
      if (
          train
          and self.stop_gradient_at_layer is not None
          and lyr == self.stop_gradient_at_layer + 1
          and modality_embedding is not None
      ):
        x = x.at[:, 0, :].add(modality_embedding)

      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1))
          * self.stochastic_depth,
          activation_fn_str=self.activation_fn_str,
          name=f'encoderblock_{lyr}',
      )(x, deterministic=not train)

      if (
          train
          and self.stop_gradient_at_layer is not None
          and lyr == self.stop_gradient_at_layer
      ):
        x = jax.lax.stop_gradient(x)

    prenorm_x = x
    if apply_norm:
      x = nn.LayerNorm(name='encoder_norm')(x)

    final_pooling = pooling if pooling is not None else self.pooling

    if final_pooling == 'tok':
      backbone_output = x[:, 0]
    elif final_pooling == 'gap':
      if self.tokenize and self.pooling == 'tok':
        backbone_output = jnp.mean(x[:, 1:], axis=1)
      else:
        backbone_output = jnp.mean(x, axis=1)
    elif final_pooling == 'none':
      backbone_output = jnp.zeros((x.shape[0], x.shape[-1]))
    else:
      raise ValueError(f'Unknown pooling: {final_pooling}')

    return dict(
        final_tokens=x,
        pooled_output=backbone_output,
        prenorm_tokens=prenorm_x,
    )
