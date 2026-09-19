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

"""WAN 2.1 VAE (Variational AutoEncoder) for video encoding/decoding.

This module implements the 3D causal VAE used in the WAN 2.1 video diffusion
model. The VAE encodes video frames into a compact latent representation and
decodes latents back into video frames.

Encoding: [B, T, H, W, 3] -> [B, T', H', W', C]
  - C = 16 (latent channels)
  - Spatial downsampling: 8x (H' = H/8, W' = W/8)
  - Temporal downsampling: 4x (T' ≈ T/4)

Decoding: [B, T', H', W', C] -> [B, T, H, W, 3]

Forked from maxdiffusion/models/wan/autoencoder_kl_wan.py.
"""

import dataclasses
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

# pylint: disable=g-bare-generic,g-importing-member,protected-access,g-doc-return-or-yield,g-doc-args
# pylint: disable=out_dim,dangerous-default-value,undefined-loop-variable,g-complex-comprehension,g-long-ternary
from flax import nnx
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
import jax.numpy as jnp
import safetensors

# Number of frames to cache for causal convolution temporal context.
CACHE_T = 2

# ---------------------------------------------------------------------------
# Activation helper
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
  """Return JAX activation function by name."""
  func = _ACTIVATIONS.get(name)
  if func is None:
    raise ValueError(f"Unknown activation function: {name}")
  return func


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DecoderOutput:
  """Output of the VAE decoder.

  Attributes:
    sample: Decoded video tensor of shape [B, T, H, W, 3].
  """

  sample: jnp.ndarray


@dataclasses.dataclass
class AutoencoderKLOutput:
  """Output of the VAE encoder.

  Attributes:
    latent_dist: Encoded outputs represented as a diagonal Gaussian distribution
      from which latents can be sampled.
  """

  latent_dist: "DiagonalGaussianDistribution"


class DiagonalGaussianDistribution:
  """Diagonal Gaussian distribution parameterized by mean and log-variance.

  Used to represent the VAE latent posterior distribution.
  """

  def __init__(self, parameters: jnp.ndarray, deterministic: bool = False):
    """Initialize from concatenated [mean, logvar] along the last axis.

    Args:
      parameters: Tensor of shape [..., 2*C] where the last axis is split into
        mean and log-variance.
      deterministic: If True, std is set to zero (mode sampling only).
    """
    self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
    self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
    self.deterministic = deterministic
    self.std = jnp.exp(0.5 * self.logvar)
    self.var = jnp.exp(self.logvar)
    if self.deterministic:
      self.var = self.std = jnp.zeros_like(self.mean)

  def sample(self, key: jax.Array) -> jnp.ndarray:
    """Sample from the distribution using the reparameterization trick."""
    return self.mean + self.std * jax.random.normal(key, self.mean.shape)

  def kl(
      self, other: Optional["DiagonalGaussianDistribution"] = None
  ) -> jnp.ndarray:
    """Return one KL divergence per batch item, summing all event dimensions.

    Video posteriors have shape [B, T, H, W, C], so the channel axis is part
    of the event alongside the temporal and spatial axes.
    """
    if self.deterministic:
      return jnp.zeros((self.mean.shape[0],), dtype=self.mean.dtype)
    event_axes = tuple(range(1, self.mean.ndim))
    if other is None:
      return 0.5 * jnp.sum(
          self.mean**2 + self.var - 1.0 - self.logvar, axis=event_axes
      )
    return 0.5 * jnp.sum(
        jnp.square(self.mean - other.mean) / other.var
        + self.var / other.var
        - 1.0
        - self.logvar
        + other.logvar,
        axis=event_axes,
    )

  def mode(self) -> jnp.ndarray:
    """Return the mode (mean) of the distribution."""
    return self.mean


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _canonicalize_tuple(
    x: Union[int, Sequence[int]], rank: int, name: str
) -> Tuple[int, ...]:
  """Canonicalize a value to a tuple of `rank` integers."""
  if isinstance(x, int):
    return (x,) * rank
  elif isinstance(x, Sequence) and len(x) == rank:
    return tuple(x)
  else:
    raise ValueError(
        f"Argument '{name}' must be an integer or a sequence of "
        f"{rank} integers. Got {x}"
    )


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------


class WanCausalConv3d(nnx.Module):
  """3D causal convolution that pads only in the past along the temporal axis.

  This ensures the convolution is causal: outputs at time t depend only on
  inputs at times <= t.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, int, int]],
      stride: Union[int, Tuple[int, int, int]] = 1,
      padding: Union[int, Tuple[int, int, int]] = 0,
      use_bias: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    self.kernel_size = _canonicalize_tuple(kernel_size, 3, "kernel_size")
    self.stride = _canonicalize_tuple(stride, 3, "stride")
    padding_tuple = _canonicalize_tuple(padding, 3, "padding")

    # Causal padding: pad only before in the depth/time dimension.
    self._causal_padding = (
        (0, 0),  # Batch
        (2 * padding_tuple[0], 0),  # Depth (causal)
        (padding_tuple[1], padding_tuple[1]),  # Height (symmetric)
        (padding_tuple[2], padding_tuple[2]),  # Width (symmetric)
        (0, 0),  # Channel
    )
    self._depth_padding_before = self._causal_padding[1][0]

    self.conv = nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=self.kernel_size,
        strides=self.stride,
        use_bias=use_bias,
        padding="VALID",
        rngs=rngs,
        kernel_init=nnx.initializers.xavier_uniform(),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(
      self,
      x: jax.Array,
      cache_x: Optional[jax.Array] = None,
      idx: int = -1,
  ) -> jax.Array:
    """Forward pass with optional temporal cache.

    Args:
      x: Input tensor of shape [B, T, H, W, C].
      cache_x: Optional cached frames from previous chunks for causal context.
      idx: Unused index parameter (kept for API compatibility).

    Returns:
      Output tensor after causal 3D convolution.
    """
    current_padding = list(self._causal_padding)
    padding_needed = self._depth_padding_before

    if cache_x is not None and padding_needed > 0:
      assert (
          cache_x.shape[0] == x.shape[0] and cache_x.shape[2:] == x.shape[2:]
      ), "Cache spatial/channel dims mismatch"
      cache_len = cache_x.shape[1]
      x = jnp.concatenate([cache_x, x], axis=1)

      padding_needed -= cache_len
      if padding_needed < 0:
        x = x[:, -padding_needed:, ...]
        current_padding[1] = (0, 0)
      else:
        current_padding[1] = (padding_needed, 0)

    padding_to_apply = tuple(current_padding)
    if any(p > 0 for dim_pads in padding_to_apply for p in dim_pads):
      x_padded = jnp.pad(
          x, padding_to_apply, mode="constant", constant_values=0.0
      )
    else:
      x_padded = x
    return self.conv(x_padded)


class WanRMSNorm(nnx.Module):
  """RMS normalization layer.

  Normalizes inputs by their L2 norm and scales by a learnable parameter.
  """

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      channel_first: bool = True,
      images: bool = True,
      eps: float = 1e-6,
      use_bias: bool = False,
  ):
    broadcastable_dims = (1, 1, 1) if not images else (1, 1)
    shape = (dim, *broadcastable_dims) if channel_first else (dim,)
    self.eps = eps
    self.channel_first = channel_first
    self.scale = dim**0.5
    self.gamma = nnx.Param(jnp.ones(shape))
    if use_bias:
      self.bias = nnx.Param(jnp.zeros(shape))
    else:
      self.bias = 0

  def __call__(self, x: jax.Array) -> jax.Array:
    normalized = jnp.linalg.norm(
        x, ord=2, axis=(1 if self.channel_first else -1), keepdims=True
    )
    normalized = x / jnp.maximum(normalized, self.eps)
    normalized = normalized * self.scale * self.gamma
    if self.bias:
      return normalized + self.bias.value
    return normalized


class WanUpsample(nnx.Module):
  """2D spatial upsampling using image resize."""

  def __init__(
      self, scale_factor: Tuple[float, float], method: str = "nearest"
  ):
    self.scale_factor = scale_factor
    self.method = method

  def __call__(self, x: jax.Array) -> jax.Array:
    """Upsample spatial dimensions of a 4D tensor [N, H, W, C]."""
    input_dtype = x.dtype
    in_shape = x.shape
    assert len(in_shape) == 4, "This module only takes tensors with shape of 4."
    n, h, w, c = in_shape
    target_h = int(h * self.scale_factor[0])
    target_w = int(w * self.scale_factor[1])
    out = jax.image.resize(
        x.astype(jnp.float32),
        (n, target_h, target_w, c),
        method=self.method,
    )
    return out.astype(input_dtype)


class Identity(nnx.Module):
  """Identity module that passes input through unchanged."""

  def __call__(self, x: jax.Array) -> jax.Array:
    return x


class ZeroPaddedConv2D(nnx.Module):
  """2D convolution wrapper (no additional padding beyond the conv itself)."""

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      kernel_size: Union[int, Tuple[int, int, int]],
      stride: Union[int, Tuple[int, int, int]] = 1,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    self.conv = nnx.Conv(
        dim,
        dim,
        kernel_size=kernel_size,
        strides=stride,
        use_bias=True,
        rngs=rngs,
        kernel_init=nnx.initializers.xavier_uniform(),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return self.conv(x)


class WanResample(nnx.Module):
  """Spatial and temporal resampling module.

  Supports 2D/3D upsampling and downsampling modes.
  """

  def __init__(
      self,
      dim: int,
      mode: str,
      rngs: nnx.Rngs,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    self.dim = dim
    self.mode = mode
    self.time_conv = nnx.data(None)

    if mode == "upsample2d":
      self.resample = nnx.Sequential(
          WanUpsample(scale_factor=(2.0, 2.0), method="nearest"),
          nnx.Conv(
              dim,
              dim // 2,
              kernel_size=(3, 3),
              padding="SAME",
              use_bias=True,
              rngs=rngs,
              kernel_init=nnx.initializers.xavier_uniform(),
              dtype=dtype,
              param_dtype=weights_dtype,
              precision=precision,
          ),
      )
    elif mode == "upsample3d":
      self.resample = nnx.Sequential(
          WanUpsample(scale_factor=(2.0, 2.0), method="nearest"),
          nnx.Conv(
              dim,
              dim // 2,
              kernel_size=(3, 3),
              padding="SAME",
              use_bias=True,
              rngs=rngs,
              kernel_init=nnx.initializers.xavier_uniform(),
              dtype=dtype,
              param_dtype=weights_dtype,
              precision=precision,
          ),
      )
      self.time_conv = WanCausalConv3d(
          rngs=rngs,
          in_channels=dim,
          out_channels=dim * 2,
          kernel_size=(3, 1, 1),
          padding=(1, 0, 0),
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    elif mode == "downsample2d":
      self.resample = ZeroPaddedConv2D(
          dim=dim,
          rngs=rngs,
          kernel_size=(3, 3),
          stride=(2, 2),
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    elif mode == "downsample3d":
      self.resample = ZeroPaddedConv2D(
          dim=dim,
          rngs=rngs,
          kernel_size=(3, 3),
          stride=(2, 2),
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
      self.time_conv = WanCausalConv3d(
          rngs=rngs,
          in_channels=dim,
          out_channels=dim,
          kernel_size=(3, 1, 1),
          stride=(2, 1, 1),
          padding=(0, 0, 0),
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    else:
      self.resample = Identity()

  def __call__(
      self,
      x: jax.Array,
      feat_cache: Optional[list] = None,
      feat_idx: Optional[list] = None,
  ) -> jax.Array:
    """Resample input tensor.

    Args:
      x: Input of shape [B, T, H, W, C].
      feat_cache: Optional list for caching intermediate temporal features.
      feat_idx: Optional mutable list [int] tracking the current cache index.

    Returns:
      Resampled tensor.
    """
    if feat_idx is None:
      feat_idx = [0]
    b, t, h, w, c = x.shape
    assert c == self.dim

    if self.mode == "upsample3d":
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = "Rep"
          feat_idx[0] += 1
        else:
          cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
          if (
              cache_x.shape[1] < 2
              and feat_cache[idx] is not None
              and feat_cache[idx] != "Rep"
          ):
            cache_x = jnp.concatenate(
                [
                    jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1),
                    cache_x,
                ],
                axis=1,
            )
          if (
              cache_x.shape[1] < 2
              and feat_cache[idx] is not None
              and feat_cache[idx] == "Rep"
          ):
            cache_x = jnp.concatenate(
                [jnp.zeros(cache_x.shape), cache_x], axis=1
            )
          if feat_cache[idx] == "Rep":
            x = self.time_conv(x)
          else:
            x = self.time_conv(x, feat_cache[idx])
          feat_cache[idx] = cache_x
          feat_idx[0] += 1
          x = x.reshape(b, t, h, w, 2, c)
          x = jnp.stack([x[:, :, :, :, 0, :], x[:, :, :, :, 1, :]], axis=1)
          x = x.reshape(b, t * 2, h, w, c)
    t = x.shape[1]
    x = x.reshape(b * t, h, w, c)
    x = self.resample(x)
    h_new, w_new, c_new = x.shape[1:]
    x = x.reshape(b, t, h_new, w_new, c_new)

    if self.mode == "downsample3d":
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = jnp.copy(x)
          feat_idx[0] += 1
        else:
          cache_x = jnp.copy(x[:, -1:, :, :, :])
          x = self.time_conv(
              jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1)
          )
          feat_cache[idx] = cache_x
          feat_idx[0] += 1

    return x


class WanResidualBlock(nnx.Module):
  """3D residual block with causal convolutions and RMS normalization."""

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      non_linearity: str = "silu",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    self.nonlinearity = get_activation(non_linearity)

    self.norm1 = WanRMSNorm(
        dim=in_dim, rngs=rngs, images=False, channel_first=False
    )
    self.conv1 = WanCausalConv3d(
        rngs=rngs,
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=3,
        padding=1,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.norm2 = WanRMSNorm(
        dim=out_dim, rngs=rngs, images=False, channel_first=False
    )
    self.conv2 = WanCausalConv3d(
        rngs=rngs,
        in_channels=out_dim,
        out_channels=out_dim,
        kernel_size=3,
        padding=1,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.conv_shortcut = (
        WanCausalConv3d(
            rngs=rngs,
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
        )
        if in_dim != out_dim
        else Identity()
    )

  def __call__(
      self,
      x: jax.Array,
      feat_cache: Optional[list] = None,
      feat_idx: Optional[list] = None,
  ) -> jax.Array:
    if feat_idx is None:
      feat_idx = [0]

    h = self.conv_shortcut(x)

    x = self.norm1(x)
    x = self.nonlinearity(x)

    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv1(x, feat_cache[idx], idx)
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv1(x)

    x = self.norm2(x)
    x = self.nonlinearity(x)

    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv2(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv2(x)

    x = x + h
    return x


class WanAttentionBlock(nnx.Module):
  """Spatial self-attention block applied per-frame."""

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    self.dim = dim
    self.norm = WanRMSNorm(rngs=rngs, dim=dim, channel_first=False)
    self.to_qkv = nnx.Conv(
        in_features=dim,
        out_features=dim * 3,
        kernel_size=(1, 1),
        rngs=rngs,
        kernel_init=nnx.initializers.xavier_uniform(),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )
    self.proj = nnx.Conv(
        in_features=dim,
        out_features=dim,
        kernel_size=(1, 1),
        rngs=rngs,
        kernel_init=nnx.initializers.xavier_uniform(),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Apply spatial self-attention per frame.

    Args:
      x: Input of shape [B, T, H, W, C].

    Returns:
      Output of shape [B, T, H, W, C] with residual connection.
    """
    identity = x
    batch_size, time, height, width, channels = x.shape

    x = x.reshape(batch_size * time, height, width, channels)
    x = self.norm(x)

    qkv = self.to_qkv(x)
    qkv = qkv.reshape(batch_size * time, 1, -1, channels * 3)
    qkv = jnp.transpose(qkv, (0, 1, 3, 2))
    q, k, v = jnp.split(qkv, 3, axis=-2)
    q = jnp.transpose(q, (0, 1, 3, 2))
    k = jnp.transpose(k, (0, 1, 3, 2))
    v = jnp.transpose(v, (0, 1, 3, 2))
    x = jax.nn.dot_product_attention(q, k, v)
    x = jnp.squeeze(x, 1).reshape(batch_size * time, height, width, channels)

    x = self.proj(x)
    x = x.reshape(batch_size, time, height, width, channels)

    return x + identity


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------


class WanMidBlock(nnx.Module):
  """Middle block with residual blocks and spatial attention."""

  def __init__(
      self,
      dim: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      non_linearity: str = "silu",
      num_layers: int = 1,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    self.dim = dim
    resnets = [
        WanResidualBlock(
            in_dim=dim,
            out_dim=dim,
            rngs=rngs,
            dropout=dropout,
            non_linearity=non_linearity,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
        )
    ]
    attentions = []
    for _ in range(num_layers):
      attentions.append(
          WanAttentionBlock(
              dim=dim,
              rngs=rngs,
              dtype=dtype,
              weights_dtype=weights_dtype,
              precision=precision,
          )
      )
      resnets.append(
          WanResidualBlock(
              in_dim=dim,
              out_dim=dim,
              rngs=rngs,
              dropout=dropout,
              non_linearity=non_linearity,
              dtype=dtype,
              weights_dtype=weights_dtype,
              precision=precision,
          )
      )
    self.attentions = nnx.data(attentions)
    self.resnets = nnx.data(resnets)

  def __call__(
      self,
      x: jax.Array,
      feat_cache: Optional[list] = None,
      feat_idx: Optional[list] = None,
  ) -> jax.Array:
    if feat_idx is None:
      feat_idx = [0]
    x = self.resnets[0](x, feat_cache, feat_idx)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
      if attn is not None:
        x = attn(x)
      x = resnet(x, feat_cache, feat_idx)
    return x


class WanUpBlock(nnx.Module):
  """Decoder upsampling block with residual layers and optional upsampling."""

  def __init__(
      self,
      in_dim: int,
      out_dim: int,
      num_res_blocks: int,
      rngs: nnx.Rngs,
      dropout: float = 0.0,
      upsample_mode: Optional[str] = None,
      non_linearity: str = "silu",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    resnets = []
    current_dim = in_dim
    for _ in range(num_res_blocks + 1):
      resnets.append(
          WanResidualBlock(
              in_dim=current_dim,
              out_dim=out_dim,
              dropout=dropout,
              non_linearity=non_linearity,
              rngs=rngs,
              dtype=dtype,
              weights_dtype=weights_dtype,
              precision=precision,
          )
      )
      current_dim = out_dim
    self.resnets = nnx.data(resnets)

    self.upsamplers = nnx.data(None)
    if upsample_mode is not None:
      self.upsamplers = [
          WanResample(
              dim=out_dim,
              mode=upsample_mode,
              rngs=rngs,
              weights_dtype=weights_dtype,
              dtype=dtype,
              precision=precision,
          )
      ]

  def __call__(
      self,
      x: jax.Array,
      feat_cache: Optional[list] = None,
      feat_idx: Optional[list] = None,
  ) -> jax.Array:
    if feat_idx is None:
      feat_idx = [0]
    for resnet in self.resnets:
      if feat_cache is not None:
        x = resnet(x, feat_cache, feat_idx)
      else:
        x = resnet(x)
    if self.upsamplers is not None:
      if feat_cache is not None:
        x = self.upsamplers[0](x, feat_cache, feat_idx)
      else:
        x = self.upsamplers[0](x)
    return x


class WanEncoder3d(nnx.Module):
  """3D encoder that maps video frames to a latent representation.

  Takes input of shape [B, T, H, W, 3] and produces latents of shape
  [B, T', H', W', z_dim] with spatial 8x and temporal 4x downsampling.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int = 128,
      z_dim: int = 4,
      dim_mult: List[int] = [1, 2, 4, 4],
      num_res_blocks: int = 2,
      attn_scales: List[float] = [],
      temperal_downsample: List[bool] = [True, True, False],
      dropout: float = 0.0,
      non_linearity: str = "silu",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    self.dim = dim
    self.z_dim = z_dim
    self.dim_mult = dim_mult
    self.num_res_blocks = num_res_blocks
    self.attn_scales = attn_scales
    self.temperal_downsample = temperal_downsample
    self.nonlinearity = get_activation(non_linearity)

    dims = [dim * u for u in [1] + dim_mult]
    scale = 1.0

    # Input convolution.
    self.conv_in = WanCausalConv3d(
        rngs=rngs,
        in_channels=3,
        out_channels=dims[0],
        kernel_size=3,
        padding=1,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # Downsample blocks.
    self.down_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      for _ in range(num_res_blocks):
        self.down_blocks.append(
            WanResidualBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                dropout=dropout,
                rngs=rngs,
                dtype=dtype,
                weights_dtype=weights_dtype,
                precision=precision,
            )
        )
        if scale in attn_scales:
          self.down_blocks.append(
              WanAttentionBlock(
                  dim=out_dim,
                  rngs=rngs,
                  dtype=dtype,
                  weights_dtype=weights_dtype,
                  precision=precision,
              )
          )
        in_dim = out_dim

      if i != len(dim_mult) - 1:
        mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
        self.down_blocks.append(
            WanResample(
                out_dim,
                mode=mode,
                rngs=rngs,
                dtype=dtype,
                weights_dtype=weights_dtype,
                precision=precision,
            )
        )
        scale /= 2.0
    self.down_blocks = nnx.data(self.down_blocks)

    # Middle block.
    self.mid_block = WanMidBlock(
        dim=out_dim,
        rngs=rngs,
        dropout=dropout,
        non_linearity=non_linearity,
        num_layers=1,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # Output blocks.
    self.norm_out = WanRMSNorm(
        out_dim, channel_first=False, images=False, rngs=rngs
    )
    self.conv_out = WanCausalConv3d(
        rngs=rngs,
        in_channels=out_dim,
        out_channels=z_dim,
        kernel_size=3,
        padding=1,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(
      self,
      x: jax.Array,
      feat_cache: Optional[list] = None,
      feat_idx: Optional[list] = None,
  ) -> jax.Array:
    if feat_idx is None:
      feat_idx = [0]

    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv_in(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv_in(x)

    for layer in self.down_blocks:
      if feat_cache is not None:
        x = layer(x, feat_cache, feat_idx)
      else:
        x = layer(x)

    x = self.mid_block(x, feat_cache, feat_idx)

    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv_out(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv_out(x)
    return x


class WanDecoder3d(nnx.Module):
  """3D decoder that maps latents back to video frames.

  Takes latents of shape [B, T', H', W', z_dim] and produces video
  of shape [B, T, H, W, 3] with spatial 8x and temporal 4x upsampling.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int = 128,
      z_dim: int = 4,
      dim_mult: List[int] = [1, 2, 4, 4],
      num_res_blocks: int = 2,
      attn_scales: List[float] = [],
      temperal_upsample: List[bool] = [False, True, True],
      dropout: float = 0.0,
      non_linearity: str = "silu",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    self.dim = dim
    self.z_dim = z_dim
    self.dim_mult = dim_mult
    self.num_res_blocks = num_res_blocks
    self.attn_scales = attn_scales
    self.temperal_upsample = temperal_upsample

    self.nonlinearity = get_activation(non_linearity)

    dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    scale = 1.0 / 2 ** (len(dim_mult) - 2)

    # Input convolution.
    self.conv_in = WanCausalConv3d(
        rngs=rngs,
        in_channels=z_dim,
        out_channels=dims[0],
        kernel_size=3,
        padding=1,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # Middle block.
    self.mid_block = WanMidBlock(
        dim=dims[0],
        rngs=rngs,
        dropout=dropout,
        non_linearity=non_linearity,
        num_layers=1,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

    # Upsample blocks.
    self.up_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      if i > 0:
        in_dim = in_dim // 2

      upsample_mode = None
      if i != len(dim_mult) - 1:
        upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"

      up_block = WanUpBlock(
          in_dim=in_dim,
          out_dim=out_dim,
          num_res_blocks=num_res_blocks,
          dropout=dropout,
          upsample_mode=upsample_mode,
          non_linearity=non_linearity,
          rngs=rngs,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
      self.up_blocks.append(up_block)

      if upsample_mode is not None:
        scale *= 2.0
    self.up_blocks = nnx.data(self.up_blocks)

    # Output blocks.
    self.norm_out = WanRMSNorm(
        dim=out_dim, images=False, rngs=rngs, channel_first=False
    )
    self.conv_out = WanCausalConv3d(
        rngs=rngs,
        in_channels=out_dim,
        out_channels=3,
        kernel_size=3,
        padding=1,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

  def __call__(
      self,
      x: jax.Array,
      feat_cache: Optional[list] = None,
      feat_idx: Optional[list] = None,
  ) -> jax.Array:
    if feat_idx is None:
      feat_idx = [0]

    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv_in(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv_in(x)

    # Middle.
    x = self.mid_block(x, feat_cache, feat_idx)

    # Upsamples.
    for up_block in self.up_blocks:
      x = up_block(x, feat_cache, feat_idx)

    # Head.
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate(
            [jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x],
            axis=1,
        )
      x = self.conv_out(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv_out(x)
    return x


# ---------------------------------------------------------------------------
# Cache for chunked encoding/decoding
# ---------------------------------------------------------------------------


class AutoencoderKLWanCache:
  """Cache for chunked temporal encoding and decoding.

  The WAN VAE processes video one temporal chunk at a time using causal
  convolutions. This cache stores intermediate feature maps between chunks
  to maintain temporal consistency.
  """

  def __init__(self, module: "AutoencoderKLWan"):
    self.module = module
    self.clear_cache()

  def clear_cache(self):
    """Reset all cache dictionaries and indices."""

    def _count_conv3d(module: nnx.Module) -> int:
      count = 0
      node_types = nnx.graph.iter_graph([module])
      for _, value in node_types:
        if isinstance(value, WanCausalConv3d):
          count += 1
      return count

    self._conv_num = _count_conv3d(self.module.decoder)
    self._conv_idx = [0]
    self._feat_map = [None] * self._conv_num
    # Cache for encoder.
    self._enc_conv_num = _count_conv3d(self.module.encoder)
    self._enc_conv_idx = [0]
    self._enc_feat_map = [None] * self._enc_conv_num


# ---------------------------------------------------------------------------
# Main VAE class
# ---------------------------------------------------------------------------


class AutoencoderKLWan(nnx.Module):
  """WAN 2.1 Variational Autoencoder with KL divergence loss.

  This VAE encodes video frames into a compact latent space and decodes
  latents back into video frames. It uses 3D causal convolutions for
  temporal consistency and processes videos in chunks.

  The encoder produces a diagonal Gaussian distribution (mean + logvar)
  and the decoder reconstructs video from sampled or mode latents.

  Default configuration (WAN 2.1):
    - Latent channels (z_dim): 16
    - Spatial downsampling: 8x
    - Temporal downsampling: 4x
    - Base dimension: 96
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      base_dim: int = 96,
      z_dim: int = 16,
      dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
      num_res_blocks: int = 2,
      attn_scales: List[float] = [],
      temperal_downsample: List[bool] = [False, True, True],
      dropout: float = 0.0,
      latents_mean: List[float] = [
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
      ],
      latents_std: List[float] = [
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
      ],
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
  ):
    self.z_dim = z_dim
    self.temperal_downsample = temperal_downsample
    self.temporal_upsample = temperal_downsample[::-1]
    self.latents_mean = latents_mean
    self.latents_std = latents_std

    dim_mult = list(dim_mult)

    self.encoder = WanEncoder3d(
        rngs=rngs,
        dim=base_dim,
        z_dim=z_dim * 2,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_downsample=temperal_downsample,
        dropout=dropout,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.quant_conv = WanCausalConv3d(
        rngs=rngs,
        in_channels=z_dim * 2,
        out_channels=z_dim * 2,
        kernel_size=1,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.post_quant_conv = WanCausalConv3d(
        rngs=rngs,
        in_channels=z_dim,
        out_channels=z_dim,
        kernel_size=1,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.decoder = WanDecoder3d(
        rngs=rngs,
        dim=base_dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_upsample=self.temporal_upsample,
        dropout=dropout,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )

  @classmethod
  def from_config(
      cls,
      config: Dict,
      rngs: Optional[nnx.Rngs] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
  ) -> "AutoencoderKLWan":
    """Create an AutoencoderKLWan from a configuration dictionary.

    Args:
      config: Dictionary with model hyperparameters. Expected keys: base_dim,
        z_dim, dim_mult, num_res_blocks, attn_scales, temperal_downsample,
        dropout, latents_mean, latents_std.
      rngs: Optional RNG container. If None, uses default seed 0.
      dtype: Compute dtype.
      weights_dtype: Parameter storage dtype.

    Returns:
      Initialized AutoencoderKLWan instance.
    """
    if rngs is None:
      rngs = nnx.Rngs(0)
    return cls(
        rngs=rngs,
        base_dim=config.get("base_dim", 96),
        z_dim=config.get("z_dim", 16),
        dim_mult=config.get("dim_mult", [1, 2, 4, 4]),
        num_res_blocks=config.get("num_res_blocks", 2),
        attn_scales=config.get("attn_scales", []),
        temperal_downsample=config.get(
            "temperal_downsample", [False, True, True]
        ),
        dropout=config.get("dropout", 0.0),
        latents_mean=config.get(
            "latents_mean",
            [
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
            ],
        ),
        latents_std=config.get(
            "latents_std",
            [
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
            ],
        ),
        dtype=dtype,
        weights_dtype=weights_dtype,
    )

  def _encode(
      self, x: jax.Array, feat_cache: AutoencoderKLWanCache
  ) -> jax.Array:
    """Encode video into concatenated [mean, logvar] latents.

    Uses chunked temporal processing for memory efficiency.
    """
    feat_cache.clear_cache()
    if x.shape[-1] != 3:
      x = jnp.transpose(x, (0, 2, 3, 4, 1))
      assert (
          x.shape[-1] == 3
      ), f"Expected input shape (N, D, H, W, 3), got {x.shape}"

    t = x.shape[1]
    iter_ = 1 + (t - 1) // 4
    for i in range(iter_):
      feat_cache._enc_conv_idx = [0]
      if i == 0:
        out = self.encoder(
            x[:, :1, :, :, :],
            feat_cache=feat_cache._enc_feat_map,
            feat_idx=feat_cache._enc_conv_idx,
        )
      else:
        out_ = self.encoder(
            x[:, 1 + 4 * (i - 1) : 1 + 4 * i, :, :, :],
            feat_cache=feat_cache._enc_feat_map,
            feat_idx=feat_cache._enc_conv_idx,
        )
        out = jnp.concatenate([out, out_], axis=1)
    enc = self.quant_conv(out)
    mu, logvar = enc[:, :, :, :, : self.z_dim], enc[:, :, :, :, self.z_dim :]
    enc = jnp.concatenate([mu, logvar], axis=-1)
    feat_cache.clear_cache()
    return enc

  def encode(
      self,
      x: jax.Array,
      feat_cache: AutoencoderKLWanCache,
      return_dict: bool = True,
  ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:  # pylint:disable=g-one-element-tuple
    """Encode video into a latent Gaussian distribution.

    Args:
      x: Input video tensor. Accepted shapes: - [B, T, H, W, 3] (channels-last,
        preferred) - [B, 3, T, H, W] (channels-first, auto-transposed)
      feat_cache: Cache object for chunked temporal processing.
      return_dict: If True, return AutoencoderKLOutput; else return tuple.

    Returns:
      AutoencoderKLOutput containing a DiagonalGaussianDistribution,
      or a tuple with the distribution.
    """
    h = self._encode(x, feat_cache)
    posterior = DiagonalGaussianDistribution(h)
    if not return_dict:
      return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)

  def _decode(
      self,
      z: jax.Array,
      feat_cache: AutoencoderKLWanCache,
      return_dict: bool = True,
  ) -> Union[DecoderOutput, jax.Array]:
    """Decode latents into video frames using chunked temporal processing."""
    feat_cache.clear_cache()
    iter_ = z.shape[1]  # Time dimension
    x = self.post_quant_conv(z)
    for i in range(iter_):
      feat_cache._conv_idx = [0]
      if i == 0:
        out = self.decoder(
            x[:, i : i + 1, :, :, :],
            feat_cache=feat_cache._feat_map,
            feat_idx=feat_cache._conv_idx,
        )
      else:
        out_ = self.decoder(
            x[:, i : i + 1, :, :, :],
            feat_cache=feat_cache._feat_map,
            feat_idx=feat_cache._conv_idx,
        )
        # Frame reordering to correct for internal reshape ordering.
        fm1 = out_[:, 0, :, :, :]
        fm2 = out_[:, 1, :, :, :]
        fm3 = out_[:, 2, :, :, :]
        fm4 = out_[:, 3, :, :, :]
        axis = 0
        if fm1.shape[0] > 1:
          axis = 1
        if len(fm1.shape) == 4:
          fm1 = jnp.expand_dims(fm1, axis=axis)
          fm2 = jnp.expand_dims(fm2, axis=axis)
          fm3 = jnp.expand_dims(fm3, axis=axis)
          fm4 = jnp.expand_dims(fm4, axis=axis)
        out = jnp.concatenate([out, fm1, fm3, fm2, fm4], axis=1)
    out = jnp.clip(out, min=-1.0, max=1.0)
    feat_cache.clear_cache()
    if not return_dict:
      return (out,)
    return DecoderOutput(sample=out)

  def decode(
      self,
      z: jax.Array,
      feat_cache: AutoencoderKLWanCache,
      return_dict: bool = True,
  ) -> Union[DecoderOutput, jax.Array]:
    """Decode latents into video frames.

    Args:
      z: Latent tensor. Accepted shapes: - [B, T', H', W', z_dim]
        (channels-last, preferred) - [B, z_dim, T', H', W'] (channels-first,
        auto-transposed)
      feat_cache: Cache object for chunked temporal processing.
      return_dict: If True, return DecoderOutput; else return tuple.

    Returns:
      DecoderOutput containing decoded video tensor [B, T, H, W, 3],
      or a tuple with the tensor.
    """
    if z.shape[-1] != self.z_dim:
      z = jnp.transpose(z, (0, 2, 3, 4, 1))
      assert (
          z.shape[-1] == self.z_dim
      ), f"Expected input shape (N, D, H, W, {self.z_dim}), got {z.shape}"
    decoded = self._decode(z, feat_cache).sample
    if not return_dict:
      return (decoded,)
    return DecoderOutput(sample=decoded)


# ---------------------------------------------------------------------------
# Weight loading from PyTorch safetensors
# ---------------------------------------------------------------------------


def _rename_key(key: str) -> str:
  """Rename PyTorch key by replacing dotted indices with underscores.

  For example: 'blocks.0.conv' -> 'blocks_0.conv'
  """
  regex = r"\w+[.]\d+"
  pats = re.findall(regex, key)
  for pat in pats:
    key = key.replace(pat, "_".join(pat.split(".")))
  return key


def _torch2jax(torch_tensor) -> jnp.ndarray:
  """Convert a PyTorch tensor to a JAX array.

  Handles bfloat16 by upcasting to float32 for numpy conversion,
  then downcasting back to bfloat16 in JAX.
  """
  # Check dtype using its string representation
  is_bfloat16 = str(torch_tensor.dtype) == "torch.bfloat16"

  if is_bfloat16:
    torch_tensor = torch_tensor.float()

  if torch_tensor.device.type != "cpu":
    torch_tensor = torch_tensor.cpu()

  numpy_value = torch_tensor.numpy()
  cpu_device = jax.local_devices(backend="cpu")[0]

  return jnp.array(
      numpy_value,
      dtype=jnp.bfloat16 if is_bfloat16 else None,
      device=cpu_device,
  )


def _rename_key_and_reshape_tensor(
    pt_tuple_key: Tuple[str, ...],
    pt_tensor: jnp.ndarray,
    random_flax_state_dict: Dict,
) -> Tuple[Tuple[str, ...], jnp.ndarray]:
  """Rename PyTorch weight key to Flax convention and reshape if needed.

  Handles renaming of:
  - Normalization weight/bias -> scale
  - Convolution weight -> kernel (with axis transposition)
  - Linear weight -> kernel (transposed)
  - Attention layers
  """
  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)  # pylint:disable=unused-assignment

  # Rename attention layers.
  if len(pt_tuple_key) > 1:
    for rename_from, rename_to in (
        ("to_out_0", "proj_attn"),
        ("to_k", "key"),
        ("to_v", "value"),
        ("to_q", "query"),
    ):
      if pt_tuple_key[-2] == rename_from:
        weight_name = pt_tuple_key[-1]
        weight_name = "kernel" if weight_name == "weight" else weight_name
        renamed_pt_tuple_key = pt_tuple_key[:-2] + (rename_to, weight_name)
        if renamed_pt_tuple_key in random_flax_state_dict:
          return renamed_pt_tuple_key, pt_tensor.T

  # Norm weight -> scale.
  if (
      any("norm" in str_ for str_ in pt_tuple_key)
      and pt_tuple_key[-1] == "bias"
      and pt_tuple_key[:-1] + ("bias",) not in random_flax_state_dict
      and pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict
  ):
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    return renamed_pt_tuple_key, pt_tensor
  elif (
      pt_tuple_key[-1] in ["weight", "gamma"]
      and pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict
  ):
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    return renamed_pt_tuple_key, pt_tensor

  # Embedding.
  if (
      pt_tuple_key[-1] == "weight"
      and pt_tuple_key[:-1] + ("embedding",) in random_flax_state_dict
  ):
    return pt_tuple_key[:-1] + ("embedding",), pt_tensor

  # 2D conv layer: [O, I, H, W] -> [H, W, I, O].
  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
  if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
    pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
    return renamed_pt_tuple_key, pt_tensor

  # 3D conv layer: [O, I, D, H, W] -> [D, H, W, I, O].
  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
  if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 5:
    pt_tensor = pt_tensor.transpose(2, 3, 4, 1, 0)
    return renamed_pt_tuple_key, pt_tensor

  # Linear layer: weight -> kernel (transposed).
  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
  if pt_tuple_key[-1] == "weight":
    pt_tensor = pt_tensor.T
    return renamed_pt_tuple_key, pt_tensor

  # Old PyTorch layer norm weight (gamma).
  if pt_tuple_key[-1] == "gamma":
    return pt_tuple_key, pt_tensor.flatten()

  # Old PyTorch layer norm bias (beta).
  if pt_tuple_key[-1] == "beta":
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    return renamed_pt_tuple_key, pt_tensor

  return pt_tuple_key, pt_tensor


def _tuple_str_to_int(in_tuple: Tuple) -> Tuple:
  """Convert string elements that look like integers back to ints."""
  out_list = []
  for item in in_tuple:
    try:
      out_list.append(int(item))
    except (ValueError, TypeError):
      out_list.append(item)
  return tuple(out_list)


def load_weights_from_pytorch(
    model: AutoencoderKLWan,
    safetensors_path: str,
) -> Dict:
  """Load PyTorch safetensors weights into a JAX VAE model.

  This function reads a PyTorch safetensors checkpoint, renames keys and
  reshapes tensors to match the Flax/NNX model structure, and returns
  a state dict that can be used with `nnx.update`.

  Args:
    model: An initialized AutoencoderKLWan instance (used for shape reference).
    safetensors_path: Path to the PyTorch .safetensors checkpoint file.

  Returns:
    A nested dictionary (Flax state dict) ready for use with nnx.update.

  Example:
    ```python
    model = AutoencoderKLWan(rngs=nnx.Rngs(0))
    cache = AutoencoderKLWanCache(model)
    state_dict = load_weights_from_pytorch(model,
    "vae/diffusion_pytorch_model.safetensors")
    nnx.update(model, nnx.State(state_dict))
    ```
  """

  # Get reference shapes from the initialized model.
  _, state = nnx.split(model)
  eval_shapes = flatten_dict(state.to_pure_dict())

  # Load PyTorch tensors.
  tensors = {}
  with safetensors.safe_open(safetensors_path, framework="pt") as f:
    for k in f.keys():
      tensors[k] = _torch2jax(f.get_tensor(k))

  cpu = jax.local_devices(backend="cpu")[0]
  flax_state_dict = {}

  for pt_key, tensor in tensors.items():
    renamed_pt_key = _rename_key(pt_key)

    # VAE-specific key renaming (order matters).
    renamed_pt_key = renamed_pt_key.replace("up_blocks_", "up_blocks.")
    renamed_pt_key = renamed_pt_key.replace("mid_block_", "mid_block.")
    renamed_pt_key = renamed_pt_key.replace("down_blocks_", "down_blocks.")

    renamed_pt_key = renamed_pt_key.replace("conv_in.bias", "conv_in.conv.bias")
    renamed_pt_key = renamed_pt_key.replace(
        "conv_in.weight", "conv_in.conv.weight"
    )
    renamed_pt_key = renamed_pt_key.replace(
        "conv_out.bias", "conv_out.conv.bias"
    )
    renamed_pt_key = renamed_pt_key.replace(
        "conv_out.weight", "conv_out.conv.weight"
    )
    renamed_pt_key = renamed_pt_key.replace("attentions_", "attentions.")
    renamed_pt_key = renamed_pt_key.replace("resnets_", "resnets.")
    renamed_pt_key = renamed_pt_key.replace("upsamplers_", "upsamplers.")
    renamed_pt_key = renamed_pt_key.replace("resample_", "resample.")
    renamed_pt_key = renamed_pt_key.replace("conv1.bias", "conv1.conv.bias")
    renamed_pt_key = renamed_pt_key.replace("conv1.weight", "conv1.conv.weight")
    renamed_pt_key = renamed_pt_key.replace("conv2.bias", "conv2.conv.bias")
    renamed_pt_key = renamed_pt_key.replace("conv2.weight", "conv2.conv.weight")
    renamed_pt_key = renamed_pt_key.replace(
        "time_conv.bias", "time_conv.conv.bias"
    )
    renamed_pt_key = renamed_pt_key.replace(
        "time_conv.weight", "time_conv.conv.weight"
    )
    renamed_pt_key = renamed_pt_key.replace("quant_conv", "quant_conv.conv")
    renamed_pt_key = renamed_pt_key.replace(
        "conv_shortcut", "conv_shortcut.conv"
    )
    if "decoder" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace(
          "resample.1.bias", "resample.layers.1.bias"
      )
      renamed_pt_key = renamed_pt_key.replace(
          "resample.1.weight", "resample.layers.1.weight"
      )
    if "encoder" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("resample.1", "resample.conv")

    pt_tuple_key = tuple(renamed_pt_key.split("."))
    flax_key, flax_tensor = _rename_key_and_reshape_tensor(
        pt_tuple_key, tensor, eval_shapes
    )
    flax_key = _tuple_str_to_int(flax_key)
    flax_state_dict[flax_key] = jax.device_put(
        jnp.asarray(flax_tensor), device=cpu
    )

  flax_state_dict = unflatten_dict(flax_state_dict)
  del tensors
  jax.clear_caches()

  return flax_state_dict
