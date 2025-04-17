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

"""Main model components."""

from typing import Optional, Self, Sequence
import einops
from flax import linen as nn
import jax.numpy as jnp
from kauldron import kontext
from kauldron.modules import attention
from kauldron.modules import knn_types
from kauldron.modules import transformers
from kauldron.modules import vit as kd_vit
from kauldron.typing import Dim, Float, Initializer, Shape, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import typeguard


class GeneralizedTransformer(nn.Module):
  """Generalized ViT model.

  The user must still: a) embed images as tokens, and b) add positional
  encodings

  Optionally, a second set of extra tokens can be fed midway through the model.
  """

  # Submodules
  layers: Sequence[knn_types.TransformerBlock]

  # KD Keys
  tokens: kontext.Key = kontext.REQUIRED

  n_iter: int = 1

  @typechecked
  @nn.compact
  def __call__(
      self,
      tokens: Float['*B N D'],
  ):

    aux = [jnp.reshape(tokens, Shape('*B N D'))]
    latent_state = tokens

    for h in range(self.n_iter):
      if h > 0:
        latent_state = jnp.concatenate([latent_state, tokens], axis=-2)

      # Self attention Layers.
      for layer in self.layers:
        if h == self.n_iter - 1:
          # store intermediate features on last iteration
          aux.append(latent_state)

        latent_state = layer(latent_state)

        # only keep features from last iteration for readouts
        latent_state = jnp.reshape(
            latent_state,
            [latent_state.shape[0], -1, Dim('D')],
        )

    return aux

  @classmethod
  def from_variant_str(cls, variant_str: str, **kwargs) -> Self:
    vit_spec = kd_vit.ViTSpec.from_variant_string(variant_str)
    all_kwargs = vit_spec.kwargs | kwargs
    all_kwargs.pop('patch_size', None)
    all_kwargs.pop('hidden_size', None)
    return cls.from_spec(**all_kwargs)

  @classmethod
  def from_spec(
      cls,
      num_heads: int,
      num_layers: int,
      mlp_size: Optional[int] = None,
      block_type=transformers.PreNormBlock,
      dtype=jnp.float32,
      qk_features: Optional[int] = None,
      v_features: Optional[int] = None,
      attn_kernel_init: Initializer = nn.initializers.lecun_normal(),
      mlp_kernel_init: Initializer = nn.initializers.xavier_uniform(),
      **kwargs,
  ):
    return cls(
        layers=tuple(
            block_type(  # pylint: disable=g-complex-comprehension
                attention_norm=nn.LayerNorm(dtype=dtype),
                mlp_norm=nn.LayerNorm(dtype=dtype),
                attention=attention.ImprovedMultiHeadDotProductAttention(
                    num_heads=num_heads,
                    qk_features=qk_features,
                    v_features=v_features,
                    kernel_init=attn_kernel_init,
                ),
                mlp=transformers.TransformerMLP(
                    hidden_size=mlp_size, kernel_init=mlp_kernel_init
                ),
            )
            for _ in range(num_layers)
        ),
        **kwargs,
    )


@typeguard.typechecked
class Model(nn.Module, kw_only=True):
  """Main model composed of encoder and processor."""

  # Sub-modules
  encoder: nn.Module
  processor: nn.Module

  @nn.compact
  @typechecked
  def __call__(
      self,
      video: Float['*B T H W C'],
  ) -> list[Float['...']]:
    tokens = self.encoder(video)

    check_type(tokens, Float['*B N_in D'])

    # Run global self-attention transformer.
    features = self.processor(tokens)

    return features


class Tokenizer(nn.Module, kw_only=True):
  """Tokenizes the input images and adds positional encodings."""
  # Required sub modules.
  patch_embedding: nn.Module
  posenc: nn.Module

  posenc_axes: tuple[int, ...] = (-4, -3, -2)

  @nn.compact
  @typechecked
  def __call__(
      self,
      images: Float['*B T H W C'],
  ) -> Float['*B N D'] | Float['*B T h w D'] | Float['*B T N D']:

    # Tokenize.
    tokens = self.patch_embedding(images)

    # Add token posenc.
    posenc = self.posenc(tokens.shape, axis=self.posenc_axes)
    tokens += posenc

    # Flatten tokens.
    tokens = einops.rearrange(tokens, '... T h w D -> ... (T h w) D')

    return tokens


class PatchEmbedding(nn.Module):
  """Extracts patches with a single learned linear projection."""

  # Hparams.
  patch_size: Sequence[int]
  num_features: int

  # KD keys.
  images: kontext.Key = kontext.REQUIRED

  @nn.compact
  @typechecked
  def __call__(
      self, images: Float['*B T H W C'] | Float['*B H W C']
  ) -> Float['*B t h w D'] | Float['*B h w D']:
    # Patchify videos.
    return nn.Conv(
        features=self.num_features,
        kernel_size=self.patch_size,
        strides=self.patch_size,
        padding='VALID',
        dtype=images.dtype,
    )(images)
