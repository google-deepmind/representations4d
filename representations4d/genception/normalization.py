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

"""Layer normalization utilities for the GenCeption model.

Provides FP32LayerNorm which casts to float32 for numerical stability
during normalization, then casts back to the original dtype.
"""

from flax import nnx
import jax
import jax.numpy as jnp


class FP32LayerNorm(nnx.Module):
  """LayerNorm that casts inputs to float32 for numerical stability.

  This wraps flax.nnx.LayerNorm but ensures that the normalization
  computation is always performed in float32, regardless of the input dtype.
  The output is cast back to the original input dtype.

  Attributes:
    rngs: Random number generators for parameter initialization.
    dim: Number of features (last dimension of the input).
    eps: Epsilon for numerical stability in normalization.
    elementwise_affine: Whether to use learnable scale and bias parameters.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      eps: float,
      elementwise_affine: bool,
  ):
    self.layer_norm = nnx.LayerNorm(
        rngs=rngs,
        num_features=dim,
        epsilon=eps,
        use_bias=elementwise_affine,
        use_scale=elementwise_affine,
        param_dtype=jnp.float32,
        dtype=jnp.float32,
    )

  def __call__(self, inputs: jax.Array) -> jax.Array:
    """Apply layer normalization in float32.

    Args:
      inputs: Input tensor of any dtype.

    Returns:
      Normalized tensor cast back to the original input dtype.
    """
    origin_dtype = inputs.dtype
    return self.layer_norm(inputs.astype(dtype=jnp.float32)).astype(
        dtype=origin_dtype
    )
