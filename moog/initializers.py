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

"""Initializers for latent states."""

from __future__ import annotations

from typing import Any, Dict, Optional

from flax import linen as nn
import jax
from kauldron.typing import Float, Shape, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class RandomStateInit(nn.Module):
  """Random, non-learnable state initialization."""

  shape: tuple[int, ...] | None
  random_init_scale: Optional[float] = 0.001
  random_fn_kwargs: Optional[Dict[Any, Any]] = None
  random_fn: str = "normal"  # Default init with unit variance.

  @typechecked
  @nn.compact
  def __call__(
      self, inputs: Optional[Float["*B N D"]], batch_shape: Shape
  ) -> Float["*B n d"]:
    if self.shape is None:
      if inputs is None:
        raise ValueError("Either shape or inputs must be specified.")
      else:
        shape = inputs.shape[-2:]
    else:
      shape = self.shape

    if self.random_fn == "normal":
      random_fn = getattr(jax.random, self.random_fn)
    else:
      raise ValueError("Unknown random_fn: {}.".format(self.random_fn))
    random_fn_kwargs = self.random_fn_kwargs or {}
    state = self.random_init_scale * random_fn(
        key=self.make_rng("default"),
        shape=batch_shape + shape,
        **random_fn_kwargs,
    )
    return state
