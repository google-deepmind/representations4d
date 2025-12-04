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

"""Convolutional networks modules."""

from __future__ import annotations

from flax import linen as nn
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class ConvNet(nn.Module):
  """A simple ConvNet."""

  features: tuple[int, ...]
  kernel_sizes: tuple[tuple[int, ...], ...]
  strides: tuple[tuple[int, ...], ...]
  transposed_conv: bool = False
  normalize_output: bool = False

  @typechecked
  @nn.compact
  def __call__(self, images: Float['*B H W C']) -> Float['*B h w c']:
    transpose = self.transposed_conv
    x = images
    for n, (f, k, s) in enumerate(
        zip(self.features, self.kernel_sizes, self.strides)
    ):
      x = nn.ConvTranspose(f, k, s)(x) if transpose else nn.Conv(f, k, s)(x)
      if n < len(self.features) - 1:
        x = nn.gelu(x)
    if self.normalize_output:
      x = nn.LayerNorm(use_bias=False, use_scale=True, dtype=x.dtype)(x)
    return x
