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

"""Position embeddings for video models."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp
from kauldron.typing import Axes, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@typechecked
def _convert_to_fourier_features(
    inputs: Float['... D'], basis_degree: int
) -> Float['... d']:
  """Convert inputs to Fourier features, e.g. for positional encoding."""

  # inputs.shape = (..., n_dims).
  # inputs should be in range [-pi, pi] or [0, 2pi].
  n_dims = inputs.shape[-1]

  # Generate frequency basis.
  freq_basis = jnp.concatenate(  # shape = (n_dims, n_dims * basis_degree)
      [2**i * jnp.eye(n_dims) for i in range(basis_degree)], 1
  )

  # x.shape = (..., n_dims * basis_degree)
  x = inputs @ freq_basis  # Project inputs onto frequency basis.

  # Obtain Fourier features as [sin(x), cos(x)] = [sin(x), sin(x + 0.5 * pi)].
  return jnp.sin(jnp.concatenate([x, x + 0.5 * jnp.pi], axis=-1))


@typechecked
def _create_gradient_grid(
    samples_per_dim: tuple[int, ...],
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> Float['...']:
  """Creates a tensor with equidistant entries from -1 to +1 in each dim.

  Args:
    samples_per_dim: Number of points to have along each dimension.
    value_range: In each dimension, points will go from range[0] to range[1]

  Returns:
    A tensor of shape [samples_per_dim] + [len(samples_per_dim)].
  """
  s = [jnp.linspace(value_range[0], value_range[1], n) for n in samples_per_dim]
  return jnp.stack(jnp.meshgrid(*s, sparse=False, indexing='ij'), axis=-1)


class FourierEmbedding(nn.Module):
  """Apply Fourier position embedding to a grid of coordinates.

  Attr:
    num_fourier_bases: The number of Fourier bases to use. The embedding
      dimensionality is 2 x len(axes) x num_fourier_bases.
    axes: Axes for which to compute position embeddings (excl. feature axis).
    update_type: Concat or project_add.

  Return:
    Array with same shape as input up to final axis.
  """

  num_fourier_bases: int
  update_type: str
  axes: Axes = (-2,)

  @typechecked
  @nn.compact
  def __call__(self, inputs: Float['... d']) -> Float['...']:
    if max(self.axes) >= 0:
      raise ValueError(f'Axes must be negative. Provided axes: {self.axes}.')
    if -1 in self.axes:
      raise ValueError(
          f'Do not include feature axis (-1). Provided axes: {self.axes}.'
      )

    emb_shape = tuple([inputs.shape[axis] for axis in self.axes])

    # NeRF-style Fourier/sinusoidal position encoding.
    coords = _create_gradient_grid(emb_shape, value_range=(-1.0, 1.0))
    pos_embedding = _convert_to_fourier_features(
        coords * jnp.pi, basis_degree=self.num_fourier_bases
    )  # TODO(tkipf): Consider removing pi scaling if we see border artifacts.

    # Re-add any removed axes (excl. leading axes, these are broadcasted).
    all_axes = list(range(min(self.axes), -1))  # Excl. leading & feature axes.
    axes_to_add = tuple(
        [axis - min(self.axes) for axis in all_axes if axis not in self.axes]
    )
    pos_embedding = jnp.expand_dims(pos_embedding, axis=axes_to_add)

    # Apply position encoding to inputs.
    if self.update_type == 'project_add':
      n_features = inputs.shape[-1]
      x = inputs + nn.Dense(n_features, name='dense_pe')(pos_embedding)
    elif self.update_type == 'concat':
      # Repeat the position embedding along the first (batch) dimension.
      pos_embedding = jnp.broadcast_to(
          pos_embedding, shape=inputs.shape[:-1] + pos_embedding.shape[-1:]
      )
      # Concatenate along the channel dimension.
      x = jnp.concatenate((inputs, pos_embedding), axis=-1)
    else:
      raise ValueError('Invalid update type provided.')

    return x


class SampleFourierEmbedding(nn.Module):
  """Fourier position embedding to sampled coordinates in [-1, 1].

  Attr:
    num_fourier_bases: The number of Fourier bases to use. The embedding
      dimensionality is 2 x number of position dimensions x num_fourier_bases.
    update_type: Concat or project_add.

  Return:
    Array with same shape as input up to final axis.
  """

  num_fourier_bases: int
  update_type: str

  @typechecked
  @nn.compact
  def __call__(
      self, inputs: Float['... d'], coords: Float['... D'] | None = None
  ) -> Float['...']:
    # NOTE: `coords` is assumed to be tensor of samples with values [-1, +1]
    # for each input dimension, corresponding to individual coordinate samples.
    # It has shape [..., n_dims]. `inputs` has shape [..., n_features].

    # Use inputs as coords if no coords are provided. Useful e.g. for bboxes.
    if coords is None:
      coords = inputs

    if inputs.shape[:-1] != coords.shape[:-1]:
      raise ValueError(
          'Inputs and coords need to have matching shape (up to final dim).'
          f' Provided inputs of shape {inputs.shape}, coords of shape'
          f' {coords.shape}'
      )

    # NeRF-style Fourier/sinusoidal position encoding.
    pos_embedding = _convert_to_fourier_features(
        coords * jnp.pi, basis_degree=self.num_fourier_bases
    )  # TODO(tkipf): Consider removing pi scaling if we see border artifacts.

    # Apply position encoding to inputs.
    if self.update_type == 'project_add':
      n_features = inputs.shape[-1]
      x = inputs + nn.Dense(n_features, name='dense_pe')(pos_embedding)
    elif self.update_type == 'concat':
      # Repeat the position embedding along the first (batch) dimension.
      pos_embedding = jnp.broadcast_to(
          pos_embedding, shape=inputs.shape[:-1] + pos_embedding.shape[-1:]
      )
      # Concatenate along the channel dimension.
      x = jnp.concatenate((inputs, pos_embedding), axis=-1)
    elif self.update_type == 'replace':
      x = jnp.broadcast_to(
          pos_embedding, shape=inputs.shape[:-1] + pos_embedding.shape[-1:]
      )
    else:
      raise ValueError('Invalid update type provided.')

    return x
