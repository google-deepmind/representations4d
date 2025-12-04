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

"""MooG modules."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
from typing import Any, Optional, TypedDict

import einops
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
from kauldron import kontext
from kauldron.modules import misc
from kauldron.typing import Bool, Float, Int, Integer, PRNGKey, Shape, typechecked  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import train_property

from moog import attention


@typechecked
def _spatial_broadcast(
    x: Float['*b c'], resolution: tuple[int, int]
) -> Float['*b h w c']:
  """Broadcast flat inputs to a 2D grid of a given resolution."""
  num_batch_axes = len(x.shape) - 1
  x = x[..., jnp.newaxis, jnp.newaxis, :]
  return jnp.tile(x, [1] * num_batch_axes + [resolution[0], resolution[1], 1])


@typechecked
def _random_roll_resize_subsampling(
    x: Float['*B H W C'],
    rng: PRNGKey,
    subsample_factor: int,
    eval_subsample_factor: int,
    train: bool,
) -> tuple[Float['*B h w C'], Int['*B 2']]:
  """Spatially subsamples an image tensor using a randomized roll-reshape.

  Args:
    x: Input array.
    rng: PRNGKey.
    subsample_factor: Integer defining the factor by which we subsample.
    eval_subsample_factor: Same as subsample_factor, but for inference.
    train: Boolean indicating whether we are in training mode.

  Returns:
    Tuple of subsampled tensor and offsets used for roll.
  """
  if not train:
    subsample_factor = eval_subsample_factor

  if subsample_factor < 0:
    raise ValueError(
        f'subsample_factor must be >= 0, but is {subsample_factor}'
    )

  if subsample_factor == 0 or subsample_factor == 1:
    # No subsampling.
    batch_axes = x.shape[:-3]
    offsets = jnp.zeros(batch_axes + (2,), dtype=jnp.int32)
    return x, offsets

  *batch_shape, h, w, num_features = x.shape
  new_h, new_w = h // subsample_factor, w // subsample_factor
  assert (
      h % subsample_factor == 0 and w % subsample_factor == 0
  ), 'Image dimensions must be evenly divisible by subsample_factor."'
  new_shape = tuple(batch_shape) + (new_h, new_w, num_features)

  # Sample random offsets by which we shift ("roll") each image.
  offsets = jax.random.randint(
      key=rng,
      shape=tuple(batch_shape) + (2,),
      dtype=jnp.int32,
      minval=0,
      maxval=subsample_factor,
  )
  if not train:
    offsets = jnp.zeros_like(offsets)
  # TODO(tkipf): Consider sampling between [-1, subsample_factor-1] if we want
  # to avoid roll-over after resizing, since there seems to be an offset of 1
  # during resize w/ jnp.image.resize. This doesn't matter for model training
  # using L2 loss and ARI metrics, but other losses and metrics (e.g. spatial
  # object size regularizers, mIoU metric) could be affected.

  # Apply random "roll" (x/y offset) across all batch elements.
  x = _batch_roll_resize(x, offsets, new_shape)

  return x, offsets


@typechecked
def _batch_roll_resize(
    x: Float['*B H W C'] | Integer['*B H W C'],
    offsets: Int['*B 2'] | None,
    new_shape: Shape,
) -> Float['*B h w C'] | Integer['*B h w C']:
  """Batched roll followed by strided downsampling (resize)."""

  # x.shape = (..., h, w, n_features)
  # offsets.shape = (..., 2)
  num_batch_axes = len(x.shape[:-3])

  # vmap jnp.roll (for image dims) across all batch axes.
  batch_roll = functools.partial(jnp.roll, axis=(-3, -2))
  for i in range(num_batch_axes):
    batch_roll = jax.vmap(
        batch_roll, in_axes=(0, 0), out_axes=0, axis_name=str(i)
    )

  if offsets is not None:
    assert num_batch_axes == len(
        offsets.shape[:-1]
    ), 'Inputs and offsets need to have same number of batch axes.'
    x = batch_roll(x, offsets)

  x = jax.image.resize(x, new_shape, method='nearest')
  return x


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


class SRTEncoder(nn.Module):
  """SRT-style hybrid CNN+Transformer encoder."""

  backbone: nn.Module
  transformer: attention.ImprovedTransformer | None
  pos_embedding: nn.Module

  @typechecked
  @nn.compact
  def __call__(self, image: Float['*b H W c']) -> Float['*b n d']:
    x = self.backbone(image)
    x = self.pos_embedding(x)
    x = einops.rearrange(x, '... h w d -> ... (h w) d')
    if self.transformer is not None:
      x = self.transformer(x)
    return x


class SRTDecoder(nn.Module):
  """SRT-style spatial transformer decoder."""

  transformer: attention.ImprovedTransformer
  pos_embedding: nn.Module
  subsample_factor: int = 0
  eval_subsample_factor: int = 0
  is_training = train_property.train_property()

  @typechecked
  @nn.compact
  def __call__(
      self,
      inputs: Float['*b n d'],
      targets: Float['*b W H C'],
      mask: Optional[Float['*b W H 1']] = None,
  ) -> tuple[Float['*b h w C'], Float['*b h w C'], Optional[Float['*b h w 1']]]:
    # Create full-resolution spatial coordinate grid.
    resolution = (targets.shape[-3], targets.shape[-2])
    coords = _create_gradient_grid(resolution, (-1.0, 1.0))

    # Repeat across batch.
    batch_axes = inputs.shape[:-2]
    coords = jnp.broadcast_to(coords, batch_axes + coords.shape)

    # Subsample spatial coordinate grid.
    coords, offsets = _random_roll_resize_subsampling(
        coords,
        rng=self.make_rng('default'),
        subsample_factor=self.subsample_factor,
        eval_subsample_factor=self.eval_subsample_factor,
        train=self.is_training,
    )
    new_h, new_w = coords.shape[-3], coords.shape[-2]
    offsets = misc.Identity(name='roll_resize_subsampling_offsets')(offsets)

    # Broadcast and pos emb.
    x = _spatial_broadcast(jnp.zeros_like(inputs[..., 0, :]), (new_h, new_w))
    x = self.pos_embedding(inputs=x, coords=coords)

    # Transformer decoder on flattened (new_h, new_w) token axis.
    x = self.transformer(x, inputs_kv=inputs, num_token_axes=2)

    # Final dense layer.
    recon = nn.Dense(
        features=targets.shape[-1],
        name='recon',
    )(x)

    # Subsample the targets.
    subsampled_targets = _batch_roll_resize(targets, offsets, recon.shape)
    subsampled_mask = None
    if mask is not None:
      subsampled_mask = _batch_roll_resize(mask, offsets, recon.shape)
    return recon, subsampled_targets, subsampled_mask


class MoogResults(TypedDict):
  """Output shapes of MooG."""

  video_predicted: Float['*b T h w c']
  subsampled_targets_predicted: Float['*b T h w c']
  states_corrected: Float['*b T n d']
  states_predicted: Float['*b T n d']
  states_concat: Float['*b T N d']
  state: Float['*b n d']


class Moog(nn.Module):
  """Deterministic MooG model with separate predictor corrector calls."""

  encoder: SRTEncoder
  initializer: nn.Module
  corrector: nn.Module
  predictor: nn.Module
  decoder: SRTDecoder
  correction_layer_norm: Optional[nn.Module] = None
  stop_gradient_prob: Optional[float] = None
  use_loss_gradient: Optional[bool] = True
  state_grad_projector: Optional[nn.Module] = None

  is_training = train_property.train_property()

  video: kontext.Key = 'batch.video'

  # NOTE: For chunked inference, we pass additional fields (e.g. "state") as
  # input to the model. As these are only available during chunked inference,
  # this is not supported by the kontext.Key system and we need to feed the full
  # batch instead as a workaround.
  batch: kontext.Key = 'batch'  # Always feed full batch as a workaround.
  # state: Optional[kontext.Key] = None

  def loss_fn_for_inner_gradient(self, s, t):
    # decode
    recon, target, _ = self.decoder(inputs=s, targets=t)
    # get loss - sum over batch dimension, average over space and channels
    loss = jnp.sum(jnp.mean(jnp.square(recon - target), axis=[-3, -2, -1]))
    return loss, (recon, target)

  def frame_step(self, encoded_inputs, frame, state):
    # First predict
    state_predicted = state + self.predictor(state)

    (_, aux), grads = nn.value_and_grad(
        self.loss_fn_for_inner_gradient,
        self,  # always takes module as first arg
        state_predicted,
        frame,
        has_aux=True,
    )  # other args
    recon, target = aux
    state_gradient = jax.lax.stop_gradient(grads[0])

    if self.use_loss_gradient:
      # Then correct (if we have observations)
      state_grad_projection = self.state_grad_projector(
          jnp.concatenate([state_predicted, state_gradient], axis=-1)
      )
    else:
      state_grad_projection = state_predicted

    state_corrected = state_predicted + self.corrector(
        state_grad_projection, inputs_kv=encoded_inputs
    )

    if self.correction_layer_norm is not None:
      state_corrected = self.correction_layer_norm(state_corrected)

    return recon, target, state_predicted, state_corrected

  @typechecked
  @nn.compact
  def __call__(
      self,
      video: Float['*b T H W c'],
      batch: Any,  # Always feed full batch as a workaround for state access.
  ) -> MoogResults:

    state = batch.get('state')
    x = self.encoder(video)

    if state is None:
      # Input encoded features of first time step into initializer.
      state = self.initializer(x[..., 0, :, :], batch_shape=x.shape[:-3])
      if self.correction_layer_norm is not None:
        state = self.correction_layer_norm(state)

    # Scan over time axis.
    seq_len = x.shape[-3]
    states_corrected = []
    states_predicted = []
    recons = []
    targets = []

    for step in range(seq_len):

      encoded_inputs = x[..., step, :, :]
      frame = video[..., step, :, :, :]

      recon, target, state_predicted, state_corrected = self.frame_step(
          encoded_inputs, frame, state
      )

      recons.append(recon)
      targets.append(target)
      states_predicted.append(state_predicted)
      states_corrected.append(state_corrected)

      state = state_corrected
      if self.stop_gradient_prob is not None:
        stop_grad_state = jax.lax.stop_gradient(state)
        state = jnp.where(
            jax.random.bernoulli(
                self.make_rng('default'),
                self.stop_gradient_prob,
                state.shape[:-2] + (1,) * len(state.shape[-2:]),
            ),
            stop_grad_state,
            state,
        )

    states_corrected = jnp.stack(states_corrected, axis=-3)
    states_predicted = jnp.stack(states_predicted, axis=-3)
    recon_predicted = jnp.stack(recons, axis=-4)
    subsampled_targets_predicted = jnp.stack(targets, axis=-4)

    return {
        'video_predicted': recon_predicted,
        'subsampled_targets_predicted': subsampled_targets_predicted,
        'states_predicted': states_predicted,
        'states_corrected': states_corrected,
        'states_concat': jnp.concatenate(
            [states_predicted, states_corrected], axis=-2
        ),
        'state': state,
    }


class TransformerReadout(nn.Module):
  """Transformer readout module."""

  transformer: attention.ImprovedTransformer
  num_token_axes: int = 1

  @typechecked
  @nn.compact
  def __call__(
      self,
      inputs: Float['*b n d'],
      queries: Float['... e'],  # [*b *k e]
  ) -> Float['... f']:  # [*b *k f]
    # Transformer decoder.
    x = self.transformer(
        queries, inputs_kv=inputs, num_token_axes=self.num_token_axes
    )
    return x


class DenseReadoutResults(TypedDict):
  """Output shapes of DenseReadout."""

  values: Float['*b T h_sub w_sub C']
  subsampled_targets: Float['*b T h_sub w_sub C']
  subsampled_mask: Optional[Float['*b T h_sub w_sub C']] = None


class DenseReadout(nn.Module):
  """A dense readout module."""

  backbone: nn.Module
  finetune: bool = False  # Whether to finetune the pre-readout model.

  @typechecked
  @nn.compact
  def __call__(
      self,
      features: Float['*b T n d'],
      target: Float['*b T h w C'],
      mask: Optional[Float['*b T h w C']] = None,
  ) -> DenseReadoutResults:
    if not self.finetune:
      features = jax.lax.stop_gradient(features)

    length = features.shape[-3]
    recons = []
    subsampled_targets = []
    subsampled_masks = []
    for step in range(length):
      recon, sub_target, sub_mask = self.backbone(
          features[..., step, :, :],
          target[..., step, :, :, :],
          mask[..., step, :, :, :] if mask is not None else None,
      )
      recons.append(recon)
      subsampled_targets.append(sub_target)
      subsampled_masks.append(sub_mask)

    recons = jnp.stack(recons, axis=-4)
    subsampled_targets = jnp.stack(subsampled_targets, axis=-4)
    if mask is not None:
      subsampled_masks = jnp.stack(subsampled_masks, axis=-4)
    else:
      subsampled_masks = None

    return {
        'values': recons,
        'subsampled_targets': subsampled_targets,
        'subsampled_mask': subsampled_masks,
    }


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


class ConditionalAutoregressiveReadoutResult(TypedDict):
  """Output shapes of ConditionalAutoregressiveReadout."""

  values: Optional[Float['*b T m e']]
  logits: Optional[Float['*b T m f']]
  visible: Optional[Float['*b T m 1']]
  states: Float['*b T m l']
  states_corrected: Float['*b T m l']
  state: Float['*b m l']  # Final state (i.e. of last time step).


class ConditionalAutoregressiveReadout(nn.Module):
  """A stateful autoregresive readout module with conditioning inputs."""

  backbone: nn.Module
  predictor: nn.Module
  initializer: nn.Module
  output_value_size: int  # These values go through the activation function.
  output_value_activation: Optional[Callable[[Float['*b']], Float['*b']]] = None
  output_logit_size: int = 0  # Logits to be used e.g. for presence probability.
  state_stop_gradient_prob: Optional[float] = None
  state_layer_norm: Optional[nn.Module] = None
  readout_state_key: str = 'readout_state'
  finetune: bool = False  # Whether to finetune the pre-readout model.
  use_certainty: bool = True
  certainty_threshold: float = 0.5

  @typechecked
  @nn.compact
  def __call__(
      self,
      features: Float['*b T n d'],
      conditioning: Float['*b m k'],
      batch: Any,  # Always feed full batch as a workaround for state access.
      # state: Optional[Float['*b m l']] = None,
      conditioning_mask: Optional[Bool['*b T m']] = None,
  ) -> ConditionalAutoregressiveReadoutResult:
    """Runs the readout forward pass.

    Args:
      features: The features to read out from, [*b T n d].
      conditioning: The conditioning to feed to the readout, [*b m k].
      batch: The full batch, used to access the state.
      conditioning_mask: A mask denoting the frame the conditioning is for. For
        example, when point tracking, the query point might be from any frame,
        so we need to feed the conditioning to the readout for a query point on
        its frame, [*b T m].

    Returns:
      A ConditionalAutoregressiveReadoutResult with the following fields:
        values: Float['*b T m e']
        logits: Float['*b T m f']
        visible: Float['*b T m 1']
        states: Float['*b T m l']
        states_corrected: Float['*b T m l']
        state: Float['*b m l'], final state (i.e. of last time step).
    """
    if not self.finetune:
      features = jax.lax.stop_gradient(features)

    if batch is not None:
      state = batch.get(self.readout_state_key)
    else:
      state = None

    conditioning = self.initializer(conditioning)
    if state is None:
      state = conditioning

    num_steps = features.shape[-3]

    states = []
    states_corrected = []
    for step in range(num_steps):
      # Construct the correct conditioning for the current step.
      # If the conditioning is for current timestep, use it, otherwise use the
      # previous state.
      if conditioning_mask is not None:
        state = jnp.where(
            conditioning_mask[..., step, :][..., jnp.newaxis],
            conditioning,
            state,
        )
      state_corrected = state + self.backbone(features[..., step, :, :], state)

      state = state_corrected + self.predictor(state_corrected)

      if self.state_layer_norm is not None:
        state = self.state_layer_norm(state)

      if self.state_stop_gradient_prob is not None:
        stop_grad_state = jax.lax.stop_gradient(state)
        state = jnp.where(
            jax.random.bernoulli(
                self.make_rng('default'),
                self.state_stop_gradient_prob,
                state.shape[:-2] + (1,) * len(state.shape[-2:]),
            ),
            stop_grad_state,
            state,
        )

      states_corrected.append(state_corrected)
      states.append(state)

    states_corrected = jnp.stack(states_corrected, axis=-3)
    states = jnp.stack(states, axis=-3)

    if self.output_value_size:
      output_values = MLP(
          hidden_size=states_corrected.shape[-1],
          output_size=self.output_value_size,
          name='output_values',
      )(states_corrected)
      output_values = self.output_value_activation(output_values)
    else:
      output_values = None

    if self.output_logit_size:
      output_logits = MLP(
          hidden_size=states_corrected.shape[-1],
          output_size=self.output_logit_size,
          name='output_logits',
      )(states_corrected)
    else:
      output_logits = None

    visibility = jax.nn.sigmoid(output_logits[..., :1])
    if self.use_certainty:
      certainty = jax.nn.sigmoid(output_logits[..., 1:])
      visible = (visibility * certainty > self.certainty_threshold).astype(
          jnp.float32
      )
    else:
      visible = visibility

    return {
        'values': output_values,
        'logits': output_logits,
        'visible': visible,
        'states_corrected': states_corrected,
        'states': states,
        'state': states[..., -1, :, :],
    }


# TODO(svansteenkiste): delete and import from kd.contrib.nn instead
@dataclasses.dataclass(kw_only=True)
class ReadoutWrapper(nn.Module, kw_only=True):  # pytype: disable=invalid-function-definition
  """Wrapper for adding readout heads to a model.

  This model adds arbitrary readout heads to a given model. It is intended to be
  used with load_model_from_xid so that the readout config can be run with
  arbitrary models.

  Attributes:
    model: The model onto which the readout head should be attached.
    model_inputs: Optional dictionary of model input key paths. Used to fill the
      keyword arguments for the `model.__call__` function from the context. For
      example, if the model expects two inputs `images` and `masks` then
      `model_inputs={"images": "batch.images", "masks": "batch.masks"}` would
      ensure that `batch["images"]` and `batch["masks"]` are fed as inputs. If
      set to None then the model inputs are filled using the `kontext.Key`
      annotations of the model.
    readout_heads: Dict of arbitrary nn.Modules to be used as trainable readout
      heads.
    readout_inputs: Dict of optional dictionaries of input key paths for the
      readout head. Similar to `model_inputs`, but in addition to "batch" also
      has access to the model predictions under "preds.model" and the model
      intermediates in "interms.model". If set to None then the readout inputs
      are filled using the `kontext.Key` annotations of the readout head.
    finetune: Whether to finetune the model. Defaults to False in which case
      there is a stop_gradient between the model and the readout head.
  """

  model: nn.Module
  model_inputs: Optional[dict[str, str | None]] = None

  readout_heads: dict[str, nn.Module]
  readout_inputs: dict[str, dict[str, str | None] | None]

  finetune: bool = False

  @typechecked
  @nn.compact
  def __call__(self, **kwargs) -> Any:
    model_inputs = self._get_model_inputs(kwargs)
    model_preds = self.model(**model_inputs)

    if not isinstance(model_preds, dict):
      raise ValueError(
          'The provided model must return a dict of predictions to be'
          ' compatible with ReadoutWrapper.'
      )

    if 'readouts' in model_preds:
      raise ValueError(
          'Key `readouts` found in model predictions. This key is reserved for'
          ' readout heads when using ReadoutWrapper.'
      )

    readouts = {}
    for head in self.readout_heads:
      readout_inputs = self._get_readout_inputs(kwargs, model_preds, head)
      if not self.finetune:
        readout_inputs = lax.stop_gradient(readout_inputs)
      readout = self.readout_heads[head](**readout_inputs)
      readouts[head] = readout

    return model_preds | {'readouts': readouts}

  @staticmethod
  def _is_external_keypath(keypath: str | None) -> bool:
    if keypath is None:
      return False  # Treat optional keypaths as internal
    return kontext.Path.from_str(keypath)[0] not in ['interms', 'preds']

  def __kontext_keys__(self) -> dict[str, str | None]:
    # Forward all the model keys.
    model_keypaths = {f'model_{k}': v for k, v in self._model_keypaths.items()}

    # As well as the readout keys that refer to external context.
    readout_batch_keypaths = {}
    for head in self.readout_heads:
      readout_batch_keypaths.update({
          f'readout_{head}_{k}': v
          for k, v in self._get_readout_keypaths(head).items()
          if self._is_external_keypath(v)
      })
    return model_keypaths | readout_batch_keypaths

  @property
  def _model_keypaths(self) -> dict[str, str | None]:
    if self.model_inputs is None:
      return kontext.get_keypaths(self.model)
    else:
      return self.model_inputs

  def _get_readout_keypaths(self, head: str) -> dict[str, str | None]:
    if self.readout_inputs[head] is None:
      return kontext.get_keypaths(self.readout_heads[head])
    else:
      return self.readout_inputs[head]

  def _get_model_inputs(self, kwargs):
    # separate out the model kwargs
    return {
        k.removeprefix('model_'): v
        for k, v in kwargs.items()
        if k.startswith('model_')
    }

  def _get_readout_inputs(self, kwargs, preds, head):
    # separate out the readout batch keys from kwargs
    readout_batch_inputs = {
        k.removeprefix(f'readout_{head}_'): v
        for k, v in kwargs.items()
        if k.startswith(f'readout_{head}_')
    }
    # get other (non-batch) keys
    readout_other_keypaths = {
        k: v
        for k, v in self._get_readout_keypaths(head).items()
        if not self._is_external_keypath(v)
    }
    # resolve (non-batch) readout keys
    interms = self.scope.root.variables()['intermediates']
    ctx = {'preds': preds, 'interms': interms}

    readout_other_inputs = kontext.resolve_from_keypaths(
        ctx, readout_other_keypaths
    )
    return readout_batch_inputs | readout_other_inputs
