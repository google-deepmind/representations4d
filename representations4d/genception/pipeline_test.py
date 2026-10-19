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

"""Tests for the GenCeption pipeline's VAE latent layout."""

import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from representations4d.genception import pipeline
from representations4d.genception import utils
from representations4d.genception import vae


class DecodeVideoLayoutTest(unittest.TestCase):

  def test_decoder_always_receives_channels_last(self):
    for batch, width in [(1, 3), (1, 16), (2, 16)]:
      with self.subTest(batch=batch, width=width):
        shape = (batch, 16, 2, 3, width)
        latents = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
        decoder = mock.Mock()
        decoder.decode.return_value = vae.DecoderOutput(
            sample=jnp.zeros((batch, 1, 1, 1, 3))
        )
        cache = mock.Mock()
        pipe = pipeline.GenCeptionPipeline(
            nnx.Linear(1, 1, rngs=nnx.Rngs(0)),
            decoder,
            cache,
            {'weights_dtype': jnp.float32},
        )
        pipe.decode_video(latents)
        actual, actual_cache = decoder.decode.call_args.args
        np.testing.assert_array_equal(
            actual, jnp.transpose(latents, (0, 2, 3, 4, 1))
        )
        self.assertIs(actual_cache, cache)

  def test_real_vae_matches_explicit_channels_last_decode(self):
    # Small random weights exercise the actual VAE, without a transformer
    # forward pass or downloading pretrained model weights.
    decoder = vae.AutoencoderKLWan(
        rngs=nnx.Rngs(0), base_dim=2, z_dim=16, num_res_blocks=1
    )
    cache = vae.AutoencoderKLWanCache(decoder)
    pipe = pipeline.GenCeptionPipeline(
        nnx.Linear(1, 1, rngs=nnx.Rngs(1)),
        decoder,
        cache,
        {'weights_dtype': jnp.float32},
    )
    for batch, width in [(1, 3), (1, 16), (2, 3), (2, 16)]:
      with self.subTest(batch=batch, width=width):
        latents = jax.random.normal(
            jax.random.key(width), (batch, 16, 2, 1, width)
        )
        expected = utils.postprocess_video(
            decoder.decode(
                jnp.transpose(latents, (0, 2, 3, 4, 1)), cache
            ).sample
        )
        actual = pipe.decode_video(latents)
        self.assertEqual(actual.shape, (batch, 5, 8, width * 8, 3))
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
  unittest.main()
