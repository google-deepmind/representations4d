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

"""Numerical tests for the video VAE's diagonal Gaussian posterior."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from representations4d.genception import vae


class DiagonalGaussianKLTest(unittest.TestCase):

  def test_video_kl_sums_channels_and_preserves_batch(self):
    shape = (2, 2, 3, 4, 16)
    mean = jnp.broadcast_to(
        jnp.array([1.0, 2.0])[:, None, None, None, None], shape
    )
    posterior = vae.DiagonalGaussianDistribution(
        jnp.concatenate([mean, jnp.zeros(shape)], axis=-1)
    )
    # Each independent coordinate contributes mean**2 / 2 when variance is 1.
    expected = np.array([192.0, 768.0])
    np.testing.assert_allclose(posterior.kl(), expected)

  def test_video_kl_against_other_gaussian(self):
    shape = (2, 2, 3, 4, 3)
    rng = np.random.default_rng(0)
    mean = rng.normal(size=shape).astype(np.float32)
    other_mean = rng.normal(size=shape).astype(np.float32)
    logvar = rng.uniform(-1, 1, size=shape).astype(np.float32)
    other_logvar = rng.uniform(-1, 1, size=shape).astype(np.float32)
    posterior = vae.DiagonalGaussianDistribution(
        jnp.concatenate([mean, logvar], axis=-1)
    )
    other = vae.DiagonalGaussianDistribution(
        jnp.concatenate([other_mean, other_logvar], axis=-1)
    )
    terms = 0.5 * (
        np.exp(logvar - other_logvar)
        + (mean - other_mean) ** 2 * np.exp(-other_logvar)
        - 1
        + other_logvar
        - logvar
    )
    expected = terms.reshape(shape[0], -1).sum(axis=1)
    np.testing.assert_allclose(posterior.kl(other), expected, rtol=1e-6)
    np.testing.assert_allclose(
        posterior.kl(posterior), np.zeros(shape[0]), atol=1e-6
    )

  def test_mean_kl_gradient_does_not_average_latent_channels(self):
    mean = jnp.ones((2, 2, 3, 4, 16), dtype=jnp.float32)

    def loss(mu):
      posterior = vae.DiagonalGaussianDistribution(
          jnp.concatenate([mu, jnp.zeros_like(mu)], axis=-1)
      )
      return posterior.kl().mean()

    gradient = jax.jit(jax.grad(loss))(mean)
    # d KL / d mu = mu; only the batch mean contributes a factor of 1 / B.
    np.testing.assert_allclose(gradient, np.full(mean.shape, 0.5), rtol=1e-6)

  def test_deterministic_kl_returns_one_zero_per_batch_item(self):
    for batch in (1, 3):
      with self.subTest(batch=batch):
        params = jnp.ones((batch, 2, 3, 4, 6), dtype=jnp.float32)
        posterior = vae.DiagonalGaussianDistribution(params, deterministic=True)
        other = vae.DiagonalGaussianDistribution(params)
        self.assertEqual(posterior.kl().shape, (batch,))
        np.testing.assert_array_equal(posterior.kl(), np.zeros(batch))
        np.testing.assert_array_equal(posterior.kl(other), np.zeros(batch))

  def test_image_posterior_reduction_is_unchanged(self):
    params = jnp.concatenate(
        [jnp.ones((2, 3, 4, 5)), jnp.zeros((2, 3, 4, 5))], axis=-1
    )
    posterior = vae.DiagonalGaussianDistribution(params)
    np.testing.assert_allclose(posterior.kl(), np.full(2, 30.0))


if __name__ == '__main__':
  unittest.main()
