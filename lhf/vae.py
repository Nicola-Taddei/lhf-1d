from typing import Sequence, Optional, Tuple, Sequence, Callable, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np

@dataclass
class VAEParams:
    features: Sequence[int]

class MLP(nn.Module):
    features: Tuple[int, ...]
    output_dim: int
    kernel_inits: Optional[Tuple[Callable, ...]] = None
    bias_inits: Optional[Tuple[Callable, ...]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        for i, feat in enumerate(self.features):
            kernel_init = (
                self.kernel_inits[i]
                if self.kernel_inits is not None
                else nn.initializers.lecun_normal()
            )
            bias_init = (
                self.bias_inits[i]
                if self.bias_inits is not None
                else nn.initializers.zeros
            )

            x = nn.relu(
                nn.Dense(
                    feat,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                )(x)
            )

        output_kernel_init = (
            self.kernel_inits[-1]
            if self.kernel_inits is not None
            else nn.initializers.lecun_normal()
        )
        output_bias_init = (
            self.bias_inits[-1]
            if self.bias_inits is not None
            else nn.initializers.zeros
        )

        return nn.Dense(
            self.output_dim,
            kernel_init=output_kernel_init,
            bias_init=output_bias_init,
        )(x)

@struct.dataclass
class ConditionalVAE:
    encoder: nn.Module        # q(z | x, y)
    decoder: nn.Module        # p(y | x, z)
    d_z: int
    d_y: int
    sigma_y: float = 0.1

    # ------------------------------------------------------------------
    # Encoder (same interface)
    # ------------------------------------------------------------------
    def _encode(self, enc_params, x, y):
        inputs = jnp.concatenate([x, y], axis=-1)
        outputs = self.encoder.apply(enc_params, inputs)

        mu = outputs[:, : self.d_z]
        log_std = outputs[:, self.d_z :]
        std = jnp.exp(log_std)

        return mu, std

    # ------------------------------------------------------------------
    # Decoder (same interface)
    # ------------------------------------------------------------------
    def _decode(self, dec_params, x, z):
        inputs = jnp.concatenate([x, z], axis=-1)
        mu = self.decoder.apply(dec_params, inputs)
        return mu

    # ------------------------------------------------------------------
    # Sampling y ~ p(y | x)
    # ------------------------------------------------------------------
    def sample(
        self,
        params,
        x,
        key,
        *,
        sigma_y: float | None = None,
        deterministic: bool = True,
    ):
        B = x.shape[0]
        key_z, key_y = jax.random.split(key)

        z = jax.random.normal(key_z, (B, self.d_z))
        mu_y = self._decode(params["decoder"], x, z)

        if deterministic:
            return mu_y

        if sigma_y is None:
            sigma_y = self.sigma_y

        eps = jax.random.normal(key_y, mu_y.shape)
        return mu_y + sigma_y * eps

    # ------------------------------------------------------------------
    # Reconstruction loss  -log p(y | x, z)
    # ------------------------------------------------------------------
    def rec_loss(self, params, x, y, key, *, sigma_y: float):
        mu_z, std_z = self._encode(params["encoder"], x, y)

        eps = jax.random.normal(key, mu_z.shape)
        z = mu_z + std_z * eps

        mu_y = self._decode(params["decoder"], x, z)

        resid = y - mu_y
        d = self.d_y

        log_py = (
            -0.5 * jnp.sum(resid**2, axis=-1) / (sigma_y**2)
            -0.5 * d * jnp.log(2.0 * jnp.pi * sigma_y**2)
        )

        return -log_py

    # ------------------------------------------------------------------
    # KL(q(z|x,y) || N(0, I))
    # ------------------------------------------------------------------
    def reg_loss(self, params, x, y, key):
        mu_z, std_z = self._encode(params["encoder"], x, y)

        kl = 0.5 * jnp.sum(
            mu_z**2 + std_z**2 - 1.0 - 2.0 * jnp.log(std_z),
            axis=-1,
        )

        return kl

    # ------------------------------------------------------------------
    # ELBO
    # ------------------------------------------------------------------
    def elbo(self, params, x, y, key, *, sigma_y: float):
        key_rec, key_reg = jax.random.split(key)

        log_py = -self.rec_loss(
            params, x, y, key_rec, sigma_y=sigma_y
        )

        kl = self.reg_loss(params, x, y, key_reg)

        return log_py - kl

    # ------------------------------------------------------------------
    # d_kl (same signature, simplified meaning)
    # ------------------------------------------------------------------
    def d_kl(
        self,
        params: Any,
        x: jnp.ndarray,
        ref_gen_model: Any,      # kept for interface compatibility
        ref_params: Any,
        key: jax.Array,
        *,
        sigma_y: float,
    ) -> jnp.ndarray:
        """
        KL between two conditional decoders with shared z.
        """

        B = x.shape[0]
        z = jax.random.normal(key, (B, self.d_z))

        mu_y = self._decode(params["decoder"], x, z)
        ref_mu_y = ref_gen_model._decode(
            ref_params["decoder"], x, z
        )

        resid = mu_y - ref_mu_y

        kl = 0.5 * jnp.sum(resid**2, axis=-1) / (sigma_y**2)

        return kl
