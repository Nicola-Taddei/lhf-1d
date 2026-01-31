from typing import Sequence, Optional, Tuple, Sequence, Callable, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np

@struct.dataclass
class PrefModel:
    y2_fn: nn.Module

    def logpdf(self, params, x, y, tau=None):
        """
        Args:
            x: (B,m,1)
            y: (B,m,2)

        Return:
            logits: (B,m,2)
        """
        y1 = y[:,:,0][..., None]
        y2 = y[:,:,1][..., None]
        y2_hat = self.y2_fn(params["y2_fn"], x, y1)
        u = -(y2 - y2_hat)*2
        u_mean = jnp.mean(u, axis=1)
        delta_u = u - u_mean[..., None]
        if tau is None:
            tau = jax.nn.softplus(params["log_tau"]) + 1e-6
        logits = delta_u / tau                        # (B, m, 1)

        logprob_like = jax.nn.log_sigmoid(logits)       # (B, m, 1)
        logprob_dislike = jax.nn.log_sigmoid(-logits)   # (B, m, 1)

        logpdf = jnp.concatenate(
            [logprob_dislike, logprob_like],
            axis=-1,
        )                                               # (B, m, 2)

        return logpdf
    


