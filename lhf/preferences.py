from typing import Sequence, Optional, Tuple, Sequence, Callable, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np

class AgentSelfAttentionMLPBlock(nn.Module):
    """
    A permutation-equivariant block:
      1. Self-attention across agents
      2. Agent-wise MLP
      3. Residual connections + LayerNorm

    Input : (B, N, T, C)
    Output: (B, N, T, C_out)
    """
    num_heads: int
    hidden_dim: int
    mlp_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, h):
        B, N, T, C = h.shape

        # ------------------------------------------------------------
        # Reshape so SelfAttention only sees (N) as sequence axis
        # ------------------------------------------------------------
        h_bt = h.transpose(0, 2, 1, 3).reshape(B * T, N, C)

        # ------------------------------------------------------------
        # SELF-ATTENTION ACROSS AGENTS
        # ------------------------------------------------------------
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            deterministic=True,   # or use train flag
        )(h_bt)

        # Residual + norm
        h_bt = nn.LayerNorm()(h_bt + attn)

        # ------------------------------------------------------------
        # AGENT-WISE MLP
        # ------------------------------------------------------------
        mlp = nn.relu(nn.Dense(self.mlp_dim)(h_bt))
        mlp = nn.Dense(self.out_dim)(mlp)

        # Residual + norm
        h_bt = nn.LayerNorm()(h_bt + mlp)

        # ------------------------------------------------------------
        # Restore shape (B, N, T, C_out)
        # ------------------------------------------------------------
        return h_bt.reshape(B, T, N, self.out_dim).transpose(0, 2, 1, 3)



class TrajectoryAttentionCNN(nn.Module):
    """
    Backbone architecture for multi-agent trajectories:

        1. Temporal CNN stack (per agent)
        2. Stacked agent-attention blocks (permutation equivariant)
        3. Mean pooling over agents (permutation invariant)
        4. Temporal CNN stack after pooling
        5. Linear head producing a scalar per time step

    Input:
        x: (B, N, T, 2)

    Output:
        u: (B, T)
    """

    # Temporal convolutions before attention
    pre_conv_channels: Sequence[int]
    pre_conv_kernel: int

    # Agent attention block parameters
    num_agent_attn_layers: int
    num_heads: int
    attn_hidden_dim: int
    attn_mlp_dim: int
    attn_out_dim: int

    # Temporal convolutions after pooling
    post_conv_channels: Sequence[int]
    post_conv_kernel: int

    @nn.compact
    def __call__(self, x):
        B, N, T, C_in = x.shape

        # ------------------------------------------------------------
        # Step 1 — Temporal CNN stack BEFORE attention
        # ------------------------------------------------------------
        h = x
        for channels in self.pre_conv_channels:
            h = nn.Conv(
                features=channels,
                kernel_size=(1, self.pre_conv_kernel),  # time-wise only
                padding="SAME",
            )(h)
            h = nn.relu(h)

        # h: (B, N, T, C_pre)
        #print("Pre-conv")
        #print("h.shape = ", h.shape)

        # ------------------------------------------------------------
        # Step 2 — Stacked Agent Attention Blocks
        # ------------------------------------------------------------
        for _ in range(self.num_agent_attn_layers):
            h = AgentSelfAttentionMLPBlock(
                num_heads=self.num_heads,
                hidden_dim=self.attn_hidden_dim,
                mlp_dim=self.attn_mlp_dim,
                out_dim=self.attn_out_dim,
            )(h)

        # h: (B, N, T, C_att)
        #print("After attn")
        #print("h.shape = ", h.shape)

        # ------------------------------------------------------------
        # Step 3 — Mean pooling over agents -> permutation invariance
        # ------------------------------------------------------------
        h = jnp.mean(h, axis=1)   # → (B, T, C_att)

        #print("After mean-pooling")
        #print("h.shape = ", h.shape)

        # ------------------------------------------------------------
        # Step 4 — Temporal CNN stack AFTER pooling
        # ------------------------------------------------------------
        for channels in self.post_conv_channels:
            # Add dummy agent dimension: (B, T, C) -> (B, 1, T, C)
            h_4d = h[:, None, :, :]

            # Apply a 2D convolution with kernel (1, k) along time
            h_4d = nn.Conv(
                features=channels,
                kernel_size=(1, self.post_conv_kernel),
                padding="SAME"
            )(h_4d)
            h_4d = nn.relu(h_4d)

            # Remove dummy dimension -> back to (B, T, C)
            h = h_4d[:, 0, :, :]


        # h: (B, T, C_post)
        #print("Post-conv")
        #print("h.shape = ", h.shape)

        # ------------------------------------------------------------
        # Step 5 — Linear prediction head (per time step)
        # ------------------------------------------------------------
        u = nn.Dense(1)(h)      # (B, T, 1)
        return u.squeeze(-1)    # (B, T)


@struct.dataclass
class PrefModel:
    utility_fn: nn.Module

    def logpdf(self, params, traj, tau=None):
        """
        Args:
            traj: (B,N,T,2)

        Return:
            logits: (B,T,2)
        """
        u = self.utility_fn(params["u_params"], traj) # (B,T)
        print("u: ", u.shape)
        u_mean = jnp.mean(u, axis=-1)
        delta_u = u - u_mean[..., None]
        print("delta_u: ", delta_u.shape)
        if tau is None:
            tau = jax.nn.softplus(params["log_tau"]) + 1e-6
        logits = delta_u / tau                        # (B, T, 1)

        print("logits: ", logits.shape)
        logprob_like = jax.nn.log_sigmoid(logits)       # (B, T, 1)
        logprob_dislike = jax.nn.log_sigmoid(-logits)   # (B, T, 1)

        print("logprob_like: ", logprob_like.shape)
        print("logprob_dislike: ", logprob_dislike.shape)

        logpdf = jnp.stack(
            [logprob_dislike, logprob_like],
            axis=-1,
        )                                               # (B, T, 2)

        print("logpdf: ", logpdf.shape)

        return logpdf