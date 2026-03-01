from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def to_x_cond(x_0: jnp.ndarray, x_T: jnp.ndarray) -> jnp.ndarray:
    """
    Convert (x_0, x_T) into conditioning vector.

    Supports:
        (N,2)        -> (4N,)
        (B,N,2)      -> (B,4N)
    """

    if x_0.shape != x_T.shape:
        raise ValueError("x_0 and x_T must have the same shape")

    if x_0.ndim == 2:
        # Unbatched
        N = x_0.shape[0]
        x_0_flat = x_0.reshape(2 * N)
        x_T_flat = x_T.reshape(2 * N)
        return jnp.concatenate([x_0_flat, x_T_flat], axis=0)

    elif x_0.ndim == 3:
        # Batched
        B, N, _ = x_0.shape
        x_0_flat = x_0.reshape(B, 2 * N)
        x_T_flat = x_T.reshape(B, 2 * N)
        return jnp.concatenate([x_0_flat, x_T_flat], axis=1)

    else:
        raise ValueError("Invalid shape for x_0")


def from_x_cond(x_cond: jnp.ndarray, N: int):
    """
    Recover x_0 and x_T from conditioning vector.

    Supports:
        (4N,)    -> (N,2)
        (B,4N)   -> (B,N,2)
    """

    if x_cond.ndim == 1:
        # Unbatched
        if x_cond.shape[0] != 4 * N:
            raise ValueError("Invalid conditioning dimension")

        split = 2 * N
        x_0 = x_cond[:split].reshape(N, 2)
        x_T = x_cond[split:].reshape(N, 2)
        return x_0, x_T

    elif x_cond.ndim == 2:
        # Batched
        B, D = x_cond.shape
        if D != 4 * N:
            raise ValueError("Invalid conditioning dimension")

        split = 2 * N
        x_0 = x_cond[:, :split].reshape(B, N, 2)
        x_T = x_cond[:, split:].reshape(B, N, 2)
        return x_0, x_T

    else:
        raise ValueError("Invalid shape for x_cond")
    
def to_y_out(y: jnp.ndarray) -> jnp.ndarray:
    """
    Flatten trajectory tensor.

    Converts:
        (B, N, T-2, 2) -> (B, N*(T-2)*2)
    """

    if y.ndim != 4:
        raise ValueError("y must have shape (B, N, T-2, 2)")

    B = y.shape[0]
    return y.reshape(B, -1)

def from_y_out(y_out: jnp.ndarray, N: int, T: int) -> jnp.ndarray:
    """
    Recover trajectory tensor.

    Converts:
        (B, N*(T-2)*2) -> (B, N, T-2, 2)
    """

    if y_out.ndim != 2:
        raise ValueError("y_out must have shape (B, D)")

    B, D = y_out.shape
    expected_dim = N * (T - 2) * 2

    if D != expected_dim:
        raise ValueError(
            f"Invalid flattened dimension. Expected {expected_dim}, got {D}"
        )

    return y_out.reshape(B, N, T - 2, 2)

def assemble_traj(
    x_0: jnp.ndarray,
    y: jnp.ndarray,
    x_T: jnp.ndarray,
) -> jnp.ndarray:
    """
    Assemble trajectories from endpoints and intermediate states.

    Supports:
        Unbatched:
            x_0: (N, 2)
            y:   (N, T-2, 2)
            x_T: (N, 2)
            -> (N, T, 2)

        Batched:
            x_0: (B, N, 2)
            y:   (B, N, T-2, 2)
            x_T: (B, N, 2)
            -> (B, N, T, 2)
    """

    if x_0.shape != x_T.shape:
        raise ValueError("x_0 and x_T must have identical shape.")

    # Detect batched vs unbatched
    unbatched = (x_0.ndim == 2)

    if unbatched:
        # Lift to batched
        x_0 = x_0[None, ...]   # (1, N, 2)
        y = y[None, ...]       # (1, N, T-2, 2)
        x_T = x_T[None, ...]   # (1, N, 2)

    # Expand time dimension
    x_0_exp = x_0[:, :, None, :]   # (B, N, 1, 2)
    x_T_exp = x_T[:, :, None, :]   # (B, N, 1, 2)

    # Concatenate along time axis
    traj = jnp.concatenate([x_0_exp, y, x_T_exp], axis=2)

    if unbatched:
        traj = traj[0]  # remove batch dimension

    return traj


def sample_context(
    key: jax.random.PRNGKey,
    N: int,
    B: int | None = None,
    low: float = -1.0,
    high: float = 1.0,
):
    """
    Sample start and end positions uniformly in [low, high].

    Args:
        key: JAX PRNG key
        N: number of agents
        B: optional batch size
        low: lower bound
        high: upper bound

    Returns:
        x_0, x_T, new_key

        If B is None:
            x_0, x_T: (N, 2)

        If B is provided:
            x_0, x_T: (B, N, 2)
    """

    key1, key2, new_key = jax.random.split(key, 3)

    if B is None:
        shape = (N, 2)
    else:
        shape = (B, N, 2)

    x_0 = jax.random.uniform(key1, shape, minval=low, maxval=high)
    x_T = jax.random.uniform(key2, shape, minval=low, maxval=high)

    return x_0, x_T, new_key

'''
def procedural_traj(
    key: jax.random.PRNGKey,
    N: int,
    T: int,
    B: int | None = None,
    v_eps: float = 0.05,
):
    """
    Generate procedural trajectories via random polar velocity sampling.

    Args:
        key: JAX PRNGKey
        N: number of agents
        T: number of timesteps
        B: optional batch size
        v_eps: velocity magnitude

    Returns:
        trajectories:
            if unbatched: (N, T, 2)
            if batched:   (B, N, T, 2)

        new_key
    """

    key_x0, key_theta, new_key = jax.random.split(key, 3)

    # -------------------------------------------------
    # Initial positions
    # -------------------------------------------------
    if B is None:
        x0_shape = (N, 2)
        theta_shape = (T - 1, N)
    else:
        x0_shape = (B, N, 2)
        theta_shape = (B, T - 1, N)

    x_0 = jax.random.uniform(
        key_x0,
        shape=x0_shape,
        minval=-1.0,
        maxval=1.0,
    )

    # -------------------------------------------------
    # Sample random angles
    # -------------------------------------------------
    theta = jax.random.uniform(
        key_theta,
        shape=theta_shape,
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )

    # -------------------------------------------------
    # Convert polar to Cartesian velocities
    # -------------------------------------------------
    vx = v_eps * jnp.cos(theta)
    vy = v_eps * jnp.sin(theta)

    v = jnp.stack([vx, vy], axis=-1)  # (..., 2)

    # -------------------------------------------------
    # Integrate via cumulative sum
    # -------------------------------------------------
    if B is None:
        # v: (T-1, N, 2)
        # Need time dimension second for easier cumsum
        v = jnp.transpose(v, (1, 0, 2))  # (N, T-1, 2)

        increments = jnp.concatenate(
            [jnp.zeros((N, 1, 2)), v],
            axis=1,
        )  # (N, T, 2)

        traj = x_0[:, None, :] + jnp.cumsum(increments, axis=1)

    else:
        # v: (B, T-1, N)
        v = jnp.transpose(v, (0, 2, 1, 3))  # (B, N, T-1, 2)

        increments = jnp.concatenate(
            [jnp.zeros((B, N, 1, 2)), v],
            axis=2,
        )  # (B, N, T, 2)

        traj = x_0[:, :, None, :] + jnp.cumsum(increments, axis=2)

    return traj
'''

def procedural_traj(
    key: jax.random.PRNGKey,
    N: int,
    T: int,
    B: int | None = None,
    scale: float = 1.0,
    v_eps: float = 0.05,
    lengthscale: float = 5.0,
):
    """
    Generate globally smooth trajectories by sampling
    a GP conditioned on start and end points.

    The trajectory is:
        linear path + GP noise
    where the GP is conditioned to be zero at t=0 and t=T-1.
    """

    key_x0, key_trans, key_gp, new_key = jax.random.split(key, 4)

    # -------------------------------------------------
    # Shapes
    # -------------------------------------------------
    if B is None:
        x0_shape = (N, 2)
        trans_shape = (1, 2)
    else:
        x0_shape = (B, N, 2)
        trans_shape = (B, 1, 2)

    # -------------------------------------------------
    # Initial positions
    # -------------------------------------------------
    x_0 = jax.random.uniform(
        key_x0,
        shape=x0_shape,
        minval=-scale,
        maxval=scale,
    )

    translation = jax.random.uniform(
        key_trans,
        shape=trans_shape,
        minval=-scale,
        maxval=scale,
    )

    x_T = x_0 + translation

    # -------------------------------------------------
    # Linear path
    # -------------------------------------------------
    t_grid = jnp.linspace(0.0, 1.0, T)

    if B is None:
        traj = x_0[:, None, :] + t_grid[None, :, None] * (x_T - x_0)[:, None, :]
    else:
        traj = x_0[:, :, None, :] + t_grid[None, None, :, None] * (x_T - x_0)[:, :, None, :]

    # -------------------------------------------------
    # Build full GP covariance over all time steps
    # -------------------------------------------------
    t = jnp.arange(T)
    dt = t[:, None] - t[None, :]
    K = v_eps**2 * jnp.exp(-0.5 * (dt / lengthscale) ** 2)
    jitter = 1e-3
    K = K + jitter * jnp.eye(T)

    # Partition indices
    idx_b = jnp.array([0, T - 1])
    idx_i = jnp.arange(1, T - 1)

    K_bb = K[idx_b[:, None], idx_b]
    K_ib = K[idx_i[:, None], idx_b]
    K_bi = K_ib.T
    K_ii = K[idx_i[:, None], idx_i]

    # Conditional covariance (Schur complement)
    K_bb_inv = jnp.linalg.inv(K_bb)
    K_cond = K_ii - K_ib @ K_bb_inv @ K_bi

    L = jnp.linalg.cholesky(K_cond + 1e-6 * jnp.eye(T - 2))

    # -------------------------------------------------
    # Sample GP noise
    # -------------------------------------------------
    def sample_one(key):
        eps = jax.random.normal(key, (T - 2,))
        return L @ eps

    if B is None:
        keys = jax.random.split(key_gp, N * 2).reshape(N, 2, 2)[:, :, 0]

        def sample_agent(k):
            kx, ky = jax.random.split(k)
            noise_x = sample_one(kx)
            noise_y = sample_one(ky)
            return jnp.stack([noise_x, noise_y], axis=-1)

        noise = jax.vmap(sample_agent)(keys)
        traj = traj.at[:, 1:-1, :].add(noise)

    else:
        keys = jax.random.split(key_gp, B * N).reshape(B, N, 2)

        def sample_agent(k):
            kx, ky = jax.random.split(k)
            noise_x = sample_one(kx)
            noise_y = sample_one(ky)
            return jnp.stack([noise_x, noise_y], axis=-1)

        noise = jax.vmap(jax.vmap(sample_agent))(keys)
        traj = traj.at[:, :, 1:-1, :].add(noise)

    return traj

'''
def trajectory_utility(traj: jnp.ndarray) -> jnp.ndarray:
    """
    Compute time-wise utility:

        U_t = - sum_i sum_j ||v_i - v_j||^2

    Supports:
        (N, T, 2)
        (B, N, T, 2)

    Returns:
        (T,)        or
        (B, T)
    """

    # -------------------------------------------------
    # Ensure batched format
    # -------------------------------------------------
    is_unbatched = (traj.ndim == 3)

    if is_unbatched:
        traj = traj[None, ...]  # (1, N, T, 2)

    B, N, T, _ = traj.shape

    # -------------------------------------------------
    # Compute velocities
    # -------------------------------------------------
    v = traj[:, :, 1:, :] - traj[:, :, :-1, :]  # (B, N, T-1, 2)

    # -------------------------------------------------
    # Efficient quadratic form:
    # -2N sum_i ||v_i||^2 + 2 ||sum_i v_i||^2
    # -------------------------------------------------
    sum_sq = jnp.sum(jnp.sum(v ** 2, axis=-1), axis=1)  # (B, T-1)

    sum_v = jnp.sum(v, axis=1)  # (B, T-1, 2)
    norm_sum_v_sq = jnp.sum(sum_v ** 2, axis=-1)  # (B, T-1)

    utility = -2.0 * N * sum_sq + 2.0 * norm_sum_v_sq  # (B, T-1)

    # -------------------------------------------------
    # Pad time 0
    # -------------------------------------------------
    utility = jnp.concatenate(
        [jnp.zeros((B, 1), dtype=traj.dtype), utility],
        axis=1,
    )

    # -------------------------------------------------
    # Restore original shape
    # -------------------------------------------------
    if is_unbatched:
        utility = utility.squeeze(0)

    return utility
'''

def trajectory_utility(
    traj: jnp.ndarray,
    accel_weight: float = 0.1,
) -> jnp.ndarray:
    """
    Mean-normalized time-wise utility:

        U_t =
        - (1/N^2) sum_{i,j} ||v_i - v_j||^2
        - accel_weight * (1/N) sum_i ||a_i||^2

    Supports:
        (N, T, 2)
        (B, N, T, 2)

    Returns:
        (T,)        or
        (B, T)
    """

    is_unbatched = (traj.ndim == 3)

    if is_unbatched:
        traj = traj[None, ...]

    B, N, T, _ = traj.shape

    # -------------------------------------------------
    # Velocities
    # -------------------------------------------------
    v = traj[:, :, 1:, :] - traj[:, :, :-1, :]  # (B, N, T-1, 2)

    # ∑_i ||v_i||^2
    sum_sq = jnp.sum(jnp.sum(v ** 2, axis=-1), axis=1)  # (B, T-1)

    # ||∑_i v_i||^2
    sum_v = jnp.sum(v, axis=1)  # (B, T-1, 2)
    norm_sum_v_sq = jnp.sum(sum_v ** 2, axis=-1)

    vel_utility = (
        -2.0 / N * sum_sq
        + 2.0 / (N ** 2) * norm_sum_v_sq
    )  # (B, T-1)

    # -------------------------------------------------
    # Acceleration penalty
    # -------------------------------------------------
    if T > 2:
        a = v[:, :, 1:, :] - v[:, :, :-1, :]  # (B, N, T-2, 2)

        accel_penalty = (
            jnp.sum(jnp.sum(a ** 2, axis=-1), axis=1) / N
        )  # (B, T-2)

        accel_penalty = jnp.concatenate(
            [
                jnp.zeros((B, 2), dtype=traj.dtype),
                accel_penalty
            ],
            axis=1
        )
    else:
        accel_penalty = jnp.zeros((B, T), dtype=traj.dtype)

    # -------------------------------------------------
    # Combine
    # -------------------------------------------------
    utility = jnp.concatenate(
        [jnp.zeros((B, 1), dtype=traj.dtype), vel_utility],
        axis=1,
    )

    utility = utility - accel_weight * accel_penalty

    if is_unbatched:
        utility = utility.squeeze(0)

    return utility

def logpdf_labels_traj(
    traj: jnp.ndarray,
    tau: float,
):
    """
    Compute log-probabilities for binary labels based on
    delta-utility sigmoid model.

    Args:
        traj:
            (N, T, 2)        or
            (B, N, T, 2)

        tau: temperature parameter

    Returns:
        logpdf:
            (T, 2)           or
            (B, T, 2)

        Where last dimension is:
            [:, :, 0] -> log P(dislike)
            [:, :, 1] -> log P(like)
    """

    # -------------------------------------------------
    # Ensure batched format
    # -------------------------------------------------
    is_unbatched = (traj.ndim == 3)

    if is_unbatched:
        traj = traj[None, ...]

    # -------------------------------------------------
    # Compute utility
    # -------------------------------------------------
    u = trajectory_utility(traj)  # (B, T)

    # -------------------------------------------------
    # Center utility per trajectory
    # -------------------------------------------------
    u_mean = jnp.mean(u, axis=1, keepdims=True)  # (B,1)

    deltas = u - u_mean                          # (B,T)
    logits = deltas / tau                        # (B,T)

    # -------------------------------------------------
    # Binary log-probs
    # -------------------------------------------------
    logprob_like = jax.nn.log_sigmoid(logits)       # (B,T)
    logprob_dislike = jax.nn.log_sigmoid(-logits)   # (B,T)

    logpdf = jnp.stack(
        [logprob_dislike, logprob_like],
        axis=-1,
    )  # (B,T,2)

    # -------------------------------------------------
    # Restore unbatched shape
    # -------------------------------------------------
    if is_unbatched:
        logpdf = logpdf.squeeze(0)

    return logpdf


class TrajectoryVisualizer:
    def __init__(
        self,
        xlim=(-5.0, 5.0),
        ylim=(-5.0, 5.0)
    ):
        """
        Args:
            xlim: tuple (min, max) for y1 axis
            ylim: tuple (min, max) for y2 axis
            num_curve_points: resolution for manifold curves
        """
        self.xlim = xlim
        self.ylim = ylim

    def visualize(
        self,
        trajectories,
        labels=None,
        scale="free",
    ):
        """
        Visualize multi-agent planar trajectories.

        Args:
            trajectories: np.ndarray of shape (N, T, 2)
                N = number of agents
                T = number of timesteps
                2 = (x, y)

            labels: np.ndarray of shape (T,) with {0,1} or None
                0 -> dislike (red)
                1 -> like (green)
                Label refers to a timestep and applies to all agents.

            scale: "fixed" or "free"
        """

        trajectories = np.asarray(trajectories)
        assert trajectories.ndim == 3 and trajectories.shape[2] == 2, \
            "trajectories must have shape (N, T, 2)"

        N, T, _ = trajectories.shape

        if labels is not None:
            labels = np.asarray(labels)
            assert labels.shape == (T,), \
                "labels must have shape (T,)"

        fig, ax = plt.subplots(figsize=(5, 5), dpi=200)

        # -------------------------------------------------
        # Plot trajectories (black lines)
        # -------------------------------------------------
        for n in range(N):
            ax.plot(
                trajectories[n, :, 0],
                trajectories[n, :, 1],
                color="black",
                linewidth=1.5,
                alpha=0.9,
            )

        # -------------------------------------------------
        # Plot all setpoints (including start/end)
        # -------------------------------------------------
        for t in range(T):

            if labels is None:
                color = "black"
            else:
                color = "green" if labels[t] == 1 else "red"

            # START
            if t == 0:
                ax.scatter(
                    trajectories[:, t, 0],
                    trajectories[:, t, 1],
                    c=color,
                    marker="o",
                    s=60,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=5,
                )

            # END
            elif t == T - 1:
                ax.scatter(
                    trajectories[:, t, 0],
                    trajectories[:, t, 1],
                    c=color,
                    marker="X",
                    s=80,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=6,
                )

            # INTERMEDIATE
            else:
                ax.scatter(
                    trajectories[:, t, 0],
                    trajectories[:, t, 1],
                    c=color,
                    s=6,
                    alpha=0.9,
                    zorder=3,
                )

        # -------------------------------------------------
        # Determine plotting limits
        # -------------------------------------------------
        if scale == "fixed":
            xlim = self.xlim
            ylim = self.ylim
        else:
            x_vals = trajectories[:, :, 0].reshape(-1)
            y_vals = trajectories[:, :, 1].reshape(-1)

            x_min, x_max = x_vals.min(), x_vals.max()
            y_min, y_max = y_vals.min(), y_vals.max()

            margin_ratio = 0.1
            x_range = x_max - x_min + 1e-8
            y_range = y_max - y_min + 1e-8

            x_min -= margin_ratio * x_range
            x_max += margin_ratio * x_range
            y_min -= margin_ratio * y_range
            y_max += margin_ratio * y_range

            min_width = self.xlim[1] - self.xlim[0]
            min_height = self.ylim[1] - self.ylim[0]

            center_x = 0.5 * (x_min + x_max)
            center_y = 0.5 * (y_min + y_max)

            width = max(x_max - x_min, min_width)
            height = max(y_max - y_min, min_height)

            xlim = (center_x - width / 2, center_x + width / 2)
            ylim = (center_y - height / 2, center_y + height / 2)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if scale == "free":
            ax.set_aspect("equal", adjustable="box")

        # -------------------------------------------------
        # Legend
        # -------------------------------------------------
        handles = []

        if labels is not None:
            handles.extend([
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="green",
                    markeredgecolor="none",
                    markersize=6,
                    label="like",
                ),
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="red",
                    markeredgecolor="none",
                    markersize=6,
                    label="dislike",
                ),
            ])

        handles.extend([
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=8,
                label="start",
            ),
            Line2D(
                [0], [0],
                marker="X",
                linestyle="None",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=8,
                label="end",
            ),
        ])

        ax.legend(handles=handles, loc="best")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Multi-Agent Trajectories")
        ax.grid(True)

        plt.show()

        return fig


