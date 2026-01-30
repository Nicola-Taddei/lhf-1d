from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class TaskParams:
    alpha: float
    beta: float
    gamma: float

def manifold(x, y1, alpha, beta, gamma):
    return x + gamma*x**2 + alpha*y1 + beta*y1**2

def sample_manifold(key, x, alpha, beta, gamma):
    B = x.shape[0]
    z = jax.random.normal(key, shape=(B,1))
    y1 = z
    y2 = manifold(x, z, alpha, beta, gamma)
    y = jnp.concatenate(
        [y1, y2],
        axis=-1
    )
    return y

def utility(x, y, alpha, beta, gamma):
    """
    x: (B,1)
    y: (B,2)
    return: (B,)
    """
    y1, y2 = y[:,0][...,None], y[:,1][...,None]
    y2_hat = manifold(x, y1, alpha, beta, gamma)
    return jnp.squeeze(-(y2 - y2_hat)**2, axis=1)

def logpdf_labels(x, y, alpha, beta, gamma, tau):
    """
    x: (B, m, 1)
    y: (B, m, 2)
    return: (B,m,2)
    """
    utility_vmapped = jax.vmap(
        utility,
        in_axes=(1, 1, None, None, None),  # map over m
        out_axes=1,
    )

    u = utility_vmapped(
        x,          # (B, m, 1)
        y,          # (B, m, 2)
        alpha,
        beta,
        gamma,
    )               # (B, m)

    u_mean = jnp.mean(u, axis=1, keepdims=True)  # (B,)

    deltas = u - u_mean                          # (B, m)
    logits = deltas / tau                        # (B, m)

    logprob_like = jax.nn.log_sigmoid(logits)       # (B, m)
    logprob_dislike = jax.nn.log_sigmoid(-logits)   # (B, m)

    logpdf = jnp.stack(
        [logprob_dislike, logprob_like],
        axis=-1,
    )                                               # (B, m, 2)

    return logpdf

class ManifoldVisualizer:
    def __init__(
        self,
        xlim=(-5.0, 5.0),
        ylim=(-5.0, 5.0),
        num_curve_points=400,
        marker_size=30
    ):
        """
        Args:
            xlim: tuple (min, max) for y1 axis
            ylim: tuple (min, max) for y2 axis
            num_curve_points: resolution for manifold curves
        """
        self.xlim = xlim
        self.ylim = ylim
        self.num_curve_points = num_curve_points
        self.marker_size=marker_size

    def visualize(
        self,
        x,
        ys,
        base_manifold=None,
        target_manifold=None,
        labels=None,
    ):
        """
        Args:
            x: scalar (conditioning value)
            ys: (B, 2) array of samples
            base_manifold: TaskParams or None
            target_manifold: TaskParams or None
            labels: (B,) array with {0,1} or None
        """
        ys = np.asarray(ys)

        fig, ax = plt.subplots(figsize=(5, 5))

        # -------------------------------------------------
        # Plot sampled points
        # -------------------------------------------------
        if labels is None:
            ax.scatter(
                ys[:, 0],
                ys[:, 1],
                c="black",
                s=self.marker_size,
                alpha=0.8,
                label="samples",
            )
        else:
            labels = np.asarray(labels)
            colors = np.where(labels == 1, "green", "red")
            ax.scatter(
                ys[:, 0],
                ys[:, 1],
                c=colors,
                s=self.marker_size,
                alpha=0.8,
            )

        # -------------------------------------------------
        # Plot base manifold
        # -------------------------------------------------
        y1_grid = np.linspace(self.xlim[0], self.xlim[1], self.num_curve_points)

        if base_manifold is not None:
            y2_base = manifold(
                x,
                y1_grid,
                base_manifold.alpha,
                base_manifold.beta,
                base_manifold.gamma,
            )
            ax.plot(
                y1_grid,
                y2_base,
                color="blue",
                linewidth=2.0,
                label="base manifold",
            )

        # -------------------------------------------------
        # Plot target manifold
        # -------------------------------------------------
        if target_manifold is not None:
            y2_target = manifold(
                x,
                y1_grid,
                target_manifold.alpha,
                target_manifold.beta,
                target_manifold.gamma,
            )
            ax.plot(
                y1_grid,
                y2_target,
                color="pink",
                linewidth=2.0,
                label="target manifold",
            )

        # -------------------------------------------------
        # Formatting
        # -------------------------------------------------
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xlabel(r"$y_1$")
        ax.set_ylabel(r"$y_2$")
        ax.set_title(rf"$y \sim p(y \mid x={x:.3f})$")
        ax.grid(True)
        ax.legend()

        plt.show()