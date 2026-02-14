from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

@dataclass
class TaskParams:
    alpha: float
    beta: float
    gamma: float

def manifold(x, y1, alpha, beta, gamma):
    return (gamma*x + beta) * y1**2 + alpha*y1

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

utility_vmapped = jax.vmap(
    utility,
    in_axes=(1, 1, None, None, None),  # map over m
    out_axes=1,
)

def logpdf_labels(x, y, alpha, beta, gamma, tau):
    """
    x: (B, m, 1)
    y: (B, m, 2)
    return: (B,m,2)
    """
    u = utility_vmapped(
        x,          # (B, m, 1)
        y,          # (B, m, 2)
        alpha,
        beta,
        gamma,
    )               # (B, m)

    u_mean = jnp.mean(u, axis=1, keepdims=True)  # (B, 1)

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
        learned_manifold=None,
        labels=None,
        scale="fixed",
    ):
        """
        Args:
            x: scalar conditioning value
            ys: (B, 2) array of samples
            base_manifold: TaskParams or None
            target_manifold: TaskParams or None
            learned_manifold: callable or None
            labels: (B,) array with {0,1} or None
            scale: "fixed" or "free"
        """
        ys = np.asarray(ys)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=200)

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
        # Determine plotting limits
        # -------------------------------------------------
        if scale == "fixed":
            xlim = self.xlim
            ylim = self.ylim
        else:
            y1_min, y1_max = ys[:, 0].min(), ys[:, 0].max()
            y2_min, y2_max = ys[:, 1].min(), ys[:, 1].max()

            margin_ratio = 0.1
            y1_range = y1_max - y1_min + 1e-8
            y2_range = y2_max - y2_min + 1e-8

            y1_min -= margin_ratio * y1_range
            y1_max += margin_ratio * y1_range
            y2_min -= margin_ratio * y2_range
            y2_max += margin_ratio * y2_range

            # Enforce minimum FoV from constructor
            min_width = self.xlim[1] - self.xlim[0]
            min_height = self.ylim[1] - self.ylim[0]

            center_y1 = 0.5 * (y1_min + y1_max)
            center_y2 = 0.5 * (y2_min + y2_max)

            width = max(y1_max - y1_min, min_width)
            height = max(y2_max - y2_min, min_height)

            xlim = (center_y1 - width / 2, center_y1 + width / 2)
            ylim = (center_y2 - height / 2, center_y2 + height / 2)

        # -------------------------------------------------
        # Build grid AFTER limits are known
        # -------------------------------------------------
        y1_grid = np.linspace(xlim[0], xlim[1], self.num_curve_points)

        # -------------------------------------------------
        # Plot base manifold
        # -------------------------------------------------
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
                label="supp(p_data)",
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
                label="supp(p_preference)",
            )

        # -------------------------------------------------
        # Plot learned manifold
        # -------------------------------------------------
        if learned_manifold is not None:
            y2_learned = jnp.squeeze(
                learned_manifold(
                    jnp.broadcast_to(
                        x[None, None, None],
                        y1_grid[None, ..., None].shape,
                    ),
                    y1_grid[None, ..., None],
                )
            )
            ax.plot(
                y1_grid,
                np.asarray(y2_learned),
                color="orange",
                linewidth=2.0,
                label="learned manifold",
            )

        # -------------------------------------------------
        # Apply limits
        # -------------------------------------------------
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Preserve geometry in free mode
        if scale == "free":
            ax.set_aspect("equal", adjustable="box")

        # -------------------------------------------------
        # Formatting
        # -------------------------------------------------
        ax.set_xlabel(r"$y_1$")
        ax.set_ylabel(r"$y_2$")
        ax.set_title(rf"$y \sim p(y \mid x={x:.3f})$")
        ax.grid(True)

        handles, labels_legend = ax.get_legend_handles_labels()

        # Add proxy artists if labels were provided
        if labels is not None:
            proxy_handles = [
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="green",
                    markeredgecolor="none",
                    markersize=8,
                    alpha=0.8,
                    label="liked samples",
                ),
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="red",
                    markeredgecolor="none",
                    markersize=8,
                    alpha=0.8,
                    label="disliked samples",
                ),
            ]
            handles += proxy_handles

        ax.legend(handles=handles, loc="best")

        plt.show()

        return fig
