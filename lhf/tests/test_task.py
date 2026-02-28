import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from lhf import *


def create_test_trajectory():
    """
    Create a deterministic multi-agent trajectory with:
        - 3 agents
        - 50 timesteps
        - 2D planar coordinates
    Returns:
        trajectories: (3, T, 2)
        labels: (T,)
    """
    N = 3
    T = 50

    t = np.linspace(0, 2 * np.pi, T)

    trajectories = np.zeros((N, T, 2))

    # Agent 1: circular motion
    trajectories[0, :, 0] = np.cos(t)
    trajectories[0, :, 1] = np.sin(t)

    # Agent 2: shifted spiral
    trajectories[1, :, 0] = 0.5 * t * np.cos(t) / (2 * np.pi) + 2.0
    trajectories[1, :, 1] = 0.5 * t * np.sin(t) / (2 * np.pi)

    # Agent 3: sinusoidal trajectory
    trajectories[2, :, 0] = np.linspace(-2, 2, T)
    trajectories[2, :, 1] = 0.5 * np.sin(2 * t) - 2.0

    # Labels: alternate like/dislike over time
    labels = (np.arange(T) % 2).astype(int)

    return trajectories, labels


def test_visualization():
    """
    Test that:
        - visualization runs without error
        - output file is saved
    """

    trajectories, labels = create_test_trajectory()

    visualizer = TrajectoryVisualizer(
        xlim=(-4, 4),
        ylim=(-4, 4),
    )

    fig = visualizer.visualize(
        trajectories=trajectories,
        labels=labels,
        scale="fixed",
    )

    output_path = os.path.join(
        os.path.dirname(__file__),
        "test_vis.png"
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    assert os.path.exists(output_path), "Visualization file was not created."

def test_round_trip():
    key = jax.random.PRNGKey(0)
    B, N = 4, 3

    x_0, x_T, key = sample_context(key, N=N, B=B)
    x_cond = to_x_cond(x_0, x_T)
    x_0_rec, x_T_rec = from_x_cond(x_cond, N)

    assert jnp.allclose(x_0, x_0_rec)
    assert jnp.allclose(x_T, x_T_rec)

def print_shape(name, x):
    print(f"{name:<20} shape = {tuple(x.shape)}")

def test_conditioning():
    print("\n=== Conditioning Test ===")

    key = jax.random.PRNGKey(0)
    x0, xT, _ = sample_context(key, N=3, B=4)

    x_cond = to_x_cond(x0, xT)
    x0_rec, xT_rec = from_x_cond(x_cond, N=3)

    print_shape("x0", x0)
    print_shape("x_T", xT)
    print_shape("x_cond", x_cond)

    assert jnp.allclose(x0, x0_rec)
    assert jnp.allclose(xT, xT_rec)

def test_procedural_traj():
    print("\n=== Procedural Trajectory Test ===")

    key = jax.random.PRNGKey(1)
    traj, _ = procedural_traj(key, N=3, T=40, B=2)

    print_shape("traj", traj)

    assert traj.shape == (2, 3, 40, 2)

def test_utility():
    print("\n=== Utility Test ===")

    key = jax.random.PRNGKey(2)
    traj, _ = procedural_traj(key, N=3, T=40, B=2)

    util = trajectory_utility(traj)

    print_shape("utility", util)

    assert util.shape == (2, 40)
    assert jnp.all(util <= 0.0)  # utility is non-positive

def test_logpdf():
    print("\n=== LogPDF Test ===")

    key = jax.random.PRNGKey(3)
    traj, _ = procedural_traj(key, N=3, T=40, B=2)

    logpdf = logpdf_labels_traj(traj, tau=0.1)

    print_shape("logpdf", logpdf)

    assert logpdf.shape == (2, 40, 2)

    # probabilities sum to 1
    probs = jnp.exp(logpdf)
    assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-6)

def test_visualization():
    print("\n=== Visualization Test ===")

    key = jax.random.PRNGKey(4)
    traj, _ = procedural_traj(key, N=3, T=50)

    logpdf = logpdf_labels_traj(traj, tau=0.1)
    labels = jnp.argmax(logpdf, axis=-1)  # (T,)

    print_shape("traj (vis)", traj)
    print_shape("labels", labels)

    vis = TrajectoryVisualizer(
        xlim=(-3, 3),
        ylim=(-3, 3),
    )

    fig = vis.visualize(
        trajectories=traj,
        labels=labels,
        scale="fixed",
    )

    fig.savefig("test_traj_vis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved test_traj_vis.png")

def test_jit():
    print("\n=== JIT Compilation Test ===")

    key = jax.random.PRNGKey(5)

    jit_proc = jax.jit(
        procedural_traj,
        static_argnames=("N", "T", "B"),
    )
    jit_util = jax.jit(trajectory_utility)
    jit_logpdf = jax.jit(logpdf_labels_traj)

    traj, key = jit_proc(key, N=3, T=30, B=2)
    util = jit_util(traj)
    logpdf = jit_logpdf(traj, tau=0.2)

    print_shape("jit traj", traj)
    print_shape("jit util", util)
    print_shape("jit logpdf", logpdf)

if __name__ == "__main__":
    test_conditioning()
    test_procedural_traj()
    test_utility()
    test_logpdf()
    test_visualization()
    test_jit()

    print("\nAll tests completed successfully.")
