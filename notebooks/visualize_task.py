# %% Import packages
from typing import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
import optax
from flax.nnx import split, merge
import yaml
from pathlib import Path
import wandb

from lhf import *

# %% Load config
path = Path("../configs/config.yaml")
with path.open("r") as f:
    config = yaml.safe_load(f)

# %% Define task
base_p = TaskParams(
    alpha = config["base_manifold"]["alpha"],
    beta = config["base_manifold"]["beta"],
    gamma = config["base_manifold"]["gamma"]
)

target_p = TaskParams(
    alpha = config["target_manifold"]["alpha"],
    beta = config["target_manifold"]["beta"],
    gamma = config["target_manifold"]["gamma"]
)

task_vis = ManifoldVisualizer(
    xlim=(-10,10),
    ylim=(-1,5)
)

# %% Visualize
task_vis.visualize(
    0,
    jnp.array([[0,0]]),
    base_manifold=base_p,
    target_manifold=target_p,
    scale="fixed"
)