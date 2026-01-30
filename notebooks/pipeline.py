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

from lhf import *

seed = 42
key = jax.random.PRNGKey(seed)

# %% Task
base_p = TaskParams(
    alpha = 0.3,
    beta = 0.2,
    gamma = 0.0
)

target_p = TaskParams(
    alpha = 0.3,
    beta = 2.0,
    gamma = 0.0
)

task_vis = ManifoldVisualizer(ylim=(-1,5))


# %% Create data
n = 10000
m = 50
tau = 0.3

(
    key,
    x_key,
    y_key,
    l_key
) = jax.random.split(key, 4)

z = jax.random.uniform(x_key, shape=(n,1), minval=-1, maxval=1)
x = jnp.broadcast_to(
    z[:,None,:],
    (n, m, 1)
)
vmapped_sample_manifold = jax.vmap(
    sample_manifold,
    in_axes=(0,1,None,None,None),
    out_axes=1
)
y_keys = jax.random.split(y_key, (m,))
y = vmapped_sample_manifold(
    y_keys,
    x,
    base_p.alpha,
    base_p.beta,
    base_p.gamma,
)
logits = logpdf_labels(
    x,
    y,
    alpha=target_p.alpha,
    beta=target_p.beta,
    gamma=target_p.gamma,
    tau=tau
)
# Sample labels
labels = jax.random.categorical(
    l_key,
    logits,
    axis=-1
)
task_vis.visualize(
    x[0,0,0],
    y[0],
    base_manifold=base_p,
    target_manifold=target_p,
    labels=labels[0]
)

# %% Pre-train
pre_train_epochs = 2000
lr = 1e-3
batch_dim = 256
sigma_y_0 = 1.5
sigma_y_T = 0.05

(
    key,
    x_key,
    y_key,
    init_key,
    ys_key
) = jax.random.split(key, 5)

x = z
y = sample_manifold(
    x_key,
    x,
    base_p.alpha,
    base_p.beta,
    base_p.gamma
)

features = [256, 256]
n_features = len(features)
d_z=1
d_y=2

encoder_mlp = MLP(
    features=features,
    output_dim=d_z + (d_z*(d_z+1)) // 2,
    kernel_inits=[jax.nn.initializers.normal(1e-2)] * n_features,
    bias_inits=[jax.nn.initializers.zeros] * n_features
)
decoder_mlp = MLP(
    features=features,
    output_dim=d_y,
    kernel_inits=[jax.nn.initializers.normal(1e-2)] * n_features,
    bias_inits=[jax.nn.initializers.zeros] * n_features
)

model = ConditionalVAE(
    encoder=encoder_mlp,
    decoder=decoder_mlp,
    d_z=1,
    d_y=2
)
x_batch = x[:batch_dim,:]
y_batch = y[:batch_dim,:]
print("x_batch.shape = ", x_batch.shape)
print("y_batch.shape = ", y_batch.shape)
xz_batch = jnp.concatenate(
    [x_batch, jnp.zeros((batch_dim,1))],
    axis=-1
)
xy_batch = jnp.concatenate(
    [x_batch, y_batch],
    axis=-1
)

print("x_batch.shape = ", x_batch.shape)
print("y_batch.shape = ", xy_batch.shape)
print("z_batch.shape = ", xz_batch.shape)
params = {
    "encoder": encoder_mlp.init(init_key, xy_batch),
    "decoder": decoder_mlp.init(init_key, xz_batch)
}

# Optimizer
opt = optax.adam(lr)
opt_state = opt.init(params)

def loss_fn(params, step, key, x, y):
    return -jnp.mean(
        model.elbo(
            params,
            x,
            y,
            key,
            sigma_y=sigma_y_0 + (sigma_y_T - sigma_y_0)*(step+1)/pre_train_epochs,
        )
    )

@jax.jit
def train_step(params, step, opt_state, key, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, step, key, x, y)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

loss_history = []

for step in range(pre_train_epochs):
    key, subkey = jax.random.split(key)
    params, opt_state, loss = train_step(
        params, step, opt_state, subkey, x, y
    )

    loss_val = float(loss)          # convert from JAX scalar
    loss_history.append(loss_val)

    print(f"[{step+1}/{pre_train_epochs}] Loss = {loss_val}")

def sample_many(params, key, x, m):
    keys = jax.random.split(key, m)
    return jax.vmap(
        lambda k: model.sample(
            params,
            x,                     # (B, 1)
            k,
            sigma_y=sigma_y_T,
            deterministic=False,
        ),
        out_axes=1,
    )(keys)

x_broadcast = jnp.broadcast_to(
    x[:, None, :],
    (n, m, 1)
)

ys = sample_many(params, ys_key, x, m)

logits = logpdf_labels(
    x_broadcast,
    ys,
    alpha=target_p.alpha,
    beta=target_p.beta,
    gamma=target_p.gamma,
    tau=tau,
)

labels = jax.random.categorical(
    l_key,
    logits,
    axis=-1,
)

plt.figure(figsize=(5, 3))
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("VAE training loss")
plt.grid(True)
plt.tight_layout()
plt.show()


task_vis.visualize(
    x[0, 0],
    ys[0],
    base_manifold=base_p,
    target_manifold=target_p,
    labels=labels[0],
)


# %% Create queries and annotate

# %% Learn preference model

# %% Improve generative model