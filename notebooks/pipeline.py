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
    gamma = 1.0
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
tau = 1.0

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
)   # (B,m,1)
vmapped_sample_manifold = jax.vmap(
    sample_manifold,
    in_axes=(0,1,None,None,None),
    out_axes=1
)
y_keys = jax.random.split(y_key, (m,))
gt_y = vmapped_sample_manifold(
    y_keys,
    x,
    base_p.alpha,
    base_p.beta,
    base_p.gamma,
)   # (B,m,2)
gt_logits = logpdf_labels(
    x,
    gt_y,
    alpha=target_p.alpha,
    beta=target_p.beta,
    gamma=target_p.gamma,
    tau=tau
)
# Sample labels
gt_labels = jax.random.categorical(
    l_key,
    gt_logits,
    axis=-1
)
task_vis.visualize(
    x[0,0,0],
    gt_y[0],
    base_manifold=base_p,
    target_manifold=target_p,
    labels=gt_labels[0]
)

# %% Pre-train
pre_train_epochs = 2000
pre_train_lr = 1e-3
pre_train_batch_dim = 256
sigma_y_0 = 1.5
sigma_y_T = 0.05

(
    key,
    x_key,
    y_key,
    init_key,
    ys_key
) = jax.random.split(key, 5)

pre_train_features = [256, 256]
pre_train_n_features = len(pre_train_features)
d_z=1
d_y=2

encoder_mlp = MLP(
    features=pre_train_features,
    output_dim=d_z + (d_z*(d_z+1)) // 2,
    kernel_inits=[jax.nn.initializers.normal(1e-2)] * pre_train_n_features,
    bias_inits=[jax.nn.initializers.zeros] * pre_train_n_features
)
decoder_mlp = MLP(
    features=pre_train_features,
    output_dim=d_y,
    kernel_inits=[jax.nn.initializers.normal(1e-2)] * pre_train_n_features,
    bias_inits=[jax.nn.initializers.zeros] * pre_train_n_features
)

model = ConditionalVAE(
    encoder=encoder_mlp,
    decoder=decoder_mlp,
    d_z=1,
    d_y=2
)
x_batch = x[:pre_train_batch_dim,0]  # (B,1)
y_batch = gt_y[:pre_train_batch_dim,0]  # (B,2)
print("x_batch.shape = ", x_batch.shape)
print("y_batch.shape = ", y_batch.shape)
xz_batch = jnp.concatenate(
    [x_batch, jnp.zeros((pre_train_batch_dim,1))],
    axis=-1
)
xy_batch = jnp.concatenate(
    [x_batch, y_batch],
    axis=-1
)

print("x_batch.shape = ", x_batch.shape)
print("y_batch.shape = ", xy_batch.shape)
print("z_batch.shape = ", xz_batch.shape)
vae_params = {
    "encoder": encoder_mlp.init(init_key, xy_batch),
    "decoder": decoder_mlp.init(init_key, xz_batch)
}

# Optimizer
opt = optax.adam(pre_train_lr)
opt_state = opt.init(vae_params)

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
    vae_params, opt_state, loss = train_step(
        vae_params, step, opt_state, subkey, x[:,0], gt_y[:,0]
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

pre_train_y = sample_many(vae_params, ys_key, x[:,0], m)

logits = logpdf_labels(
    x,
    pre_train_y,
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
plt.ylabel("-ELBO")
plt.title("VAE training loss")
plt.grid(True)
plt.tight_layout()
plt.show()


task_vis.visualize(
    x[0, 0, 0],
    pre_train_y[0],
    base_manifold=base_p,
    target_manifold=target_p,
    labels=labels[0],
)


# %% Learn preference model
pref_lr = 1e-3
pref_batch_dim = 256
pref_train_epochs = 1000

(
    key,
    init_key,
    l_key,
    gt_l_key
) = jax.random.split(key, 4)

mlp = MLP(
    features=[256, 256],
    output_dim=1
)

init_batch = jnp.zeros((pref_batch_dim, m, 2))
pref_params = {
    "y2_fn": mlp.init(init_key, init_batch),
    #"log_tau": jnp.array(1.0)
}

y2_learned = lambda p, x, y1: mlp.apply(p, jnp.concatenate([x,y1], axis=2))

pref_model = PrefModel(
    y2_fn = y2_learned
)

def pref_nll(params, x, y, labels):
    logpdf = pref_model.logpdf(params, x, y, tau=tau)      # (B, m, 2)

    # Gather log-probabilities of the observed labels
    logp = jnp.take_along_axis(
        logpdf,
        labels[..., None],     # (B, m, 1)
        axis=-1,
    )[..., 0]                  # (B, m)

    return -jnp.mean(logp)

opt = optax.adam(pref_lr)
opt_state = opt.init(pref_params)

@jax.jit
def train_step(params, opt_state, x, y, labels):
    loss, grads = jax.value_and_grad(pref_nll)(
        params,
        x,
        y,
        labels,
    )
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

loss_history = []

for step in range(pref_train_epochs):
    pref_params, opt_state, loss = train_step(
        pref_params,
        opt_state,
        x,        # (B, m, 1)
        pre_train_y,        # (B, m, 2)
        labels,   # (B, m)
    )

    loss_val = float(loss)
    loss_history.append(loss_val)

    print(f"[{step+1}/{pref_train_epochs}] NLL = {loss_val:.4f}")

# %% Visualize pref model
plt.figure(figsize=(5, 3))
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("nll")
plt.title("Pref model loss")
plt.grid(True)
plt.tight_layout()
plt.show()

pred_logits = pref_model.logpdf(
    pref_params,
    x,
    pre_train_y,
    tau=tau
)

pred_labels = jax.random.categorical(
    l_key,
    logits,
    axis=-1,
)

gt_logits = logpdf_labels(
    x,
    pre_train_y,
    alpha=target_p.alpha,
    beta=target_p.beta,
    gamma=target_p.gamma,
    tau=tau,
)

gt_new_labels = jax.random.categorical(
    gt_l_key,
    logits,
    axis=-1,
)

acc_gt = jnp.mean(gt_new_labels == labels)
acc_learned = jnp.mean(pred_labels == labels)

perc_likes = jnp.mean(labels == 1)

print(f"Dataset composition (likes/dislikes):   {perc_likes} / {1 - perc_likes}")

print(f"ACC(learned vs GT):   {acc_learned} / {acc_gt}")

#y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y) / (jax.nn.softplus(pref_params["log_tau"]) + 1e-6)
y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y)

task_vis.visualize(
    x[0, 0, 0],
    pre_train_y[0],
    base_manifold=base_p,
    target_manifold=target_p,
    learned_manifold=y2_fn,
    labels=labels[0],
    scale="free"
)

# %% Improve generative model
align_lr = 1e-3
align_batch_dim = 256
align_epochs = 1000

beta = 0.01

(
    key,
    _
) = jax.random.split(key, 2)

base_vae_params = jax.lax.stop_gradient(vae_params)
aligned_vae_params = vae_params

#y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y) / (jax.nn.softplus(pref_params["log_tau"]) + 1e-6)
y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y)

def avg_u(params, key, x, sigma_y):
    y = model.sample(
        params,
        x,
        key,
        sigma_y=sigma_y,
        deterministic=False
    )      # (B, m, 2)

    y1 = y[:, None, 0][...,None]
    y2 = y[:, None, 1][...,None]
    y2_hat = y2_fn(x[:, None, :], y1)
    u = -(y2 - y2_hat)*2

    return jnp.mean(u)

def kl_div(params, key, x, sigma_y):
    kl = model.d_kl(
        params,
        base_vae_params,
        x,
        key,
        sigma_y=sigma_y
    )

    return jnp.mean(kl)

def align_loss(params, key, x, sigma_y):
    u_bar = avg_u(
        params,
        key, 
        x,
        sigma_y
    )
    kl = kl_div(
        params,
        key,
        x,
        sigma_y
    )
    
    return -u_bar + beta * kl

opt = optax.adam(align_lr)
opt_state = opt.init(vae_params)

@jax.jit
def train_step(params, opt_state, key, x, sigma_y):
    loss, grads = jax.value_and_grad(align_loss)(
        params,
        key,
        x,
        sigma_y
    )
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

loss_history = []
u_history = []
kl_history = []

for step in range(align_epochs):
    key, subkey = jax.random.split(key, 2)
    aligned_vae_params, opt_state, loss = train_step(
        aligned_vae_params,
        opt_state,
        subkey,
        x[:,0],        # (B, m, 1)
        sigma_y_T
    )

    loss_val = float(loss)
    u_val = float(
        avg_u(
            aligned_vae_params,
            key,
            x[:,0],
            sigma_y_T
        )
    )
    kl_val = float(
        kl_div(
            aligned_vae_params,
            key,
            x[:,0],
            sigma_y_T
        )
    )
    loss_history.append(loss_val)
    u_history.append(u_val)
    kl_history.append(kl_val)

    print(f"[{step+1}/{pref_train_epochs}] loss = {loss_val:.4f}")

# %% Visualize improved model
# Loss
plt.figure(figsize=(5, 3))
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("loss")
plt.title("Alignement loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Utility
plt.figure(figsize=(5, 3))
plt.plot(u_history)
plt.xlabel("Training step")
plt.ylabel("avg_u")
plt.title("Average utility")
plt.grid(True)
plt.tight_layout()
plt.show()

# KL divergence
plt.figure(figsize=(5, 3))
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("kl_div")
plt.title("KL divergence")
plt.grid(True)
plt.tight_layout()
plt.show()

(
    key,
    y_key,
    l_key,
    gt_l_key
) = jax.random.split(key, 4)

# Sample new y
vmapped_vae_sample = jax.vmap(
    model.sample,
    in_axes=(None, 1, 0),
    out_axes=1
)

aligned_y = vmapped_vae_sample(
    aligned_vae_params,
    x,
    jax.random.split(y_key, (m,))
)

gt_logits = logpdf_labels(
    x,
    aligned_y,
    alpha=target_p.alpha,
    beta=target_p.beta,
    gamma=target_p.gamma,
    tau=tau,
)

gt_new_labels = jax.random.categorical(
    gt_l_key,
    logits,
    axis=-1,
)

perc_likes = jnp.mean(gt_new_labels == 1)

print(f"Dataset composition (likes/dislikes):   {perc_likes} / {1 - perc_likes}")

print(f"ACC(learned vs GT):   {acc_learned} / {acc_gt}")

#y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y) / (jax.nn.softplus(pref_params["log_tau"]) + 1e-6)
y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y)

task_vis.visualize(
    x[0, 0, 0],
    aligned_y[0],
    base_manifold=base_p,
    target_manifold=target_p,
    learned_manifold=y2_fn,
    labels=gt_new_labels[0],
    #scale="free"
)

# %%
