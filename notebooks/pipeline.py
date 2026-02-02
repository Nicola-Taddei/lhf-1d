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

from lhf import *

path = Path("../configs/config.yaml")
with path.open("r") as f:
    config = yaml.safe_load(f)

key = jax.random.PRNGKey(config["seed"])

# %% Task
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

task_vis = ManifoldVisualizer(ylim=(-1,5))


# %% Create data
n_queries = config["n_queries"]
n_internal = config["n_internal"]
m = config["m"]
tau = config["tau"]

(
    key,
    x_key,
    y_key,
    l_key
) = jax.random.split(key, 4)

x = jax.random.uniform(x_key, shape=(n_internal,1), minval=-1, maxval=1)
xs = jnp.broadcast_to(
    x[:,None,:],
    (n_internal, m, 1)
)   # (B,m,1)
vmapped_sample_manifold = jax.vmap(
    sample_manifold,
    in_axes=(0,1,None,None,None),
    out_axes=1
)
y_keys = jax.random.split(y_key, (m,))
ys = vmapped_sample_manifold(
    y_keys,
    xs,
    base_p.alpha,
    base_p.beta,
    base_p.gamma,
)   # (B,m,2)
gt_logits = logpdf_labels(
    xs,
    ys,
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
    xs[0,0,0],
    ys[0],
    base_manifold=base_p,
    target_manifold=target_p,
    labels=gt_labels[0]
)


# %% Training loop: Initialization

num_iter = config["num_iter"]

# Logging:
gt_u_history = []
y_history = []
ys_tot = None
x_tot = None
l_tot = None
l_history = []

# Step 1: Pre-training
pre_train_epochs = config["pre_train_epochs"]
pre_train_lr = config["pre_train_lr"]
pre_train_batch_dim = config["pre_train_batch_dim"]
sigma_y_0 = config["sigma_y_0"]
sigma_y_T = config["sigma_y_T"]

(
    key,
    x_key,
    y_key,
    init_key,
    ys_key
) = jax.random.split(key, 5)

pre_train_features = config["pre_train_features"]
pre_train_n_features = len(pre_train_features)
d_z=config["d_z"]
d_y=config["d_y"]

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

gen_model = ConditionalVAE(
    encoder=encoder_mlp,
    decoder=decoder_mlp,
    d_z=d_z,
    d_y=d_y
)
x_batch = xs[:pre_train_batch_dim,0]  # (B,1)
y_batch = ys[:pre_train_batch_dim,0]  # (B,2)

xz_batch = jnp.concatenate(
    [x_batch, jnp.zeros((pre_train_batch_dim,1))],
    axis=-1
)
xy_batch = jnp.concatenate(
    [x_batch, y_batch],
    axis=-1
)

vae_params = {
    "encoder": encoder_mlp.init(init_key, xy_batch),
    "decoder": decoder_mlp.init(init_key, xz_batch)
}

def pre_train_loss_fn(params, step, key, x, y):
        return -jnp.mean(
            gen_model.elbo(
                params,
                x,
                y,
                key,
                sigma_y=sigma_y_0 + (sigma_y_T - sigma_y_0)*(step+1)/pre_train_epochs,
            )
        )

@jax.jit
def pre_train_step(params, step, opt_state, key, x, y):
    loss, grads = jax.value_and_grad(pre_train_loss_fn)(params, step, key, x, y)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def sample_many(params, key, x, m):
    keys = jax.random.split(key, m)
    return jax.vmap(
        lambda k: gen_model.sample(
            params,
            x,                     # (B, 1)
            k,
            sigma_y=sigma_y_T,
            deterministic=False,
        ),
        out_axes=1,
    )(keys)

# Step 2: Learn preference model
pref_lr = config["pref_lr"]
pref_batch_dim = config["pref_batch_dim"]
pref_train_epochs = config["pref_train_epochs"]

(
    key,
    init_key,
    l_key,
    gt_l_key
) = jax.random.split(key, 4)

mlp = MLP(
    features=config["pref_model_features"],
    output_dim=1
)

init_batch = jnp.zeros((pref_batch_dim, m, 2))
pref_params = {
    "y2_fn": mlp.init(init_key, init_batch),
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

@jax.jit
def pref_train_step(params, opt_state, x, y, labels):
    loss, grads = jax.value_and_grad(pref_nll)(
        params,
        x,
        y,
        labels,
    )
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Step 3: Improve VAE
align_lr = config["align_lr"]
align_batch_dim = config["align_batch_dim"]
align_epochs = config["align_epochs"]
beta = config["beta"]

def avg_u(params, y2_fn, key, x, sigma_y):
    y = gen_model.sample(
        params,
        x,
        key,
        sigma_y=sigma_y,
        deterministic=False
    )      # (B, m, 2)

    y1 = y[:, None, 0][...,None]
    y2 = y[:, None, 1][...,None]
    y2_hat = y2_fn(x[:, None, :], y1)
    u = -(y2 - y2_hat)**2

    return jnp.mean(u)

def kl_div(params, base_vae_params, key, x, sigma_y):
    kl = gen_model.d_kl(
        params,
        base_vae_params,
        x,
        key,
        sigma_y=sigma_y
    )

    return jnp.mean(kl)

def align_loss(params, base_vae_params,y2_fn,  key, x, sigma_y):
    u_bar = avg_u(
        params,
        y2_fn,
        key, 
        x,
        sigma_y
    )
    kl = kl_div(
        params,
        base_vae_params,
        key,
        x,
        sigma_y
    )
    
    return -u_bar + beta * kl

#@jax.jit
def align_train_step(params, base_vae_params, y2_fn, opt_state, key, x, sigma_y):
    loss, grads = jax.value_and_grad(align_loss)(
        params,
        base_vae_params,
        y2_fn,
        key,
        x,
        sigma_y
    )
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

vmapped_vae_sample = jax.vmap(
    gen_model.sample,
    in_axes=(None, 1, 0),
    out_axes=1
)

# %% Training loop: Iterations

for iter in range(num_iter):
    # Step 1: Pre-training
    opt = optax.adam(pre_train_lr)
    opt_state = opt.init(vae_params)

    loss_history = []

    for step in range(pre_train_epochs):
        key, subkey = jax.random.split(key)
        vae_params, opt_state, loss = pre_train_step(
            vae_params, step, opt_state, subkey, xs[:,0], ys[:,0]
        )
        loss_val = float(loss)          # convert from JAX scalar
        loss_history.append(loss_val)

        print(f"[{step+1}/{pre_train_epochs}] -ELBO = {loss_val}")

    # Plot loss
    plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("-ELBO")
    plt.title("VAE training loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ys = sample_many(vae_params, ys_key, xs[:,0], m)
    ys_query = ys[:n_queries]
    y_history.append(ys_query)   # ys for queries

    logits = logpdf_labels(
        xs,
        ys,
        alpha=target_p.alpha,
        beta=target_p.beta,
        gamma=target_p.gamma,
        tau=tau,
    )
    gt_labels = jax.random.categorical(
        l_key,
        logits,
        axis=-1,
    )
    ls = gt_labels[:n_queries]
    l_history.append(ls)   # labels used for queries

    if ys_tot is None:
        ys_tot = ys_query
        l_tot = ls
        xs_tot = xs[:n_queries]
    else:
        ys_tot = jnp.concatenate([ys_tot, ys_query], axis=0)
        l_tot = jnp.concatenate([l_tot, ls], axis=0)
        xs_tot = jnp.concatenate([xs_tot, xs[:n_queries]], axis=0)

    u = utility_vmapped(
        xs,
        ys,
        target_p.alpha,
        target_p.beta,
        target_p.gamma,
    )

    mean_u = jnp.mean(u)
    gt_u_history.append(mean_u)

    task_vis.visualize(
        xs[0, 0, 0],
        ys[0],
        base_manifold=base_p,
        target_manifold=target_p,
        labels=gt_labels[0],
    )


    # Step 2: Learn preference model
    opt = optax.adam(pref_lr)
    opt_state = opt.init(pref_params)

    loss_history = []

    n_tot = xs_tot.shape[0]
    for step in range(pref_train_epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(
            subkey,
            (pref_batch_dim,),
            minval=0,
            maxval=n_tot
        )
        xs_batch = xs_tot[idx]
        ys_batch = ys_tot[idx]
        l_batch = l_tot[idx]
        pref_params, opt_state, loss = pref_train_step(
            pref_params,
            opt_state,
            xs_batch,        # (pref_batch_dim, m, 1)
            ys_batch,        # (pref_batch_dim, m, 2)
            l_batch,   # (pref_batch_dim, m)
        )

        loss_val = float(loss)
        loss_history.append(loss_val)

        print(f"[{step+1}/{pref_train_epochs}] NLL = {loss_val:.4f}")

    # Plot loss
    plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("nll")
    plt.title("Pref model loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    pred_logits = pref_model.logpdf(
        pref_params,
        xs,
        ys,
        tau=tau
    )

    pred_labels = jax.random.categorical(
        l_key,
        pred_logits,
        axis=-1,
    )

    gt_logits = logpdf_labels(
        xs,
        ys,
        alpha=target_p.alpha,
        beta=target_p.beta,
        gamma=target_p.gamma,
        tau=tau,
    )

    gt_new_labels = jax.random.categorical(
        gt_l_key,
        gt_logits,
        axis=-1,
    )

    acc_gt = jnp.mean(gt_new_labels == gt_labels)
    acc_learned = jnp.mean(pred_labels == gt_labels)

    perc_likes = jnp.mean(gt_labels == 1)

    print(f"Dataset composition (likes/dislikes):   {perc_likes} / {1 - perc_likes}")

    print(f"ACC(learned vs GT):   {acc_learned} / {acc_gt}")

    #y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y) / (jax.nn.softplus(pref_params["log_tau"]) + 1e-6)
    y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y)

    task_vis.visualize(
        xs[0, 0, 0],
        ys[0],
        base_manifold=base_p,
        target_manifold=target_p,
        learned_manifold=y2_fn,
        labels=gt_labels[0],
        #scale="free"
    )


    # Step 3: Improve VAE

    base_vae_params = jax.lax.stop_gradient(vae_params)

    #y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y) / (jax.nn.softplus(pref_params["log_tau"]) + 1e-6)
    y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y)
    """ y2_fn = lambda x, y1: manifold(
        x,
        y1,
        target_p.alpha,
        target_p.beta,
        target_p.gamma,
    ) """

    opt = optax.adam(align_lr)
    opt_state = opt.init(vae_params)

    loss_history = []
    u_history = []
    kl_history = []

    for step in range(align_epochs):
        key, subkey = jax.random.split(key, 2)
        vae_params, opt_state, loss = align_train_step(
            vae_params,
            base_vae_params,
            y2_fn,
            opt_state,
            subkey,
            xs[:,0],        # (B, 1)
            sigma_y_T
        )

        loss_val = float(loss)
        u_val = float(
            avg_u(
                vae_params,
                y2_fn,
                key,
                xs[:,0],
                sigma_y_T
            )
        )
        kl_val = float(
            kl_div(
                vae_params,
                base_vae_params,
                key,
                xs[:,0],
                sigma_y_T
            )
        )
        loss_history.append(loss_val)
        u_history.append(u_val)
        kl_history.append(kl_val)

        print(f"[{step+1}/{pref_train_epochs}] loss = {loss_val:.4f}")

    # Plot loss
    plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("loss")
    plt.title("Alignement loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot utility
    plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(u_history)
    plt.xlabel("Training step")
    plt.ylabel("avg_u")
    plt.title("Average utility")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot KL divergence
    plt.figure(figsize=(5, 3), dpi=200)
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

    ys = vmapped_vae_sample(
        vae_params,
        xs,
        jax.random.split(y_key, (m,))
    )

    gt_logits = logpdf_labels(
        xs,
        ys,
        alpha=target_p.alpha,
        beta=target_p.beta,
        gamma=target_p.gamma,
        tau=tau,
    )

    gt_new_labels = jax.random.categorical(
        gt_l_key,
        gt_logits,
        axis=-1,
    )

    #y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y) / (jax.nn.softplus(pref_params["log_tau"]) + 1e-6)
    y2_fn = lambda x, y: y2_learned(pref_params["y2_fn"], x, y)

    task_vis.visualize(
        xs[0, 0, 0],
        ys[0],
        base_manifold=base_p,
        target_manifold=target_p,
        learned_manifold=y2_fn,
        labels=gt_new_labels[0],
        #scale="free"
    )

# %% Retrieve VAE from from data

# Step 1: Pre-training
opt = optax.adam(pre_train_lr)
opt_state = opt.init(vae_params)

loss_history = []

for step in range(pre_train_epochs):
    key, subkey = jax.random.split(key)
    vae_params, opt_state, loss = pre_train_step(
        vae_params, step, opt_state, subkey, xs[:,0], ys[:,0]
    )
    loss_val = float(loss)          # convert from JAX scalar
    loss_history.append(loss_val)

    print(f"[{step+1}/{pre_train_epochs}] -ELBO = {loss_val}")

# Plot loss
plt.figure(figsize=(5, 3), dpi=200)
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("-ELBO")
plt.title("VAE training loss")
plt.grid(True)
plt.tight_layout()
plt.show()

ys = sample_many(vae_params, ys_key, xs[:,0], m)

u = utility_vmapped(
    xs,
    ys,
    target_p.alpha,
    target_p.beta,
    target_p.gamma,
)

mean_u = jnp.mean(u)
gt_u_history.append(mean_u)

logits = logpdf_labels(
    xs,
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

l_history.append(labels)   # labels used for queries

task_vis.visualize(
    xs[0, 0, 0],
    ys[0],
    base_manifold=base_p,
    target_manifold=target_p,
    labels=labels[0],
)

# %% Plot ground truth utility
plt.figure(figsize=(5, 3), dpi=200)
plt.plot(gt_u_history)
plt.xlabel("Iteration")
plt.ylabel("u")
plt.title("Utility vs iteration")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
