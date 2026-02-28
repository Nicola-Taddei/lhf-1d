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

path = Path("../configs/config.yaml")
#path = Path("configs/config.yaml")
with path.open("r") as f:
    config = yaml.safe_load(f)

interactive = config.get("interactive", False)

if not interactive:
    import matplotlib
    matplotlib.use("Agg")

wandb_flag = config.get("use_wandb", False)

if wandb_flag:
    run = wandb.init(
        project="traj-alignment",
        name=config.get("run_name", None),
        config=config,
        mode="online",  # "offline"
    )
    config = dict(wandb.config)

    log_dir = Path("../logs") / run.id
    logger = WandbLogger(
        run,
        str(log_dir),
        config,
        "data",
        "data"
    )
else:
    logger = Logger(
        log_dir="../logs/new_run",
        config=config
    )

key = jax.random.PRNGKey(config["seed"])

# %% Visualizer
vis = TrajectoryVisualizer()

# %% Create data
N = config["N"]
T = config["T"]
scale = config["scale"]
v_eps = config["v_eps"]
lengthscale = config["lengthscale"]
n_queries = config["n_queries"]
n_internal = config["n_internal"]
tau = config["tau"]

(
    key,
    x_key,
    y_key,
    l_key
) = jax.random.split(key, 4)

traj = procedural_traj(
    y_key,
    N,
    T,
    n_internal,
    scale=scale,
    v_eps=v_eps,
    lengthscale=lengthscale
)
x_0, ys_traj, x_T = traj[:,:,0,:], traj[:,:,1:-1,:], traj[:,:,-1,:]
xs = to_x_cond(x_0, x_T)
ys = to_y_out(ys_traj)

logger.log_data(np.array(xs), "xs.npy")
logger.log_data(np.array(ys), "ys.npy")

gt_logits = logpdf_labels_traj(
    traj,
    tau=tau
)

# Sample labels
gt_labels = jax.random.categorical(
    l_key,
    gt_logits,
    axis=-1
)

for i in range(10):
    fig = vis.visualize(
        traj[i],
        labels=gt_labels[i]
    )
    logger.log_data(fig, Path("base") / f"samples_{i}.png")


# %% Training loop: Initialization

num_iter = config["num_iter"]

# Logging:
gt_u_history = []
traj_buffer = None
labels_buffer = None

# Step 1: Pre-training
pre_train_epochs = config["pre_train_epochs"]
pre_train_lr = config["pre_train_lr"]
clip_norm = config["clip_norm"]
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
d_y= N * (T-2) * 2

encoder_mlp = MLP(
    features=pre_train_features,
    #output_dim=d_z + (d_z*(d_z+1)) // 2,
    output_dim=2*d_z,
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
x_batch = xs[:pre_train_batch_dim]  # (B,d_x)
y_batch = ys[:pre_train_batch_dim]  # (B,d_y)

xz_batch = jnp.concatenate(
    [x_batch, jnp.zeros((pre_train_batch_dim, d_z))],
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
logger.log_data(vae_params, "base/vae_params.flax")

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

w = config["w"]
traj_utility_model = TrajectoryAttentionCNN(
    pre_conv_channels=[w, w, w],
    pre_conv_kernel=config["pre_conv_kernel"],

    # Agent attention block parameters
    num_agent_attn_layers=config["num_agent_attn_layers"],
    num_heads=config["num_heads"],
    attn_hidden_dim=w,
    attn_mlp_dim=w,
    attn_out_dim=w,

    # Temporal convolutions after pooling
    post_conv_channels=[w, w],
    post_conv_kernel=config["post_conv_kernel"],
)

learned_traj_utility = lambda params, traj: traj_utility_model.apply(params, traj)

traj_batch = jnp.zeros((pref_batch_dim, N, T, 2))
u_params = traj_utility_model.init(init_key, traj_batch)

if config["learning_tau"]:
    pref_params = {
        "u_params": u_params,
        "log_tau": jnp.array(1.0)
    }
else:
    pref_params = {
        "u_params": u_params,
    }

logger.log_data(pref_params, "base/pref_params.flax")

pref_model = PrefModel(
    utility_fn=learned_traj_utility
)

def pref_nll(params, traj, labels):
    logpdf = pref_model.logpdf(params, traj, tau=tau)      # (B, T, 2)

    # Gather log-probabilities of the observed labels
    logp = jnp.take_along_axis(
        logpdf,
        labels[..., None],     # (B, T, 1)
        axis=-1,
    )[..., 0]                  # (B, T)

    return -jnp.mean(logp)

@jax.jit
def pref_train_step(params, opt_state, traj, labels):
    loss, grads = jax.value_and_grad(pref_nll)(
        params,
        traj,
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

def avg_u(params, key, u_params, x, sigma_y):
    y_out = gen_model.sample(
        params,
        x,
        key,
        sigma_y=sigma_y,
        deterministic=False
    )      # (B, N*(T-2)*2)

    y = from_y_out(y_out, N, T)

    x_0, x_T = from_x_cond(x, N)

    traj = assemble_traj(x_0, y, x_T)

    u = learned_traj_utility(u_params, traj)

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

def disp(params, x, key, n_z=8):
    d = gen_model.dispersion(
        params,
        x,
        key,
        n_z=n_z
    )
    return jnp.mean(d)

@jax.jit
def align_loss(params, base_vae_params, u_params, key, x, sigma_y):
    u_bar = avg_u(
        params,
        key, 
        u_params,
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
    if config["creativity-incentive"] == "max-var":
        gamma = config["gamma"]
        dispersion = disp(
            params,
            x,
            key,
            n_z=config["disp_samples"]
        )
        loss = -u_bar + beta * kl - gamma * jnp.log(dispersion)
    elif config["creativity-incentive"] == "var-target":
        gamma = config["gamma"]
        dispersion = disp(
            params,
            x,
            key,
            n_z=config["disp_samples"]
        )
        loss = -u_bar + beta *kl + gamma * (dispersion - config["var-target"])**2
    else:
        loss = -u_bar + beta * kl

    return loss

@jax.jit
def align_train_step(params, base_vae_params, u_params, opt_state, key, x, sigma_y):
    loss, grads = jax.value_and_grad(align_loss)(
        params,
        base_vae_params,
        u_params,
        key,
        x,
        sigma_y
    )
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

global_step = 0


# %% Training loop: Iterations

for iter in range(num_iter):
    log_folder = Path(f"iter_{iter}")
    # Step 1: Pre-training
    opt = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(pre_train_lr),
    )
    opt_state = opt.init(vae_params)

    vae_params_best = vae_params
    key, subkey = jax.random.split(key)
    best_loss = float(pre_train_loss_fn(
        vae_params_best,
        0,
        subkey,
        xs,
        jax.lax.stop_gradient(ys)
    ))

    loss_history = []

    for step in range(pre_train_epochs):
        key, subkey = jax.random.split(key)
        # TODO: We are not using batches anymore. Decide what to do
        vae_params, opt_state, loss = pre_train_step(
            vae_params, step, opt_state, subkey, xs, jax.lax.stop_gradient(ys)
        )
        loss_val = float(loss)          # convert from JAX scalar
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            vae_params_best = vae_params

        if wandb_flag:
            wandb.log(
                {
                    f"iter_{iter}/elbo" : -loss_val
                },
                step=global_step
            )
            global_step += 1
        print(f"[{step+1}/{pre_train_epochs}] -ELBO = {loss_val}")

    vae_params = vae_params_best
    logger.log_data(vae_params, log_folder / "vae_params.flax")

    # Plot loss
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("-ELBO")
    plt.title("VAE training loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    logger.log_data(fig, log_folder / "vae_pre_training_loss.png")

    ys = gen_model.sample(
        vae_params,
        xs,
        ys_key
    )
    traj = assemble_traj(
        x_0,
        from_y_out(ys, N, T),
        x_T
    )
    # TODO: Improve query selection method: random, acquisition function, ...
    traj_query = traj[:n_queries]

    gt_logits = logpdf_labels_traj(
        traj,
        tau=tau,
    )
    gt_labels = jax.random.categorical(
        l_key,
        gt_logits,
        axis=-1,
    )
    # TODO: Replicate here the query selection process
    # used for trajectories (if it changes)
    ls = gt_labels[:n_queries]

    if traj_buffer is None:
        traj_buffer = traj_query
        labels_buffer = ls
    else:
        traj_buffer = jnp.concatenate(
            [traj_buffer, traj_query],
            axis=0
        )
        labels_buffer = jnp.concatenate(
            [labels_buffer, ls],
            axis=0
        )

    var = disp(
        vae_params,
        xs,
        ys_key,
        n_z=config["disp_samples"]
    )

    if wandb_flag:
        wandb.log(
            {
                "variance": float(var)
            },
            step=global_step
        )

    u = trajectory_utility(traj)

    for i in range(10):
        fig = vis.visualize(
            traj[i],
            scale="free"
        )
        logger.log_data(fig, log_folder / f"pre_trained_samples_{i}.png")

    mean_u = jnp.mean(u)
    gt_u_history.append(float(mean_u))
    if wandb_flag:
        wandb.log(
            {
                "gt_u": float(mean_u)
            },
            step=global_step
        )


    # Step 2: Learn preference model
    opt = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(pref_lr),
    )
    opt_state = opt.init(pref_params)

    loss_history = []

    n_tot = traj_buffer.shape[0]
    for step in range(pref_train_epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(
            subkey,
            (pref_batch_dim,),
            minval=0,
            maxval=n_tot
        )
        traj_batch = traj_buffer[idx]
        l_batch = labels_buffer[idx]
        pref_params, opt_state, loss = pref_train_step(
            pref_params,
            opt_state,
            traj_batch,        # (pref_batch_dim, N, T, 2)
            l_batch,   # (pref_batch_dim, T)
        )

        loss_val = float(loss)
        loss_history.append(loss_val)

        if wandb_flag:
            wandb.log(
                {
                    f"iter_{iter}/nll" : loss_val
                },
                step=global_step
            )
            global_step += 1
        print(f"[{step+1}/{pref_train_epochs}] NLL = {loss_val:.4f}")

    logger.log_data(pref_params, log_folder / "pref_params.flax")

    # Plot loss
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("nll")
    plt.title("Pref model loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    logger.log_data(fig, log_folder / "pref_training_loss.png")

    pred_logits = pref_model.logpdf(
        pref_params,
        traj,
        tau=tau
    )

    pred_labels = jax.random.categorical(
        l_key,
        pred_logits,
        axis=-1,
    )

    gt_logits = logpdf_labels_traj(
        traj,
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

    # Step 3: Improve VAE
    base_vae_params = jax.lax.stop_gradient(vae_params)

    opt = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(align_lr),
    )
    opt_state = opt.init(vae_params)

    loss_history = []
    u_history = []
    kl_history = []
    disp_history = []

    for step in range(align_epochs):
        key, subkey = jax.random.split(key, 2)
        # TODO: Again, we are using the whole dataset instead of minibatches
        vae_params, opt_state, loss = align_train_step(
            vae_params,
            base_vae_params,
            pref_params["u_params"],
            opt_state,
            subkey,
            xs,        # (B, N*2*2)
            sigma_y_T
        )

        loss_val = float(loss)
        u_val = float(
            avg_u(
                vae_params,
                key,
                pref_params["u_params"],
                xs,
                sigma_y_T
            )
        )
        kl_val = float(
            kl_div(
                vae_params,
                base_vae_params,
                key,
                xs,
                sigma_y_T
            )
        )
        disp_val = float(
            disp(
                vae_params,
                xs,
                key,
                n_z=config["disp_samples"]
            )
        )
        loss_history.append(loss_val)
        u_history.append(u_val)
        kl_history.append(kl_val)
        disp_history.append(disp_val)

        if wandb_flag:
            wandb.log(
                {
                    f"iter_{iter}/align_loss" : loss_val,
                    f"iter_{iter}/u_hat" : u_val,
                    f"iter_{iter}/kl" : kl_val,
                    f"iter_{iter}/disp" : disp_val,
                },
                step=global_step
            )
            global_step += 1

        print(f"[{step+1}/{align_epochs}] loss = {loss_val:.4f}")

    logger.log_data(vae_params, log_folder / "improved_vae_params.flax")

    # Plot loss
    loss_fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("loss")
    plt.title("Alignement loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot utility
    u_fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(u_history)
    plt.xlabel("Training step")
    plt.ylabel("avg_u")
    plt.title("Average utility")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot KL divergence
    d_kl_fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(kl_history)
    plt.xlabel("Training step")
    plt.ylabel("kl_div")
    plt.title("KL divergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot dispersion
    disp_fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(disp_history)
    plt.xlabel("Training step")
    plt.ylabel("disp")
    plt.title("Dispersion")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    logger.log_data(loss_fig, log_folder / "vae_alignement_loss.png")
    logger.log_data(u_fig, log_folder / "align_u.png")
    logger.log_data(d_kl_fig, log_folder / "align_d_kl.png")
    logger.log_data(disp_fig, log_folder / "align_disp.png")

    (
        key,
        y_key,
        l_key,
        gt_l_key
    ) = jax.random.split(key, 4)

    ys = gen_model.sample(
        vae_params,
        xs,
        y_key
    )

    traj = assemble_traj(
        x_0,
        from_y_out(ys, N, T),
        x_T
    )

    gt_logits = logpdf_labels_traj(
        traj,
        tau=tau
    )

    gt_new_labels = jax.random.categorical(
        gt_l_key,
        gt_logits,
        axis=-1,
    )

    for i in range(10):
        fig = vis.visualize(
            traj[i],
            labels=gt_new_labels[i],
            scale="free"
        )

    logger.log_data(fig, log_folder / f"aligned_samples_{i}.png")

# %% Retrieve VAE from data

# Step 1: Pre-training
opt = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.adam(pre_train_lr),
)
opt_state = opt.init(vae_params)

loss_history = []

for step in range(pre_train_epochs):
    key, subkey = jax.random.split(key)
    vae_params, opt_state, loss = pre_train_step(
        vae_params, step, opt_state, subkey, xs, ys
    )
    loss_val = float(loss)          # convert from JAX scalar
    loss_history.append(loss_val)

    if wandb_flag:
        wandb.log(
            {
                f"final/elbo" : -loss_val,
            },
            step=global_step
        )
        global_step += 1
    print(f"[{step+1}/{pre_train_epochs}] -ELBO = {loss_val}")

logger.log_data(vae_params, Path("final") / "vae_params.flax")

# Plot loss
fig = plt.figure(figsize=(5, 3), dpi=200)
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("-ELBO")
plt.title("VAE training loss")
plt.grid(True)
plt.tight_layout()
plt.show()
logger.log_data(fig, Path("final") / "elbo.png")

ys = gen_model.sample(
    vae_params,
    xs,
    y_key
)

traj = assemble_traj(
    x_0,
    from_y_out(ys, N, T),
    x_T
)

u = trajectory_utility(traj)

var = disp(
    vae_params,
    xs,
    ys_key,
    n_z=config["disp_samples"]
)

mean_u = jnp.mean(u)
gt_u_history.append(float(mean_u))
if wandb_flag:
    wandb.log(
        {
            "gt_u": float(mean_u)
        },
        step=global_step
    )

gt_logits = logpdf_labels_traj(
    traj,
    tau,
)

gt_labels = jax.random.categorical(
    l_key,
    gt_logits,
    axis=-1,
)

for i in range(10):
    fig = vis.visualize(
        traj[i],
        labels=gt_labels[i],
        scale="free"
    )
    logger.log_data(fig, Path("final") / f"samples_{i}.png")

# %% Plot ground truth utility
fig = plt.figure(figsize=(5, 3), dpi=200)
plt.plot(gt_u_history)
plt.xlabel("Iteration")
plt.ylabel("u")
plt.title("Utility vs iteration")
plt.grid(True)
plt.tight_layout()
plt.show()
logger.log_data(fig, "gt_u_vs_iter.png")

# %% Print stuff

print("gt_u_history = ", gt_u_history)

if wandb_flag:
    wandb.log(
        {
            "delta_gt_u" : gt_u_history[-1] - gt_u_history[0],
            "final_variance": float(var),
            "variance": float(var)
        },
        step=global_step
    )

if wandb_flag:
    logger.upload_artifact()
