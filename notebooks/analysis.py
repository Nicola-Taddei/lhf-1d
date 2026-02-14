# %%
from pathlib import Path
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb

from lhf import *

# %%
analysis_config_path = Path("../configs/analysis.yaml")

with analysis_config_path.open("r") as f:
    analysis_cfg = yaml.safe_load(f)

wandb_cfg = analysis_cfg["wandb"]
analysis_params = analysis_cfg["analysis"]
vis_cfg = analysis_cfg["visualization"]

PROJECT = wandb_cfg["project"]
ARTIFACT_NAME = wandb_cfg["artifact_name"]
ARTIFACT_VERSION = wandb_cfg["artifact_version"]
ARTIFACT_TYPE = wandb_cfg["artifact_type"]

ARTIFACT_REF = f"{PROJECT}/{ARTIFACT_NAME}:{ARTIFACT_VERSION}"

ITERATION = analysis_params["iteration"]
VIS_M = analysis_params["vis_m"]
seed = analysis_params["seed"]

# %%
# %%
run = wandb.init(
    project=PROJECT,
    job_type="analysis",
)
artifact = run.use_artifact(
    ARTIFACT_REF,
    type="data"
)
artifact_dir = Path(artifact.download())

print("Artifact downloaded to:", artifact_dir)

# %%
with open(artifact_dir / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)

# %%
iter_folder = artifact_dir / f"iter_{ITERATION}"

vae_params = load_flax_pytree(iter_folder / "improved_vae_params.flax")
pref_params = load_flax_pytree(iter_folder / "pref_params.flax")

print("Loaded VAE and preference model.")

# %%
d_z = config["d_z"]
d_y = config["d_y"]

# Encoder
encoder_mlp = MLP(
    features=config["pre_train_features"],
    output_dim=d_z + (d_z*(d_z+1)) // 2,
)

# Decoder
decoder_mlp = MLP(
    features=config["pre_train_features"],
    output_dim=d_y,
)

gen_model = ConditionalVAE(
    encoder=encoder_mlp,
    decoder=decoder_mlp,
    d_z=d_z,
    d_y=d_y,
)

# Preference model
mlp = MLP(
    features=config["pref_model_features"],
    output_dim=1
)

y2_learned = lambda p, x, y1: mlp.apply(p, jnp.concatenate([x, y1], axis=2))

pref_model = PrefModel(y2_fn=y2_learned)

# %%
key = jax.random.PRNGKey(config["seed"])

n_internal = config["n_internal"]
m = config["m"]

x = jax.random.uniform(key, shape=(n_internal, 1), minval=-1, maxval=1)
xs = jnp.broadcast_to(x[:, None, :], (n_internal, m, 1))


# %%
def sample_many(params, key, x, m):
    keys = jax.random.split(key, m)
    return jax.vmap(
        lambda k: gen_model.sample(
            params,
            x,
            k,
            sigma_y=config["sigma_y_T"],
            deterministic=False,
        ),
        out_axes=1,
    )(keys)

ys = sample_many(vae_params, key, xs[:, 0], VIS_M)


# %%
base_p = TaskParams(**config["base_manifold"])
target_p = TaskParams(**config["target_manifold"])

task_vis = ManifoldVisualizer(ylim=(-1, 5))

fig = task_vis.visualize(
    xs[0, 0, 0],
    ys[0],
    base_manifold=base_p,
    target_manifold=target_p,
    learned_manifold=lambda x, y: y2_learned(
        pref_params["y2_fn"],
        x,
        y
    ),
    scale="free"
)

plt.show()
