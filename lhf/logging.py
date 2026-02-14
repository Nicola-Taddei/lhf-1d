from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import jax
import jax.numpy as jnp
import yaml
import matplotlib.figure
import wandb
import shutil

from .serialization import save_flax_pytree

class Logger:
    """
    Simple filesystem logger for experiments.

    Supports:
    - Configuration logging (YAML)
    - NumPy arrays (.npy)
    - Matplotlib figures (.png)
    """

    def __init__(self, log_dir: Union[str, Path], config: Dict[str, Any]):
        """
        Args:
            log_dir: Directory where all logs will be stored.
            config: Configuration dictionary to be saved as config.yaml.
        """
        self.log_dir = Path(log_dir)
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=False)

        self._save_config(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_data(self, data: Any, filename: Union[str, Path]) -> None:
        """
        Log data to disk.

        Supported types:
        - numpy.ndarray -> .npy
        - matplotlib.figure.Figure -> .png

        Args:
            data: Data to log.
            filename: Relative path (may include subfolders) with extension.

        Raises:
            ValueError: If data type or file extension is unsupported.
        """
        path = self.log_dir / Path(filename)

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        suffix = path.suffix.lower()

        if isinstance(data, np.ndarray):
            self._log_numpy(data, path, suffix)
        elif isinstance(data, matplotlib.figure.Figure):
            self._log_figure(data, path, suffix)
        elif isinstance(data, dict):
            self._log_flax_pytree(data, path, suffix)
        else:
            raise ValueError(
                f"Unsupported data type: {type(data)}. "
                "Only numpy.ndarray and matplotlib.figure.Figure are supported."
            )


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _save_config(self, config: Dict[str, Any]) -> None:
        path = self.log_dir / "config.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    def _log_numpy(self, array: np.ndarray, path: Path, suffix: str) -> None:
        if suffix != ".npy":
            raise ValueError(
                f"NumPy arrays must be saved with '.npy' extension, got '{suffix}'."
            )
        np.save(path, array)

    def _log_figure(
        self, fig: matplotlib.figure.Figure, path: Path, suffix: str
    ) -> None:
        if suffix != ".png":
            raise ValueError(
                f"Matplotlib figures must be saved with '.png' extension, got '{suffix}'."
            )
        fig.savefig(path, bbox_inches="tight", dpi=200)
    
    def _log_flax_pytree(self, pytree: Dict[str, Any], path: Path, suffix: str) -> None:
        if suffix != ".flax":
            raise ValueError("Flax PyTrees must be saved with '.flax' extension.")
        save_flax_pytree(path, pytree)



class WandbLogger:
    """
    Filesystem + Weights & Biases logger.

    - Logs everything locally
    - Uses an externally initialized wandb.Run
    - Uploads the entire directory as an artifact
    """

    def __init__(
        self,
        run: wandb.sdk.wandb_run.Run,
        log_dir: Union[str, Path],
        config: Dict[str, Any],
        artifact_name: str,
        artifact_type: str = "experiment",
    ):
        """
        Args:
            run: Already initialized wandb run.
            log_dir: Local logging directory.
            config: Experiment configuration.
            artifact_name: Name of artifact to upload.
            artifact_type: W&B artifact type.
        """

        self.run = run
        self.log_dir = Path(log_dir)
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=False)

        self.artifact_name = artifact_name
        self.artifact_type = artifact_type

        self._save_config(config)

    # ============================================================
    # Public API
    # ============================================================

    def log_data(self, data: Any, filename: Union[str, Path]) -> None:
        """
        Log data locally.
        """

        path = self.log_dir / Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        suffix = path.suffix.lower()

        if isinstance(data, np.ndarray):
            self._log_numpy(data, path, suffix)
        elif isinstance(data, matplotlib.figure.Figure):
            self._log_figure(data, path, suffix)
        elif isinstance(data, dict):
            self._log_flax_pytree(data, path, suffix)
        else:
            raise ValueError(
                f"Unsupported data type: {type(data)}. "
                "Supported: numpy.ndarray, matplotlib.figure.Figure, dict."
            )

    def upload_artifact(self) -> None:
        """
        Create and upload artifact from log directory.
        Does NOT call run.finish().
        """

        artifact = wandb.Artifact(
            name=self.artifact_name,
            type=self.artifact_type,
        )

        artifact.add_dir(str(self.log_dir))
        self.run.log_artifact(artifact)

    # ============================================================
    # Internal helpers
    # ============================================================

    def _save_config(self, config: Dict[str, Any]) -> None:
        path = self.log_dir / "config.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    def _log_numpy(self, array: np.ndarray, path: Path, suffix: str) -> None:
        if suffix != ".npy":
            raise ValueError(
                f"NumPy arrays must use '.npy' extension, got '{suffix}'."
            )
        np.save(path, array)

    def _log_figure(
        self, fig: matplotlib.figure.Figure, path: Path, suffix: str
    ) -> None:
        if suffix != ".png":
            raise ValueError(
                f"Matplotlib figures must use '.png' extension, got '{suffix}'."
            )
        fig.savefig(path, bbox_inches="tight", dpi=200)

    def _log_flax_pytree(self, pytree: Dict[str, Any], path: Path, suffix: str) -> None:
        if suffix != ".flax":
            raise ValueError("Flax PyTrees must be saved with '.flax' extension.")
        save_flax_pytree(path, pytree)

