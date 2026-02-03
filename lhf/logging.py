from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import jax
import jax.numpy as jnp
import yaml
import matplotlib.figure


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
        self.log_dir.mkdir(parents=True, exist_ok=True)

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
            self._log_pytree(data, path, suffix)
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
    
    def _log_pytree(self, pytree: Dict[str, Any], path: Path, suffix: str) -> None:
        if suffix != ".npz":
            raise ValueError(
                "PyTrees / dicts must be saved with '.npz' extension."
            )

        leaves, treedef = jax.tree_util.tree_flatten(pytree)

        # Convert all leaves to NumPy
        leaves_np = [
            np.asarray(leaf) if isinstance(leaf, (jnp.ndarray, np.ndarray)) else leaf
            for leaf in leaves
        ]

        # Store with stable indexing
        arrays = {f"leaf_{i}": leaf for i, leaf in enumerate(leaves_np)}

        np.savez(path, **arrays)

        # Save tree structure alongside
        with open(path.with_suffix(".treedef.yaml"), "w") as f:
            yaml.safe_dump(str(treedef), f)
