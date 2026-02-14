from pathlib import Path
from typing import Any
from flax.serialization import to_bytes, from_bytes


def save_flax_pytree(path: Path, pytree: Any) -> None:
    """
    Save a Flax/JAX PyTree to disk using Flax serialization.

    Args:
        path: Target file path (should end with .flax)
        pytree: Arbitrary nested PyTree (e.g. model parameters)
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix != ".flax":
        raise ValueError("Flax checkpoints must use '.flax' extension.")

    bytes_data = to_bytes(pytree)

    with open(path, "wb") as f:
        f.write(bytes_data)


def load_flax_pytree(path: Path) -> Any:
    """
    Load a Flax/JAX PyTree from disk.

    Args:
        path: Path to .flax file

    Returns:
        Restored PyTree
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, "rb") as f:
        bytes_data = f.read()

    # Important: pass None as target to allow full reconstruction
    pytree = from_bytes(None, bytes_data)

    return pytree
