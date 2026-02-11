from collections.abc import Mapping
from copy import deepcopy


def merge_configs(base_cfg: dict, run_cfg: Mapping) -> dict:
    """
    Recursively overwrite base_cfg with values from run_cfg.

    Args:
        base_cfg: dict loaded from YAML (authoritative defaults)
        run_cfg: mapping (e.g. wandb.config) with override values

    Returns:
        A new dict containing the merged configuration.
    """
    merged = deepcopy(base_cfg)

    def _merge(dst, src):
        for k, v in src.items():
            if isinstance(v, Mapping) and k in dst and isinstance(dst[k], Mapping):
                _merge(dst[k], v)
            else:
                dst[k] = v

    _merge(merged, dict(run_cfg))
    return merged
