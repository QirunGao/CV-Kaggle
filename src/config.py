import os
import yaml

# Path to the default configuration file: points to configs/default.yaml
BASE_CFG = os.path.join(os.path.dirname(__file__), "../configs/default.yaml")

# Path to the override configuration file: optionally specified via environment variable CFG_FILE
OVR_CFG = os.getenv("CFG_FILE", "")


def _deep_merge(a: dict, b: dict) -> dict:
    """
    Recursively merge dictionary b into dictionary a (in-place), and return the result.

    - If values corresponding to a key are both dictionaries, merge them recursively;
    - Otherwise, overwrite a's value with b's.
    """
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


# Load the default configuration
with open(BASE_CFG, "r", encoding="utf-8") as f:
    _raw_cfg = yaml.safe_load(f)

# If an override config file is specified, load and merge it into the default config
if OVR_CFG:
    with open(OVR_CFG, "r", encoding="utf-8") as f:
        _ovr_cfg = yaml.safe_load(f)
    _raw_cfg = _deep_merge(_raw_cfg, _ovr_cfg)


class _CfgDict(dict):
    """
    A configuration dictionary class that allows attribute-style access.
    Recursively converts nested dictionaries to _CfgDict instances.
    """

    def __getattr__(self, name):
        val = self.get(name)
        if isinstance(val, dict):
            return _CfgDict(val)
        return val


# Create the final config object; nested fields can be accessed via cfg.xxx.yyy
cfg = _CfgDict(_raw_cfg)
