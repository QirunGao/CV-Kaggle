import os
import yaml

# 默认配置文件路径
BASE_CFG = os.path.join(os.path.dirname(__file__), "../configs/default.yaml")
# 覆盖配置，通过环境变量 CFG_FILE 传入；如果不设置则不覆盖
OVR_CFG = os.getenv("CFG_FILE", "")


def _deep_merge(a: dict, b: dict) -> dict:
    """
    递归合并 b 到 a（原地），返回合并后的 a。
    """
    for k, v in b.items():
        # 如果 a[k] 和 v 都是 dict，就递归合并；否则直接覆盖
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


# 1. 加载默认配置
with open(BASE_CFG, "r", encoding="utf-8") as f:
    _raw_cfg = yaml.safe_load(f)

# 2. 如果指定了覆盖配置文件，则加载并合并
if OVR_CFG:
    with open(OVR_CFG, "r", encoding="utf-8") as f:
        _ovr_cfg = yaml.safe_load(f)
    _raw_cfg = _deep_merge(_raw_cfg, _ovr_cfg)


class _CfgDict(dict):
    """允许用点号访问字典属性"""

    def __getattr__(self, name):
        val = self.get(name)
        if isinstance(val, dict):
            return _CfgDict(val)
        return val


# 最终全局 cfg 对象
cfg = _CfgDict(_raw_cfg)
