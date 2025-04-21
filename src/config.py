import os
import yaml

# 默认配置文件路径：指向 configs/default.yaml
BASE_CFG = os.path.join(os.path.dirname(__file__), "../configs/default.yaml")

# 覆盖配置文件路径：通过环境变量 CFG_FILE 指定（可选）
OVR_CFG = os.getenv("CFG_FILE", "")


def _deep_merge(a: dict, b: dict) -> dict:
    """
    深度合并字典 b 到字典 a 中（原地操作），返回合并后的结果。

    - 若对应键的值均为字典，则递归合并；
    - 否则以 b 中的值覆盖 a。
    """
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


# 加载默认配置
with open(BASE_CFG, "r", encoding="utf-8") as f:
    _raw_cfg = yaml.safe_load(f)

# 若指定了覆盖配置文件，则读取后合并到默认配置
if OVR_CFG:
    with open(OVR_CFG, "r", encoding="utf-8") as f:
        _ovr_cfg = yaml.safe_load(f)
    _raw_cfg = _deep_merge(_raw_cfg, _ovr_cfg)


class _CfgDict(dict):
    """
    可使用属性方式访问的配置字典类。
    支持递归转换内部嵌套字典为 _CfgDict 实例。
    """

    def __getattr__(self, name):
        val = self.get(name)
        if isinstance(val, dict):
            return _CfgDict(val)
        return val


# 构造最终配置对象，可通过 cfg.xxx.yyy 访问嵌套字段
cfg = _CfgDict(_raw_cfg)
