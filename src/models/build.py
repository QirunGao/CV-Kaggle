import timm
from src.config import cfg


def build_model():
    """
    构建图像分类模型。

    - 使用 timm 库创建指定主干网络（例如：resnet, swin, efficientnet 等）；
    - 支持加载预训练权重；
    - 设置输出类别数量与随机 drop 路径率。

    返回
    ----
    一个由 timm 创建的图像分类模型实例。
    """
    model = timm.create_model(
        cfg.model.backbone,           # 主干网络结构，如 'swin_base_patch4_window7_384'
        pretrained=cfg.model.pretrained,  # 是否加载预训练权重
        num_classes=cfg.model.num_classes,  # 最后全连接层输出类别数
        drop_rate=0.0,                # dropout 概率
        drop_path_rate=0.1            # drop path 概率（适用于 transformer）
    )
    return model
