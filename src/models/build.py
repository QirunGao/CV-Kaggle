import timm
from src.config import cfg


def build_model():
    m = timm.create_model(
        cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1
    )
    return m
