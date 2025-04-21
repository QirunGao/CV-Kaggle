import timm
from src.config import cfg


def build_model():
    """
    Build an image classification model.

    - Uses the `timm` library to create a specified backbone (e.g., ResNet, Swin, EfficientNet, etc.);
    - Supports loading pretrained weights;
    - Sets the number of output classes and the stochastic drop path rate.

    Returns
    -------
    An image classification model instance created by `timm`.
    """
    model = timm.create_model(
        cfg.model.backbone,            # Backbone architecture, e.g., 'swin_base_patch4_window7_384'
        pretrained=cfg.model.pretrained,  # Whether to load pretrained weights
        num_classes=cfg.model.num_classes,  # Number of output classes
        drop_rate=0.0,                 # Dropout probability
        drop_path_rate=0.1             # Drop path probability (for transformer-based models)
    )
    return model
