import timm
from src.config import cfg
from src.models.dual_head import DualHead


def build_model():
    """
    Create and return an image classification model based on configuration.

    - Supports a dual-head architecture for separate binary and severity outputs;
    - Otherwise builds a single-head classifier with a chosen backbone from timm;
    - Allows loading of pretrained weights, setting output classes, and adjusting dropout rates.

    Returns
    -------
    A PyTorch model instance configured for the specified task.
    """
    # Choose dual-head model if configured
    if cfg.model.get("type", "single") == "dual_head":
        return DualHead(
            backbone_name=cfg.model.backbone,  # Backbone architecture name
            num_classes_bin=2,  # Binary classification head outputs (disease/no disease)
            num_classes_sev=4,  # Severity classification head outputs (levels 1-4)
            pretrained=cfg.model.pretrained,  # Load pretrained backbone weights
        )

    # Build a standard single-head classifier via timm
    model = timm.create_model(
        cfg.model.backbone,  # Backbone architecture, e.g., 'swin_base_patch4_window7_384'
        pretrained=cfg.model.pretrained,  # Load pretrained weights if True
        num_classes=cfg.model.num_classes,  # Total number of output classes
        drop_rate=cfg.model.get("drop_rate", 0.0),  # Dropout probability before final layer
        drop_path_rate=cfg.model.get("drop_path_rate", 0.1),  # Stochastic depth rate for transformer models
    )
    return model
