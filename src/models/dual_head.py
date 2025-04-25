import torch.nn as nn
import timm


class DualHead(nn.Module):
    """
    Dual-head image classifier:
    - Shared backbone for feature extraction;
    - Binary head for disease detection (disease vs. no disease);
    - Severity head for classifying disease severity into 4 levels (applied only if disease present).
    """

    def __init__(
            self,
            backbone_name: str,
            num_classes_bin: int = 2,
            num_classes_sev: int = 4,
            pretrained: bool = True,
    ):
        super().__init__()
        # Initialize backbone without its original classification head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # remove default head to use custom heads
        )
        in_features = self.backbone.num_features

        # Binary classification head: disease vs. no disease
        self.head_bin = nn.Linear(in_features, num_classes_bin)
        # Severity classification head: 4-level severity (computed on diseased samples only)
        self.head_sev = nn.Linear(in_features, num_classes_sev)

    def forward(self, x):
        # Extract features from backbone
        feats = self.backbone(x)
        # Compute logits for binary disease detection
        logit_bin = self.head_bin(feats)
        # Compute logits for disease severity levels
        logit_sev = self.head_sev(feats)
        return logit_bin, logit_sev
