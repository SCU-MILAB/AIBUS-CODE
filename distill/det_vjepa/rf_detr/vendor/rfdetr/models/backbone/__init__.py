"""DINO builder is disabled: this integration supplies its own ViT-G joiner."""
from torch import nn
Joiner = nn.Module
def build_backbone(*args, **kwargs):
    raise RuntimeError("Use VJepaRFBackbone; upstream DINO initialization is disabled")
