"""
CLIP Intermediate Layer Feature Extraction using PyTorch Hooks

Extract intermediate layer features from CLIP models without modifying source code.

Usage:
    from utils.clip_hooks import IntermediateLayerExtractor

    extractor = IntermediateLayerExtractor(model, layer_indices=[1, 7, 10])
    with torch.no_grad():
        features = model.encode_image(images)
    intermediate_features = extractor.get_features()
    extractor.remove_hooks()
"""

import torch
import torch.nn as nn


class IntermediateLayerExtractor:
    """
    Extract intermediate layer features from CLIP visual encoder using PyTorch hooks.

    Attributes:
        model: CLIP model instance
        layer_indices: List of layer indices to extract (1-indexed)
        features: Dictionary storing captured features
        hooks: List of hook handles
    """

    def __init__(self, model, layer_indices):
        """
        Initialize intermediate layer feature extractor.

        Args:
            model: CLIP model instance
            layer_indices: List of layer indices to extract (1-indexed, e.g., [1, 7, 10])
        """
        self.model = model
        self.layer_indices = layer_indices
        self.features = {idx: [] for idx in layer_indices}
        self.hooks = []
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for specified layers."""
        for idx, resblock in enumerate(self.model.visual.transformer.resblocks, 1):
            if idx in self.layer_indices:
                handle = resblock.register_forward_hook(self._create_hook_fn(idx))
                self.hooks.append(handle)

    def _create_hook_fn(self, layer_idx):
        """Create hook function to capture layer output."""
        def hook_fn(module, input, output):
            # output shape: [seq_len, batch_size, hidden_dim], e.g., [197, 32, 768] for ViT-B/16
            feature = output.clone().detach()
            feature = feature.permute(1, 2, 0)  # [batch_size, hidden_dim, seq_len]
            pooled_feature = self.adaptive_pool(feature).squeeze(-1)  # [batch_size, hidden_dim]
            self.features[layer_idx].append(pooled_feature)

        return hook_fn

    def get_features(self):
        """Get captured intermediate layer features."""
        return self.features

    def clear_features(self):
        """Clear captured features for next forward pass."""
        self.features = {idx: [] for idx in self.layer_indices}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __del__(self):
        """Destructor to ensure hooks are removed."""
        self.remove_hooks()


def extract_intermediate_features_with_hooks(model, images, layer_indices):
    """
    Extract intermediate features from CLIP using hooks (convenience function).

    Args:
        model: CLIP model instance
        images: Input image tensor
        layer_indices: List of layer indices to extract (1-indexed)

    Returns:
        image_features: Final layer image features
        intermediate_features: Dict mapping layer indices to feature lists

    Example:
        >>> image_features, intermediate = extract_intermediate_features_with_hooks(
        ...     model, images, [1, 7, 10]
        ... )
        >>> print(intermediate[7][0].shape)
        torch.Size([32, 768])
    """
    extractor = IntermediateLayerExtractor(model, layer_indices)

    with torch.no_grad():
        image_features = model.encode_image(images)

    intermediate_features = extractor.get_features()
    extractor.remove_hooks()

    return image_features, intermediate_features
