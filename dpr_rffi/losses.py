from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss for one normalized view per sample."""

    if features.ndim != 2:
        raise ValueError(f"Expected features shape (N, D), got {tuple(features.shape)}.")
    labels = labels.reshape(-1)
    if labels.shape[0] != features.shape[0]:
        raise ValueError("features and labels must have matching first dimension.")
    if features.shape[0] <= 1:
        return features.sum() * 0.0

    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / float(temperature)
    logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()

    device = features.device
    eye = torch.eye(features.shape[0], dtype=torch.bool, device=device)
    label_mask = labels[:, None].eq(labels[None, :])
    positive_mask = label_mask & ~eye
    denominator_mask = ~eye

    exp_logits = torch.exp(logits) * denominator_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return features.sum() * 0.0
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()

