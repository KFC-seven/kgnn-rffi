from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dpr_rffi.training import (
    IqDataset,
    make_model,
    normalize_iq,
    set_seed,
    stratified_train_val_split,
)


@dataclass
class HyperRsiTrainResult:
    model: HyperRsiModel
    train_indices: np.ndarray
    val_indices: np.ndarray
    best_val_accuracy: float
    best_epoch: int
    trained_epochs: int
    stopped_epoch: int
    early_stopped: bool


class CosineMarginHead(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, radius: float = 8.0, margin: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(num_classes), int(embedding_dim)))
        nn.init.xavier_uniform_(self.weight)
        self.radius = float(radius)
        self.margin = float(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        features = nn.functional.normalize(embeddings, p=2, dim=1)
        weights = nn.functional.normalize(self.weight, p=2, dim=1)
        cosine = features @ weights.t()
        if labels is not None:
            margin = torch.zeros_like(cosine)
            margin.scatter_(1, labels.view(-1, 1), self.margin)
            cosine = cosine - margin
        return cosine * self.radius


class HyperRsiModel(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        embedding_dim: int,
        radius: float = 8.0,
        margin: float = 0.2,
    ):
        super().__init__()
        self.backbone = make_model(
            model_name=model_name,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )
        self.head = CosineMarginHead(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            radius=radius,
            margin=margin,
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _, embeddings = self.backbone(x)
        logits = self.head(embeddings, labels=labels)
        return logits, embeddings


def train_hyperrsi(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 0,
    embedding_dim: int = 64,
    model_name: str = "tiny",
    radius: float = 8.0,
    margin: float = 0.2,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    device: str = "cpu",
) -> HyperRsiTrainResult:
    set_seed(seed)
    random.seed(int(seed))
    train_idx, val_idx = stratified_train_val_split(y, val_frac=val_frac, seed=seed)
    model = HyperRsiModel(
        model_name=model_name,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        radius=radius,
        margin=margin,
    ).to(device)
    train_loader = DataLoader(
        IqDataset(x[train_idx], y[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        IqDataset(x[val_idx], y[val_idx]),
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_epoch = 0
    best_state = None
    no_improve_epochs = 0
    trained_epochs = 0
    stopped_epoch = int(epochs)
    patience = max(0, int(early_stop_patience))
    min_delta = max(0.0, float(early_stop_min_delta))
    for epoch in range(1, int(epochs) + 1):
        trained_epochs = int(epoch)
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb, labels=yb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        val_acc = evaluate_hyperrsi_accuracy(model, val_loader, device=device)
        improved = best_state is None or val_acc >= best_val + min_delta
        if improved:
            best_val = val_acc
            best_epoch = int(epoch)
            no_improve_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve_epochs += 1
            if patience > 0 and no_improve_epochs >= patience:
                stopped_epoch = int(epoch)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return HyperRsiTrainResult(
        model=model,
        train_indices=train_idx,
        val_indices=val_idx,
        best_val_accuracy=float(best_val),
        best_epoch=int(best_epoch),
        trained_epochs=int(trained_epochs),
        stopped_epoch=int(stopped_epoch if trained_epochs < int(epochs) else int(epochs)),
        early_stopped=bool(trained_epochs < int(epochs)),
    )


def evaluate_hyperrsi_accuracy(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits, _ = model(xb.to(device))
            pred = torch.argmax(logits.cpu(), dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.shape[0])
    return float(correct) / float(total) if total else float("nan")


def infer_hyperrsi_logits_embeddings(
    model: nn.Module,
    x: np.ndarray,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        torch.from_numpy(normalize_iq(x).astype(np.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    logits_parts = []
    emb_parts = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            logits, embeddings = model(xb.to(device))
            logits_parts.append(logits.cpu().numpy())
            norm_embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            emb_parts.append(norm_embeddings.cpu().numpy())
    return (
        np.concatenate(logits_parts, axis=0).astype(np.float32),
        np.concatenate(emb_parts, axis=0).astype(np.float32),
    )
