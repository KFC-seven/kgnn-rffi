from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import weibull_min
from torch import nn
from torch.utils.data import DataLoader

from diagnostic.sourceonly import (
    IqDataset,
    make_model,
    normalize_iq,
    set_seed,
    stratified_train_val_split,
)


@dataclass
class CenterLossTrainResult:
    model: nn.Module
    train_indices: np.ndarray
    val_indices: np.ndarray
    best_val_accuracy: float
    best_epoch: int
    trained_epochs: int
    stopped_epoch: int
    early_stopped: bool


@dataclass(frozen=True)
class FdmEvtModel:
    centers: np.ndarray
    weibull_params: list[tuple[float, float, float]]


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.empty(int(num_classes), int(embedding_dim)))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target_centers = self.centers.index_select(0, labels)
        diff = embeddings - target_centers
        return torch.mean(torch.sum(diff * diff, dim=1))


def train_center_loss_source(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    center_lr: float = 5e-4,
    center_loss_weight: float = 0.01,
    val_frac: float = 0.2,
    seed: int = 0,
    embedding_dim: int = 64,
    model_name: str = "tiny",
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    device: str = "cpu",
) -> CenterLossTrainResult:
    set_seed(seed)
    random.seed(int(seed))
    train_idx, val_idx = stratified_train_val_split(y, val_frac=val_frac, seed=seed)
    model = make_model(
        model_name=model_name,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
    ).to(device)
    center_loss = CenterLoss(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
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
    center_optimizer = torch.optim.Adam(center_loss.parameters(), lr=center_lr)
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_epoch = 0
    best_model_state = None
    best_center_state = None
    no_improve_epochs = 0
    trained_epochs = 0
    stopped_epoch = int(epochs)
    patience = max(0, int(early_stop_patience))
    min_delta = max(0.0, float(early_stop_min_delta))
    for epoch in range(1, int(epochs) + 1):
        trained_epochs = int(epoch)
        model.train()
        center_loss.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            center_optimizer.zero_grad(set_to_none=True)
            logits, embeddings = model(xb)
            loss = criterion(logits, yb) + float(center_loss_weight) * center_loss(embeddings, yb)
            loss.backward()
            optimizer.step()
            center_optimizer.step()
        val_acc = evaluate_accuracy(model, val_loader, device=device)
        improved = best_model_state is None or val_acc >= best_val + min_delta
        if improved:
            best_val = val_acc
            best_epoch = int(epoch)
            no_improve_epochs = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_center_state = {k: v.detach().cpu().clone() for k, v in center_loss.state_dict().items()}
        else:
            no_improve_epochs += 1
            if patience > 0 and no_improve_epochs >= patience:
                stopped_epoch = int(epoch)
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_center_state is not None:
        center_loss.load_state_dict(best_center_state)
    return CenterLossTrainResult(
        model=model,
        train_indices=train_idx,
        val_indices=val_idx,
        best_val_accuracy=float(best_val),
        best_epoch=int(best_epoch),
        trained_epochs=int(trained_epochs),
        stopped_epoch=int(stopped_epoch if trained_epochs < int(epochs) else int(epochs)),
        early_stopped=bool(trained_epochs < int(epochs)),
    )


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
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


def infer_logits_embeddings(
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
            emb_parts.append(embeddings.cpu().numpy())
    return (
        np.concatenate(logits_parts, axis=0).astype(np.float32),
        np.concatenate(emb_parts, axis=0).astype(np.float32),
    )


def fit_fdm_evt(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    tail_size: int = 20,
) -> FdmEvtModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    centers = _class_centers(embeddings, labels, num_classes=num_classes)
    params: list[tuple[float, float, float]] = []
    for cls in range(int(num_classes)):
        cls_embeddings = embeddings[labels == cls]
        distances = np.linalg.norm(cls_embeddings - centers[cls][None, :], axis=1)
        tail_count = max(2, min(int(tail_size), distances.size))
        tail = np.sort(distances)[-tail_count:]
        fit = weibull_min.fit(np.maximum(tail, 1e-6), floc=0.0)
        params.append(tuple(float(x) for x in fit))
    return FdmEvtModel(centers=centers.astype(np.float32), weibull_params=params)


def fdm_evt_unknown_score(
    embeddings: np.ndarray,
    model: FdmEvtModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    distances = np.linalg.norm(embeddings[:, None, :] - model.centers[None, :, :], axis=2)
    pred = np.argmin(distances, axis=1).astype(np.int64)
    scores = np.empty(embeddings.shape[0], dtype=np.float32)
    for idx, cls in enumerate(pred.tolist()):
        shape, loc, scale = model.weibull_params[int(cls)]
        scores[idx] = float(weibull_min.cdf(max(distances[idx, cls], 1e-6), shape, loc=loc, scale=scale))
    return scores.astype(np.float32), pred


def _class_centers(embeddings: np.ndarray, labels: np.ndarray, *, num_classes: int) -> np.ndarray:
    centers = []
    for cls in range(int(num_classes)):
        mask = labels == cls
        if not np.any(mask):
            raise ValueError(f"Cannot fit center for class {cls}: no samples.")
        centers.append(np.mean(embeddings[mask], axis=0))
    return np.stack(centers, axis=0).astype(np.float32)
