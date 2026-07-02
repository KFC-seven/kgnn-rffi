from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.svm import OneClassSVM, SVC
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
class OpenSvddTrainResult:
    model: OpenSvddArplModel
    train_indices: np.ndarray
    val_indices: np.ndarray
    best_val_accuracy: float
    best_epoch: int
    trained_epochs: int
    stopped_epoch: int
    early_stopped: bool


@dataclass(frozen=True)
class OpenSvddOneClassModel:
    models: list[OneClassSVM]
    scales: np.ndarray


@dataclass(frozen=True)
class OpenSvddClosestSvmModel:
    models: list[SVC]
    scales: np.ndarray
    closest_classes: np.ndarray


@dataclass(frozen=True)
class OpenSvddClosestUnionModel:
    models: list[OneClassSVM]
    scales: np.ndarray
    closest_classes: np.ndarray


class ReciprocalPointHead(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.reciprocal_points = nn.Parameter(torch.empty(int(num_classes), int(embedding_dim)))
        self.radius_logits = nn.Parameter(torch.zeros(int(num_classes)))
        nn.init.xavier_uniform_(self.reciprocal_points)

    def distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        dim = max(1, embeddings.shape[1])
        diff = embeddings[:, None, :] - self.reciprocal_points[None, :, :]
        euclidean = torch.sum(diff * diff, dim=2) / float(dim)
        cosine_like = embeddings @ self.reciprocal_points.t()
        return euclidean - cosine_like

    def radii(self) -> torch.Tensor:
        return nn.functional.softplus(self.radius_logits) + 1e-6

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.distances(embeddings)


class OpenSvddArplModel(nn.Module):
    def __init__(self, *, model_name: str, num_classes: int, embedding_dim: int):
        super().__init__()
        self.backbone = make_model(
            model_name=model_name,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )
        self.head = ReciprocalPointHead(num_classes=num_classes, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, embeddings = self.backbone(x)
        distances = self.head(embeddings)
        return distances, embeddings


def train_opensvdd_arpl(
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
    open_loss_weight: float = 0.1,
    radius_penalty_weight: float = 1e-4,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    device: str = "cpu",
) -> OpenSvddTrainResult:
    set_seed(seed)
    random.seed(int(seed))
    train_idx, val_idx = stratified_train_val_split(y, val_frac=val_frac, seed=seed)
    model = OpenSvddArplModel(
        model_name=model_name,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
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
            distances, _ = model(xb)
            class_loss = criterion(distances, yb)
            target_distances = distances.gather(1, yb.view(-1, 1)).squeeze(1)
            target_radii = model.head.radii().gather(0, yb)
            open_loss = torch.relu(target_distances - target_radii).mean()
            radius_penalty = model.head.radii().mean()
            loss = class_loss + float(open_loss_weight) * open_loss + float(radius_penalty_weight) * radius_penalty
            loss.backward()
            optimizer.step()
        val_acc = evaluate_opensvdd_accuracy(model, val_loader, device=device)
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
    return OpenSvddTrainResult(
        model=model,
        train_indices=train_idx,
        val_indices=val_idx,
        best_val_accuracy=float(best_val),
        best_epoch=int(best_epoch),
        trained_epochs=int(trained_epochs),
        stopped_epoch=int(stopped_epoch if trained_epochs < int(epochs) else int(epochs)),
        early_stopped=bool(trained_epochs < int(epochs)),
    )


def evaluate_opensvdd_accuracy(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            distances, _ = model(xb.to(device))
            pred = torch.argmax(distances.cpu(), dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.shape[0])
    return float(correct) / float(total) if total else float("nan")


def infer_opensvdd_logits_embeddings(
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
            distances, embeddings = model(xb.to(device))
            logits_parts.append(distances.cpu().numpy())
            emb_parts.append(embeddings.cpu().numpy())
    return (
        np.concatenate(logits_parts, axis=0).astype(np.float32),
        np.concatenate(emb_parts, axis=0).astype(np.float32),
    )


def fit_opensvdd_oneclass(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    nu: float = 0.03,
    gamma: str | float = "scale",
) -> OpenSvddOneClassModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    models: list[OneClassSVM] = []
    scales = []
    for cls in range(int(num_classes)):
        cls_embeddings = embeddings[labels == cls]
        if cls_embeddings.size == 0:
            raise ValueError(f"Cannot fit OpenSVDD one-class boundary for class {cls}: no samples.")
        model = OneClassSVM(kernel="rbf", nu=max(float(nu), 1e-4), gamma=gamma)
        model.fit(cls_embeddings)
        decision = model.decision_function(cls_embeddings).reshape(-1)
        scale = float(np.quantile(np.maximum(decision, 1e-6), 0.95))
        models.append(model)
        scales.append(max(scale, 1e-6))
    return OpenSvddOneClassModel(models=models, scales=np.asarray(scales, dtype=np.float32))


def opensvdd_oneclass_unknown_score(
    embeddings: np.ndarray,
    model: OpenSvddOneClassModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    decisions = np.stack(
        [boundary.decision_function(embeddings).reshape(-1) for boundary in model.models],
        axis=1,
    )
    normalized = decisions / model.scales.reshape(1, -1)
    pred = np.argmax(normalized, axis=1).astype(np.int64)
    return (-np.max(normalized, axis=1)).astype(np.float32), pred


def fit_opensvdd_closest_union_ocsvm(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    nu: float = 0.03,
    gamma: str | float = "scale",
) -> OpenSvddClosestUnionModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    centers = _class_centers(embeddings, labels, num_classes=num_classes)
    closest_classes = _closest_classes(centers)
    models: list[OneClassSVM] = []
    scales = []
    for cls, closest in enumerate(closest_classes.tolist()):
        fit_mask = (labels == cls) | (labels == int(closest))
        fit_embeddings = embeddings[fit_mask]
        if fit_embeddings.size == 0:
            raise ValueError(f"Cannot fit OpenSVDD OVC union boundary for class {cls}: no samples.")
        model = OneClassSVM(kernel="rbf", nu=max(float(nu), 1e-4), gamma=gamma)
        model.fit(fit_embeddings)
        decision = model.decision_function(embeddings[labels == cls]).reshape(-1)
        scale = float(np.quantile(np.maximum(decision, 1e-6), 0.95))
        models.append(model)
        scales.append(max(scale, 1e-6))
    return OpenSvddClosestUnionModel(
        models=models,
        scales=np.asarray(scales, dtype=np.float32),
        closest_classes=closest_classes.astype(np.int64),
    )


def opensvdd_closest_union_unknown_score(
    embeddings: np.ndarray,
    model: OpenSvddClosestUnionModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    decisions = np.stack(
        [boundary.decision_function(embeddings).reshape(-1) for boundary in model.models],
        axis=1,
    )
    normalized = decisions / model.scales.reshape(1, -1)
    pred = np.argmax(normalized, axis=1).astype(np.int64)
    return (-np.max(normalized, axis=1)).astype(np.float32), pred


def fit_opensvdd_closest_svm(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    gamma: str | float = "scale",
) -> OpenSvddClosestSvmModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    centers = _class_centers(embeddings, labels, num_classes=num_classes)
    closest_classes = _closest_classes(centers)
    models: list[SVC] = []
    scales = []
    for cls, closest in enumerate(closest_classes.tolist()):
        keep = (labels == cls) | (labels == int(closest))
        x_fit = embeddings[keep]
        y_fit = (labels[keep] == cls).astype(np.int64)
        if len(np.unique(y_fit)) != 2:
            raise ValueError(f"Cannot fit OpenSVDD closest boundary for class {cls}: missing closest class.")
        model = SVC(kernel="rbf", gamma=gamma, class_weight="balanced")
        model.fit(x_fit, y_fit)
        decision = model.decision_function(embeddings[labels == cls]).reshape(-1)
        scale = float(np.quantile(np.maximum(decision, 1e-6), 0.95))
        models.append(model)
        scales.append(max(scale, 1e-6))
    return OpenSvddClosestSvmModel(
        models=models,
        scales=np.asarray(scales, dtype=np.float32),
        closest_classes=closest_classes.astype(np.int64),
    )


def opensvdd_closest_svm_unknown_score(
    embeddings: np.ndarray,
    model: OpenSvddClosestSvmModel,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    decisions = np.stack(
        [boundary.decision_function(embeddings).reshape(-1) for boundary in model.models],
        axis=1,
    )
    normalized = decisions / model.scales.reshape(1, -1)
    pred = np.argmax(normalized, axis=1).astype(np.int64)
    return (-np.max(normalized, axis=1)).astype(np.float32), pred


def _class_centers(embeddings: np.ndarray, labels: np.ndarray, *, num_classes: int) -> np.ndarray:
    centers = []
    for cls in range(int(num_classes)):
        mask = labels == cls
        if not np.any(mask):
            raise ValueError(f"Cannot fit center for class {cls}: no samples.")
        centers.append(np.mean(embeddings[mask], axis=0))
    return np.stack(centers, axis=0).astype(np.float32)


def _closest_classes(centers: np.ndarray) -> np.ndarray:
    diff = centers[:, None, :] - centers[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    return np.argmin(distances, axis=1).astype(np.int64)
