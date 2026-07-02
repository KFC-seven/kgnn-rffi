from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TinyIQNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, int(num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"Expected input shape (N, L, 2), got {tuple(x.shape)}.")
        x_ch = x.permute(0, 2, 1).contiguous()
        features = self.encoder(x_ch)
        embeddings = self.embedding(features)
        logits = self.classifier(embeddings)
        return logits, embeddings


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layers = nn.Sequential(
            BasicBlock1D(64, 64),
            BasicBlock1D(64, 64),
            BasicBlock1D(64, 128, stride=2),
            BasicBlock1D(128, 128),
            BasicBlock1D(128, 256, stride=2),
            BasicBlock1D(256, 256),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, int(num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"Expected input shape (N, L, 2), got {tuple(x.shape)}.")
        x_ch = x.permute(0, 2, 1).contiguous()
        features = self.layers(self.stem(x_ch))
        embeddings = self.embedding(features)
        logits = self.classifier(embeddings)
        return logits, embeddings


def make_model(model_name: str, num_classes: int, embedding_dim: int = 64) -> nn.Module:
    if model_name == "tiny":
        return TinyIQNet(num_classes=num_classes, embedding_dim=embedding_dim)
    if model_name == "resnet1d":
        return ResNet1D(num_classes=num_classes, embedding_dim=embedding_dim)
    raise ValueError(f"Unknown model_name={model_name!r}.")


class IqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(normalize_iq(x).astype(np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


@dataclass
class TrainResult:
    model: TinyIQNet
    train_indices: np.ndarray
    val_indices: np.ndarray
    best_val_accuracy: float
    best_epoch: int
    trained_epochs: int
    stopped_epoch: int
    early_stopped: bool


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def normalize_iq(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    power = np.sqrt(np.mean(np.sum(x * x, axis=-1), axis=1, keepdims=True) + 1e-12)
    return x / power[:, None, :]


def stratified_train_val_split(
    labels: np.ndarray,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for cls in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        val_n = max(1, int(round(float(val_frac) * len(idx))))
        val_n = min(len(idx) - 1, val_n) if len(idx) > 1 else 0
        val_parts.append(idx[:val_n])
        train_parts.append(idx[val_n:])
    train_idx = np.concatenate(train_parts).astype(np.int64)
    val_idx = np.concatenate(val_parts).astype(np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def train_sourceonly(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 0,
    embedding_dim: int = 64,
    model_name: str = "tiny",
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    device: str = "cpu",
) -> TrainResult:
    set_seed(seed)
    train_idx, val_idx = stratified_train_val_split(y, val_frac=val_frac, seed=seed)
    model = make_model(
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
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        val_acc = evaluate_accuracy(model, val_loader, device=device)
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
    return TrainResult(
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
