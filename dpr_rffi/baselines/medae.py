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
from dpr_rffi.baselines.metric_learning import CenterLoss


@dataclass
class MeDaeTrainResult:
    model: nn.Module
    train_indices: np.ndarray
    val_indices: np.ndarray
    best_val_accuracy: float
    best_epoch: int
    trained_epochs: int
    stopped_epoch: int
    early_stopped: bool


class ResidualShrinkageBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        hidden = max(4, out_channels // 4)
        self.threshold = nn.Sequential(
            nn.Linear(out_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels),
            nn.Sigmoid(),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv(x)
        pooled = torch.mean(torch.abs(residual), dim=2)
        scales = self.threshold(pooled).unsqueeze(2)
        thresholds = scales * pooled.unsqueeze(2)
        shrunk = torch.sign(residual) * torch.relu(torch.abs(residual) - thresholds)
        return self.activation(shrunk + self.shortcut(x))


class MeDaeShrinkageModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        embedding_dim: int,
        input_length: int,
    ):
        super().__init__()
        self.input_length = int(input_length)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            ResidualShrinkageBlock1D(32, 32),
            ResidualShrinkageBlock1D(32, 32),
            ResidualShrinkageBlock1D(32, 64, stride=2),
            ResidualShrinkageBlock1D(64, 64),
            ResidualShrinkageBlock1D(64, 128, stride=2),
            ResidualShrinkageBlock1D(128, 128),
            ResidualShrinkageBlock1D(128, 128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, int(embedding_dim)),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(int(embedding_dim), int(num_classes))
        self.decoder_fc = nn.Sequential(
            nn.Linear(int(embedding_dim), 128 * 16),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"Expected input shape (N, L, 2), got {tuple(x.shape)}.")
        x_ch = x.permute(0, 2, 1).contiguous()
        features = self.encoder(x_ch)
        embeddings = self.embedding(features)
        logits = self.classifier(embeddings)
        decoded = self.decoder_fc(embeddings).reshape(-1, 128, 16)
        reconstruction = self.decoder(decoded)
        if reconstruction.shape[-1] != self.input_length:
            reconstruction = nn.functional.interpolate(
                reconstruction,
                size=self.input_length,
                mode="linear",
                align_corners=False,
            )
        return logits, embeddings, reconstruction.permute(0, 2, 1).contiguous()


class MeDaeModel(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        embedding_dim: int,
        input_length: int,
    ):
        super().__init__()
        self.input_length = int(input_length)
        self.backbone = make_model(
            model_name=model_name,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(embedding_dim), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.input_length * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, embeddings = self.backbone(x)
        reconstruction = self.decoder(embeddings).reshape(-1, self.input_length, 2)
        return logits, embeddings, reconstruction


def train_medae(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    center_lr: float = 5e-4,
    val_frac: float = 0.2,
    seed: int = 0,
    embedding_dim: int = 64,
    model_name: str = "tiny",
    ce_weight: float = 1.0,
    mse_weight: float = 0.5,
    metric_weight: float = 0.005,
    noise_std: float = 0.05,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    device: str = "cpu",
) -> MeDaeTrainResult:
    set_seed(seed)
    random.seed(int(seed))
    train_idx, val_idx = stratified_train_val_split(y, val_frac=val_frac, seed=seed)
    input_length = int(np.asarray(x).shape[1])
    model = MeDaeModel(
        model_name=model_name,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        input_length=input_length,
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
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
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
            clean = xb.to(device)
            yb = yb.to(device)
            if noise_std > 0.0:
                noisy = clean + torch.randn_like(clean) * float(noise_std)
            else:
                noisy = clean
            optimizer.zero_grad(set_to_none=True)
            center_optimizer.zero_grad(set_to_none=True)
            logits, embeddings, reconstruction = model(noisy)
            loss = (
                float(ce_weight) * ce(logits, yb)
                + float(mse_weight) * mse(reconstruction, clean)
                + float(metric_weight) * center_loss(embeddings, yb)
            )
            loss.backward()
            optimizer.step()
            center_optimizer.step()
        val_acc = evaluate_medae_accuracy(model, val_loader, device=device)
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
    return MeDaeTrainResult(
        model=model,
        train_indices=train_idx,
        val_indices=val_idx,
        best_val_accuracy=float(best_val),
        best_epoch=int(best_epoch),
        trained_epochs=int(trained_epochs),
        stopped_epoch=int(stopped_epoch if trained_epochs < int(epochs) else int(epochs)),
        early_stopped=bool(trained_epochs < int(epochs)),
    )


def train_medae_shrinkage_full(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    center_lr: float = 5e-4,
    val_frac: float = 0.2,
    seed: int = 0,
    embedding_dim: int = 64,
    mse_weight: float = 0.5,
    metric_weight: float = 0.005,
    noise_std: float = 0.05,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    device: str = "cpu",
) -> MeDaeTrainResult:
    set_seed(seed)
    random.seed(int(seed))
    train_idx, val_idx = stratified_train_val_split(y, val_frac=val_frac, seed=seed)
    input_length = int(np.asarray(x).shape[1])
    model = MeDaeShrinkageModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        input_length=input_length,
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
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
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
            clean = xb.to(device)
            yb = yb.to(device)
            noisy = clean + torch.randn_like(clean) * float(noise_std) if noise_std > 0.0 else clean
            optimizer.zero_grad(set_to_none=True)
            center_optimizer.zero_grad(set_to_none=True)
            logits, embeddings, reconstruction = model(noisy)
            loss = (
                ce(logits, yb)
                + float(mse_weight) * mse(reconstruction, clean)
                + float(metric_weight) * center_loss(embeddings, yb)
            )
            loss.backward()
            optimizer.step()
            center_optimizer.step()
        val_acc = evaluate_medae_accuracy(model, val_loader, device=device)
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
    return MeDaeTrainResult(
        model=model,
        train_indices=train_idx,
        val_indices=val_idx,
        best_val_accuracy=float(best_val),
        best_epoch=int(best_epoch),
        trained_epochs=int(trained_epochs),
        stopped_epoch=int(stopped_epoch if trained_epochs < int(epochs) else int(epochs)),
        early_stopped=bool(trained_epochs < int(epochs)),
    )


def evaluate_medae_accuracy(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits, _, _ = model(xb.to(device))
            pred = torch.argmax(logits.cpu(), dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.shape[0])
    return float(correct) / float(total) if total else float("nan")


def infer_medae_logits_embeddings_recon(
    model: nn.Module,
    x: np.ndarray,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = normalize_iq(x).astype(np.float32)
    loader = DataLoader(
        torch.from_numpy(clean),
        batch_size=batch_size,
        shuffle=False,
    )
    logits_parts = []
    emb_parts = []
    recon_error_parts = []
    model.eval()
    cursor = 0
    with torch.no_grad():
        for xb in loader:
            batch_n = int(xb.shape[0])
            logits, embeddings, reconstruction = model(xb.to(device))
            logits_parts.append(logits.cpu().numpy())
            emb_parts.append(embeddings.cpu().numpy())
            target = torch.from_numpy(clean[cursor : cursor + batch_n]).to(device)
            error = torch.mean((reconstruction - target) ** 2, dim=(1, 2))
            recon_error_parts.append(error.cpu().numpy())
            cursor += batch_n
    return (
        np.concatenate(logits_parts, axis=0).astype(np.float32),
        np.concatenate(emb_parts, axis=0).astype(np.float32),
        np.concatenate(recon_error_parts, axis=0).astype(np.float32),
    )
