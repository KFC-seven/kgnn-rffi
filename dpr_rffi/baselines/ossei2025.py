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
    stratified_train_val_split,
)
from dpr_rffi.losses import supervised_contrastive_loss


@dataclass
class Ossei2025TrainResult:
    model: nn.Module
    train_indices: np.ndarray
    val_indices: np.ndarray
    best_val_accuracy: float
    best_epoch: int
    trained_epochs: int
    stopped_epoch: int
    early_stopped: bool
    history: list[dict]


@dataclass(frozen=True)
class Ossei2025ScoreModel:
    centers: np.ndarray
    recon_mean: np.ndarray
    recon_std: np.ndarray
    stat_mean: np.ndarray
    stat_std: np.ndarray


@dataclass(frozen=True)
class Ossei2025PaperScoreModel:
    mu_vectors: np.ndarray
    recon_scales: np.ndarray


class VectorVAE(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__()
        hidden = max(32, int(feature_dim))
        self.encoder = nn.Sequential(
            nn.Linear(int(feature_dim), hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, int(latent_dim))
        self.logvar = nn.Linear(hidden, int(latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(int(latent_dim), hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, int(feature_dim)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        logvar = torch.clamp(self.logvar(hidden), min=-6.0, max=6.0)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu
        return self.decoder(z), mu, logvar


class Ossei2025Model(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        embedding_dim: int,
        projection_dim: int,
        vae_latent_dim: int,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.backbone = make_model(
            model_name=model_name,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )
        self.projection = nn.Sequential(
            nn.Linear(int(embedding_dim), int(projection_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(projection_dim), int(projection_dim)),
        )
        self.vaes = nn.ModuleList(
            [VectorVAE(feature_dim=embedding_dim, latent_dim=vae_latent_dim) for _ in range(int(num_classes))]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, embeddings = self.backbone(x)
        projection = self.projection(embeddings)
        return logits, embeddings, projection

    def vae_scores(self, embeddings: torch.Tensor, *, beta: float = 0.05) -> torch.Tensor:
        scores = []
        for vae in self.vaes:
            reconstruction, mu, logvar = vae(embeddings)
            mse = torch.mean((reconstruction - embeddings) ** 2, dim=1)
            kl = _kl_per_sample(mu, logvar) / float(max(1, embeddings.shape[1]))
            scores.append(mse + float(beta) * kl)
        return torch.stack(scores, dim=1)


def train_ossei2025_source(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 0,
    embedding_dim: int = 64,
    projection_dim: int = 128,
    vae_latent_dim: int = 64,
    model_name: str = "tiny",
    ce_weight: float = 1.0,
    recon_ce_weight: float = 0.2,
    vae_weight: float = 0.5,
    supcon_start: float = 0.1,
    supcon_end: float = 1.0,
    kl_start: float = 0.005,
    kl_end: float = 0.5,
    temperature: float = 0.07,
    gamma: float = 0.1,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    device: str = "cpu",
) -> Ossei2025TrainResult:
    _set_seed(seed)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    train_idx, val_idx = stratified_train_val_split(y, val_frac=val_frac, seed=seed)
    model = Ossei2025Model(
        model_name=model_name,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        vae_latent_dim=vae_latent_dim,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best_val = -1.0
    best_epoch = 0
    best_state = None
    no_improve_epochs = 0
    stopped_epoch = int(epochs)
    patience = max(0, int(early_stop_patience))
    min_delta = max(0.0, float(early_stop_min_delta))
    history: list[dict] = []
    for epoch in range(1, int(epochs) + 1):
        sup_weight = _linear_schedule(epoch, epochs, supcon_start, supcon_end)
        kl_weight = _linear_schedule(epoch, epochs, kl_start, kl_end)
        loss_value = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            ce=ce,
            ce_weight=ce_weight,
            recon_ce_weight=recon_ce_weight,
            vae_weight=vae_weight,
            supcon_weight=sup_weight,
            kl_weight=kl_weight,
            temperature=temperature,
            gamma=gamma,
            device=device,
        )
        val_acc = _evaluate_accuracy(model, val_loader, device=device)
        history.append(
            {
                "epoch": epoch,
                "loss": loss_value,
                "val_accuracy": val_acc,
                "supcon_weight": float(sup_weight),
                "kl_weight": float(kl_weight),
            }
        )
        improved = best_state is None or val_acc >= best_val + min_delta
        if improved:
            best_val = val_acc
            best_epoch = int(epoch)
            no_improve_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improve_epochs += 1
            if patience > 0 and no_improve_epochs >= patience:
                stopped_epoch = int(epoch)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return Ossei2025TrainResult(
        model=model,
        train_indices=train_idx,
        val_indices=val_idx,
        best_val_accuracy=float(best_val),
        best_epoch=int(best_epoch),
        trained_epochs=int(len(history)),
        stopped_epoch=int(stopped_epoch if len(history) < int(epochs) else int(epochs)),
        early_stopped=bool(len(history) < int(epochs)),
        history=history,
    )


def infer_ossei2025_outputs(
    model: Ossei2025Model,
    x: np.ndarray,
    *,
    batch_size: int = 512,
    score_beta: float = 0.05,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(normalize_iq(x).astype(np.float32))
    loader = DataLoader(tensor, batch_size=batch_size, shuffle=False)
    logits_parts = []
    embedding_parts = []
    score_parts = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            logits, embeddings, _projection = model(xb.to(device))
            scores = model.vae_scores(embeddings, beta=score_beta)
            logits_parts.append(logits.cpu().numpy())
            embedding_parts.append(embeddings.cpu().numpy())
            score_parts.append(scores.cpu().numpy())
    return (
        np.concatenate(logits_parts, axis=0).astype(np.float32),
        np.concatenate(embedding_parts, axis=0).astype(np.float32),
        np.concatenate(score_parts, axis=0).astype(np.float32),
    )


def fit_ossei2025_score_model(
    embeddings: np.ndarray,
    labels: np.ndarray,
    vae_scores: np.ndarray,
    *,
    num_classes: int,
) -> Ossei2025ScoreModel:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    vae_scores = np.asarray(vae_scores, dtype=np.float32)
    centers = []
    recon_mean = []
    recon_std = []
    stat_mean = []
    stat_std = []
    for cls in range(int(num_classes)):
        mask = labels == cls
        if not np.any(mask):
            raise ValueError(f"Cannot fit OSSEI-2025 score model for class {cls}: no samples.")
        cls_embeddings = embeddings[mask]
        center = np.mean(cls_embeddings, axis=0)
        centers.append(center.astype(np.float32))
        cls_recon = vae_scores[mask, cls]
        cls_dist = np.linalg.norm(cls_embeddings - center[None, :], axis=1)
        recon_mean.append(float(np.mean(cls_recon)))
        recon_std.append(float(max(np.std(cls_recon), 1e-6)))
        stat_mean.append(float(np.mean(cls_dist)))
        stat_std.append(float(max(np.std(cls_dist), 1e-6)))
    return Ossei2025ScoreModel(
        centers=np.stack(centers, axis=0).astype(np.float32),
        recon_mean=np.asarray(recon_mean, dtype=np.float32),
        recon_std=np.asarray(recon_std, dtype=np.float32),
        stat_mean=np.asarray(stat_mean, dtype=np.float32),
        stat_std=np.asarray(stat_std, dtype=np.float32),
    )


def ossei2025_unknown_scores(
    embeddings: np.ndarray,
    vae_scores: np.ndarray,
    model: Ossei2025ScoreModel,
    *,
    recon_weight: float = 1.0,
    stat_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    vae_scores = np.asarray(vae_scores, dtype=np.float32)
    recon_z = (vae_scores - model.recon_mean[None, :]) / model.recon_std[None, :]
    distances = np.linalg.norm(embeddings[:, None, :] - model.centers[None, :, :], axis=2)
    stat_z = (distances - model.stat_mean[None, :]) / model.stat_std[None, :]
    fused = float(recon_weight) * recon_z + float(stat_weight) * stat_z
    pred = np.argmin(fused, axis=1).astype(np.int64)
    score = fused[np.arange(fused.shape[0]), pred]
    return score.astype(np.float32), pred


def fit_ossei2025_paper_score_model(
    embeddings: np.ndarray,
    labels: np.ndarray,
    vae_scores: np.ndarray,
    *,
    num_classes: int,
) -> Ossei2025PaperScoreModel:
    """Fit source-only statistics for the paper-formula approximation.

    The original OSSEI detector combines S1=P_R(r,V_y)||r||_1^2 and
    S2=r^T mu(y).  Our VAE produces an energy rather than a Gaussian
    reconstruction probability, so P_R is approximated by exp(-energy/scale_y)
    with scale_y fitted from source-train correct-class VAE energies.
    """

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    vae_scores = np.asarray(vae_scores, dtype=np.float32)
    class_sums = []
    recon_scales = []
    for cls in range(int(num_classes)):
        mask = labels == cls
        if not np.any(mask):
            raise ValueError(f"Cannot fit OSSEI-2025 paper score model for class {cls}: no samples.")
        class_embeddings = np.maximum(embeddings[mask], 0.0)
        class_sums.append(np.sum(class_embeddings, axis=0))
        class_energy = vae_scores[mask, cls]
        recon_scales.append(float(max(np.mean(class_energy), 1e-6)))
    sums = np.stack(class_sums, axis=0).astype(np.float32)
    total = np.sum(sums, axis=0, keepdims=True)
    mu_vectors = sums / np.maximum(total, 1e-6)
    return Ossei2025PaperScoreModel(
        mu_vectors=mu_vectors.astype(np.float32),
        recon_scales=np.asarray(recon_scales, dtype=np.float32),
    )


def ossei2025_paper_unknown_scores(
    embeddings: np.ndarray,
    vae_scores: np.ndarray,
    model: Ossei2025PaperScoreModel,
    *,
    recon_weight: float = 1.0,
    stat_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.maximum(np.asarray(embeddings, dtype=np.float32), 0.0)
    vae_scores = np.asarray(vae_scores, dtype=np.float32)
    recon_prob = np.exp(-vae_scores / np.maximum(model.recon_scales[None, :], 1e-6))
    activation = np.sum(np.abs(embeddings), axis=1, keepdims=True) ** 2
    s1 = recon_prob * activation
    s2 = embeddings @ model.mu_vectors.T
    known_score = float(recon_weight) * s1 + float(stat_weight) * s2
    pred = np.argmax(known_score, axis=1).astype(np.int64)
    unknown_score = -known_score[np.arange(known_score.shape[0]), pred]
    return unknown_score.astype(np.float32), pred


def _train_epoch(
    *,
    model: Ossei2025Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce: nn.Module,
    ce_weight: float,
    recon_ce_weight: float,
    vae_weight: float,
    supcon_weight: float,
    kl_weight: float,
    temperature: float,
    gamma: float,
    device: str,
) -> float:
    model.train()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, embeddings, projection = model(xb)
        logits_ce = ce(logits, yb)
        supcon = supervised_contrastive_loss(projection, yb, temperature=temperature)
        correct_recon, correct_kl = _correct_class_vae_loss(model, embeddings, yb)
        all_scores = model.vae_scores(embeddings, beta=kl_weight)
        recon_ce = ce(-float(gamma) * all_scores, yb)
        loss = (
            float(ce_weight) * logits_ce
            + float(supcon_weight) * supcon
            + float(vae_weight) * (correct_recon + float(kl_weight) * correct_kl)
            + float(recon_ce_weight) * recon_ce
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def _correct_class_vae_loss(
    model: Ossei2025Model,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    recon_terms = []
    kl_terms = []
    total = 0
    for cls in torch.unique(labels).detach().cpu().tolist():
        mask = labels == int(cls)
        if not bool(mask.any()):
            continue
        cls_embeddings = embeddings[mask]
        reconstruction, mu, logvar = model.vaes[int(cls)](cls_embeddings)
        weight = int(cls_embeddings.shape[0])
        total += weight
        recon_terms.append(torch.mean((reconstruction - cls_embeddings) ** 2) * weight)
        kl_terms.append(torch.mean(_kl_per_sample(mu, logvar)) * weight)
    if total == 0:
        zero = embeddings.sum() * 0.0
        return zero, zero
    return sum(recon_terms) / float(total), sum(kl_terms) / float(total)


def _kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1.0 + logvar - mu * mu - torch.exp(logvar), dim=1)


def _evaluate_accuracy(model: Ossei2025Model, loader: DataLoader, *, device: str) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits, _embeddings, _projection = model(xb.to(device))
            pred = torch.argmax(logits.cpu(), dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.shape[0])
    return float(correct) / float(total) if total else 0.0


def _linear_schedule(epoch: int, epochs: int, start: float, end: float) -> float:
    if int(epochs) <= 1:
        return float(end)
    ratio = (int(epoch) - 1) / float(max(1, int(epochs) - 1))
    return float(start) + ratio * (float(end) - float(start))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
