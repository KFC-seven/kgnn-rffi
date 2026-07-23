from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TinyIQNet(nn.Module):
    """Two-block 1-D CNN used for the ManySig dataset."""

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
            nn.Linear(64, int(embedding_dim)),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(int(embedding_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_tensor_shape(x)
        features = self.encoder(x.permute(0, 2, 1).contiguous())
        embedding = self.embedding(features)
        return self.classifier(embedding), embedding


class ResidualBlock1D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(output_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(output_channels),
            )
            if stride != 1 or input_channels != output_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.residual(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    """Six-block 1-D ResNet used for the ManyTx dataset."""

    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layers = nn.Sequential(
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 256, stride=2),
            ResidualBlock1D(256, 256),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, int(embedding_dim)),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(int(embedding_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_tensor_shape(x)
        features = self.layers(self.stem(x.permute(0, 2, 1).contiguous()))
        embedding = self.embedding(features)
        return self.classifier(embedding), embedding


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    train_indices: np.ndarray
    validation_indices: np.ndarray
    best_validation_accuracy: float
    best_epoch: int
    trained_epochs: int
    early_stopped: bool


class IQDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(normalize_iq(x))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def make_encoder(
    architecture: str,
    *,
    num_classes: int,
    embedding_dim: int,
) -> nn.Module:
    if architecture == "tiny":
        return TinyIQNet(num_classes, embedding_dim)
    if architecture == "resnet1d":
        return ResNet1D(num_classes, embedding_dim)
    raise ValueError(f"Unknown architecture: {architecture!r}.")


def make_model(
    model_name: str,
    num_classes: int,
    embedding_dim: int = 64,
) -> nn.Module:
    """Compatibility entry point shared by the baseline trainers."""

    return make_encoder(
        model_name,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
    )


def train_source_encoder(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    architecture: str,
    embedding_dim: int,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.20,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.001,
    seed: int = 42,
    device: str = "cpu",
) -> TrainingResult:
    """Train the source classifier and retain the best source-validation epoch."""

    set_seed(seed)
    train_indices, validation_indices = stratified_train_validation_split(
        y,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    model = make_encoder(
        architecture,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
    ).to(device)
    train_loader = DataLoader(
        IQDataset(np.asarray(x)[train_indices], np.asarray(y)[train_indices]),
        batch_size=int(batch_size),
        shuffle=True,
    )
    validation_loader = DataLoader(
        IQDataset(np.asarray(x)[validation_indices], np.asarray(y)[validation_indices]),
        batch_size=int(batch_size),
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    objective = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    trained_epochs = 0
    for epoch in range(1, int(epochs) + 1):
        trained_epochs = epoch
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_x.to(device))
            loss = objective(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
        accuracy = evaluate_accuracy(model, validation_loader, device=device)
        if best_state is None or accuracy >= best_accuracy + float(early_stopping_min_delta):
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                int(early_stopping_patience) > 0
                and epochs_without_improvement >= int(early_stopping_patience)
            ):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        train_indices=train_indices,
        validation_indices=validation_indices,
        best_validation_accuracy=float(best_accuracy),
        best_epoch=int(best_epoch),
        trained_epochs=int(trained_epochs),
        early_stopped=bool(trained_epochs < int(epochs)),
    )


def infer(
    model: nn.Module,
    x: np.ndarray,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        torch.from_numpy(normalize_iq(x)),
        batch_size=int(batch_size),
        shuffle=False,
    )
    logits_parts: list[np.ndarray] = []
    embedding_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_x in loader:
            logits, embedding = model(batch_x.to(device))
            logits_parts.append(logits.cpu().numpy())
            embedding_parts.append(embedding.cpu().numpy())
    return (
        np.concatenate(logits_parts, axis=0).astype(np.float32),
        np.concatenate(embedding_parts, axis=0).astype(np.float32),
    )


def encoder_function(
    model: nn.Module,
    *,
    batch_size: int = 512,
    device: str = "cpu",
):
    def encode(x: np.ndarray) -> np.ndarray:
        return infer(model, x, batch_size=batch_size, device=device)[1]

    return encode


def classifier_function(
    model: nn.Module,
    *,
    batch_size: int = 512,
    device: str = "cpu",
):
    def predict(x: np.ndarray) -> np.ndarray:
        logits, _ = infer(model, x, batch_size=batch_size, device=device)
        return np.argmax(logits, axis=1).astype(np.int64)

    return predict


def normalize_iq(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"Expected shape (N, L, 2), got {values.shape}.")
    rms = np.sqrt(
        np.mean(np.sum(values * values, axis=-1), axis=1, keepdims=True) + 1e-12
    )
    return (values / rms[:, None, :]).astype(np.float32)


def stratified_train_validation_split(
    labels: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    training: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    for cls in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == cls)
        if indices.size < 2:
            raise ValueError("Every source class requires at least two samples.")
        rng.shuffle(indices)
        validation_count = max(1, int(round(float(validation_fraction) * indices.size)))
        validation_count = min(indices.size - 1, validation_count)
        validation.append(indices[:validation_count])
        training.append(indices[validation_count:])
    train_indices = np.concatenate(training).astype(np.int64)
    validation_indices = np.concatenate(validation).astype(np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def stratified_train_val_split(
    labels: np.ndarray,
    val_frac: float = 0.20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    return stratified_train_validation_split(
        labels,
        validation_fraction=val_frac,
        seed=seed,
    )


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits, _ = model(batch_x.to(device))
            prediction = torch.argmax(logits.cpu(), dim=1)
            correct += int((prediction == batch_y).sum().item())
            total += int(batch_y.shape[0])
    return float(correct) / float(total) if total else 0.0


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _validate_tensor_shape(x: torch.Tensor) -> None:
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f"Expected input shape (N, L, 2), got {tuple(x.shape)}.")


IqDataset = IQDataset
