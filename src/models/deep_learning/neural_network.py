"""Feed-forward Neural Network for tabular churn data using PyTorch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseChurnModel, ModelMetadata


class _MLP(nn.Module):
    """Multi-layer perceptron with BatchNorm, dropout, and residual connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        batch_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim

        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class NeuralNetworkModel(BaseChurnModel):
    """PyTorch MLP with early stopping and class-weight balancing.

    Training uses BCEWithLogitsLoss with a pos_weight computed from the
    training set label distribution so no under-/oversampling is required.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 100,
        patience: int = 10,
    ) -> None:
        self._hidden_dims = hidden_dims or [256, 128, 64]
        self._dropout = dropout
        self._batch_norm = batch_norm
        self._lr = learning_rate
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._net: _MLP | None = None
        self._metadata = ModelMetadata(
            name="neural_network",
            version="1.0.0",
            params={
                "hidden_dims": self._hidden_dims,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
            },
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> NeuralNetworkModel:
        self._metadata.feature_names = X_train.columns.tolist()
        input_dim = X_train.shape[1]

        self._net = _MLP(input_dim, self._hidden_dims, self._dropout, self._batch_norm)
        self._net.to(self._device)

        pos_weight = torch.tensor(
            [(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32
        ).to(self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = (
            self._make_loader(X_val, y_val, shuffle=False)
            if X_val is not None and y_val is not None
            else None
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, self._max_epochs + 1):
            train_loss = self._train_epoch(self._net, train_loader, criterion, optimizer)

            if val_loader:
                val_loss = self._eval_epoch(self._net, val_loader, criterion)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    self._best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self._patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    self._net.load_state_dict(self._best_state)
                    break

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}/{self._max_epochs} — train_loss: {train_loss:.4f}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._net is not None, "Call fit() first"
        self._net.eval()
        tensor = torch.tensor(X.values, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._net(tensor)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    # ------------------------------------------------------------------

    def _make_loader(
        self, X: pd.DataFrame, y: pd.Series, shuffle: bool
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32)
        dataset: TensorDataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle)

    def _train_epoch(
        self,
        net: _MLP,
        loader: DataLoader[tuple[torch.Tensor, ...]],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        net.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self._device)
            y_batch = y_batch.to(self._device)
            optimizer.zero_grad()
            logits = net(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        n = max(int(len(loader.dataset)), 1)  # type: ignore[arg-type]
        return total_loss / n

    def _eval_epoch(
        self,
        net: _MLP,
        loader: DataLoader[tuple[torch.Tensor, ...]],
        criterion: nn.Module,
    ) -> float:
        net.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                logits = net(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * len(X_batch)
        n = max(int(len(loader.dataset)), 1)  # type: ignore[arg-type]
        return total_loss / n
