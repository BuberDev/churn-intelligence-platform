"""Tabular Transformer — attention-based model for structured data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseChurnModel, ModelMetadata


class _ColumnEmbedding(nn.Module):
    """Projects each feature dimension to a shared d_model space."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) → (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weight + self.bias


class _TabTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = _ColumnEmbedding(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, n_features, d_model)
        emb = self.embedding(x)
        out = self.transformer(emb)
        # CLS-style: mean-pool over feature dimension
        pooled = self.norm(out.mean(dim=1))
        return self.classifier(pooled).squeeze(1)


class TabularTransformerModel(BaseChurnModel):
    """Attention-based model treating each feature as a token.

    Inspired by TabTransformer (Huang et al., 2020). Applies multi-head
    self-attention across feature columns so the model can learn complex
    feature interactions without explicit engineering.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 5e-4,
        batch_size: int = 256,
        max_epochs: int = 80,
        patience: int = 15,
    ) -> None:
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._dropout = dropout
        self._lr = learning_rate
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: _TabTransformer | None = None
        self._metadata = ModelMetadata(
            name="tabular_transformer",
            version="1.0.0",
            params={
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
            },
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> TabularTransformerModel:
        n_features = X_train.shape[1]
        self._metadata.feature_names = X_train.columns.tolist()

        # d_model must be divisible by n_heads
        d_model = self._d_model
        while d_model % self._n_heads != 0:
            d_model += 1

        self._net = _TabTransformer(
            n_features, d_model, self._n_heads, self._n_layers, self._dropout
        )
        self._net.to(self._device)

        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
            dtype=torch.float32,
        ).to(self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self._lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._max_epochs
        )

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = (
            self._make_loader(X_val, y_val, shuffle=False)
            if X_val is not None and y_val is not None
            else None
        )

        best_val_loss = math.inf
        patience_count = 0
        best_state: dict[str, torch.Tensor] = {}

        for epoch in range(1, self._max_epochs + 1):
            self._net.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self._device), y_b.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._net(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if val_loader:
                val_loss = self._eval_loss(criterion, val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                else:
                    patience_count += 1
                if patience_count >= self._patience:
                    logger.info(f"TabTransformer early stop at epoch {epoch}")
                    self._net.load_state_dict(best_state)
                    break

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._net is not None
        self._net.eval()
        tensor = torch.tensor(X.values, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            proba = torch.sigmoid(self._net(tensor)).cpu().numpy()
        return proba

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def _make_loader(
        self, X: pd.DataFrame, y: pd.Series, shuffle: bool
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        dataset: TensorDataset = TensorDataset(
            torch.tensor(X.values, dtype=torch.float32),
            torch.tensor(y.values, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle)

    def _eval_loss(
        self,
        criterion: nn.Module,
        loader: DataLoader[tuple[torch.Tensor, ...]],
    ) -> float:
        assert self._net is not None
        self._net.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self._device), y_b.to(self._device)
                total += criterion(self._net(X_b), y_b).item() * len(X_b)
                n += len(X_b)
        return total / max(n, 1)
