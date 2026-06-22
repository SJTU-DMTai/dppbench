import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .tabular_model import TabularModel


class TorchTabularModel(TabularModel):
    """Small PyTorch tabular model base compatible with TabularModel."""

    def __init__(self, task="binary", seed=42, epochs=10, batch_size=256,
                 lr=1e-3, weight_decay=0.0, hidden_units=(256, 128),
                 dropout=0.1, device=None, max_train_rows=None,
                 max_features=None,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                 row_attention=False, **kwargs):
        super().__init__(task=task, seed=seed)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_units = tuple(hidden_units)
        self.dropout = float(dropout)
        self.device = device or "cpu"
        self.max_train_rows = int(max_train_rows) if max_train_rows else None
        self.max_features = int(max_features) if max_features else None
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.row_attention = bool(row_attention)
        self.feature_names = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def _make_net(self, n_features):
        raise NotImplementedError

    def _to_array(self, X, fit=False):
        if isinstance(X, pd.DataFrame):
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if fit or self.feature_names is None:
                if self.max_features and X.shape[1] > self.max_features:
                    variances = X.var(axis=0).fillna(0.0).sort_values(ascending=False)
                    self.feature_names = list(variances.head(self.max_features).index)
                else:
                    self.feature_names = list(X.columns)
            X = X.reindex(columns=self.feature_names, fill_value=0)
            return X.values.astype(np.float32)
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _sample_train(self, X, y):
        if self.max_train_rows and len(X) > self.max_train_rows:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X), size=self.max_train_rows, replace=False)
            idx.sort()
            return X[idx], y[idx]
        return X, y

    def _prepare_y_train(self, y):
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if self.task == "binary":
            return y
        self._y_mean = float(y.mean())
        self._y_std = float(y.std())
        if self._y_std < 1e-6:
            self._y_std = 1.0
        return (y - self._y_mean) / self._y_std

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = self._to_array(X_train, fit=True)
        y_raw = np.asarray(y_train, dtype=np.float32).reshape(-1)
        X, y_raw = self._sample_train(X, y_raw)
        y = self._prepare_y_train(y_raw)

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std < 1e-6] = 1.0
        X = (X - self._x_mean) / self._x_std

        self.model = self._make_net(X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCELoss() if self.task == "binary" else nn.MSELoss()
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total = 0.0
            count = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(xb)
                count += len(xb)
            print(f"  [{self.__class__.__name__}] epoch {epoch}/{self.epochs} loss={total / max(count, 1):.4f}")
        return self

    def predict(self, X):
        X = self._to_array(X, fit=False)
        X = (X - self._x_mean) / self._x_std
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i:i + self.batch_size]).float().to(self.device)
                preds.append(self.model(xb).detach().cpu().numpy().reshape(-1))
        out = np.concatenate(preds, axis=0)
        if self.task != "binary":
            out = out * self._y_std + self._y_mean
        return out

    def feature_importance(self):
        return {}


class _MLPNet(nn.Module):
    def __init__(self, n_features, hidden_units, dropout, task):
        super().__init__()
        layers = []
        dims = [n_features] + list(hidden_units)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        if task == "binary":
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP(TorchTabularModel):
    def _make_net(self, n_features):
        return _MLPNet(n_features, self.hidden_units, self.dropout, self.task)


class _FeatureTokenTransformer(nn.Module):
    def __init__(self, n_features, d_model, nhead, num_layers, dim_feedforward,
                 dropout, task, use_cls=True, saint_rows=False):
        super().__init__()
        if d_model % nhead != 0:
            d_model = max(nhead, (d_model // nhead) * nhead)
        self.use_cls = use_cls
        self.saint_rows = saint_rows
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.col_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        if saint_rows:
            row_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            )
            self.row_encoder = nn.TransformerEncoder(row_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid() if task == "binary" else nn.Identity(),
        )

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        if self.use_cls:
            cls = self.cls.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        z = self.col_encoder(tokens)
        pooled = z[:, 0, :] if self.use_cls else z.mean(dim=1)
        if self.saint_rows and x.size(0) > 1:
            pooled = self.row_encoder(pooled.unsqueeze(0)).squeeze(0)
        return self.head(pooled)


class TabTransformer(TorchTabularModel):
    def _make_net(self, n_features):
        return _FeatureTokenTransformer(
            n_features, self.d_model, self.nhead, self.num_layers,
            self.dim_feedforward, self.dropout, self.task,
            use_cls=False, saint_rows=False,
        )


class FTTransformer(TorchTabularModel):
    def _make_net(self, n_features):
        return _FeatureTokenTransformer(
            n_features, self.d_model, self.nhead, self.num_layers,
            self.dim_feedforward, self.dropout, self.task,
            use_cls=True, saint_rows=False,
        )


class SAINT(TorchTabularModel):
    def _make_net(self, n_features):
        return _FeatureTokenTransformer(
            n_features, self.d_model, self.nhead, self.num_layers,
            self.dim_feedforward, self.dropout, self.task,
            use_cls=True, saint_rows=self.row_attention,
        )
