import math
import numpy as np
import pandas as pd
from .tabular_model import TabularModel


class TransformerForecaster(TabularModel):
    """Small Transformer-encoder seq-to-one regressor.

    Uses sliding windows of length ``seq_len`` over the input feature
    matrix. Each window is projected to ``d_model``, augmented with sinusoidal
    positional encoding, passed through ``num_layers`` of
    ``nn.TransformerEncoderLayer``, then a linear head reads the last token.
    """

    def __init__(self, task="regression", seed=42,
                 seq_len=24, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1,
                 lr=1e-3, batch_size=128, epochs=5, device=None, **kwargs):
        super().__init__(task=task, seed=seed)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.device = device
        self.feature_names = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._train_tail = None

    def _make_windows(self, X, y=None):
        n = len(X)
        if n < self.seq_len:
            pad = np.repeat(X[:1], self.seq_len - n, axis=0)
            X = np.concatenate([pad, X], axis=0)
            n = len(X)
        windows = np.lib.stride_tricks.sliding_window_view(
            X, window_shape=self.seq_len, axis=0
        )
        windows = windows.transpose(0, 2, 1)
        if y is not None:
            y_aligned = y[self.seq_len - 1:]
            return windows, y_aligned
        return windows

    def _to_array(self, X):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self.device or "cpu"

        X = self._to_array(X_train)
        y = np.asarray(y_train, dtype=np.float32).reshape(-1)

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std < 1e-6] = 1.0
        X_norm = (X - self._x_mean) / self._x_std

        self._y_mean = float(y.mean())
        self._y_std = float(y.std())
        if self._y_std < 1e-6:
            self._y_std = 1.0
        y_norm = (y - self._y_mean) / self._y_std

        self._train_tail = X_norm[-(self.seq_len - 1):].copy() if self.seq_len > 1 else np.empty((0, X_norm.shape[1]), dtype=np.float32)

        Xw, yw = self._make_windows(X_norm, y_norm)
        d_in = Xw.shape[2]

        class _PosEnc(nn.Module):
            def __init__(self, d_model, max_len):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1).float()
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, : x.size(1)]

        nhead = self.nhead
        d_model = self.d_model
        # Make d_model divisible by nhead.
        if d_model % nhead != 0:
            d_model = (d_model // nhead) * nhead
            if d_model == 0:
                d_model = nhead
        seq_len = self.seq_len
        dim_ff = self.dim_feedforward
        dropout = self.dropout
        num_layers = self.num_layers

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(d_in, d_model)
                self.pe = _PosEnc(d_model, seq_len)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=dim_ff, dropout=dropout,
                    batch_first=True,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
                self.head = nn.Linear(d_model, 1)

            def forward(self, x):
                z = self.proj(x)
                z = self.pe(z)
                z = self.enc(z)
                return self.head(z[:, -1, :]).squeeze(-1)

        self.model = _Net().to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        ds = TensorDataset(
            torch.from_numpy(Xw).float(),
            torch.from_numpy(yw).float(),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            count = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(xb)
                count += len(xb)
            avg = total / max(count, 1)
            print(f"  [Transformer] epoch {epoch + 1}/{self.epochs} loss={avg:.4f}")
        return self

    def predict(self, X):
        import torch
        device = self.device or "cpu"
        X = self._to_array(X)
        X_norm = (X - self._x_mean) / self._x_std

        if self._train_tail is not None and len(self._train_tail) > 0:
            X_norm = np.concatenate([self._train_tail, X_norm], axis=0)

        Xw = self._make_windows(X_norm)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xw), self.batch_size):
                batch = torch.from_numpy(Xw[i:i + self.batch_size]).float().to(device)
                preds.append(self.model(batch).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        keep = preds[-len(X):]
        return keep * self._y_std + self._y_mean
