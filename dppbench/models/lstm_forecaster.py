import numpy as np
import pandas as pd
from .tabular_model import TabularModel


class LSTMForecaster(TabularModel):
    """Simple PyTorch LSTM seq-to-one regressor for time-series forecasting.

    Builds sliding windows of length ``seq_len`` over the input feature matrix
    and trains a single-layer LSTM with a linear regression head. Mirrors the
    ``TabularModel`` interface so it plugs into ``scripts/train.py``.
    """

    def __init__(self, task="regression", seed=42,
                 seq_len=24, hidden_size=64, num_layers=1, dropout=0.0,
                 lr=1e-3, batch_size=128, epochs=5, device=None, **kwargs):
        super().__init__(task=task, seed=seed)
        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
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
        self._train_tail = None  # last seq_len-1 train rows for warm start

    def _make_windows(self, X, y=None):
        n = len(X)
        if n < self.seq_len:
            # pad at the front by repeating the first row
            pad = np.repeat(X[:1], self.seq_len - n, axis=0)
            X = np.concatenate([pad, X], axis=0)
            n = len(X)
        windows = np.lib.stride_tricks.sliding_window_view(
            X, window_shape=self.seq_len, axis=0
        )
        windows = windows.transpose(0, 2, 1)  # (N-seq_len+1, seq_len, D)
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

        # Standardize for stable LSTM training.
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

        class _Net(nn.Module):
            def __init__(self, d_in, hidden, layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=d_in, hidden_size=hidden,
                    num_layers=layers, batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self.model = _Net(d_in, self.hidden_size, self.num_layers, self.dropout).to(device)
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
            print(f"  [LSTM] epoch {epoch + 1}/{self.epochs} loss={avg:.4f}")
        return self

    def predict(self, X):
        import torch
        device = self.device or "cpu"
        X = self._to_array(X)
        X_norm = (X - self._x_mean) / self._x_std

        # prepend training tail so first prediction also gets a full window
        if self._train_tail is not None and len(self._train_tail) > 0:
            X_norm = np.concatenate([self._train_tail, X_norm], axis=0)
            offset = len(self._train_tail)
        else:
            offset = 0

        Xw = self._make_windows(X_norm)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xw), self.batch_size):
                batch = torch.from_numpy(Xw[i:i + self.batch_size]).float().to(device)
                preds.append(self.model(batch).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        # align: predictions correspond to indices [seq_len-1 + offset : ]
        # we want output length == len(X) (original input length)
        # because we prepended (seq_len-1) tail rows, the first prediction
        # already corresponds to the first original X row.
        keep = preds[-len(X):]
        return keep * self._y_std + self._y_mean
