from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import joblib
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Compatibility alias for joblib pickles saved as "tabnet_surrogate"
sys.modules.setdefault("tabnet_surrogate", sys.modules[__name__])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _TabNetLite(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int, n_steps: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_steps = n_steps

        self.selector_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if step == 0 else hidden_dim, input_dim),
                nn.Softmax(dim=-1),
            )
            for step in range(n_steps)
        ])

        self.transformer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * n_steps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        outputs = []
        for selector in self.selector_layers:
            mask = selector(state)
            step_input = x * mask
            state = self.transformer(step_input)
            outputs.append(state)
        features = torch.cat(outputs, dim=1)
        return self.head(features)


class TabNetLiteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        n_classes: Optional[int] = None,
        hidden_dim: int = 64,
        n_steps: int = 3,
        dropout: float = 0.25,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 150,
        patience: int = 20,
        min_delta: float = 1e-4,
        validation_split: float = 0.15,
        random_state: int = 42,
        device: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.validation_split = validation_split
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _prepare_data(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if self.input_dim is None:
            self.input_dim = X.shape[1]
        if self.n_classes is None:
            self.n_classes = int(np.max(y)) + 1

        if self.validation_split and len(X) > 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_split,
                random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = X[:0], y[:0]

        return X_train, X_val, y_train, y_val

    def fit(self, X, y, X_val=None, y_val=None):
        _seed_everything(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if self.input_dim is None:
            self.input_dim = X.shape[1]
        if self.n_classes is None:
            self.n_classes = int(np.max(y)) + 1

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = self._prepare_data(X, y)
        else:
            X_train = X
            y_train = y
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)) if len(X_val) else None

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False) if val_ds is not None else None

        self.device_ = torch.device(self.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_ = _TabNetLite(
            input_dim=self.input_dim,
            n_classes=self.n_classes,
            hidden_dim=self.hidden_dim,
            n_steps=self.n_steps,
            dropout=self.dropout,
        ).to(self.device_)

        counts = np.bincount(y_train, minlength=self.n_classes).astype(np.float32)
        weights = counts.sum() / np.clip(counts, 1.0, None)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device_)

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state = None
        best_val = math.inf
        bad_epochs = 0
        history = []

        for epoch in range(self.max_epochs):
            self.model_.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 2.0)
                optimizer.step()
                running_loss += float(loss.item()) * len(xb)

            train_loss = running_loss / max(1, len(train_ds))

            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                preds = []
                trues = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        logits = self.model_(xb)
                        loss = criterion(logits, yb)
                        val_loss += float(loss.item()) * len(xb)
                        preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                        trues.extend(yb.cpu().numpy().tolist())
                val_loss = val_loss / max(1, len(val_ds))
                val_acc = accuracy_score(trues, preds) if trues else 0.0
            else:
                val_loss = train_loss
                val_acc = 0.0

            history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})

            if self.verbose:
                print(f'Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

            if val_loss + self.min_delta < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        self.history_ = history
        self.classes_ = np.arange(self.n_classes)
        return self

    def _predict_tensor(self, X: np.ndarray, mc_samples: int = 1, return_samples: bool = False):
        check_is_fitted(self, 'model_')
        X = np.asarray(X, dtype=np.float32)
        tensor = torch.from_numpy(X).to(self.device_)
        outputs = []

        with torch.no_grad():
            if mc_samples <= 1:
                self.model_.eval()
                logits = self.model_(tensor)
                probs = torch.softmax(logits, dim=1)
                arr = probs.cpu().numpy()
                return (arr, arr[None, :, :]) if return_samples else arr

            self.model_.train()
            for _ in range(mc_samples):
                logits = self.model_(tensor)
                probs = torch.softmax(logits, dim=1)
                outputs.append(probs.cpu().numpy())
            self.model_.eval()

        stacked = np.stack(outputs, axis=0)
        mean = np.mean(stacked, axis=0)
        return (mean, stacked) if return_samples else mean

    def predict_proba(self, X):
        return self._predict_tensor(X, mc_samples=1)

    def predict_proba_mc(self, X, mc_samples: int = 25, return_samples: bool = False):
        return self._predict_tensor(X, mc_samples=mc_samples, return_samples=return_samples)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

