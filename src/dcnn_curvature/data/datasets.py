from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

EPS = 1e-8


@dataclass
class Standardizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float | np.ndarray
    y_std: float | np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "Standardizer":
        if y.ndim > 1:
            y_mean, y_std = y.mean(axis=0), y.std(axis=0) + EPS
        else:
            y_mean, y_std = float(y.mean()), float(y.std() + EPS)
        return cls(
            x_mean=x.mean(axis=0),
            x_std=x.std(axis=0) + EPS,
            y_mean=y_mean,
            y_std=y_std,
        )

    def transform(self, x: np.ndarray, y: np.ndarray | None = None):
        xn = (x - self.x_mean) / self.x_std
        if y is None:
            return xn
        yn = (y - self.y_mean) / self.y_std
        return xn, yn

    def inverse_transform_y(self, yn: np.ndarray) -> np.ndarray:
        return yn * self.y_std + self.y_mean

    def inverse_transform_x(self, xn: np.ndarray) -> np.ndarray:
        return xn * self.x_std + self.x_mean


def make_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )


def make_dataloader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(make_dataset(x, y), batch_size=batch_size, shuffle=shuffle)
