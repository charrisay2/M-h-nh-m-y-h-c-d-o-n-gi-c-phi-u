"""Thiết kế thực nghiệm: walk-forward validation có purge + embargo.

Tham khảo: López de Prado, "Advances in Financial Machine Learning" (2018), chương 7.

Lý do không dùng KFold / train_test_split(shuffle=True):
  * Chuỗi thời gian có tự tương quan -> shuffle làm rò rỉ thông tin tương lai.
  * Đặc trưng dùng cửa sổ 21 ngày + nhãn dùng T+1 -> các mẫu ở ranh giới
    train/validation bị chồng lấn -> purge xóa vùng chồng lấn, embargo tạo
    khoảng cách an toàn sau tập validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    meta: dict = field(default_factory=dict)


def _purge_embargo(n_samples: int, train_pos: np.ndarray, val_pos: np.ndarray,
                   purge: int, embargo: int) -> np.ndarray:
    """Loại khỏi tập huấn luyện các mẫu nằm quá gần tập validation."""
    val_lo, val_hi = int(val_pos.min()), int(val_pos.max())
    keep = (train_pos < val_lo - purge) | (train_pos > val_hi + embargo)
    return train_pos[keep]


def walk_forward_folds(dates: pd.DatetimeIndex | pd.Series, n_folds: int = 5,
                       val_window_days: int = 252, purge_days: int = 21,
                       embargo_days: int = 5, expanding: bool = True,
                       min_train_days: int = 500) -> list[Fold]:
    """Chia walk-forward trên trục thời gian.

    Trả về danh sách Fold, các fold xếp theo thứ tự thời gian.
    """
    idx = pd.DatetimeIndex(np.asarray(dates))
    n = len(idx)
    if n < min_train_days + val_window_days:
        raise ValueError(f"Dữ liệu quá ngắn: {n} dòng")

    folds: list[Fold] = []
    for i in range(n_folds):
        val_end_pos = n - 1 - (n_folds - 1 - i) * val_window_days
        val_start_pos = val_end_pos - val_window_days + 1
        if val_start_pos <= 0:
            continue

        val_pos = np.arange(val_start_pos, val_end_pos + 1)
        train_pos_full = np.arange(0, val_start_pos)
        train_pos = _purge_embargo(n, train_pos_full, val_pos, purge_days, embargo_days)

        if not expanding:  # cửa sổ trượt cố định
            train_pos = train_pos[train_pos >= train_pos.max() - 4 * val_window_days]

        if len(train_pos) < min_train_days:
            continue

        folds.append(Fold(
            train_idx=train_pos, val_idx=val_pos,
            train_start=idx[train_pos.min()], train_end=idx[train_pos.max()],
            val_start=idx[val_start_pos], val_end=idx[val_end_pos],
            meta={"fold": i + 1},
        ))
    return folds


def holdout_indices(dates: pd.DatetimeIndex | pd.Series, start: str, end: str):
    """Tách tập test out-of-sample theo khoảng ngày."""
    idx = pd.DatetimeIndex(np.asarray(dates))
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    test_pos = np.where(mask)[0]
    train_pos = np.where(~mask)[0]
    return train_pos, test_pos


def describe_folds(folds: list[Fold]) -> pd.DataFrame:
    return pd.DataFrame([{
        "fold": f.meta["fold"],
        "train_start": f.train_start.date(),
        "train_end": f.train_end.date(),
        "n_train": len(f.train_idx),
        "val_start": f.val_start.date(),
        "val_end": f.val_end.date(),
        "n_val": len(f.val_idx),
    } for f in folds])
