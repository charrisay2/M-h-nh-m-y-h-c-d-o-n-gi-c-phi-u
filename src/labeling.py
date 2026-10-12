"""Định nghĩa nhãn (biến phụ thuộc).

Quy tắc vàng: nhãn phải được tạo bằng shift(-horizon) TRƯỚC KHI chia tập,
và hàng cuối cùng của mỗi mã (không có tương lai) phải bị loại bỏ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_labels(panel: pd.DataFrame, horizon: int = 1, mode: str = "threshold",
               threshold: float = 0.005) -> pd.DataFrame:
    """Thêm cột fwd_ret và y vào panel đã có cột close.

    mode="binary"     : y = 1 nếu return T+h > 0
    mode="threshold"  : y = 1 nếu return > +threshold, 0 nếu < -threshold,
                        bỏ qua các ngày ở giữa (vùng nhiễu)
    """
    panel = panel.copy()
    grp = panel.groupby("ticker", group_keys=False)["close"]
    fwd = (grp.shift(-horizon) / panel["close"]) - 1.0
    panel["fwd_ret"] = fwd

    if mode == "binary":
        panel["y"] = (fwd > 0).astype(float)
    elif mode == "threshold":
        panel["y"] = np.where(fwd > threshold, 1.0, np.where(fwd < -threshold, 0.0, np.nan))
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")

    # Không có tương lai -> không có nhãn
    panel.loc[fwd.isna(), "y"] = np.nan
    return panel


def drop_unlabeled(panel: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ các dòng chưa có nhãn và các đặc trưng còn NaN."""
    feature_cols = [c for c in panel.columns
                    if c not in {"y", "fwd_ret", "ticker", "close", "row_id"}]
    before = len(panel)
    out = panel.dropna(subset=["y"] + feature_cols)
    return out
