"""Sinh dữ liệu GIẢ LẬP - CHỈ dùng để smoke-test pipeline khi không có mạng.

⚠️ CẢNH BÁO: Kết quả chạy trên dữ liệu giả lập KHÔNG có giá trị học thuật.
   Không được dùng số liệu từ chế độ này trong báo cáo nộp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(ticker: str, start: str, end: str, seed: int | None = None,
                         mu: float = 0.0002, sigma0: float = 0.015) -> pd.DataFrame:
    """GBM + phân cụm biến động (GARCH-like), tạo OHLCV nhất quán."""
    seed = seed if seed is not None else (abs(hash(ticker)) % (2**31))
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start, end)
    n = len(dates)

    # Biến động có tính dai dẵn (volatility clustering)
    vol = np.empty(n)
    vol[0] = sigma0
    for t in range(1, n):
        vol[t] = 0.02 + 0.92 * vol[t - 1] + 0.05 * rng.normal(0, sigma0)

    shock = rng.normal(0, 1, n)
    ret = mu + vol * shock
    close = 100 * np.exp(np.cumsum(ret))

    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = np.exp(rng.normal(13, 0.6, n)) * (1 + 2 * np.abs(ret) / vol)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )
