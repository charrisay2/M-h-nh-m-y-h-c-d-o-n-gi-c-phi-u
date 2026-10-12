"""Tạo đặc trưng kỹ thuật.

NGUYÊN TẮC BẮT BUỘC: mọi đặc trưng tại ngày t chỉ được dùng dữ liệu đến hết ngày t.
Không dùng rolling(center=True), không dùng shift âm, không fit scaler trên toàn tập.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------- chỉ báo cơ sở
def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = _ema(close, fast) - _ema(close, slow)
    sig = _ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger_pctb(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    mid = _sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper, lower = mid + k * sd, mid - k * sd
    return (close - lower) / (upper - lower).replace(0.0, np.nan)


def true_range(high, low, close) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high, low, close, n: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def stochastic(high, low, close, n: int = 14) -> pd.Series:
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    return 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)


def adx(high, low, close, n: int = 14) -> pd.Series:
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_n = true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean() / tr_n
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean() / tr_n
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def parkinson_vol(high, low, n: int = 21) -> pd.Series:
    """Ước lượng biến động từ biên độ H-L, hiệu quả hơn close-to-close."""
    hl = np.log(high / low) ** 2
    var = hl.rolling(n, min_periods=n).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * 252.0)


def obv(close, volume) -> pd.Series:
    return (np.sign(close.diff().fillna(0.0)) * volume).cumsum()


# --------------------------------------------------------------- bộ đặc trưng
def compute_features(df: pd.DataFrame, market_ret: pd.Series | None,
                     fcfg: dict) -> pd.DataFrame:
    """Tính toàn bộ đặc trưng cho MỘT mã."""
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    windows = fcfg.get("windows", [5, 10, 21, 63, 252])
    n_rsi = int(fcfg.get("rsi_period", 14))
    n_atr = int(fcfg.get("atr_period", 14))
    n_adx = int(fcfg.get("adx_period", 14))
    n_bb = int(fcfg.get("bb_period", 20))
    n_stoch = int(fcfg.get("stoch_period", 14))
    w_beta = int(fcfg.get("beta_window", 60))

    ret = np.log(c / c.shift(1))
    feats: dict[str, pd.Series] = {}

    # 1) Lợi suất quá khứ (đã dịch về quá khứ, không chứa tương lai)
    for w in windows:
        feats[f"ret_{w}"] = c / c.shift(w) - 1.0
    for k in range(1, 6):
        feats[f"ret_lag{k}"] = ret.shift(k)

    # 2) Xu hướng
    for w in (10, 20, 50, 200):
        feats[f"px_over_sma{w}"] = c / _sma(c, w) - 1.0
    feats["sma20_over_sma60"] = _sma(c, 20) / _sma(c, 60) - 1.0
    feats["sma50_over_sma200"] = _sma(c, 50) / _sma(c, 200) - 1.0
    feats.update(macd(c).to_dict("series"))

    # 3) Động lượng
    feats[f"rsi_{n_rsi}"] = rsi(c, n_rsi)
    feats[f"stoch_k"] = stochastic(h, l, c, n_stoch)
    for w in (5, 21):
        feats[f"roc_{w}"] = c / c.shift(w) - 1.0

    # 4) Biến động
    atr_n = atr(h, l, c, n_atr)
    feats["atr_pct"] = atr_n / c
    feats[f"bb_pctb"] = bollinger_pctb(c, n_bb)
    for w in (5, 21, 63):
        feats[f"realized_vol_{w}"] = ret.rolling(w, min_periods=w).std() * np.sqrt(252.0)
    feats["parkinson_vol_21"] = parkinson_vol(h, l, 21)
    feats["vol_ratio_5_21"] = (
        feats["realized_vol_5"] / feats["realized_vol_21"].replace(0.0, np.nan)
    )

    # 5) Khối lượng (đặc biệt quan trọng với thị trường VN)
    logv = np.log1p(v)
    feats["log_volume"] = logv
    feats["volume_z20"] = (logv - logv.rolling(20, min_periods=20).mean()) / \
        logv.rolling(20, min_periods=20).std().replace(0.0, np.nan)
    feats["volume_over_ma20"] = v / _sma(v, 20).replace(0.0, np.nan)
    obv_s = obv(c, v)
    feats["obv_slope_10"] = (obv_s - obv_s.shift(10)) / \
        obv_s.diff().rolling(50, min_periods=50).std().replace(0.0, np.nan)

    # 6) Cấu trúc vị thế giá
    feats["adx"] = adx(h, l, c, n_adx)
    feats["dist_from_52w_high"] = c / h.rolling(252, min_periods=252).max() - 1.0
    feats["dist_from_52w_low"] = c / l.rolling(252, min_periods=252).min() - 1.0
    feats["hl_range"] = (h - l) / c

    # 7) Đặc trưng thị trường (nhân tố chỉ số)
    if market_ret is not None:
        mkt = market_ret.reindex(c.index)
        for w in (1, 5, 21):
            feats[f"mkt_ret_{w}"] = (1 + mkt.fillna(0)).rolling(w, min_periods=1).apply(
                np.prod, raw=True) - 1.0 if w > 1 else mkt
        cov = ret.rolling(w_beta, min_periods=w_beta).cov(mkt)
        var = mkt.rolling(w_beta, min_periods=w_beta).var()
        feats["beta_60"] = cov / var.replace(0.0, np.nan)
        feats["excess_ret_21"] = feats["ret_21"] - (
            (1 + mkt.fillna(0)).rolling(21, min_periods=21).apply(np.prod, raw=True) - 1.0
        )

    # 8) Lịch
    idx = pd.DatetimeIndex(c.index)
    feats["day_of_week"] = idx.dayofweek.astype(float).values
    feats["month"] = idx.month.astype(float).values

    out = pd.DataFrame(feats, index=c.index)
    out["ret_1d"] = ret                       # dùng cho baseline / backtest
    return out.replace([np.inf, -np.inf], np.nan)


def build_panel(frames: dict[str, pd.DataFrame], market_ret: pd.Series,
                fcfg: dict) -> pd.DataFrame:
    """Ghép đặc trưng của tất cả mã thành một bảng panel."""
    per_ticker = []
    for ticker, df in frames.items():
        f = compute_features(df, market_ret, fcfg)
        f["ticker"] = ticker
        f["close"] = df["close"].astype(float)
        per_ticker.append(f)

    panel = pd.concat(per_ticker)
    panel["row_id"] = np.arange(len(panel))

    # Loại bỏ giai đoạn khởi động của từng mã (rolling window chưa đủ dữ liệu).
    # Dùng cumcount thay vì groupby.apply: pandas >= 3.0 mặc định loại bỏ
    # cột group khỏi kết quả của apply.
    warmup = int(fcfg.get("warmup_days", 260))
    seq = panel.groupby("ticker").cumcount()
    panel = panel[seq >= warmup]
    return panel.sort_index()
