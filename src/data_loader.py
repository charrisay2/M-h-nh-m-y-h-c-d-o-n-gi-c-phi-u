"""Lớp dữ liệu: tải OHLCV từ VN (vnstock) và US (yfinance), lưu raw bất biến.

Nguyên tắc:
  * data/raw/ CHỈ GHI MỘT LẦN, không bao giờ chỉnh sửa tay.
  * Mọi lần tải đều tạo ra báo cáo chất lượng dữ liệu -> đây là đầu ra bắt buộc
    của Chương 3 (phần mô tả dữ liệu) và là nơi phát hiện lỗi điều chỉnh giá.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import ROOT, ensure_dir, get_logger

log = get_logger("data")

RAW_COLUMNS = ["open", "high", "low", "close", "volume"]


# ----------------------------------------------------------------- chuẩn hóa
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Đưa mọi nguồn về chung schema: index=date (datetime), cột OHLCV thường."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=RAW_COLUMNS)

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).strip().lower() for c in df.columns]

    renamed = {"time": "date", "datetime": "date", "tradingdate": "date",
               "adj close": "close", "adjclose": "close", "adj_close": "close",
               "vol": "volume", "totalvolume": "volume", "matchvolume": "volume"}
    df = df.rename(columns={k: v for k, v in renamed.items() if k in df.columns})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]

    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df = df[(df["close"] > 0)]
    df.index.name = "date"
    return df


# ----------------------------------------------------------------- nguồn US
def _download_us(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _normalize(df.reset_index())


# ----------------------------------------------------------------- nguồn VN
def _download_vn(ticker: str, start: str, end: str) -> pd.DataFrame:
    """vnstock thay đổi API nhiều lần -> thử lần lượt các biến thể."""
    errors = []
    start_s, end_s = pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")

    try:  # vnstock >= 3.x
        from vnstock import Vnstock

        stock = Vnstock().stock(symbol=ticker, source="VCI")
        df = stock.quote.history(start=start_s, end=end_s, interval="1D")
        return _normalize(df.reset_index())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"vnstock3: {type(exc).__name__}: {exc}")

    try:  # vnstock 0.2.x
        from vnstock import stock_historical_data

        df = stock_historical_data(symbol=ticker, start_date=start_s, end_date=end_s)
        return _normalize(df.reset_index())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"vnstock_legacy: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"Không tải được {ticker} bằng vnstock. Chi tiết: {errors}")


def _download_one(ticker: str, start: str, end: str, source: str) -> pd.DataFrame:
    if source == "yfinance":
        return _download_us(ticker, start, end)
    if source == "vnstock":
        return _download_vn(ticker, start, end)
    if source == "vnstock":
        return _download_vn(ticker, start, end)
    if source == "synthetic":
        from synthetic import make_synthetic_ohlcv

        return make_synthetic_ohlcv(ticker, start, end)
    raise ValueError(f"Nguồn không hỗ trợ: {source}")


# ----------------------------------------------------------------- API chính
def download_market(cfg: dict, market: str, force: bool = False) -> dict[str, pd.DataFrame]:
    """Tải (hoặc đọc lại) toàn bộ mã của một thị trường."""
    mcfg = cfg["markets"][market]
    raw_dir = ensure_dir(Path(cfg["project"]["raw_dir"]) / market)
    start, end = cfg["project"]["start_date"], cfg["project"]["end_date"]
    source = mcfg["source"]
    suffix = mcfg.get("ticker_suffix", "")
    min_days = int(cfg["project"].get("min_history_days", 1500))

    if source == "cache":  # chỉ đọc lại từ đĩa, không gọi mạng
        return load_raw(cfg, market)

    out: dict[str, pd.DataFrame] = {}
    skipped: list[tuple[str, int]] = []
    for i, ticker in enumerate(mcfg["tickers"], 1):
        path = raw_dir / f"{ticker}.csv"
        if path.exists() and not force:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if len(df) >= min_days:
                out[ticker] = df
            continue
        try:
            df = _download_one(f"{ticker}{suffix}", start, end, source)
            if len(df) < min_days:
                skipped.append((ticker, len(df)))
                log.warning(f"[{market}] {ticker}: chỉ {len(df)} dòng (< {min_days}) -> loại")
                continue
            df.to_csv(path)
            out[ticker] = df
            log.info(f"[{market}] ({i}/{len(mcfg['tickers'])}) {ticker}: {len(df)} dòng")
        except Exception as exc:  # noqa: BLE001
            log.error(f"[{market}] {ticker} lỗi: {type(exc).__name__}: {exc}")

    if skipped:
        log.warning(f"[{market}] Loại {len(skipped)} mã do lịch sử quá ngắn: {skipped}")
    return out


def load_raw(cfg: dict, market: str, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    raw_dir = ROOT / cfg["project"]["raw_dir"] / market
    tickers = tickers or cfg["markets"][market]["tickers"]
    min_days = int(cfg["project"].get("min_history_days", 1500))
    data = {}
    for t in tickers:
        path = raw_dir / f"{t}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if len(df) >= min_days:
                data[t] = df
    return data


def build_market_factor(frames: dict[str, pd.DataFrame], index_ticker: str | None,
                        cfg: dict, market: str) -> pd.Series:
    """Lợi suất của chỉ số thị trường; nếu không tải được chỉ số thì dùng
    trung bình equal-weight của các mã đã tải làm biến đại diện."""
    raw_dir = ROOT / cfg["project"]["raw_dir"] / market
    idx_path = raw_dir / f"_{index_ticker.replace('^', 'IDX_')}.csv"
    series = None
    suffix = cfg["markets"][market].get("ticker_suffix", "")
    symbol = index_ticker if index_ticker.startswith("^") else f"{index_ticker}{suffix}"

    if idx_path.exists():
        series = pd.read_csv(idx_path, index_col=0, parse_dates=True)["close"]
    elif index_ticker:
        try:
            source = cfg["markets"][market]["source"]
            df = _download_one(symbol, cfg["project"]["start_date"],
                               cfg["project"]["end_date"],
                               "yfinance" if source == "cache" else source)
            if len(df) > 100:
                df.to_csv(idx_path)
                series = df["close"]
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[{market}] Không tải được chỉ số {index_ticker}: {exc}")

    if series is None or len(series) < 100:
        log.warning(f"[{market}] Dùng equal-weight của {len(frames)} mã làm biến thị trường")
        rets = pd.concat({k: v["close"].pct_change() for k, v in frames.items()}, axis=1)
        series = (1 + rets.mean(axis=1, skipna=True)).cumprod()

    return np.log(series / series.shift(1)).dropna()


# ----------------------------------------------------------------- làm sạch
def clean_frames(frames: dict[str, pd.DataFrame], max_daily_std: float = 0.08,
                 max_jump_frac: float = 0.01) -> tuple[dict[str, pd.DataFrame], list[tuple]]:
    """Loại các mã có chuỗi giá không hợp lý.

    Yahoo đôi khi trả về chuỗi đã điều chỉnh sai (ví dụ mã FPT của VN có giá ~1e-8
    và hơn 1.500 ngày biến động > 15%), sẽ làm hỏng toàn bộ panel nếu không lọc.
    Tiêu chí (độc lập với đơn vị tiền tệ):
      * std lợi suất ngày > 8%   (cổ phiếu thật hiếm khi vượt 4-5%)
      * tỷ lệ ngày |lợi suất| > 15% vượt 1%
      * giá trung vị <= 0
    """
    keep: dict[str, pd.DataFrame] = {}
    dropped: list[tuple[str, str]] = []
    for ticker, df in frames.items():
        close = df["close"].astype(float)
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(ret) < 500:
            dropped.append((ticker, f"chỉ {len(ret)} ngày có lợi suất"))
            continue
        std = float(ret.std())
        jump = float((ret.abs() > 0.15).mean())
        if not np.isfinite(std) or std > max_daily_std:
            dropped.append((ticker, f"std ngày = {std:.3f} (> {max_daily_std})"))
            continue
        if jump > max_jump_frac:
            dropped.append((ticker, f"{jump:.1%} ngày biến động > 15%"))
            continue
        if close.median() <= 0:
            dropped.append((ticker, "giá trung vị <= 0"))
            continue
        keep[ticker] = df
    return keep, dropped


# ----------------------------------------------------------------- chất lượng
def build_quality_report(cfg: dict, market: str,
                         frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Bảng chất lượng dữ liệu từng mã -> đầu ra cho Chương 3."""
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
    rows = []
    for ticker, df in frames.items():
        close = df["close"].astype(float)
        ret = np.log(close / close.shift(1))
        rows.append({
            "ticker": ticker,
            "n_days": len(df),
            "first": df.index.min().date(),
            "last": df.index.max().date(),
            "missing_vs_union_pct": 1 - len(df) / max(len(all_dates), 1),
            "zero_volume_days": int((df["volume"].fillna(0) == 0).sum()),
            "dup_dates": int(df.index.duplicated().sum()),
            "flat_price_days": int((ret.abs() < 1e-12).sum()),
            "jump_gt_15pct_days": int((ret.abs() > 0.15).sum()),
            "ann_vol": float(ret.std() * np.sqrt(252)) if len(ret) > 20 else np.nan,
            "median_volume": float(df["volume"].median()),
        })
    report = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    out_dir = ensure_dir(Path(cfg["project"]["reports_dir"]) / "tables")
    report.to_csv(out_dir / f"data_quality_{market}.csv", index=False)
    return report
