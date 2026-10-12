"""Mô phỏng giao dịch (backtest) có tính chi phí.

Lý do bắt buộc: Accuracy/AUC cao không có nghĩa là kiếm được tiền.
Sau khi trừ phí môi giới, thuế và trượt giá, rất nhiều "tín hiệu tốt" biến mất.

Chiến lược: quyết định vị thế tại đóng cửa ngày t, giữ đến đóng cửa ngày t+1.
  position_t = 1 nếu P(up) > threshold (long-only) ; ngược lại đứng ngoài.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def perf_stats(daily_ret: pd.Series, periods_per_year: int = 252) -> dict:
    r = daily_ret.dropna()
    if len(r) < 20:
        return {"n_days": len(r)}
    equity = (1 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    years = len(r) / periods_per_year
    cagr = float((equity.iloc[-1]) ** (1 / years) - 1.0) if years > 0 else np.nan
    sd = float(r.std())
    sharpe = float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan
    downside = r[r < 0]
    dsd = float(downside.std()) if len(downside) > 1 else np.nan
    sortino = float(r.mean() / dsd * np.sqrt(periods_per_year)) if dsd and dsd > 0 else np.nan
    mdd = max_drawdown(equity)
    return {
        "n_days": int(len(r)),
        "total_return": total,
        "cagr": cagr,
        "ann_vol": float(sd * np.sqrt(periods_per_year)),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd and mdd < 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
    }


def backtest_panel(panel: pd.DataFrame, prob_col: str, cost_bps: float,
                   allow_short: bool = False, threshold: float = 0.5,
                   periods_per_year: int = 252) -> tuple[dict, pd.DataFrame]:
    """Backtest trên bảng panel (nhiều mã), danh mục equal-weight mỗi ngày."""
    df = panel.copy()
    df = df.dropna(subset=["fwd_ret", prob_col])

    per_ticker = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_index()
        prob = g[prob_col].astype(float).values
        pos = np.sign(prob - threshold) if allow_short else (prob > threshold).astype(float)
        pos = pd.Series(pos, index=g.index)

        turnover = pos.diff().abs().fillna(pos.abs())
        gross = pos * g["fwd_ret"].astype(float)
        cost = turnover * (cost_bps / 10_000.0)
        net = gross - cost

        per_ticker.append(pd.DataFrame({
            "ticker": ticker, "pos": pos, "gross": gross, "cost": cost, "net": net,
            "bh": g["fwd_ret"].astype(float),
        }))

    bt = pd.concat(per_ticker)
    daily = bt.groupby(level=0).mean(numeric_only=True)

    stats = perf_stats(daily["net"], periods_per_year)
    stats.update({f"bh_{k}": v for k, v in perf_stats(daily["bh"], periods_per_year).items()})
    stats.update({
        "mean_turnover_daily": float(bt.groupby(level=0)["cost"].mean().mean() /
                                     (cost_bps / 10_000.0)) if cost_bps > 0 else 0.0,
        "exposure": float(bt["pos"].mean()),
        "total_cost_drag": float(bt["cost"].groupby(level=0).mean().sum()),
    })
    stats["sharpe_gross"] = perf_stats(daily["gross"], periods_per_year)["sharpe"]
    return stats, daily


def backtest_by_year(daily: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Hiệu quả từng năm -> dùng cho phần phân tích độ ổn định."""
    rows = []
    for year, g in daily.groupby(daily.index.year):
        s = perf_stats(g["net"], periods_per_year)
        b = perf_stats(g["bh"], periods_per_year)
        rows.append({"year": int(year), "sharpe": s.get("sharpe"),
                     "total_return": s.get("total_return"),
                     "max_drawdown": s.get("max_drawdown"),
                     "bh_total_return": b.get("total_return"),
                     "bh_sharpe": b.get("sharpe"), "n_days": s.get("n_days")})
    return pd.DataFrame(rows)
