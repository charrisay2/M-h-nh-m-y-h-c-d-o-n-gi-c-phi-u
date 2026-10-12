"""Sinh biểu đồ cho báo cáo (matplotlib, backend Agg -> chạy được không cần màn hình)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity_curves(curves: dict[str, pd.Series], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, r in curves.items():
        ax.plot((1 + r.fillna(0)).cumprod().index, (1 + r.fillna(0)).cumprod().values, label=name, lw=1.5)
    ax.set_title(title)
    ax.set_ylabel("Giá trị danh mục (khởi điểm = 1)")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_drawdown(daily_ret: pd.Series, title: str, path: Path) -> Path:
    eq = (1 + daily_ret.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.35)
    ax.set_title(title)
    ax.set_ylabel("Sụt giảm (%)")
    return _save(fig, path)


def plot_roc(y_true, scores: dict[str, np.ndarray], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="Ngẫu nhiên (0.500)")
    ax.set_xlabel("Tỷ lệ dương tính giả")
    ax.set_ylabel("Tỷ lệ dương tính thật")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def plot_feature_importance(series: pd.Series, title: str, path: Path, top: int = 20) -> Path:
    s = series.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.28 * len(s))))
    ax.barh(s.index, s.values, color="#2c7fb8")
    ax.set_title(title)
    ax.set_xlabel("Mức độ quan trọng (gain / |coef|)")
    return _save(fig, path)


def plot_metric_by_year(df: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.0))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["total_return"] * 100, w, label="Chiến lược", color="#27ae60")
    ax.bar(x + w / 2, df["bh_total_return"] * 100, w, label="Mua & nắm giữ", color="#95a5a6")
    ax.set_xticks(x)
    ax.set_xticklabels(df["year"].astype(int))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Lợi suất (%)")
    ax.set_title(title)
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_cv_auc(df: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.0))
    for model, g in df.groupby("model"):
        g = g.sort_values("fold")
        ax.plot(g["fold"], g["roc_auc"], marker="o", lw=1.5, label=model)
    ax.set_xlabel("Fold (walk-forward)")
    ax.set_ylabel("ROC-AUC")
    ax.axhline(0.5, color="black", ls="--", lw=0.9, label="Ngẫu nhiên")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)
