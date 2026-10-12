"""Đánh giá mô hình: chỉ số phân loại + kiểm định thống kê.

Điểm khác biệt so với cách làm thông thường:
  * Bootstrap khoảng tin cậy 95% cho AUC (điểm số đơn lẻ là vô nghĩa).
  * McNemar test: hai mô hình có thực sự khác nhau hay chỉ là nhiễu?
  * Diebold-Mariano test: so sánh chuỗi sai số dự báo.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, average_precision_score)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob > threshold).astype(int)
    out = {"n": int(len(y_true)), "positive_rate": float(y_true.mean())}
    try:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError as exc:
        out["error"] = str(exc)
    return out


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000,
                     seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Khoảng tin cậy của AUC bằng bootstrap (percentile)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n < 30 or len(np.unique(y_true)) < 2:
        return (np.nan, np.nan)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    if not aucs:
        return (np.nan, np.nan)
    return (float(np.percentile(aucs, 100 * alpha / 2)),
            float(np.percentile(aucs, 100 * (1 - alpha / 2))))


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                 exact: bool = True) -> dict:
    """McNemar: so sánh hai mô hình trên CÙNG tập dữ liệu.

    H0: hai mô hình có tỷ lệ lỗi như nhau.
    """
    y_true = np.asarray(y_true).astype(int)
    a = (np.asarray(pred_a).astype(int) == y_true)
    b = (np.asarray(pred_b).astype(int) == y_true)
    n01 = int(np.sum(~a & b))    # A sai, B đúng
    n10 = int(np.sum(a & ~b))    # A đúng, B sai
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0, "note": "hai mô hình dự đoán giống hệt nhau"}
    if exact or n_disc < 25:
        p = float(stats.binomtest(n10, n_disc, 0.5).pvalue)
        stat = float(min(n01, n10))
    else:
        stat = float((abs(n10 - n01) - 1) ** 2 / n_disc)
        p = float(1 - stats.chi2.cdf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": stat, "p_value": p}


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, h: int = 1,
                    power: int = 2) -> dict:
    """Diebold-Mariano: dãy sai số nào nhỏ hơn một cách có ý nghĩa?

    H0: hai mô hình có độ chính xác dự báo như nhau.
    """
    e1 = np.abs(np.asarray(errors_a, dtype=float))
    e2 = np.abs(np.asarray(errors_b, dtype=float))
    m = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[m], e2[m]
    if len(e1) < 30:
        return {"statistic": np.nan, "p_value": np.nan, "note": "không đủ mẫu"}

    d = e1 ** power - e2 ** power
    n = len(d)
    dm_stat = float(d.mean())

    def autocov(x, k):
        x = x - x.mean()
        return float(np.dot(x[k:], x[:-k]) / n) if k > 0 else float(np.dot(x, x) / n)

    gamma0 = autocov(d, 0)
    var = gamma0
    for k in range(1, h):
        var += 2 * (1 - k / h) * autocov(d, k)
    var = var / n
    if var <= 0:
        return {"statistic": np.nan, "p_value": np.nan, "note": "phương sai không xác định"}
    stat = dm_stat / np.sqrt(var)
    p = float(2 * (1 - stats.norm.cdf(abs(stat))))
    return {"statistic": float(stat), "p_value": p,
            "mean_loss_a": float((e1 ** power).mean()),
            "mean_loss_b": float((e2 ** power).mean())}


def compare_models(y_true: np.ndarray, scores: dict[str, np.ndarray],
                   base: str, ecfg: dict) -> pd.DataFrame:
    """Bảng so sánh từng mô hình với mô hình gốc (base)."""
    import pandas as pd

    rows = []
    y_true = np.asarray(y_true).astype(int)
    base_prob = np.asarray(scores[base], dtype=float)
    base_pred = (base_prob > 0.5).astype(int)
    for name, prob in scores.items():
        prob = np.asarray(prob, dtype=float)
        m = classification_metrics(y_true, prob)
        lo, hi = bootstrap_auc_ci(y_true, prob, n_boot=int(ecfg.get("n_bootstrap", 1000)))
        m["auc_ci_low"], m["auc_ci_high"] = lo, hi
        if name != base:
            mc = mcnemar_test(y_true, base_pred, (prob > 0.5).astype(int))
            m["mcnemar_p"] = mc["p_value"]
            dm = diebold_mariano(1 - prob, 1 - base_prob,
                                 power=int(ecfg.get("dm_power", 2)))
            m["dm_p"] = dm["p_value"]
        else:
            m["mcnemar_p"], m["dm_p"] = np.nan, np.nan
        m["model"] = name
        rows.append(m)
    cols = ["model", "n", "accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "mcc", "roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc",
            "mcnemar_p", "dm_p"]
    return pd.DataFrame(rows)[cols]
