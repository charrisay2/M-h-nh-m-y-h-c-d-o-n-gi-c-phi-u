"""Điều phối toàn bộ thực nghiệm cho MỘT thị trường.

Luồng: dữ liệu -> đặc trưng -> nhãn -> walk-forward CV -> chọn mô hình
       -> đánh giá out-of-sample -> backtest có chi phí -> SHAP -> lưu kết quả
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import data_loader
import evaluate
import features as feat_mod
import labeling
import models as mdl
import plots
import split as split_mod
from utils import ROOT, ensure_dir, get_logger, save_json, set_seed

log = get_logger("pipeline")
FEATURE_EXCLUDE = {"y", "fwd_ret", "ticker", "close", "row_id", "ret_1d"}


def feature_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in FEATURE_EXCLUDE]


def _date_mask(panel: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(panel.index.isin(dates))


# ------------------------------------------------------------------ dữ liệu
def build_dataset(cfg: dict, market: str, force_download: bool = False):
    log.info(f"[{market}] ---- BƯỚC 1: DỮ LIỆU ----")
    frames = data_loader.download_market(cfg, market, force=force_download)
    if not frames:
        raise RuntimeError(f"[{market}] Không có dữ liệu nào. Kiểm tra nguồn/mạng.")
    frames = {t: f for t, f in frames.items() if len(f) > 500}
    qcfg = cfg.get("data_quality", {})
    frames, dropped = data_loader.clean_frames(
        frames,
        max_daily_std=float(qcfg.get("max_daily_std", 0.08)),
        max_jump_frac=float(qcfg.get("max_jump_frac", 0.01)),
    )
    if dropped:
        log.warning(f"[{market}] Loại {len(dropped)} mã có chuỗi giá bất thường:")
        for t, why in dropped:
            log.warning(f"    - {t}: {why}")
    log.info(f"[{market}] {len(frames)} mã hợp lệ")

    report = data_loader.build_quality_report(cfg, market, frames)
    market_ret = data_loader.build_market_factor(
        frames, cfg["markets"][market]["index_ticker"], cfg, market)

    log.info(f"[{market}] ---- BƯỚC 2: ĐẶC TRƯNG ----")
    panel = feat_mod.build_panel(frames, market_ret, cfg["features"])
    lcfg = cfg["labeling"]
    panel = labeling.add_labels(panel, horizon=int(lcfg["horizon"]),
                                mode=lcfg["mode"], threshold=float(lcfg["threshold"]))
    panel = labeling.drop_unlabeled(panel)
    log.info(f"[{market}] panel: {panel.shape[0]:,} dòng, {len(feature_cols(panel))} đặc trưng, "
             f"tỷ lệ nhãn 1 = {panel['y'].mean():.3f}")
    return frames, panel, report


# ------------------------------------------------------------------ CV
def cross_validate(cfg: dict, panel: pd.DataFrame, market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info(f"[{market}] ---- BƯỚC 3: WALK-FORWARD CV ----")
    vcfg = cfg["validation"]
    cols = feature_cols(panel)

    # Tập CV chỉ gồm dữ liệu TRƯỚC tập test
    test_start = pd.Timestamp(cfg["test"]["start_date"])
    cv_panel = panel[panel.index < test_start]
    all_dates = pd.DatetimeIndex(sorted(cv_panel.index.unique()))

    folds = split_mod.walk_forward_folds(
        all_dates,
        n_folds=int(vcfg["n_folds"]),
        val_window_days=int(vcfg["val_window_days"]),
        purge_days=int(vcfg["purge_days"]),
        embargo_days=int(vcfg["embargo_days"]),
        expanding=bool(vcfg["expanding"]),
    )
    fold_desc = split_mod.describe_folds(folds)
    log.info(f"[{market}] {len(folds)} fold\n{fold_desc.to_string(index=False)}")

    rows = []
    for f in folds:
        tr_dates = all_dates[f.train_idx]
        va_dates = all_dates[f.val_idx]
        tr_mask = _date_mask(cv_panel, tr_dates)
        va_mask = _date_mask(cv_panel, va_dates)
        X_tr, y_tr = cv_panel.loc[tr_mask, cols], cv_panel.loc[tr_mask, "y"].values
        X_va, y_va = cv_panel.loc[va_mask, cols], cv_panel.loc[va_mask, "y"].values

        for name in cfg["models"]["evaluate"]:
            est = mdl.get_model(name, cfg, seed=int(cfg["project"]["seed"]))
            mdl.fit_model(est, X_tr, y_tr, name, X_va, y_va, cfg)
            prob = mdl.predict_proba(est, X_va, name)
            m = evaluate.classification_metrics(y_va, prob)
            m.update({"market": market, "model": name, "fold": f.meta["fold"],
                      "n_train": len(X_tr), "n_val": len(X_va),
                      "val_start": str(va_dates.min().date()),
                      "val_end": str(va_dates.max().date())})
            rows.append(m)
            log.info(f"[{market}] fold {f.meta['fold']} | {name:9s} AUC={m.get('roc_auc', float('nan')):.4f} "
                     f"Acc={m.get('accuracy', float('nan')):.4f}")

    cv = pd.DataFrame(rows)
    summary = (cv.groupby("model")[["roc_auc", "accuracy", "f1", "mcc", "pr_auc"]]
                 .agg(["mean", "std"]).round(4))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index().sort_values("roc_auc_mean", ascending=False)
    return cv, summary


# ------------------------------------------------------------------ test
def _refit_full(model, name: str, X: pd.DataFrame, y: np.ndarray, cfg: dict) -> object:
    """Fit lại trên TOÀN BỘ dữ liệu trước test với số cây đã chọn bằng early stopping.

    Không dùng lại đối tượng cũ (xgboost >= 3.x giữ callback early_stopping),
    mà tạo mô hình mới với n_estimators = best_iteration.
    """
    if name not in {"lightgbm", "xgboost"}:
        model.fit(X, y)
        return model

    best = getattr(model, "best_iteration_", None)
    if best is None:
        best = getattr(model, "best_iteration", None)
    n_trees = int(best) if (best is not None and int(best) > 0) else \
        int(cfg.get(name, {}).get("n_estimators", 500))

    params = {k: v for k, v in cfg.get(name, {}).items() if k != "early_stopping_rounds"}
    params["n_estimators"] = n_trees
    fresh = mdl.get_model(name, {**cfg, name: params}, seed=int(cfg["project"]["seed"]))
    fresh.fit(X, y)
    return fresh


def evaluate_test(cfg: dict, panel: pd.DataFrame, frames: dict, market: str,
                  best_model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info(f"[{market}] ---- BƯỚC 4: ĐÁNH GIÁ OUT-OF-SAMPLE ----")
    cols = feature_cols(panel)
    test_start = pd.Timestamp(cfg["test"]["start_date"])
    test_end = pd.Timestamp(cfg["test"]["end_date"])
    test_mask = (panel.index >= test_start) & (panel.index <= test_end)
    train_mask = ~test_mask

    X_tr, y_tr = panel.loc[train_mask, cols], panel.loc[train_mask, "y"].values
    X_te, y_te = panel.loc[test_mask, cols], panel.loc[test_mask, "y"].values
    log.info(f"[{market}] train={len(X_tr):,} dòng | test={len(X_te):,} dòng")

    # Chia nhỏ để early stopping khi fit cuối cùng
    dates_tr = pd.DatetimeIndex(sorted(panel.loc[train_mask].index.unique()))
    cut_pos = int(len(dates_tr) * 0.85)
    purge = int(cfg["validation"]["purge_days"])
    fit_dates = dates_tr[: max(cut_pos - purge, 1)]
    es_dates = dates_tr[cut_pos:]
    fit_mask = _date_mask(panel.loc[train_mask], fit_dates)
    es_mask = _date_mask(panel.loc[train_mask], es_dates)

    scores: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}
    for name in list(cfg["models"]["evaluate"]) + (["random"] if "random" not in cfg["models"]["evaluate"] else []):
        est = mdl.get_model(name, cfg, seed=int(cfg["project"]["seed"]))
        mdl.fit_model(est, X_tr.loc[fit_mask], y_tr[fit_mask], name,
                      X_tr.loc[es_mask], y_tr[es_mask], cfg)
        est = _refit_full(est, name, X_tr, y_tr, cfg)
        prob = mdl.predict_proba(est, X_te, name)
        scores[name] = prob
        fitted[name] = est
        log.info(f"[{market}] {name:9s} test AUC={evaluate.classification_metrics(y_te, prob).get('roc_auc', float('nan')):.4f}")

    # ARIMA baseline (chỉ chạy trên tập test vì chi phí tính toán)
    acfg = cfg.get("arima", {})
    ret_by_ticker = {t: np.log(f["close"] / f["close"].shift(1)) for t, f in frames.items()}
    arima_map = mdl.arima_direction_scores(
        ret_by_ticker, test_start, order=tuple(acfg.get("order", (2, 0, 2))),
        refit_every=int(acfg.get("refit_every", 20)))
    te_index = panel.loc[test_mask]
    raw = np.array([arima_map.get((t, d), np.nan) for t, d in
                    zip(te_index["ticker"], te_index.index)])
    coverage = float(np.isfinite(raw).mean())
    scores["arima"] = 1.0 / (1.0 + np.exp(-100.0 * np.nan_to_num(raw, nan=0.0)))
    log.info(f"[{market}] arima     test AUC={evaluate.classification_metrics(y_te, scores['arima']).get('roc_auc', float('nan')):.4f} "
             f"(phủ {coverage:.1%} mẫu)")

    base = "naive" if "naive" in scores else list(scores)[0]
    table = evaluate.compare_models(y_te, scores, base, cfg["evaluation"])

    preds = te_index[["ticker", "close", "fwd_ret", "y"]].copy()
    for name, s in scores.items():
        preds[f"prob_{name}"] = s
    return table, preds, fitted


# ------------------------------------------------------------------ SHAP
def shap_importance(model, name: str, X: pd.DataFrame, cfg: dict) -> pd.Series | None:
    ecfg = cfg.get("explainability", {})
    if not ecfg.get("shap", True):
        return None
    try:
        import shap

        n = min(int(ecfg.get("shap_sample", 2000)), len(X))
        sample = X.sample(n, random_state=int(cfg["project"]["seed"])) if len(X) > n else X
        if name in {"lightgbm", "xgboost", "rf"}:
            sv = shap.TreeExplainer(model).shap_values(sample)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            if getattr(sv, "ndim", 2) == 3:
                sv = sv[:, :, 1]
        else:
            sv = shap.LinearExplainer(model, sample).shap_values(sample)
        return pd.Series(np.abs(sv).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"SHAP không khả dụng ({type(exc).__name__}: {exc}) -> dùng feature importance thường")
        return None


def model_importance(model, name: str, cols: list[str]) -> pd.Series:
    if name == "lightgbm" and hasattr(model, "booster_"):
        return pd.Series(model.booster_.feature_importance(importance_type="gain"), index=cols)
    if name == "xgboost" and hasattr(model, "get_booster"):
        gain = model.get_booster().get_score(importance_type="gain")
        return pd.Series({c: gain.get(c, 0.0) for c in cols})
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=cols)
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        return pd.Series(np.abs(model.named_steps["clf"].coef_[0]), index=cols)
    return pd.Series(dtype=float)


# ------------------------------------------------------------------ main
def run_market(cfg: dict, market: str, force_download: bool = False,
               run_cv: bool = True, run_shap: bool = True) -> dict:
    set_seed(int(cfg["project"].get("seed", 42)))
    tables_dir = ensure_dir(Path(cfg["project"]["reports_dir"]) / "tables")
    figures_dir = ensure_dir(Path(cfg["project"]["reports_dir"]) / "figures")

    frames, panel, quality = build_dataset(cfg, market, force_download)
    cols = feature_cols(panel)

    cv, cv_summary = (pd.DataFrame(), pd.DataFrame())
    if run_cv:
        cv, cv_summary = cross_validate(cfg, panel, market)
        cv.to_csv(tables_dir / f"cv_results_{market}.csv", index=False)
        cv_summary.to_csv(tables_dir / f"cv_summary_{market}.csv", index=False)
        plots.plot_cv_auc(cv, f"{cfg['markets'][market]['label']} - AUC qua các fold walk-forward",
                          figures_dir / f"cv_auc_{market}.png")

    best_model = cfg["models"]["primary"]
    if run_cv and not cv_summary.empty:
        candidates = cv_summary[~cv_summary["model"].isin({"random", "naive"})]
        if not candidates.empty:
            best_model = str(candidates.iloc[0]["model"])

    test_table, preds, fitted = evaluate_test(cfg, panel, frames, market, best_model)
    test_table.to_csv(tables_dir / f"test_results_{market}.csv", index=False)
    preds.to_csv(tables_dir / f"test_predictions_{market}.csv")

    # Backtest
    bcfg = cfg["backtest"]
    cost = float(cfg["markets"][market]["cost_bps"])
    stats, daily = backtest.backtest_panel(
        preds.assign(prob=preds[f"prob_{best_model}"]), "prob", cost,
        allow_short=bool(bcfg["allow_short"]), threshold=float(bcfg["prob_threshold"]),
        periods_per_year=int(cfg["evaluation"]["sharpe_periods_per_year"]))
    daily.to_csv(tables_dir / f"backtest_daily_{market}.csv")
    by_year = backtest.backtest_by_year(daily, int(cfg["evaluation"]["sharpe_periods_per_year"]))
    by_year.to_csv(tables_dir / f"backtest_by_year_{market}.csv", index=False)

    label = cfg["markets"][market]["label"]
    plots.plot_equity_curves(
        {"Chiến lược": daily["net"], "Mua & nắm giữ": daily["bh"]},
        f"{label} - Đường cong vốn ({best_model}, phí {cost:.0f} bps)",
        figures_dir / f"equity_{market}.png")
    plots.plot_drawdown(daily["net"], f"{label} - Sụt giảm vốn", figures_dir / f"drawdown_{market}.png")
    plots.plot_metric_by_year(by_year, f"{label} - Lợi suất từng năm",
                              figures_dir / f"by_year_{market}.png")
    roc_scores = {c.replace("prob_", ""): preds[c].values
                  for c in preds.columns if c.startswith("prob_")}
    plots.plot_roc(preds["y"].values, roc_scores, f"{label} - ROC trên tập test",
                   figures_dir / f"roc_{market}.png")

    # Giải thích mô hình (dùng lại mô hình đã fit ở bước test, không train lại)
    imp = None
    if run_shap and best_model in fitted:
        test_start = pd.Timestamp(cfg["test"]["start_date"])
        test_end = pd.Timestamp(cfg["test"]["end_date"])
        X_te = panel.loc[(panel.index >= test_start) & (panel.index <= test_end), cols]
        est = fitted[best_model]
        try:
            s_imp = shap_importance(est, best_model, X_te, cfg)
            imp = s_imp if s_imp is not None else model_importance(est, best_model, cols)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Không tính được importance: {exc}")
    if imp is not None and len(imp):
        imp.to_csv(tables_dir / f"feature_importance_{market}.csv")
        plots.plot_feature_importance(imp, f"{label} - Mức độ quan trọng của đặc trưng",
                                      figures_dir / f"importance_{market}.png")

    result = {
        "market": market, "label": label, "n_tickers": len(frames),
        "n_rows": int(len(panel)),
        "n_features": len(cols),
        "date_range": [str(panel.index.min().date()), str(panel.index.max().date())],
        "positive_rate": float(panel["y"].mean()),
        "best_model": best_model,
        "cv_mean_auc": float(cv_summary.iloc[0]["roc_auc_mean"]) if not cv_summary.empty else None,
        "test_table": test_table.to_dict("records"),
        "backtest": stats,
        "by_year": by_year.to_dict("records"),
    }
    save_json(result, Path(cfg["project"]["reports_dir"]) / f"metrics_{market}.json")
    quality.to_csv(tables_dir / f"data_quality_{market}.csv", index=False)
    return result
