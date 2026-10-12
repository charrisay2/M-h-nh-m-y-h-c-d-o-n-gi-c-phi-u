"""Định nghĩa bộ mô hình: baseline -> boosting.

Thứ tự trong báo cáo:
  0. random      - mức sàn tuyệt đối (AUC kỳ vọng 0.50)
  1. naive       - hôm nay tăng thì mai tăng (baseline khó đánh bại nhất)
  2. logreg      - baseline tuyến tính
  3. rf          - baseline phi tuyến
  4. lightgbm    - mô hình chính
  5. xgboost     - đối chiếu
  6. arima       - baseline kinh tế lượng (chỉ đánh giá trên tập test)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import get_logger

log = get_logger("models")


# ------------------------------------------------------------------ baseline
class NaivePersistence(BaseEstimator, ClassifierMixin):
    """Baseline: dự đoán ngày mai theo dấu lợi suất gần nhất.

    predict_proba trả về xác suất đơn điệu theo ret_lag1 (expit) để vừa có
    thứ tự hạng (tính được AUC) vừa có ngưỡng 0.5 tương đương quy tắc dấu.
    """

    def __init__(self, feature: str = "ret_lag1", k: float = 100.0):
        self.feature = feature
        self.k = k

    def fit(self, X, y=None):
        self.classes_ = np.array([0.0, 1.0])
        if isinstance(X, pd.DataFrame) and self.feature in X.columns:
            self.feature_idx_ = list(X.columns).index(self.feature)
        else:
            self.feature_idx_ = None
        return self

    def _score(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.feature in X.columns:
                s = X[self.feature].values
            elif getattr(self, "feature_idx_", None) is not None:
                s = X.iloc[:, self.feature_idx_].values
            else:
                s = X.iloc[:, 0].values
        else:
            X = np.asarray(X)
            s = X[:, self.feature_idx_] if getattr(self, "feature_idx_", None) is not None else X[:, 0]
        return np.nan_to_num(np.asarray(s, dtype=float), nan=0.0)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.k * self._score(X)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(float)


class RandomBaseline(BaseEstimator, ClassifierMixin):
    """Mức sàn: dự đoán ngẫu nhiên, AUC kỳ vọng = 0.50."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, y=None):
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X):
        rng = np.random.default_rng(self.seed)
        p = rng.random(len(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(float)


# ------------------------------------------------------------------ factory
def get_model(name: str, cfg: dict, seed: int = 42, n_jobs: int = -1):
    if name == "random":
        return RandomBaseline(seed=seed)
    if name == "naive":
        return NaivePersistence(feature="ret_lag1")
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=seed)),
        ])
    if name == "rf":
        p = cfg.get("randomforest", {})
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 6),
            min_samples_leaf=p.get("min_samples_leaf", 50),
            max_features=p.get("max_features", "sqrt"),
            random_state=seed, n_jobs=n_jobs,
        )
    if name == "lightgbm":
        import lightgbm as lgb

        p = {k: v for k, v in cfg.get("lightgbm", {}).items()
             if k not in {"early_stopping_rounds"}}
        return lgb.LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, **p)
    if name == "xgboost":
        from xgboost import XGBClassifier

        p = {k: v for k, v in cfg.get("xgboost", {}).items()
             if k not in {"early_stopping_rounds"}}
        # early_stopping_rounds KHÔNG đưa vào constructor (xgboost >= 3 giữ callback
        # và sẽ lỗi khi fit lại không có eval_set) -> truyền qua callbacks lúc fit.
        kwargs = dict(eval_metric="auc", tree_method="hist", random_state=seed, n_jobs=n_jobs)
        return XGBClassifier(**kwargs, **p)
    raise ValueError(f"Mô hình không xác định: {name}")


def fit_model(model, X_tr: pd.DataFrame, y_tr: np.ndarray, name: str,
              X_va: pd.DataFrame | None = None, y_va: np.ndarray | None = None,
              cfg: dict | None = None) -> object:
    """Fit có early-stopping trên tập validation (chỉ với boosting)."""
    cfg = cfg or {}
    if X_va is None or len(X_va) == 0:
        model.fit(X_tr, y_tr)
        return model

    if name == "lightgbm":
        import lightgbm as lgb

        rounds = cfg.get("lightgbm", {}).get("early_stopping_rounds", 100)
        cbs = [lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(0)]
        try:  # lightgbm >= 4.7 ưu tiên eval_X/eval_y
            model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, eval_metric="auc", callbacks=cbs)
        except TypeError:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc", callbacks=cbs)
    elif name == "xgboost":
        import xgboost as xgb

        rounds = int(cfg.get("xgboost", {}).get("early_stopping_rounds", 100))
        model.set_params(callbacks=[xgb.callback.EarlyStopping(rounds=rounds)])
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        model.fit(X_tr, y_tr)
    return model


def predict_proba(model, X: pd.DataFrame, name: str) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-np.asarray(s)))
    raise AttributeError(f"{name} không có predict_proba")


# ------------------------------------------------------------------ ARIMA
def arima_direction_scores(returns_by_ticker: dict[str, pd.Series], test_start,
                           order=(2, 0, 2), refit_every: int = 20,
                           max_hist: int = 1000) -> dict:
    """Baseline ARIMA: dự báo 1 bước với cửa sổ mở rộng trên log-return.

    Trả về dict {(ticker, date): forecast}. Mỗi bước chỉ dùng dữ liệu đến
    thời điểm đó (expanding window), không có rò rỉ dữ liệu.
    Lưu ý: chạy chậm -> chỉ dùng cho tập test, không dùng trong CV.
    """
    import warnings

    from statsmodels.tsa.arima.model import ARIMA

    out: dict = {}
    # ARIMA trên chuỗi tài chính thường không hội tụ -> tắt cảnh báo của statsmodels
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*optimization failed to converge.*")
    warnings.filterwarnings("ignore", message=".*Non-stationary starting.*")
    warnings.filterwarnings("ignore", message=".*Non-invertible starting.*")
    test_start = pd.Timestamp(test_start)
    for ticker, r in returns_by_ticker.items():
        r = r.dropna().sort_index()
        train_r = r[r.index < test_start]
        test_r = r[r.index >= test_start]
        if len(train_r) < 60 or len(test_r) == 0:
            continue
        hist = list(train_r.values[-max_hist:])
        model = None
        for i, (d, actual) in enumerate(test_r.items()):
            if model is None or i % max(int(refit_every), 1) == 0:
                try:
                    model = ARIMA(hist, order=tuple(order)).fit()
                except Exception:  # noqa: BLE001
                    model = None
            if model is None:
                fc = float(np.mean(hist[-5:])) if len(hist) >= 5 else 0.0
            else:
                try:
                    fc = float(model.forecast(steps=1)[0])
                except Exception:  # noqa: BLE001
                    fc = float(np.mean(hist[-5:]))
            out[(ticker, d)] = fc
            hist.append(float(actual))
            if len(hist) > max_hist:
                hist = hist[-max_hist:]
    return out
