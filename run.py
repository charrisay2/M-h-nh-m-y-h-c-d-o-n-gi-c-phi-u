"""Điểm vào chính: chạy toàn bộ thực nghiệm (VN và/hoặc US).

Cách dùng
---------
    python run.py                        # chạy cả hai thị trường (đầy đủ)
    python run.py --market vn            # chỉ Việt Nam
    python run.py --market us --no-cv    # bỏ qua cross-validation (nhanh)
    python run.py --skip-download        # dùng lại dữ liệu đã tải trong data/raw
    python run.py --offline-demo         # dữ liệu GIẢ LẬP để smoke-test (KHÔNG dùng nộp báo cáo)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pipeline  # noqa: E402
from utils import ROOT, ensure_dir, get_logger, load_config  # noqa: E402

log = get_logger("run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML dự đoán xu hướng giá cổ phiếu (VN vs US)")
    p.add_argument("--market", choices=["vn", "us", "both"], default="both")
    p.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    p.add_argument("--skip-download", action="store_true", help="Dùng lại dữ liệu trong data/raw")
    p.add_argument("--force-download", action="store_true", help="Tải lại dữ liệu")
    p.add_argument("--no-cv", action="store_true", help="Bỏ qua walk-forward CV")
    p.add_argument("--no-shap", action="store_true", help="Bỏ qua SHAP")
    p.add_argument("--offline-demo", action="store_true",
                   help="Dùng dữ liệu GIẢ LẬP để kiểm tra pipeline (không dùng cho báo cáo)")
    return p.parse_args()


def compare_markets(results: dict[str, dict], cfg: dict) -> pd.DataFrame:
    """Bảng so sánh VN vs US - phần cốt lõi của Chương 4."""
    rows = []
    for market, r in results.items():
        tt = pd.DataFrame(r["test_table"])
        best = tt[tt["model"] == r["best_model"]]
        bt = r["backtest"]
        rows.append({
            "market": r["label"],
            "n_tickers": r["n_tickers"],
            "n_rows": r["n_rows"],
            "date_range": " → ".join(r["date_range"]),
            "best_model": r["best_model"],
            "cv_mean_auc": r.get("cv_mean_auc"),
            "test_auc": float(best["roc_auc"].iloc[0]) if len(best) else np.nan,
            "test_auc_ci": (f"[{best['auc_ci_low'].iloc[0]:.3f}, {best['auc_ci_high'].iloc[0]:.3f}]"
                            if len(best) else "n/a"),
            "test_accuracy": float(best["accuracy"].iloc[0]) if len(best) else np.nan,
            "naive_auc": float(tt.loc[tt["model"] == "naive", "roc_auc"].iloc[0])
                         if (tt["model"] == "naive").any() else np.nan,
            "logreg_auc": float(tt.loc[tt["model"] == "logreg", "roc_auc"].iloc[0])
                          if (tt["model"] == "logreg").any() else np.nan,
            "arima_auc": float(tt.loc[tt["model"] == "arima", "roc_auc"].iloc[0])
                         if (tt["model"] == "arima").any() else np.nan,
            "sharpe": bt.get("sharpe"),
            "sharpe_gross": bt.get("sharpe_gross"),
            "bh_sharpe": bt.get("bh_sharpe"),
            "cagr": bt.get("cagr"),
            "bh_cagr": bt.get("bh_cagr"),
            "max_drawdown": bt.get("max_drawdown"),
            "calmar": bt.get("calmar"),
            "exposure": bt.get("exposure"),
        })
    return pd.DataFrame(rows)


def write_summary_md(results: dict[str, dict], comparison: pd.DataFrame, cfg: dict) -> Path:
    lines = ["# Kết quả thực nghiệm", "",
             "_Sinh tự động bởi `run.py`. Không chỉnh sửa tay._", ""]
    for market, r in results.items():
        tt = pd.DataFrame(r["test_table"])
        lines += [
            f"## {r['label']}",
            "",
            f"- Mã: **{r['n_tickers']}** | Mẫu: **{r['n_rows']:,}** | Đặc trưng: **{r['n_features']}**",
            f"- Khoảng dữ liệu: {r['date_range'][0]} → {r['date_range'][1]}",
            f"- Tỷ lệ nhãn tăng: {r['positive_rate']:.3f}",
            f"- Mô hình tốt nhất (CV): **{r['best_model']}**",
            "",
            "### Kết quả trên tập test out-of-sample",
            "",
            tt.round(4).to_markdown(index=False),
            "",
            "### Backtest (đã trừ chi phí giao dịch)",
            "",
            "| Chỉ số | Chiến lược | Mua & nắm giữ |",
            "|---|---|---|",
        ]
        bt = r["backtest"]
        for k in ["total_return", "cagr", "ann_vol", "sharpe", "sortino",
                  "max_drawdown", "calmar", "hit_rate"]:
            v, b = bt.get(k), bt.get(f"bh_{k}")
            lines.append(f"| {k} | {v:.4f} | {b:.4f} |" if isinstance(v, float)
                         else f"| {k} | {v} | {b} |")
        lines += ["", f"- Tỷ lệ thời gian nắm giữ (exposure): {bt.get('exposure', float('nan')):.3f}",
                  "- Tổng chi phí giao dịch tích lũy: "
                  f"{bt.get('total_cost_drag', float('nan')):.4f}", ""]

    lines += ["## So sánh hai thị trường", "", comparison.round(4).to_markdown(index=False), "",
              "> ⚠️ Mọi kết quả chỉ phục vụ mục đích học thuật, không phải khuyến nghị đầu tư."]
    out = ensure_dir(ROOT / cfg["project"]["reports_dir"]) / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.offline_demo:
        log.warning("⚠️  CHẾ ĐỘ GIẢ LẬP: dữ liệu không có thật, KHÔNG dùng kết quả này nộp báo cáo")
        # Ghi sang thư mục riêng để KHÔNG bao giờ làm bẩn data/raw (dữ liệu thật)
        cfg["project"]["raw_dir"] = "data/demo"
        for m in cfg["markets"]:
            cfg["markets"][m]["source"] = "synthetic"
            cfg["markets"][m]["tickers"] = cfg["markets"][m]["tickers"][:6]
    if args.skip_download:
        for m in cfg["markets"]:
            cfg["markets"][m]["source"] = "cache"

    markets = ["vn", "us"] if args.market == "both" else [args.market]
    results: dict[str, dict] = {}
    for market in markets:
        t0 = time.time()
        log.info("=" * 70)
        results[market] = pipeline.run_market(
            cfg, market,
            force_download=args.force_download,
            run_cv=not args.no_cv,
            run_shap=not args.no_shap,
        )
        log.info(f"[{market}] hoàn thành trong {time.time() - t0:.0f}s")

    comparison = compare_markets(results, cfg)
    tables_dir = ensure_dir(ROOT / cfg["project"]["reports_dir"] / "tables")
    comparison.to_csv(tables_dir / "market_comparison.csv", index=False)

    print("\n" + "=" * 90)
    print("SO SÁNH HAI THỊ TRƯỜNG")
    print("=" * 90)
    print(comparison.round(4).to_string(index=False))

    md = write_summary_md(results, comparison, cfg)
    log.info(f"Báo cáo tổng hợp: {md}")

    with open(ensure_dir(ROOT / cfg["project"]["reports_dir"]) / "results_full.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
