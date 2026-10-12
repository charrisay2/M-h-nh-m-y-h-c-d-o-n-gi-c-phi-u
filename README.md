# Mô hình máy học dự đoán xu hướng giá cổ phiếu: Việt Nam vs Hoa Kỳ

Đồ án/tiểu luận cuối kỳ. Pipeline hoàn chỉnh: tải dữ liệu thật → tạo 45 đặc trưng kỹ thuật →
huấn luyện gradient boosting → đánh giá out-of-sample có kiểm định thống kê → backtest có
chi phí giao dịch → giải thích mô hình bằng SHAP → so sánh hai thị trường.

## 1. Câu hỏi nghiên cứu

> Với cùng một pipeline học máy và cùng bộ đặc trưng, khả năng dự đoán xu hướng giá ngày hôm sau
> của nhóm cổ phiếu Việt Nam (VN30) có khác biệt **có ý nghĩa thống kê** so với nhóm cổ phiếu
> Hoa Kỳ (S&P 500) hay không?

## 2. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Đã kiểm tra chạy được trên Python 3.13 (pandas 3.x, lightgbm 4.7, xgboost 3.4).

## 3. Chạy

```bash
python run.py                       # full: VN + US, có CV + SHAP
python run.py --market vn           # chỉ Việt Nam
python run.py --no-cv               # bỏ cross-validation (nhanh hơn nhiều)
python run.py --skip-download       # dùng lại dữ liệu đã tải trong data/raw
python run.py --offline-demo        # dữ liệu GIẢ LẬP để smoke-test (không dùng nộp báo cáo)
```

## 4. Cấu trúc thư mục

```
config/config.yaml        Mọi tham số: mã, thời gian, purge, embargo, hyperparameter
src/
  data_loader.py          Tải VN (.VN trên Yahoo) và US, báo cáo chất lượng dữ liệu
  features.py             45 đặc trưng: xu hướng, động lượng, biến động, khối lượng, beta
  labeling.py             Nhãn xu hướng T+1 (binary hoặc ngưỡng ±0.5%)
  split.py                Walk-forward CV + purge + embargo   <- quan trọng nhất
  models.py               random / naive / logreg / rf / lightgbm / xgboost / ARIMA
  evaluate.py             AUC, MCC + bootstrap CI + McNemar + Diebold-Mariano
  backtest.py             Mô phỏng giao dịch có phí, thuế, trượt giá
  plots.py                Biểu đồ cho báo cáo
  pipeline.py             Điều phối toàn bộ thực nghiệm
run.py                    Điểm vào CLI
data/raw/                 Dữ liệu gốc (chỉ ghi 1 lần)
reports/                  Bảng, biểu đồ, RESULTS.md
```

## 5. Phương pháp (phần này là lý do bài được đánh giá cao)

### 5.1 Chống rò rỉ dữ liệu
| Bẫy | Cách xử lý trong code |
|---|---|
| `train_test_split(shuffle=True)` | Không dùng. Dùng `walk_forward_folds()` trong `split.py` |
| Đặc trưng dùng cửa sổ tương lai | Mọi chỉ báo chỉ dùng `rolling/ewm` nhìn về quá khứ |
| Mẫu ở ranh giới train/val bị chồng lấn | **purge** 21 ngày trước tập validation |
| Mẫu ngay sau tập validation | **embargo** 5 ngày |
| Scaler fit trên toàn bộ dữ liệu | Scaler nằm trong `Pipeline`, fit theo từng fold |
| Tune tham số trên tập test | Tập test (2024–2025) chỉ được chạm đúng 1 lần |

### 5.2 Nhãn
- `horizon = 1`: dự đoán đóng cửa ngày mai.
- `mode = threshold`: `y = 1` nếu lợi suất > +0.5%, `y = 0` nếu < −0.5%, bỏ qua vùng giữa để giảm nhiễu.
- `shift(-1)` được thực hiện **trước** khi chia tập.

### 5.3 Kiểm định thống kê
- **Bootstrap CI 95%** cho AUC (1.000 lần) — điểm số đơn lẻ là vô nghĩa.
- **McNemar test** — hai mô hình khác nhau thật hay chỉ là nhiễu?
- **Diebold-Mariano test** — so sánh chuỗi sai số dự báo.

### 5.4 Đánh giá kinh tế
Backtest long-only, chi phí một chiều: **VN 35 bps** (phí ~15 + thuế 10 + trượt giá ~10),
**US 5 bps** (slippage/spread). Báo cáo Sharpe, Sortino, CAGR, Max Drawdown, Calmar, turnover,
và so sánh với **mua & nắm giữ** cùng danh mục.

## 6. Kết quả ở đâu?

- `reports/RESULTS.md` — báo cáo tổng hợp (bảng test + backtest + so sánh 2 thị trường)
- `reports/tables/` — `cv_results_*.csv`, `test_results_*.csv`, `market_comparison.csv`,
  `backtest_daily_*.csv`, `feature_importance_*.csv`, `data_quality_*.csv`
- `reports/figures/` — đường cong vốn, drawdown, ROC, lợi suất từng năm, tầm quan trọng đặc trưng

## 7. Lưu ý quan trọng

- Dữ liệu VN lấy qua Yahoo (hậu tố `.VN`). Yahoo **không có VN-Index**, nên biến thị trường được
  thay bằng **trung bình equal-weight** của các mã đã tải (`build_market_factor()`).
- `vnstock` không cài được trên Python 3.13, nên pipeline dùng Yahoo cho cả hai thị trường.
  Nếu bạn muốn đổi nguồn, sửa `markets.vn.source` trong `config/config.yaml`.
- Chạy `--offline-demo` cho dữ liệu giả lập. Kết quả khi đó **không** được đưa vào báo cáo.
- ⚠️ Mọi kết quả chỉ phục vụ mục đích học thuật, **không phải khuyến nghị đầu tư**.
