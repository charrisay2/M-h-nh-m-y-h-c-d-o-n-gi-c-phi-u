# PHẦN 3 — THẢO LUẬN, KẾT LUẬN, HỎI ĐÁP & PHỤ LỤC (Slide 13-15)

---

## SLIDE 13 — Phát hiện chính (Trả lời 3 giả thuyết)

**Nội dung trên slide (bảng kiểm chứng giả thuyết):**

| Giả thuyết | Kết quả | Bằng chứng |
|---|---|---|
| **H1**: Mô hình đạt ý nghĩa thống kê trên VN | ✅ **CHẤP NHẬN** | XGBoost AUC 0.5695, CI 95% [0.558; 0.582] không chứa 0,50; McNemar p < 0.001 |
| **H1'**: Mô hình đạt ý nghĩa thống kê trên Mỹ | ⚠️ **BÁC BỞ** | AUC 0.5195, CI [0.508; 0.532]; chỉ hơn naive 0,6 điểm; áp lực trade-off cao |
| **H2**: Đặc trưng thị trường quan trọng hơn đặc trưng riêng mã | ✅ **CHẤP NHẬN** | Cả 2 thị trường, `mkt_ret_*` nằm trong top 3 SHAP |
| **H3**: Chi phí là rào cản chính | ✅ **CHẤP NHẬN MẠNH** | Chi phí tích lũy VN = 0,694; Sharpe 1,17 vs 2,33 của mua & nắm giữ |

**Điểm nhấn (ô lớn, giữa slide):**
> **Thị trường kém hiệu quả hơn (VN) thì dễ dự đoán hơn — nhưng khó khai thác hơn, vì chi phí giao dịch cao hơn.**

**Lời thoại:**
"Chúng em quay lại kiểm chứng ba giả thuyết đã nêu.
**Giả thuyết một**: mô hình đạt ý nghĩa thống kê trên Việt Nam — được chấp nhận. AUC 0.5695, khoảng tin cậy không chứa 0,50, và kiểm định McNemar cho p nhỏ hơn 0.001. Nhưng **trên Mỹ thì bị bác bỏ** — chúng em xin nhấn mạnh điều này.
**Giả thuyết hai**: đặc trưng thị trường quan trọng hơn đặc trưng riêng từng mã — được chấp nhận ở cả hai thị trường, như mình thấy ở phần SHAP.
**Giả thuyết ba — được chấp nhận mạnh nhất**: chi phí giao dịch là rào cản chính. Ở Việt Nam, gần 70% vốn bị ăn mòn bởi phí và trượt giá.
Tóm lại, đề tài cho một kết luận hơi nghịch lý: **thị trường kém hiệu quả hơn thì dễ dự đoán hơn, nhưng lại khó khai thác hơn vì chi phí giao dịch cao hơn**. Theo chúng em, đây là bài học thực sự đáng giá."

---

## SLIDE 14 — Hạn chế & hướng mở rộng

**Nội dung trên slide (2 cột):**

*Hạn chế — chúng em nêu thẳng, không né:*
1. **LightGBM trên Mỹ có recall = 1,0** — gần như luôn dự đoán "tăng", do 2024–2025 là bull market mạnh. → mất cân bằng thời gian, không phải lợi thế mô hình
2. **Nhãn ±0,5% loại ngày biến động nhỏ, nhưng giao dịch thực tế không biết trước** ngày nào sẽ bị loại → backtest nghiêng về phía có lợi
3. **Yahoo không có VN-Index** → biến thị trường là equal-weight, không phải chỉ số chuẩn
4. Dữ liệu điều chỉnh giá VN chưa kiểm chứng hoàn toàn ở các ngày chia tách, trả cổ tức
5. Thị trường T+ ở VN hạn chế việc đảo vị thế nhanh theo mô hình

*Hướng mở rộng:*
1. **Giảm turnover**: chỉ giao dịch khi P(tăng) vượt ngưỡng xa (ví dụ 0,6 hoặc 0,65) thay vì 0,5
2. Bật **Optuna** để tune siêu tham số trên validation
3. Thêm dữ liệu phi cấu trúc: tin tức, báo cáo tài chính, chỉ báo vĩ mô
4. Thử cách gán nhãn **triple-barrier** (biến động + thời gian) thay vì ngưỡng cố định

**Lời thoại:**
"Chúng em xin liệt kê hạn chế một cách thẳng thắn.
Thứ nhất, LightGBM trên Mỹ có recall bằng 1,0 — nghĩa là nó dự đoán "tăng" gần như mọi ngày. Đó thực ra là hệ quả của thị trường tăng mạnh trong giai đoạn test, chứ không phải lợi thế của mô hình.
Thứ hai, và đây là hạn chế nghiêm trọng nhất: chúng em **loại các ngày biến động trong khoảng ±0,5% khi huấn luyện**, nhưng trong giao dịch thực tế nhà đầu tư không thể biết trước ngày mai sẽ biến động mạnh hay yếu. Điều này làm backtest nghiêng về phía có lợi cho mô hình, và chúng em đã nêu rõ trong báo cáo.
Thứ ba, biến thị trường của chúng em là equal-weight chứ không phải VN-Index thật.
Về hướng mở rộng, việc quan trọng nhất là **giảm tần suất giao dịch**. Mô hình hiện đổi vị thế gần như mỗi ngày; nếu chỉ giao dịch khi tín hiệu đủ mạnh — ví dụ xác suất vượt 0,65 — chi phí sẽ giảm mạnh. Đó cũng là việc chúng em dự định làm tiếp."

---

## SLIDE 15 — Kết luận & Q&A

**Nội dung trên slide (3 ô kết luận + ô cảm ơn):**

**1. Đã làm gì:**
> Pipeline ML hoàn chỉnh trên 56 mã thực tế, 87.110 mẫu, 10 năm dữ liệu, cả hai thị trường; 45 đặc trưng; 7 mô hình; walk-forward có purge; kiểm định thống kê đầy đủ.

**2. Kết quả chính:**
> **VN: AUC 0.5695 — tín hiệu dự đoán thực sự, có ý nghĩa thống kê.**
> **Mỹ: AUC 0.5195 — không dự đoán được.**
> **Nhưng cả hai: sau chi phí, chiến lược đều thua mua & nắm giữ.**

**3. Bài học rút ra:**
> Trong dự đoán giá chứng khoán, **câu hỏi không chỉ là "mô hình có chính xác không" mà còn là "nó có dùng được sau khi trừ chi phí không"**. Bỏ qua câu hỏi thứ hai là lỗi phổ biến của nhiều nghiên cứu.

**Câu kết (đọc chậm, to):**
> _"Mô hình có tín hiệu. Nhưng tín hiệu chưa đủ lớn để vượt qua chi phí giao dịch."_

**Lời thoại:**
"Tóm lại, đề tài của chúng em đã xây dựng một pipeline đầy đủ và chạy trên dữ liệu thật của cả hai thị trường.
Kết quả: trên **Việt Nam**, mô hình có tín hiệu dự đoán thực sự, vượt trội so với baseline và có ý nghĩa thống kê rõ ràng. Trên **Mỹ**, mô hình gần như không dự đoán được gì. Và ở cả hai nơi, sau khi trừ chi phí giao dịch, chiến lược đều thua mua và nắm giữ.
Bài học chúng em rút ra, và cũng là điều chúng em muốn gửi gắm: **trong lĩnh vực này, câu hỏi không chỉ là mô hình có chính xác không, mà còn là nó có dùng được sau khi trừ chi phí không**. Rất nhiều nghiên cứu dừng lại ở câu hỏi đầu tiên.
Em xin hết phần trình bày. Nhóm em rất mong nhận được thắc mci và góp ý từ thầy/cô và các bạn."

---

# PHẦN HỎI ĐÁP — Câu hỏi dự kiến và cách trả lời

## 1. "Sao không dùng LSTM / Transformer?"

**Trả lời:**
"Về lý thuyết mạng hồi quy phù hợp với dữ liệu trình tự, nhưng trên **bài toán dạng bảng với 45 đặc trưng đầu vào**, gradient boosting thường cho kết quả tốt hơn hoặc tương đương trong khi cần ít dữ liệu hơn nhiều, huấn luyện nhanh hơn và kết quả ổn định hơn. Với dữ liệu chứng khoán — vốn có tỷ lệ tín hiệu trên nhiễu rất thấp, mô hình học sâu rất dễ bị overfit. Chúng em chọn boosting vì nó phù hợp với quy mô dữ liệu và dễ kiểm soát được. Dù vậy, đây là hướng so sánh chúng em để ngỏ cho nghiên cứu tiếp theo."

## 2. "Làm sao chắc chắn không rò rỉ dữ liệu?"

**Trả lời:** (đưa 4 bằng chứng cụ thể)
1. Toàn bộ đặc trưng tính bằng `rolling`/`ewm` **không** có `center=True` → chỉ nhìn về quá khứ.
2. Nhãn `shift(-1)` được tạo **trước** khi chia tập, không có thông tin tương lai trong tập huấn luyện.
3. Chia tập bằng **walk-forward có purge 21 ngày** — bằng đúng độ dài cửa sổ đặc trưng lớn nhất.
4. **Bằng chứng thực nghiệm**: chúng em đã chạy thử pipeline trên **dữ liệu ngẫu nhiên sinh tự động** và thu được AUC ≈ 0,497 — đúng bằng tung đồng xu. Nếu có rò rỉ, con số này phải cao bất thường.

## 3. "Kết quả có thực sự dùng được không?"

**Trả lời:**
"Em xin trả lời thẳng: **không, chưa dùng được cho giao dịch thực tế**. Sau khi tính đủ phí, thuế và trượt giá, chiến lược thua cả mua & nắm giữ. Điều này bản thân nó là một kết quả có giá trị: nó cho thấy tín hiệu tuy tồn tại nhưng còn nhỏ hơn rào cản chi phí. Đề tài của em mang tính học thuật, nhằm kiểm chứng câu hỏi nghiên cứu một cách trung thực, chứ không đề xuất một chiến lược đầu tư."

## 4. "Tại sao Việt Nam dễ dự đoán hơn Mỹ?"

**Trả lời:**
"Chúng em không khẳng định về mặt nhân quả, chỉ nêu quan sát phù hợp với giả thuyết: thị trường Việt Nam có tỷ trọng nhà đầu tư cá nhân cao hơn, dòng tiền tập trung vào ít mã, và thông tin phản ánh vào giá chậm hơn — những yếu tố làm cho giá có thành phần dự đoán được cao hơn. Ngoài ra, kết quả cũng bị ảnh hưởng bởi giai đoạn test 2024–2025: thị trường Mỹ trong giai đoạn này rất hiệu quả và biến động thấp, nên khó dự đoán hơn. Chúng em đã nêu đây là một hạn chế về tính khái quát."

## 5. "Tại sao lại bỏ qua các ngày biến động trong ±0,5%?"

**Trả lời:**
"Vì trong vùng biến động hẹp đó, chuyển động của giá gần như là nhiễu ngẫu nhiên — nhãn 'tăng' hay 'giảm' không mang thông tin. Việc giữ chúng sẽ làm nhiễu tập huấn luyện. Chúng em cũng đã nêu rõ trong phần hạn chế rằng **trong backtest, chiến lược vẫn phải giao dịch cả những ngày này** vì nhà đầu tư không biết trước — nên cách xử lý này có thiên kiến nghiêng về phía có lợi cho mô hình, và chúng em đã thừa nhận nó."

## 6. "Nhóm em đã phân công công việc như thế nào?"

**Trả lời:**
"Đề tài được chia theo trách nhiệm kiểm chứng, không phải chia code: nhóm trưởng phụ trách thiết kế thực nghiệm, chống rò rỉ và kết luận; một bạn phụ trách dữ liệu và EDA; một bạn phụ trách đặc trưng, mô hình và giải thích SHAP; một bạn phụ trách backtest, chi phí và tính ứng dụng. Toàn bộ code được viết như một pipeline thống nhất để đảm bảo mọi thành viên chạy trên cùng một quy trình, kết quả không phụ thuộc vào máy của từng người."

## 7. "Nếu chỉ có 10 phút thay vì 20?"

**Slide cần giữ lại (bắt buộc):** 1, 4, 5, 7, 8, 10, 12, 13, 15
**Slide cắt được:** 2 (mục lục), 3 (rút gọn còn câu hỏi nghiên cứu), 6, 9, 11, 14 (để trong phụ lục)

---

# PHỤ LỤC — Thông tin để tra khi bị hỏi sâu

## Tham số kỹ thuật theo mô hình

| Mô hình | Tham số chính |
|---|---|
| XGBoost | learning_rate 0,03; max_depth 5; min_child_weight 20; subsample 0,7; colsample_bytree 0,5; reg_lambda 5; n_estimators 1.500; early stopping 100 vòng |
| LightGBM | learning_rate 0,03; num_leaves 15; min_child_samples 200; feature_fraction 0,5; bagging_fraction 0,7; bagging_freq 1; lambda_l1 1; lambda_l2 5; max_depth 6; n_estimators 2.000 |
| Random Forest | n_estimators 300; max_depth 6; min_samples_leaf 50; max_features = sqrt |
| Logistic Regression | C = 0,1; max_iter 2.000; có StandardScaler trong Pipeline |
| ARIMA | order (2, 0, 2) trên chuỗi log-return; cửa sổ mở rộng; refit sau mỗi 20 bước |

## Thiết kế thực nghiệm (số cụ thể)

| Tham số | Giá trị |
|---|---|
| Số fold walk-forward | 5 |
| Độ dài mỗi tập validation | 252 ngày giao dịch (~1 năm) |
| Purge | 21 ngày |
| Embargo | 5 ngày |
| Tập huấn luyện | expanding (mở rộng dần) |
| Tập test | 01/01/2024 – 31/12/2025 |
| Tập test bị chạm | **đúng 1 lần** |
| Số lần bootstrap | 1.000 |
| Khoảng tin cậy | 95% |
| Chi phí VN / US | 35 bps / 5 bps (một chiều) |

## Thư viện đã dùng

`pandas` · `numpy` · `scikit-learn` · `lightgbm` · `xgboost` · `statsmodels` (ARIMA) · `shap` · `matplotlib` · `yfinance` · `scipy` (kiểm định thống kê)

## Danh sách hình ảnh có sẵn để chèn

Tất cả trong `reports/figures/`:
- `roc_vn.png`, `roc_us.png` — đường ROC tập test
- `equity_vn.png`, `equity_us.png` — đường cong vốn so với mua & nắm giữ
- `drawdown_vn.png`, `drawdown_us.png` — sụt giảm vốn
- `importance_vn.png`, `importance_us.png` — mức độ quan trọng đặc trưng (SHAP)
- `cv_auc_vn.png`, `cv_auc_us.png` — AUC qua các fold walk-forward
- `by_year_vn.png`, `by_year_us.png` — lợi suất từng năm

## Bảng dữ liệu để tra nếu bị hỏi chi tiết

Tất cả trong `reports/tables/`:
- `test_results_vn.csv`, `test_results_us.csv` — toàn bộ chỉ số phân loại
- `cv_summary_vn.csv`, `cv_summary_us.csv` — kết quả trung bình walk-forward
- `cv_results_vn.csv`, `cv_results_us.csv` — chi tiết từng fold
- `backtest_by_year_*.csv`, `backtest_daily_*.csv` — kết quả backtest
- `data_quality_vn.csv`, `data_quality_us.csv` — chất lượng dữ liệu từng mã
- `market_comparison.csv` — bảng tổng hợp so sánh hai thị trường

---

> ⚠️ **Nhắc nhở khi trả lời mọi câu hỏi:** mọi kết quả trong đề tài chỉ phục vụ mục đích học thuật, **không phải khuyến nghị đầu tư**.
