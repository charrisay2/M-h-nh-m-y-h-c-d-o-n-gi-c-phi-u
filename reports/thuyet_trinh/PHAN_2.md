# PHẦN 2 — KẾT QUẢ (Slide 9-12)

---

## SLIDE 9 — Cách đánh giá mô hình

**Nội dung trên slide (3 khối):**

**Khối 1 — Chỉ số phân loại:**
- Accuracy, Balanced Accuracy, Precision, Recall, F1
- **ROC-AUC** (chỉ số chính) — đo khả năng xếp hạng độc lập ngưỡng
- **PR-AUC** — quan trọng khi mất cân bằng lớp
- **MCC** — hệ số tương quan Matthews, đáng tin hơn accuracy khi lệch lớp

**Khối 2 — Kiểm định thống kê (phần làm nên sự khác biệt của đề tài):**
- **Bootstrap CI 95%** cho AUC, 1.000 lần lấy mẫu lại → *một con số AUC đơn lẻ là vô nghĩa*
- **McNemar test** — hai mô hình có thật sự khác nhau, hay chỉ là nhiễu?
- **Diebold-Mariano test** — so sánh chuỗi sai số dự báo giữa hai mô hình

**Khối 3 — Đánh giá kinh tế:**
- Backtest long-only, chi phí 1 chiều: **VN 35 bps** (phí ~15 + thuế ~10 + trượt giá ~10), **US 5 bps**
- Chỉ số: Sharpe, Sortino, CAGR, Max Drawdown, Calmar, tỷ lệ nắm giữ
- So sánh với **mua & nắm giữ** cùng danh mục

**Lời thoại:**
"Chúng em đánh giá ở ba tầng.
**Tầng một** là các chỉ số phân loại thông thường. Nhưng chúng em lấy ROC-AUC làm chỉ số chính thay vì accuracy, vì accuracy bị lệch nghiêm trọng khi tỷ lệ hai lớp chênh nhau.
**Tầng hai là kiểm định thống kê** — và đây là điểm chúng em cho là khác biệt nhất so với cách làm thông thường. Một con số AUC 0,57 nhìn thì đẹp, nhưng nó ổn định không? Vì vậy chúng em dùng **bootstrap với 1.000 lần lấy mẫu lại** để ra khoảng tin cậy 95%. Nếu khoảng đó có chứa 0,50, thì về mặt thống kê mô hình chẳng khác gì tung đồng xu. Chúng em cũng dùng **McNemar** và **Diebold-Mariano** để kiểm tra một mô hình có thật sự vượt mô hình khác, hay chỉ khỏang trong nhiễu lấy mẫu.
**Tầng ba là đánh giá kinh tế**: phí, thuế và trượt giá được tính vào. Đây là bước mà hầu hết các bài áp dụng ML cho chứng khoán bỏ qua."

---

## SLIDE 10 — Kết quả: Việt Nam vs Mỹ ⭐

**Nội dung trên slide (bảng AUC trên tập TEST 2024-2025):**

| Mô hình | **VN — AUC** | CI 95% | **US — AUC** | CI 95% |
|---|---|---|---|---|
| Random | 0.4937 | [0.481; 0.506] | 0.4994 | [0.487; 0.510] |
| **Naive** | 0.4673 | [0.454; 0.480] | 0.5134 | [0.502; 0.525] |
| ARIMA(2,0,2) | 0.4960 | [0.484; 0.509] | 0.5061 | [0.495; 0.518] |
| Logistic Regression | 0.5512 | [0.539; 0.564] | 0.5094 | [0.498; 0.521] |
| Random Forest | 0.5504 | [0.538; 0.564] | 0.5108 | [0.499; 0.523] |
| LightGBM | 0.5583 | [0.546; 0.572] | 0.5057 | [0.494; 0.516] |
| **XGBoost** | **0.5695** | **[0.558; 0.582]** | **0.5195** | **[0.508; 0.532]** |

**Kết quả walk-forward CV (5 fold, chọn mô hình):**
- VN: XGBoost **0.5393 ± 0.0232** | RF 0.5301 | LightGBM 0.5292 | LogReg 0.5197 | **Naive 0.4831**
- US: XGBoost **0.5242** | RF 0.5108 | LightGBM 0.5057 | LogReg 0.5094 | **Naive 0.5134**

**Hình đính kèm:** `reports/figures/roc_vn.png` và `roc_us.png` (2 hình, đặt cạnh nhau)

**3 kết luận đọc chậm (đặt dưới, cỡ chữ lớn):**
1. **Việt Nam: XGBoost 0.5695**, CI không chứa 0,50, vượt naive 10,2 điểm, McNemar **p < 0.001**
2. **Mỹ: XGBoost 0.5195**, CI dưới [0,508; 0,532] — chỉ hơn naive **0,6 điểm**, gần như không dự đoán được
3. Cả hai thị trường, **mô hình boosting đều được CV chọn tự động**

**Lời thoại:**
"Đây là bảng kết quả trọng tâm của đề tài — tập test out-of-sample 2024-2025, chỉ được chạm một lần.
Ở thị trường **Việt Nam**: XGBoost đạt AUC **0.5695**, khoảng tin cậy 95% là từ 0,558 đến 0,582 — **không chứa 0,50**. Mô hình vượt baseline naive tới 10,2 điểm AUC, và McNemar cho p-value nhỏ hơn 0.001, nghĩa là đây là khác biệt thực sự, không phải nhiễu.
Ở thị trường **Mỹ**: XGBoost chỉ đạt **0.5195**, khoảng tin cậy từ 0,508 đến 0,532. Nó chỉ hơn baseline naive 0,6 điểm. Ở đây em xin nói thẳng: **về mặt thực tiễn, mô hình không dự đoán được gì trên thị trường Mỹ**.
Như vậy câu trả lời cho câu hỏi nghiên cứu đã rõ: **có khác biệt có ý nghĩa thống kê** — thị trường Việt Nam dễ dự đoán hơn hẳn Mỹ, và điều này phù hợp với luận điểm về mức độ hiệu quả của thị trường.
Ở góc phải là đường ROC. Đường VN cong rõ ràng hơn đường Mỹ — đường Mỹ gần như nằm sát đường chéo."

---

## SLIDE 11 — Đặc trưng nào quan trọng (SHAP)

**Nội dung trên slide (2 cột hình + nhận xét):**

**Hình đính kèm:** `reports/figures/importance_vn.png` (trái) và `importance_us.png` (phải)

**Top 5 đặc trưng — Việt Nam:**
1. `mkt_ret_21` (0,081) — lợi suất chỉ số 21 ngày
2. `mkt_ret_1` (0,074)
3. `day_of_week` (0,072)
4. `mkt_ret_5` (0,060)
5. `ret_lag4` (0,040)

**Top 5 đặc trưng — Mỹ:**
1. `mkt_ret_1` (0,112)
2. `mkt_ret_5` (0,079)
3. `mkt_ret_21` (0,070)
4. `month` (0,042)
5. `ret_lag5` (0,040)

**Nhận xét nổi bật (ô vàng):**
> Ở **cả hai** thị trường, nhóm đặc trưng thị trường (lợi suất chỉ số) nằm trong top 3. Mô hình phần lớn đang học **định thời điểm thị trường**, không phải lợi thế riêng từng mã cổ phiếu.

**Lời thoại:**
"Chúng em dùng SHAP để xem mô hình thực sự học gì.
Điều đáng chú ý: mô hình **không** học chủ yếu từ các chỉ báo kỹ thuật cổ điển như RSI hay MACD. Nó học phần lớn từ **lợi suất của chính chỉ số thị trường** ở các khung 1, 5 và 21 ngày. Ở Việt Nam, `day_of_week` — ngày trong tuần — cũng lọt vào top 3.
Điều này dẫn đến một nhận định quan trọng mà em muốn chia sẻ: **mô hình phần lớn đang làm nhiệm vụ định thời điểm thị trường chung, chứ không tìm ra lợi thế riêng của từng mã cổ phiếu**. Điều này giải thích tại sao hiệu quả trên từng mã lại khiêm tốn như vậy. Đây cũng là một trong những điểm chúng em nêu ở phần hạn chế."

---

## SLIDE 12 — Backtest có chi phí ⭐

**Nội dung trên slide (bảng 2 cột):**

| Chỉ số | VN — Chiến lược | VN — Mua & nắm giữ | US — Chiến lược | US — Mua & nắm giữ |
|---|---|---|---|---|
| Lợi suất tích lũy | **+65,4%** | +277,9% | **+54,6%** | +87,0% |
| CAGR | 30,3% | 101,3% | 24,6% | 37,1% |
| **Sharpe** | **1,17** | **2,33** | **1,70** | **1,97** |
| Sortino | 3,51 | 3,84 | 2,29 | 2,86 |
| Max Drawdown | -25,2% | -24,8% | -17,3% | -17,4% |
| Calmar | 1,20 | 4,09 | 1,42 | 2,13 |
| Tỷ lệ nắm giữ | 47,0% | 100% | 71,3% | 100% |
| **Chi phí tích lũy** | **0,694** | 0 | 0,087 | 0 |

**Hình đính kèm:** `reports/figures/equity_vn.png` + `drawdown_vn.png`

**Phân tích theo từng năm (bảng nhỏ góc dưới):**

| Năm | VN chiến lược | VN mua & nắm giữ | US chiến lược | US mua & nắm giữ |
|---|---|---|---|---|
| 2024 | **−0,3%** | +40,3% | +34,2% | +44,7% |
| 2025 | +66,0% | +169,4% | +15,2% | +29,2% |

**Lời thoại:**
"Đây là slide trung thực nhất của đề tài, và cũng là slide em muốn thầy/cô lưu ý nhất.
Khi tính đủ **phí giao dịch, thuế và trượt giá**, chiến lược thua mua & nắm giữ ở **cả hai** thị trường, và thua ở **cả hai** năm. Ở Việt Nam, Sharpe của chiến lược chỉ đạt 1,17 trong khi mua & nắm giữ đạt 2,33.
Xem bảng theo năm: năm 2024, chiến lược của chúng em gần như bằng 0 — nghĩa là −0,3% — trong khi mua & nắm giữ lãi hơn 40%. Lý do nằm ở ô cuối: **chi phí giao dịch tích lũy tới 0,694**, tức là gần 70% vốn bị ăn mòn bởi phí và trượt giá, vì mô hình đổi vị thế rất thường xuyên.
Vậy nên kết luận của chúng em không phải là 'mô hình kiếm được tiền'. Kết luận là: **mô hình có tín hiệu dự đoán thống kê có ý nghĩa, nhưng rào cản chi phí giao dịch lớn đến mức biến tín hiệu đó thành không khả thi về mặt kinh tế**. Đây chính xác là lý do giả thuyết thị trường hiệu quả vẫn đứng vững trong thực tế — và đó cũng là đóng góp của đề tài khi chúng em chứng minh nó bằng số liệu thay vì chỉ đoán."

---

*→ Tiếp theo: xem file `PHAN_3.md` (Slide 13-15, phần hỏi đáp và phụ lục).*
