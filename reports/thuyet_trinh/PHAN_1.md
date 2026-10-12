# KỊCH BẢN THUYẾT TRÌNH — Mô hình ML dự đoán xu hướng giá cổ phiếu (VN vs Mỹ)

> File này gồm **3 phần** (PHAN_1, PHAN_2, PHAN_3). Muốn xem trọn bộ thì mở cả 3 file.
> Mỗi slide gồm: **Nội dung đưa lên slide** + **Lời thoại đọc** + **Hình đính kèm**.

---

# PHẦN 1 — MỞ ĐẦU VÀ PHƯƠNG PHÁP (Slide 1-8)

## Thông tin chung

| Mục | Giá trị |
|---|---|
| Thời lượng | 20 phút thuyết trình + 5-10 phút hỏi đáp |
| Số slide | 15 |
| Người trình bày | Nhóm trưởng (bạn) mở màn, chốt kết luận, giải trình phần phương pháp |
| Hình có sẵn | `reports/figures/*.png` (12 hình) — đã sinh từ kết quả chạy thật |

---

## SLIDE 1 — Trang bìa

**Nội dung trên slide:**
- Tên đề tài: *Mô hình máy học dự đoán xu hướng giá cổ phiếu: So sánh thị trường Việt Nam và Hoa Kỳ*
- Tên nhóm / tên các thành viên / lớp / môn học
- 3 ô nhấn mạnh (đặt dưới tên đề tài):
  1. 56 mã thực tế, 87.110 mẫu, 10 năm dữ liệu
  2. Walk-forward có purge — chống rò rỉ dữ liệu
  3. Kiểm định thống kê + backtest có chi phí

**Lời thoại:**
"Xin chào thầy/cô và các bạn. Nhóm em xin trình bày đề tài **dự đoán xu hướng giá cổ phiếu bằng máy học, so sánh giữa thị trường Việt Nam và Hoa Kỳ**.
Đề tài của nhóm khác biệt ở ba điểm: thứ nhất, chúng em dùng **dữ liệu thực tế** của 56 mã cổ phiếu trong 10 năm, không phải dữ liệu mô phỏng. Thứ hai, toàn bộ quy trình đánh giá được thiết kế để **chống rò rỉ dữ liệu** — đây là lỗi phổ biến nhất trong các bài làm dự đoán giá chứng khoán. Thứ ba, chúng em không dừng lại ở accuracy mà còn **kiểm định thống kê** và **mô phỏng giao dịch có chi phí thực** để xem mô hình có thực sự dùng được hay không."

---

## SLIDE 2 — Mục lục

**Nội dung trên slide (5 mục):**
1. Đặt vấn đề và câu hỏi nghiên cứu
2. Bộ dữ liệu và tiền xử lý
3. Phương pháp: đặc trưng, nhãn, thiết kế thực nghiệm
4. Các mô hình sử dụng
5. Kết quả, thảo luận và kết luận

**Lời thoại:**
"Phần thuyết trình gồm 5 phần. Em xin đi nhanh 2 phần đầu để dành thời gian cho phần mô hình và kết quả, vì đây cũng là phần chứa đóng góp chính của đề tài."

---

## SLIDE 3 — Đặt vấn đề & câu hỏi nghiên cứu

**Nội dung trên slide:**
- Ảnh hưởng thực tế: nếu dự đoán đúng hướng đi ngày mai, nhà đầu tư có thể ra quyết định tốt hơn
- Nhưng: giả thuyết thị trường hiệu quả (EMH) cho rằng giá đã phản ánh mọi thông tin sẵn có
- **Câu hỏi nghiên cứu:**
  > "Với cùng một pipeline học máy và cùng bộ đặc trưng, khả năng dự đoán xu hướng giá ngày hôm sau của nhóm cổ phiếu Việt Nam (VN30) có khác biệt **có ý nghĩa thống kê** so với nhóm cổ phiếu Hoa Kỳ (S&P 500) hay không?"
- 3 giả thuyết:
  - **H1**: Mô hình đạt ý nghĩa thống kê trên VN (AUC > 0,50 có CI không chứa 0,50)
  - **H2**: Nhóm đặc trưng thị trường (lợi suất chỉ số, beta) quan trọng hơn đặc trưng riêng từng mã
  - **H3**: Chi phí giao dịch là rào cản chính để biến tín hiệu thành lợi nhuận

**Lời thoại:**
"Câu hỏi mở đầu là: thị trường Việt Nam có dễ dự đoán hơn Mỹ không?
Về lý thuyết, giả thuyết thị trường hiệu quả cho rằng nếu thị trường thật sự hiệu quả thì không gì có thể dự đoán được giá cả. Tuy nhiên, có nhiều yếu tố khiến thị trường ở các nước mới nổi kém hiệu quả hơn: tỷ trọng nhà đầu tư cá nhân cao hơn, thông tin phản ánh chậm hơn, thanh khoản thấp hơn.
Chúng em đặt câu hỏi nghiên cứu một cách có thể kiểm chứng được: **khác biệt đó có ý nghĩa thống kê hay chỉ là nhiễu ngẫu nhiên?** Để trả lời, chúng em đưa ra ba giả thuyết kiểm chứng được — em sẽ quay lại xác nhận từng giả thuyết ở phần kết luận."

---

## SLIDE 4 — Bài toán: Input / Output

**Nội dung trên slide:**
- **Tên bài toán**: Phân loại nhị phân trên chuỗi thời gian tài chính (binary time-series classification)
- **Input**: Vector 45 đặc trưng của mã cổ phiếu i tại ngày t — chỉ dùng dữ liệu đến hết ngày t
- **Output**: P(giá tăng ngày mai), ngưỡng 0,5 → quyết định mua hoặc đứng ngoài
- **Công thức nhãn**:
  - y = 1 nếu ln(close_{t+1} / close_t) > +0,5%
  - y = 0 nếu ln(close_{t+1} / close_t) < −0,5%
  - Trong khoảng ±0,5% → **loại bỏ mẫu** (giảm nhiễu)
- **Lý do chọn phân loại thay vì hồi quy giá** (ô riêng, đặt nổi):
  - Hồi quy dự đoán giá tuyệt đối dễ tạo ảo giác: đường dự đoán "khớp" đường thật chỉ vì trễ 1 nhịp
  - Phân loại hướng đi bám sát quyết định thực tế của nhà đầu tư và đo được đúng thứ ta quan tâm
  - Tránh được bẫy "thang đo sai" khi giá có đơn vị khác nhau giữa VND và USD

**Lời thoại:**
"Đề tài của nhóm được hình thức hóa như sau. **Input** là vector 45 đặc trưng tính đến ngày hiện tại. **Output** là xác suất giá tăng ngày hôm sau, so với ngưỡng 0,5 để ra quyết định.
Một điểm quan trọng trong định nghĩa nhãn: chúng em **loại bỏ các ngày biến động trong khoảng ±0,5%**. Lý do là trong vùng này, chuyển động của giá gần như là nhiễu ngẫu nhiên — việc nhồi nhét các mẫu không có tín hiệu vào sẽ làm nhiễu tập huấn luyện.
Và đây là lý do chúng em chọn phân loại thay vì hồi quy giá: khi dự đoán giá tuyệt đối, đường dự đoán rất dễ "khớp" đường giá thật chỉ đơn thuần vì trễ một nhịp — nhìn rất đẹp trên biểu đồ nhưng vô giá trị. Phân loại hướng đi không cho phép ảo giác đó."

---

## SLIDE 5 — Bộ dữ liệu

**Nội dung trên slide (bảng 2 cột):**

| Chỉ số | Việt Nam | Mỹ |
|---|---|---|
| Nguồn | Yahoo Finance (`yfinance`) | Yahoo Finance (`yfinance`) |
| Giai đoạn | 01/01/2015 – 31/12/2025 | 01/01/2015 – 31/12/2025 |
| Số mã dùng | **26** | **30** |
| Tổng mẫu | 37.313 dòng | 49.797 dòng |
| Tỷ lệ nhãn "tăng" | 50,8% | 54,0% |
| Số ngày giao dịch/mã | ~2.846 | ~2.765 |

**3 ô phụ (đặt dưới bảng):**
- **Làm sạch**: loại mã có std lợi suất ngày > 8% hoặc > 1% ngày biến động > 15% (phát hiện lỗi điều chỉnh giá); loại mã có ít hơn 1.500 ngày giao dịch
- **Tiêu chí chọn mã**: thanh khoản cao, niêm yết liên tục ≥ 10 năm
- **Không có VN-Index**: dùng lợi suất bình quân equal-weight của các mã đã tải làm biến đại diện thị trường

**Lời thoại:**
"Về dữ liệu: cả hai thị trường đều lấy từ Yahoo Finance qua thư viện yfinance, giai đoạn 10 năm từ 2015 đến 2025, dữ liệu OHLCV theo ngày.
Sau khi lọc, chúng em giữ lại **26 mã Việt Nam và 30 mã Mỹ**, tổng cộng hơn 87.000 mẫu. Mỗi mẫu là một cặp (mã, ngày) với vector đặc trưng và nhãn tương lai của nó.
Hai điểm chúng em muốn nhấn mạnh ở slide này. Thứ nhất, chúng em có một bước **kiểm soát chất lượng dữ liệu tự động** — loại các mã có chuỗi giá bất thường, ví dụ biến động ngày vượt 8% liên tục, vì đó thường là lỗi điều chỉnh giá cổ phiếu từ phía nhà cung cấp chứ không phải biến động thật.
Thứ hai, chúng em xin nói rõ: Yahoo **không cung cấp VN-Index**, nên biến thị trường trong đặc trưng được thay bằng lợi suất bình quân của chính các mã đã tải. Chúng em không gọi biến đó là VN-Index."

---

## SLIDE 6 — Đặc trưng (Feature engineering)

**Nội dung trên slide (6 nhóm, mỗi nhóm 2-3 ví dụ):**

| Nhóm | Ví dụ đặc trưng |
|---|---|
| **Xu hướng** | close/SMA20, SMA20/SMA60, SMA50/SMA200, MACD histogram |
| **Động lượng** | RSI(14), Stochastic %K, ROC(5), ROC(21) |
| **Biến động** | ATR(14)/giá, Bollinger %B, realized vol 5/21/63 ngày, Parkinson vol |
| **Khối lượng** | log-volume z-score 20 ngày, volume/MA20, độ dốc OBV |
| **Vị thế giá** | khoảng cách đến đỉnh/đáy 52 tuần, ADX(14), HL range |
| **Thị trường & lịch** | lợi suất chỉ số 1/5/21 ngày, beta trượt 60 ngày, ngày trong tuần, tháng |

**Ghi chú quan trọng (ô đỏ/nổi):**
> Mọi đặc trưng tại ngày t **chỉ dùng dữ liệu đến hết ngày t**. Không có cửa sổ nào nhìn về tương lai.

**Lời thoại:**
"Chúng em xây dựng 45 đặc trưng, chia thành 6 nhóm. Nhóm **xu hướng** phản ánh giá đang đi lên hay xuống so với các trung bình động. Nhóm **động lượng** đo tốc độ và mức độ quá mua quá bán. Nhóm **biến động** rất quan trọng với chứng khoán — cùng một đặc trưng kỹ thuật có thể có nghĩa khác nhau tùy môi trường biến động.
Nhóm **khối lượng** đặc biệt nhạy với thị trường Việt Nam. Nhóm **thị trường** gồm lợi suất chỉ số và beta — đây là nhóm chúng em dự đoán sẽ quan trọng nhất, và em sẽ chứng minh ở phần SHAP.
Điều then chốt: **mọi đặc trưng chỉ nhìn về quá khứ**. Không một rolling window nào dùng center=True, không một hàm nào dùng dữ liệu ngày t+1. Đây là điều kiện tiên quyết để phần đánh giá sau đó có giá trị."

---

## SLIDE 7 — Nhãn & thiết kế thực nghiệm

**Nội dung trên slide (2 cột):**

*Cột trái — Thiết kế thực nghiệm:*
- Walk-forward, **5 fold**, cửa sổ trượt theo thời gian
- Huấn luyện mở rộng dần (expanding window)
- Mỗi fold: validation dài ~1 năm giao dịch
- **Purge 21 ngày**: xóa mẫu bị chồng lấn do cửa sổ đặc trưng
- **Embargo 5 ngày**: khoảng cách an toàn sau tập validation
- Tập test 2024–2025: **chạm đúng MỘT lần** khi báo cáo kết quả cuối

*Cột phải — Hình minh họa (vẽ trên slide):*
```
ĐỒNG HỒ THỜI GIAN →
[--- TRAIN 2015-2020 ---][purge][--- VAL 2021 ---]  ← folder 1
[--- TRAIN 2015-2021 ---][purge][--- VAL 2022 ---]  ← folder 2
[--- TRAIN 2015-2022 ---][purge][--- VAL 2023 ---]  ← folder 3
      ... lặp lại 5 lần, cửa sổ trượt về phía trước ...
         [===== TEST 2024-2025 — chỉ chạm 1 lần =====]
```

**Lời thoại:**
"Đây là slide quan trọng nhất trong phần phương pháp, và cũng là nơi nhiều bài làm mắc lỗi nghiêm trọng.
Thay vì chia dữ liệu một lần rồi huấn luyện, chúng em dùng **walk-forward với 5 fold**: luôn huấn luyện trên quá khứ, kiểm tra trên tương lai gần nhất, rồi trượt cửa sổ về phía trước. Tập huấn luyện mở rộng dần theo thời gian, giống cách nhà đầu tư thật sự thu thập dữ liệu.
Vùng màu xám là **purge**: đặc trưng của chúng em dùng cửa sổ dài nhất 252 ngày, nên một mẫu ở rìa tập huấn luyện vẫn chứa thông tin tràn sang tập validation. Chúng em cắt bỏ vùng đó. **Embargo** là khoảng cách an toàn phía sau.
Và quan trọng nhất: **tập test 2024-2025 chỉ được chạm đúng một lần**, khi báo cáo kết quả cuối cùng. Chúng em không tune tham số trên tập test, không xem kết quả test rồi quay lại sửa mô hình."

---

## SLIDE 8 — Các mô hình sử dụng

**Nội dung trên slide (3 cột):**

| Nhóm 1: Baseline | Nhóm 2: Mô hình chính | Nhóm 3: Công cụ |
|---|---|---|
| **Random** — mức sàn 0,50 | **XGBoost** | **SHAP** — giải thích mô hình |
| **Naive** — "hôm nay tăng ⇒ mai tăng" | **LightGBM** | **Optuna** — tune siêu tham số (chỉ trên validation) |
| **Logistic Regression** | **Random Forest** | |
| **ARIMA(2,0,2)** — dự báo 1 bước trên log-return | | |

**Tham số cấu hình (bảng nhỏ góc phải):**
- XGBoost: LR 0,03, depth 5, 1.500 cây, colsample 0,5
- LightGBM: LR 0,03, 15 lá, 2.000 cây, feature_fraction 0,5
- Random Forest: 300 cây, min_samples_leaf 50

**Lời thoại:**
"Về mô hình, chúng em chia thành hai tầng.
**Tầng thứ nhất là baseline** — và đây là điều nhiều bài bỏ qua. Nếu không có baseline, một con số accuracy 55% chẳng chứng minh được gì. Chúng em dùng bốn baseline: Random để biết mức sàn; Naive là quy tắc 'hôm nay tăng thì mai tăng' — quy tắc này **khó đánh bại đến bất ngờ** trên chuỗi thời gian tài chính; Logistic Regression để có một mô hình đơn giản có thể diễn giải; và ARIMA làm đại diện cho phương pháp kinh tế lượng truyềnền.
**Tầng thứ hai là mô hình chính**: XGBoost và LightGBM — hai thuật toán gradient boosting — cùng Random Forest để đối chiếu. Các tham số được cố tình đặt nhỏ: learning rate thấp, số lá ít, regularisation mạnh, vì dữ liệu tài chính có tỷ lệ tín hiệu trên nhiễu rất thấp, mô hình phức tạp sẽ học vẹt rất nhanh.
SHAP dùng để giải thích mô hình sau khi huấn luyện. Optuna đã cài sẵn nhưng đang tắt — chúng em chỉ bật khi tune trên tập validation, không bao giờ trên test."

---

*→ Tiếp theo: xem file `PHAN_2.md` (Slide 9-12) và `PHAN_3.md` (Slide 13-15 + Q&A).*
