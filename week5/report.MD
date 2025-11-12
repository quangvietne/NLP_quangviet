
### 📊 Bảng so sánh kết quả định lượng

Sau khi huấn luyện và đánh giá cả 4 pipeline, đây là kết quả

| Pipeline | F1-score (Macro) | Test Loss (hoặc Val\_loss) |
| :--- | :---: | :---: |
| **TF-IDF + Logistic Regression** | **0.8353** | N/A |
| Word2Vec (Avg) + Dense | 0.3032 | \~2.5184 |
| Embedding (Pre-trained) + LSTM | 0.0418 | \~3.3181 |
| Embedding (Scratch) + LSTM | 0.0005 | \~4.1240 |

**Phân tích nhanh kết quả định lượng:**

Kết quả định lượng cho thấy một điều rất rõ ràng:

1.  **Mô hình cổ điển (TF-IDF + Logistic Regression) chiến thắng áp đảo** với F1-score (macro) lên đến 83.53%.
2.  **Tất cả các mô hình Deep Learning đều thất bại thảm hại.** Mô hình "Word2Vec (Avg) + Dense" chỉ đạt F1-score 30.32%, trong khi cả hai mô hình LSTM đều cho kết quả gần như bằng 0 (F1-score 4.18% và 0.05%).
3.  **Lý do thất bại (sơ bộ):** Có hai lý do chính.
      * **Embedding (Pre-trained) + LSTM:** Mô hình này thất bại vì nó sử dụng vector Word2Vec (`w2v_model`) do chúng ta tự huấn luyện trên 8954 câu (quá ít). Chất lượng embedding này rất thấp, và việc chúng ta "đóng băng" nó (`trainable=False`) đã khiến mô hình không thể học được.
      * **Embedding (Scratch) + LSTM:** Mô hình này thất bại vì **đói dữ liệu (Data Starvation)**. Tập dữ liệu 8954 câu là quá nhỏ để mô hình có thể học đồng thời cả vector embedding lẫn quan hệ ngữ nghĩa của LSTM.

-----

### 🧠 Phân tích định tính (Tại sao LSTM không hoạt động?)

Đây là phần quan trọng nhất. Tôi đã chạy dự đoán trên 3 câu "khó" (lấy từ cell code cuối cùng) và so sánh với nhãn thật (True Labels) đã được sửa lại cho chính xác.

  * **Câu 1 (Phủ định):** `"can you remind me to not call my mom"`
      * **Nhãn thật:** `reminder_create`
  * **Câu 2 (Cấu trúc "hoặc"):** `"is it going to be sunny or rainy tomorrow"`
      * **Nhãn thật:** `weather_query`
  * **Câu 3 (Phụ thuộc xa & Phủ định):** `"find a flight from new york to london but not through paris"`
      * **Nhãn thật:** `flight_search`

**Bảng kết quả dự đoán (Từ Cell code cuối cùng):**

| Mô hình | Câu 1 ("...not call...") | Câu 2 ("...sunny or rainy...") | Câu 3 ("...but not through...") |
| :--- | :--- | :--- | :--- |
| **Nhãn thật** | **`reminder_create`** | **`weather_query`** | **`flight_search`** |
| TF-IDF + LR | `calendar_set` (Sai¹) | **`weather_query` (Đúng)** | `general_negate` (Sai) |
| W2V + Dense | `general_quirky` (Sai) | `qa_maths` (Sai) | `transport_query` (Sai²) |
| LSTM (Pre-trained) | `general_explain` (Sai) | `music_query` (Sai) | `lists_createoradd` (Sai) |
| LSTM (Scratch) | `iot_coffee` (Sai) | `iot_coffee` (Sai) | `iot_coffee` (Sai) |

-----

#### Phân tích chi tiết:

1.  **Hiện tượng "Sụp đổ mô hình" (Model Collapse) của LSTM (Scratch):**

      * Đây là phát hiện rõ ràng nhất. Mô hình "LSTM (Scratch)" đã sụp đổ hoàn toàn. Nó dự đoán **`iot_coffee` cho mọi câu**.
      * **Giải thích:** Với một tập dữ liệu quá nhỏ (8954 mẫu), mô hình có quá nhiều tham số (phải học cả Embedding và LSTM) đã không thể hội tụ. Nó chỉ học được cách dự đoán một lớp duy nhất để giảm thiểu loss. Mô hình này hoàn toàn vô dụng.

2.  **Sự thất bại của LSTM (Pre-trained) do Embedding kém:**

      * Mô hình này cũng thất bại, dự đoán các lớp sai một cách ngẫu nhiên (`general_explain`, `music_query`, `lists_createoradd`).
      * **Giải thích:** Mô hình Word2Vec (`w2v_model`) được huấn luyện ở Task 2 chỉ dựa trên 8954 câu. Đây là một embedding "rác". Khi tôi nạp nó vào LSTM và đặt `trainable=False`, tôi đã buộc mô hình LSTM phải học ngữ nghĩa dựa trên các vector đầu vào vô nghĩa. Dù kiến trúc LSTM có mạnh mẽ đến đâu, nó cũng không thể học được gì từ đầu vào kém chất lượng (Garbage In, Garbage Out).

3.  **Câu 1 & 2 (Chiến thắng cho TF-IDF):**

      * Mô hình **TF-IDF + LR** dự đoán **đúng** Câu 2 (`weather_query`) và **sai** Câu 1.
      * **(Sai¹) - Câu 1:** Nhãn thật là `reminder_create`. TF-IDF dự đoán `calendar_set`. Đây là một lỗi **sai nhưng có thể hiểu được**. Hai intent `reminder_create` (tạo nhắc nhở) và `calendar_set` (đặt lịch) rất gần nhau về mặt ngữ nghĩa và từ khóa (đều dùng "remind"). Mô hình đã bắt đúng *chủ đề* nhưng sai intent cụ thể.
      * **(Đúng) - Câu 2:** Mô hình bắt chính xác từ khóa `"sunny"`, `"rainy"`, `"tomorrow"` để dự đoán `weather_query`. Tất cả các mô hình DL khác đều sai hoàn toàn.

4.  **Câu 3 (Câu "khó" nhất: "...but not through paris"):**

      * Đây là câu mà **tất cả các mô hình đều dự đoán sai**, nhưng sai theo những cách khác nhau.
      * **TF-IDF + LR (Sai):** Dự đoán `general_negate`. Điều này cho thấy điểm yếu của nó: nó thấy "not" và bị nhầm lẫn, nghĩ rằng ý định của câu là "phủ định một điều gì đó" thay vì "tìm kiếm thông tin".
      * **W2V + Dense (Sai²):** Dự đoán `transport_query`. Đây là một lỗi **sai nhưng rất sát**. Nhãn thật là `flight_search` (tìm chuyến bay), là một intent con của `transport_query` (truy vấn vận tải). Giống như Câu 1, mô hình này đã bắt đúng *chủ đề* nhưng sai intent cụ thể. Bằng cách lấy trung bình, nó đã bỏ qua vế "but not" và chỉ tập trung vào các từ khóa "flight", "new york", "london".

**Kết luận từ phân tích định tính:**
Không có mô hình nào hiểu được các câu phức tạp. Các mô hình LSTM (vốn được kỳ vọng làm tốt việc này) đã thất bại hoàn toàn do không được huấn luyện đủ (thiếu dữ liệu, embedding kém). Mô hình TF-IDF, dù "ngây thơ", lại là mô hình duy nhất dự đoán đúng 1/3 câu và có 1/3 câu sai "chấp nhận được" (sai nhưng gần đúng).

-----

### ⚖️ Nhận xét chung: Ưu và Nhược điểm

Dựa trên các kết quả thực nghiệm của tôi trong file notebook này:

| Phương pháp | Ưu điểm (Pros) | Nhược điểm (Cons) |
| :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | - **Hiệu quả nhất** (F1 \> 83%).<br>- **Cực kỳ nhanh** và đơn giản để huấn luyện.<br>- Hoạt động tốt nhất trên các tập dữ liệu nhỏ (như 8954 mẫu), là baseline hoàn hảo. | - **Không hiểu thứ tự từ** (bị đánh lừa bởi câu "but not" và dự đoán sai `general_negate`).<br>- **Không hiểu ngữ nghĩa sâu** (nhầm lẫn giữa `reminder_create` và `calendar_set`). |
| **Word2Vec (Avg) + Dense** | - Bắt đầu nắm bắt được **ngữ nghĩa** (semantic) của từ.<br>- Dự đoán sai nhưng "gần đúng" ở Câu 3 (bắt đúng chủ đề `transport`). | - **Mất hoàn toàn thứ tự từ** do lấy trung bình. Đây là một bước thô sơ.<br>- Kết quả tổng thể rất tệ (F1 \< 31%), làm mất quá nhiều thông tin. |
| **Embedding (Pre-trained) + LSTM** | - *Lý thuyết:* Hiểu được thứ tự từ. | - **Thất bại (F1 \< 5%)**.<br>- **Embedding quá tệ:** Mô hình Word2Vec tự huấn luyện trên 8954 câu là không đủ chất lượng.<br>- **`trainable=False`:** "Đóng băng" một embedding tệ là một sai lầm chí mạng. |
| **Embedding (Scratch) + LSTM** | - *Lý thuyết:* Mô hình mạnh nhất, có thể học embedding dành riêng cho tác vụ. | - **Thất bại hoàn toàn (F1 \~ 0%)**.<br>- **Đói dữ liệu:** Mô hình quá phức tạp so với 8954 mẫu, dẫn đến "model collapse".<br>- Huấn luyện chậm nhất. |

**Kết luận chung:** Bài lab này là một ví dụ kinh điển cho thấy **"Deep Learning không phải là viên đạn bạc"**. Đối với các bài toán phân loại văn bản có tập dữ liệu nhỏ (\~9000 mẫu), một mô hình thống kê mạnh mẽ như **TF-IDF + Logistic Regression** là lựa chọn chiến thắng rõ ràng. Các kiến trúc phức tạp như LSTM, khi không có đủ dữ liệu hoặc không có embedding chất lượng cao (như GloVe, FastText đã được huấn luyện trước), sẽ không thể học được gì và cho kết quả tệ hơn đáng kể.