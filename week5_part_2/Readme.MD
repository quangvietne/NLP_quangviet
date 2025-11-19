Chào bạn, dưới đây là **Báo cáo Kết quả (Report)** cho bài toán Part-of-Speech Tagging sử dụng mô hình Simple RNN, dựa trên kết quả huấn luyện từ các Cell 6 và Cell 7 của bạn.

Báo cáo này đáp ứng đầy đủ các tiêu chí bạn yêu cầu: Hiển thị độ chính xác theo từng epoch, báo cáo kết quả tốt nhất, và cung cấp ví dụ dự đoán câu mới.

## Báo Cáo Kết Quả Mô Hình Simple RNN cho POS Tagging 📊

### 1\. Thông số Kỹ thuật

  * **Mô hình**: Simple RNN (Recurrent Neural Network).
  * **Kiến trúc**: `nn.Embedding` -\> `nn.RNN` -\> `nn.Linear`.
  * **Bộ dữ liệu**: Universal Dependencies (UD\_English-EWT).
  * **Kích thước Embedding**: $100$
  * **Kích thước Hidden State (RNN)**: $128$
  * **Số Epoch huấn luyện**: $10$

-----

### 2\. Độ Chính xác qua từng Epoch

Bảng dưới đây trình bày độ chính xác trên tập huấn luyện (Train) và tập phát triển (Dev) sau mỗi epoch. Độ chính xác được tính toán bằng cách **loại bỏ các token padding** để phản ánh hiệu suất thực tế của mô hình trên các từ hợp lệ.

| Epoch | Loss (Train) | Accuracy (Train) | Loss (Dev) | Accuracy (Dev) | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.3502 | 96.28% | 1.6361 | 87.05% | |
| 2 | 0.3051 | 96.80% | 1.7229 | 86.94% | |
| **3** | **0.2630** | **97.21%** | **1.7882** | **87.09%** | **Mô hình Tốt nhất** |
| 4 | 0.2261 | 97.66% | 1.8471 | 87.04% | |
| 5 | 0.1950 | 97.97% | 1.9042 | 87.07% | |
| 6 | 0.1712 | 98.28% | 2.0026 | 86.81% | |
| 7 | 0.1467 | 98.50% | 2.1280 | 86.72% | |
| 8 | 0.1266 | 98.75% | 2.2059 | 86.81% | |
| 9 | 0.1092 | 98.91% | 2.3160 | 86.83% | |
| 10 | 0.0947 | 99.02% | 2.5090 | 86.58% | |

#### Lựa chọn Mô hình Tốt nhất

Mô hình tốt nhất được lựa chọn dựa trên độ chính xác cao nhất đạt được trên **tập phát triển (Dev)**:

  * **Độ chính xác cao nhất trên tập Dev đạt được là $87.09\%$ tại Epoch 3.**

-----

### 3\. Độ Chính xác Cuối cùng trên Tập Dev

Sau 10 Epoch huấn luyện và đánh giá, độ chính xác tốt nhất của mô hình được ghi nhận trên tập Dev là:

$$\text{Accuracy}_{\text{Dev} (\text{Best})} = \mathbf{87.09\%}$$

-----

### 4\. (Nâng cao) Ví dụ Dự đoán Câu Mới

Hàm `predict_sentence(sentence)` đã được sử dụng để xử lý câu mới và in ra cặp `(từ, nhãn_dự_đoán)`.

| Từ khóa | Nhãn UPOS Dự đoán |
| :---: | :---: |
| **Câu** | **"I love NLP and PyTorch"** |
| I | PRON (Đại từ) |
| love | VERB (Động từ) |
| NLP | DET (Từ hạn định) |
| and | CCONJ (Liên từ kết hợp) |
| PyTorch | NOUN (Danh từ) |

**Kết quả Dự đoán:**

```
[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'DET'), ('and', 'CCONJ'), ('PyTorch', 'NOUN')]
```