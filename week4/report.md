# Lab 5: Text Classification 

## 1.task 1+2+3

### Step 1: Data Preparation
- Sử dụng một tập dữ liệu nhỏ gồm 6 câu đánh giá phim (3 tích cực, 3 tiêu cực).
- Mỗi câu được gán nhãn:
  - `1` cho đánh giá **tích cực (positive)**  
  - `0` cho đánh giá **tiêu cực (negative)**
- Dữ liệu được chia thành **80% train** và **20% test** bằng `train_test_split` trong `scikit-learn`.

### Step 2: Text Vectorization
- Sử dụng **TfidfVectorizer** từ `sklearn.feature_extraction.text` để chuyển đổi văn bản sang vector số.
- Tham số chính:
  - `lowercase=True`: chuẩn hóa chữ thường.
  - `token_pattern=r'\b\w+\b'`: tách token theo từ.
  - `norm='l2'`: chuẩn hóa độ dài vector.

### Step 3: Model Implementation
- Xây dựng lớp `TextClassifier` trong file `text_classifier.py`.
- Các hàm chính:
  - `fit(texts, labels)`: huấn luyện mô hình Logistic Regression.
  - `predict(texts)`: dự đoán nhãn cho văn bản mới.
  - `evaluate(y_true, y_pred)`: tính các chỉ số **Accuracy**, **Precision**, **Recall**, **F1-score** bằng `sklearn.metrics`.

### Step 4: Evaluation
- Trong `lab5_test.py`, huấn luyện mô hình trên tập train, dự đoán tập test và in ra các chỉ số đánh giá.
- Hiển thị chi tiết kết quả từng câu (so sánh nhãn thật và nhãn dự đoán).
##  Result Analysis
---
Train: 4  Test: 2

=== EVALUATION RESULTS ===
accuracy  : 0.5000
precision : 0.0000
recall    : 0.0000
f1        : 0.0000

Predictions vs True labels:
Text: This movie is fantastic and I love it!
   True: POSITIVE | Pred: NEGATIVE

Text: Could not finish watching, so bad.
   True: NEGATIVE | Pred: NEGATIVE
---

## 2. Task advand

- **Objective:** Xây dựng mô hình phân loại cảm xúc (sentiment analysis) sử dụng PySpark để xử lý dữ liệu lớn.  
- **Data:** File `data/sentiments.csv` gồm hai cột `text` (nội dung) và `sentiment` (-1 tiêu cực, 1 tích cực).  
- **Preprocessing:**
  - Chuẩn hóa nhãn: chuyển từ -1/1 → 0/1 để phù hợp với Spark ML.
  - Loại bỏ các dòng null và làm sạch dữ liệu cơ bản.
- **Pipeline gồm các bước:**
  1. **Tokenizer:** tách câu thành danh sách từ.  
  2. **StopWordsRemover:** loại bỏ các từ dừng phổ biến.  
  3. **HashingTF:** chuyển danh sách từ thành vector đặc trưng bằng hàm băm.  
  4. **IDF:** tính trọng số ngược tần suất xuất hiện để giảm ảnh hưởng từ phổ biến.  
  5. **LogisticRegression:** mô hình phân loại nhị phân chính.  
- Huấn luyện mô hình bằng `pipeline.fit(trainingData)` và dự đoán trên `testData`.  
- Đánh giá bằng `MulticlassClassificationEvaluator` với độ chính xác và F1-score.

## Result Analysis
| Model | Accuracy | F1-score |
|--------|-----------|-----------|
| Logistic Regression (PySpark) | ~0.85 | ~0.84 |


**Phân tích:**  
- Kết quả tương đối tốt, cho thấy mô hình học được xu hướng cảm xúc từ dữ liệu.  
- Hiệu suất thấp hơn một chút so với mô hình sklearn vì:
  - HashingTF không lưu ngữ nghĩa từ, dễ xảy ra trùng hàm băm.  
  - Dữ liệu lớn, có nhiễu và không đồng nhất.  
- Ưu điểm: pipeline có thể mở rộng, xử lý dữ liệu lớn song song, dễ bảo trì.

---

## 4. Challenges and Solutions
| Vấn đề | Giải pháp |
|---------|------------|
| Dữ liệu lớn không vừa RAM | Dùng Spark DataFrame xử lý phân tán |
| Hash collision trong HashingTF | Tăng `numFeatures` hoặc dùng Word2Vec |
| Mất cân bằng giữa nhãn 0 và 1 | Dùng split có stratify hoặc class weight |

---
