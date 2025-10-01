# Báo cáo – Lab 17: Spark NLP Pipeline

## 1. Các bước triển khai

0. **Đọc và hiểu code do trợ giảng cung cấp**

1. **Switch Tokenizers**

   - Comment phần `RegexTokenizer`.
   - Sử dụng `Tokenizer` cơ bản (phân tách theo khoảng trắng).

2. **Điều chỉnh kích thước vector đặc trưng (numFeatures)**

   - Trong `HashingTF`, thay đổi `numFeatures` từ **20000** xuống **1000**.

3. **Mở rộng Pipeline với Logistic Regression**

   - Thêm cột `label` giả lập (do dataset chưa có nhãn sẵn).
   - Thêm stage `LogisticRegression` vào pipeline sau bước IDF.
   - Train mô hình Logistic Regression trên tập dữ liệu đã vector hóa.

4. **Thử Vectorizer khác – Word2Vec**

   - Comment các stage `HashingTF` và `IDF`.
   - Thêm `Word2Vec` để sinh embedding cho mỗi văn bản.

5. **Sửa code phần ghi log và reslut**
   - Sửa code sao cho mỗi lần chạy , file log và result đc append thêm kết quả chư ko phải ghi đè kết quả

---

## 2. Cách chạy code và log kết quả (How to run the code and log the results)

1. Biên dịch và chạy chương trình bằng sbt:  
   sbt run

2. log và result ( trong file log và result , copy past vào thì dài quá nên em ko để ở đây )

## 3. Giải thích kết quả thu được

1. **Switch Tokenizers**

- Tokenizer: tách từ đơn giản theo khoảng trắng → nhanh, ít lỗi nhưng không xử lý dấu câu.
- RegexTokenizer: tách chi tiết hơn (loại bỏ ký tự đặc biệt, dấu câu).

2. **Giảm numFeatures từ 20000 → 1000**

- Vector TF-IDF ngắn hơn, ít chiều hơn.
- Khi vocab thực tế lớn hơn 1000 → xảy ra hash collision (nhiều từ khác nhau ánh xạ vào cùng một index).

3. **Logistic Regression**

- Có thể train mô hình phân loại cơ bản sau khi có vector TF-IDF.
- Vì dataset không có nhãn thật → cần tạo label giả (vd: random hoặc rule-based) để demo.

4. **Word2Vec**

- Thay vì vector TF-IDF thưa, Word2Vec sinh vector dense (embedding).
- Captures ngữ nghĩa tốt hơn TF-IDF, nhưng tốn tài nguyên tính toán hơn.

## 4. Khó khăn gặp phải và cách giải quyết (Difficulties and Solutions)

- Dataset không có nhãn → Logistic Regression không train được.
- Giải pháp: tạo cột label giả để chạy thử mô hình
