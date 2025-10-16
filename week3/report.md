# Báo Cáo Lab 4: Word Embeddings with Word2Vec

## Giải Thích Các Bước Thực Hiện

### Bước 1: Cài Đặt và Khởi Tạo Môi Trường

Đầu tiên, chúng ta cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Các thư viện chính sử dụng trong lab:

- **gensim**: Cung cấp API để tải mô hình pre-trained và huấn luyện Word2Vec
- **numpy**: Xử lý các phép toán vector và ma trận
- **scipy**: Tính toán độ tương đồng cosine
- **matplotlib**: Trực quan hóa kết quả (nếu cần)

### Bước 2: Xây Dựng Lớp WordEmbedder

Chúng ta tạo lớp `WordEmbedder` để quản lý các tác vụ liên quan đến embedding:

```python
class WordEmbedder:
    def __init__(self, model_name: str)
    def get_vector(self, word: str)
    def get_similarity(self, word1: str, word2: str)
    def get_most_similar(self, word: str, top_n: int = 10)
    def embed_document(self, document: str)
    def tokenize(self, text: str)
```

Thiết kế này cho phép:

- Tải mô hình GloVe pre-trained một lần duy nhất (tăng hiệu suất)
- Chuẩn hóa các phương thức để dễ dàng tái sử dụng
- Xử lý các trường hợp đặc biệt như OOV (Out-of-Vocabulary) words

### Bước 3: Khám Phá Word Embedding

Thực hiện các phép thử cơ bản để hiểu hành vi của mô hình pre-trained:

- Lấy vector cho các từ cụ thể (ví dụ: "king", "queen")
- Tính độ tương đồng giữa các cặp từ khác nhau
- Tìm top N từ tương tự nhất với một từ cho trước

### Bước 4: Embedding Toàn Bộ Tài Liệu

Phương pháp baseline được sử dụng là lấy trung bình (mean) các vector từ trong tài liệu:

```
document_vector = mean([vector(word) for word in tokenize(document)])
```

Phương pháp này đơn giản nhưng hiệu quả, cho phép biểu diễn một tài liệu toàn bộ thành một vector duy nhất có cùng chiều với word vectors.

### Bước 5: Huấn Luyện Mô Hình Word2Vec Tùy Chỉnh (Bonus)

Ngoài việc sử dụng pre-trained models, chúng ta cũng thử huấn luyện Word2Vec từ đầu trên tập dữ liệu UD_English-EWT. Điều này giúp chúng ta hiểu rõ hơn quá trình học embeddings.

### Bước 6: Mở Rộng với Spark (Advanced)

Sử dụng PySpark để huấn luyện Word2Vec trên dataset lớn hơn (C4 dataset). Spark cho phép xử lý dữ liệu massive bằng cách phân tán tính toán trên nhiều máy.

---

## Hướng Dẫn Chạy Code

### Chạy Notebook NLP_lab3_new.ipynb (Pre-trained Model)

**Yêu cầu**: Python 3.8+, Jupyter Notebook hoặc Google Colab

**Các bước thực hiện**:

1. Mở notebook trên colab

2. Chạy từng ô (cell) theo thứ tự:

   - Cell 1: `!pip install gensim` - Cài đặt thư viện
   - Cell 2: Import các thư viện cần thiết (gensim, numpy, matplotlib)
   - Cell 3: Định nghĩa lớp `WordEmbedder` và tất cả các phương thức
   - Cell 4: Khởi tạo embedder và lấy vector cho từ "king"
   - Cell 5: Tính độ tương đồng giữa các cặp từ
   - Cell 6: Tìm 10 từ tương tự nhất với "computer"
   - Cell 7: Embedding câu "The queen rules the country."

3. Kiểm tra kết quả: Các output sẽ được in trực tiếp trong notebook

**Thời gian chạy**: Lần đầu mất khoảng 2-5 phút để tải mô hình (65MB). Các lần chạy tiếp theo sẽ nhanh hơn nhờ bộ nhớ đệm.

### Chạy Notebook NLP_lab3_train_model.ipynb (Spark + C4 Dataset)

**Các bước thực hiện**:

1. Chuẩn bị dữ liệu:

   - Tải C4 dataset: `c4-train.00000-of-01024-30K.json`
   - Đặt vào `/content/` hoặc thay đổi đường dẫn trong code

2. Mở notebook trên colab

3. Theo dõi tiến trình:
   - Spark session sẽ khởi tạo (in "Spark session initialized")
   - Dữ liệu JSON được tải và xử lý
   - Mô hình Word2Vec được huấn luyện (thường mất 1-3 phút)
   - Kết quả tương tự từ được in ra

## Phân Tích Kết Quả

### 1. Kết Quả Độ Tương Đồng (Similarity)

Từ thực nghiệm với mô hình GloVe, chúng ta có được:

```
Similarity between 'king' and 'queen': 0.7839
Similarity between 'king' and 'man': 0.5309
```

**Phân tích chi tiết**:

- **king ↔ queen (0.7839)**: Độ tương đồng cao (~78%) cho thấy mô hình đã nắm bắt được mối quan hệ ngữ nghĩa sâu sắc giữa hai chức vị hoàng gia. Cả hai từ đều mang ý nghĩa về quyền lực, chức vị cao cấp, và bối cảnh cung điện.

- **king ↔ man (0.5309)**: Độ tương đồng trung bình (~53%) phản ánh mối liên hệ yếu hơn. Mặc dù "king" là một loại "man", nhưng chúng khác biệt đáng kể về ngữ cảnh sử dụng - "king" mang nghĩa chuyên biệt và hẹp hơn.

**Nhận xét quan trọng**: Vector embedding không chỉ giữ thông tin từ vựng cơ bản mà còn thể hiện ý nghĩa ngữ cảnh phong phú. Mô hình GloVe được huấn luyện trên 100 triệu từ từ Wikipedia đã học được các mối quan hệ này rất tốt.

### 2. Top 10 Từ Tương Tự với "Computer"

Kết quả từ mô hình pre-trained:

```
1. computers      - 0.9165  (Dạng số nhiều)
2. software       - 0.8815  (Liên quan công nghệ)
3. technology     - 0.8526  (Liên quan công nghệ)
4. electronic     - 0.8126  (Liên quan công nghệ)
5. internet       - 0.8060  (Liên quan công nghệ)
6. computing      - 0.8026  (Liên quan công nghệ)
7. devices        - 0.8016  (Liên quan công nghệ)
8. digital        - 0.7992  (Liên quan công nghệ)
9. applications   - 0.7913  (Liên quan công nghệ)
10. (tiếp tục...)
```

**Phân loại các từ tìm được**:

- **Nhóm từ vựng gần nhất** (0.90-0.92):

  - "computers": Dạng số nhiều của "computer", thường xuất hiện cùng ngữ cảnh

- **Nhóm ngữ cảnh công nghệ chính** (0.88-0.81):

  - "software", "technology", "electronic"
  - Những từ này thường xuyên đi cùng với "computer" trong các tài liệu

- **Nhóm từ liên quan gián tiếp** (0.80-0.79):
  - "internet", "devices", "digital"
  - Thể hiện hệ sinh thái công nghệ rộng lớn

**Kết luận**: Mô hình GloVe đã học được các mối quan hệ ngữ cảnh phong phú, không chỉ là từ đồng nghĩa hoàn toàn mà còn các từ có liên quan chủ đề hay hệ sinh thái.

### 3. Embedding Tài Liệu

Câu kiểm thử: "The queen rules the country."

Kết quả vector (50 chiều):

```
[0.0456, 0.3653, -0.5597, 0.0401, 0.0966, 0.1562, -0.3362, -0.1250, ...]
```

**Quá trình tính toán**:

1. Tokenize câu: ["the", "queen", "rules", "the", "country"]
2. Lấy vector cho mỗi từ (bỏ qua các từ không trong vocab)
3. Trung bình cộng: `(vector_queen + vector_rules + vector_country) / 3`

**Ý nghĩa thực tiễn**:

- Vector này đại diện cho toàn bộ ý nghĩa của câu trong không gian 50 chiều
- Có thể dùng để so sánh sự tương đồng giữa các tài liệu khác nhau
- Là nền tảng cho các tác vụ như: phân loại văn bản, tìm kiếm ngữ nghĩa, clustering tài liệu

### 4. So Sánh: Pre-trained vs Model Tự Huấn Luyện

**Mô hình Pre-trained (GloVe - Wikipedia 100M từ)**:

- ✅ Chất lượng cao, nắm bắt ngữ nghĩa tổng quát rất tốt
- ✅ Nhanh chóng, không cần huấn luyện
- ✅ Tốt cho hầu hết NLP tasks tiêu chuẩn
- ❌ Có thể không phù hợp hoàn toàn với domain chuyên biệt

**Mô hình Tự Huấn Luyện (Word2Vec trên UD English - 217K từ)**:

Từ notebook NLP_lab3_train_model.ipynb, chúng ta thấy:

```
Word: "woman" → Similar words: ["acquire", "rachel", "toy", ...]
Analogy: king - man + woman = ["acquire", "toy", ...]
```

**Kết luận**:

- Pre-trained models tốt cho các tác vụ chung
- Tự huấn luyện cần dữ liệu lớn (thường > 1M từ) nhưng có thể tùy chỉnh cho domain riêng

### 5. Kết Quả Spark Word2Vec (Advanced)

Với dữ liệu C4 (30,000 dòng):

```
Các từ tương tự với 'computer':
- uwowned     0.7178
- desktop     0.7055
- pc          0.6234
- computers   0.6192
- software    0.6016
```

**So sánh với pre-trained GloVe**:

- GloVe: software (0.8815) vs Spark: software (0.6016)
- Điểm giống nhau: Cả hai đều xác định được "desktop", "software" là từ liên quan
- Điểm khác nhau: Spark tìm được "pc" (viết tắt) mà GloVe xếp thấp hơn

**Giải thích chênh lệch**:

- Dữ liệu Spark (30K) nhỏ hơn đáng kể so với GloVe (100M)
- Cách tiền xử lý dữ liệu khác nhau
- GloVe là 100 chiều, Spark là 100 chiều nhưng học từ dữ liệu ít hơn
- Spark chỉ học từ C4, trong khi GloVe học từ Wikipedia (chất lượng cao hơn)

**Ưu điểm của Spark**:

- Có khả năng mở rộng trên dữ liệu massive (terabytes)
- Phù hợp cho các tập đoàn với infrastructure lớn
- Có thể tích hợp với các pipeline big data khác

---

## Kết Luận

### Thành Tựu Chính

1. **Hiểu rõ Word Embeddings**: Chúng ta đã chuyển từ TF-IDF (thưa) sang Word2Vec (dày đặc) và thấy được lợi ích của ngữ cảnh ngữ nghĩa.

2. **Sử dụng Pre-trained Models**: GloVe cung cấp embedding chất lượng cao ngay lập tức mà không cần huấn luyện lâu.

3. **Tính Toán Độ Tương Đồng**: Cosine similarity là công cụ hiệu quả trong không gian embedding.

4. **Embedding Tài Liệu**: Phương pháp averaging là một cách đơn giản nhưng hiệu quả để tạo document vectors.

5. **Mở Rộng với Spark**: Chúng ta đã thấy cách mở rộng Word2Vec trên dữ liệu lớn bằng distributed computing.

### Bài Học Rút Ra

- Pre-trained embeddings phù hợp cho hầu hết các tác vụ NLP thực tế
- Tự huấn luyện cần dữ liệu lớn (thường > 1M từ) để có chất lượng tốt
- Embedding vectors giữ được mối quan hệ ngữ cảnh phong phú, cho phép tính toán analogies
- Spark cung cấp khả năng mở rộng nhưng đi kèm với độ phức tạp cao hơn

### Hướng Phát Triển Tiếp Theo

1. **Áp dụng cho Tiếng Việt**: Sử dụng FastText hoặc PhoBERT cho kết quả tốt hơn
2. **Trực Quan Hóa 2D**: Dùng t-SNE hoặc UMAP để visualize clusters từ
3. **Fine-tuning**: Tùy chỉnh embeddings cho các domain chuyên biệt
4. **Contextual Embeddings**: Chuyển sang BERT/Transformer cho kết quả tốt hơn
5. **Multi-lingual Models**: Sử dụng mBERT hoặc XLM-R cho đa ngôn ngữ

---

## Tài Liệu Tham Khảo

### Dữ Liệu Sử Dụng

- GloVe Wiki Gigaword 50-D: Pre-trained trên 6 tỷ tokens từ Wikipedia
- UD English-EWT: 217K tokens từ English Web Treebank
- C4 Dataset: 30K dòng văn bản (subset nhỏ)
