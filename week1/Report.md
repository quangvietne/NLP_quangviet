# NLP Labs Report

---

## LAB 1

### Mô tả công việc

- Định nghĩa và triển khai Interface: `tokenize(self, text: str) -> list[str]`
- Cài đặt và triển khai **SimpleTokenizer** và **RegexTokenizer**
- Viết test cho cả hai tokenizer (theo yêu cầu đề bài)
- Viết file `dataset_loaders.py` để load dữ liệu
- Viết test cho tokenizer trên bộ dữ liệu **UD_English-EWT**

### Kết quả chạy code

```
--- Testing Tokenizers ---
Input text: Hello, world! This is a test.
SimpleTokenizer Tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer Tokens:   ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Input text: NLP is fascinating... isn't it?
SimpleTokenizer Tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
RegexTokenizer Tokens:   ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Input text: Let's see how it handles 123 numbers and punctuation!
SimpleTokenizer Tokens: ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
RegexTokenizer Tokens:   ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

--- Testing Tokenizers on file data ---

--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...

SimpleTokenizer Output (first 20 tokens): ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']
RegexTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
```

### Giải thích kết quả

- **SimpleTokenizer**: chỉ xử lý một số dấu câu cố định → giữ nguyên contraction và từ ghép, nhưng không bao phủ hết các trường hợp.
- **RegexTokenizer**: tổng quát hơn, tách được nhiều loại ký tự → nhưng dễ gây tách sai contraction và từ có dấu gạch ngang.
  _Ví dụ:_ `al-zaman → ['al', '-', 'zaman']`, `let's → ['let', "'", 's']`

---

## LAB 2

### Mô tả công việc

- Xây dựng **Vectorizer Interface**, định nghĩa class trừu tượng `Vectorizer` với 3 phương thức:

  - `fit(self, corpus: list[str])`
  - `transform(self, documents: list[str]) -> list[list[int]]`
  - `fit_transform(self, corpus: list[str]) -> list[list[int]]`

- Cài đặt và triển khai **CountVectorizer**.

### Kết quả chạy code

```
Learned vocabulary:
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}

Document-term matrix:
Document 1: [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
Document 2: [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
Document 3: [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```

### Giải thích kết quả

- **Vocabulary**: gồm 10 phần tử. Dấu `"."` vẫn xuất hiện trong vocabulary vì RegexTokenizer coi nó là token hợp lệ → dẫn đến từ vựng chứa ký hiệu không có nhiều giá trị ngữ nghĩa.
- **Document-Term Matrix (DTM)**:

  - Mỗi document được biểu diễn thành vector đếm số lần xuất hiện của từng token trong vocabulary.
  - Các câu có độ dài và số lượng từ khác nhau được phản ánh trực tiếp trong số lượng token được gán giá trị > 0.

- **Khó khăn gặp phải**:

  - Dấu câu được giữ lại trong vocabulary → tạo ra token nhiễu.
  - Contraction và từ ghép bị tách sai bởi RegexTokenizer (ví dụ: `isn't`, `let's`).

- **Cách giải quyết**:

  - Thêm bước tiền xử lý loại bỏ dấu câu.
  - Chuẩn hóa contraction về dạng đầy đủ (`isn't → is not`).
  - Với corpus lớn, nên chuyển sang **TF-IDF Vectorizer** hoặc áp dụng giảm chiều để xử lý tính thưa của ma trận.

---
