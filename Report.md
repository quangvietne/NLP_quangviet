--- LAB 1 ---

\*\*\* Mô tả công việc :

- Định nghĩa và triển khai Interface : tokenize(self, text: str)->list[str]
- Cài đặt , triển khai SimpleTokenizer , RegexTokenizer
- Viết test cho SimpleTokenizer và RegexTokenizer ( theo yêu cầu đề bài)
- Viết file dataset_loaders.py để load dữ liệu
- Viết test cho SimpleTokenizer và RegexTokenizer trên bộ dữ liệu UD_English-EWT

\*\*\* Kết quả chạy code :

"""

--- Testing Tokenizers ---
Input text: Hello, world! This is a test.
SimpleTokenizer Tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer Tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
Input text: NLP is fascinating... isn't it?
SimpleTokenizer Tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
RegexTokenizer Tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']
Input text: Let's see how it handles 123 numbers and punctuation!
SimpleTokenizer Tokens: ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
RegexTokenizer Tokens: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

--- Testing Tokenizers on file data ---

--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...
SimpleTokenizer Output (first 20 tokens): ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']
RegexTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']

"""

\*\*\* Giải thích kết quả:

- SimpleTokenizer thì đơn giản, chỉ xử lý một số dấu câu cố định → giữ nguyên contraction và từ ghép , nhưng ko chính xác trong nhiều trường hợp ( chưa bao phủ các dấu câu )
- RegexTokenizer thì tổng quát hơn, tách được nhiều loại ký tự → nhưng gây ra hiện tượng tách sai với contraction và từ có dấu gạch ngang. (VD : al-zaman -> 'al', '-', 'zaman' ; let's -> 'let', "'", 's',)

--- LAB 2 ---

\*\*\* Mô tả công việc :

- Xây dựng Vectorizer Interface , định nghĩa class trừu tượng Vectorizer với 3 phương thức:

* fit(self, corpus: list[str])
* transform(self, documents: list[str])-> list[list[int]]
* fit_transform(self, corpus: list[str])-> list[list[int]]

- Cài đặt , triển khai CountVectorizer :

\*\*\* Kết quả chạy code :

"""

Learned vocabulary:
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}

Document-term matrix:
Document 1: [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
Document 2: [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
Document 3: [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]

"""

\*\*\* Giải thích kết quả :

- Vocabulary thu được gồm 10 phần tử: Dấu "." vẫn xuất hiện trong vocabulary. Điều này cho thấy RegexTokenizer đang coi dấu câu là một token hợp lệ, dẫn đến việc vocabulary chứa các ký hiệu không có nhiều giá trị ngữ nghĩa.

- Document-Term Matrix (DTM) :

* Mỗi document được biểu diễn thành một vector đếm số lần xuất hiện của từng token trong vocabulary.
* Nhìn vào DTM, ta thấy rằng các câu có độ dài và số từ khác nhau sẽ được phản ánh trực tiếp trong số lượng token được kích hoạt trong vector

- Gặp khó khăn với contraction và từ ghép: Với tokenizer hiện tại, contraction (như "isn't", "let's") sẽ bị tách thành nhiều token nhỏ, làm tăng số lượng từ vựng và gây khó khăn trong xử lý ngữ nghĩa.
