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

- Mô tả công việc
