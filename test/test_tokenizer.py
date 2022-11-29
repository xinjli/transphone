import unittest
from transphone.tokenizer import read_tokenizer

class TestTokenizer(unittest.TestCase):

    def test_eng_tokenizer(self):
        eng = read_tokenizer('eng')

        self.assertTrue(eng.tokenize_words('hello world') == ['hello', 'world'])
        self.assertTrue(eng.tokenize_words('hello world!!  ') == ['hello', 'world'])
        self.assertTrue(eng.tokenize('hello world') == ['h', 'ʌ', 'l', 'o', 'w', 'w', 'ɹ̩', 'l', 'd'])