import unittest
from transphone.tokenizer import read_tokenizer
from transphone.lang.epitran_tokenizer import read_epitran_tokenzier

class TestTokenizer(unittest.TestCase):

    def test_eng_tokenizer(self):
        eng = read_tokenizer('eng')

        self.assertTrue(eng.tokenize_words('hello world') == ['hello', 'world'])
        self.assertTrue(eng.tokenize_words('hello world!!  ') == ['hello', 'world'])
        self.assertTrue(eng.tokenize('hello world') == ['h', 'ʌ', 'l', 'o', 'w', 'w', 'ɹ̩', 'l', 'd'])

    def spa_tokenizer(self):

        spa = read_epitran_tokenzier('spa-Latn')
        self.assertTrue(spa.tokenize('hola hola') == ['o', 'l', 'a', 'o', 'l', 'a'])