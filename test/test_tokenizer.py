import unittest
from transphone.tokenizer import read_tokenizer
from transphone.lang.epitran_tokenizer import read_epitran_tokenizer

class TestTokenizer(unittest.TestCase):

    def test_eng_tokenizer(self):
        eng = read_tokenizer('eng')

        self.assertEqual(eng.tokenize_words('hello world'), ['hello', 'world'])
        self.assertEqual(eng.tokenize_words('hello world!!  '), ['hello', 'world'])
        self.assertEqual(eng.tokenize('hello world'), ['h', 'ʌ', 'l', 'o', 'w', 'w', 'ɹ̩', 'l', 'd'])

    def test_spa_epitran_tokenizer(self):

        spa = read_epitran_tokenizer('spa-Latn')
        self.assertEqual(spa.tokenize('hola hola'), ['o', 'l', 'a', 'o', 'l', 'a'])
        self.assertEqual(spa.tokenize('español'),  ['e', 's', 'p', 'a', 'ɲ', 'o', 'l'])

        spa = read_epitran_tokenizer('spa-Latn', use_lexicon=False)
        self.assertEqual(spa.tokenize('hola hola'), ['o', 'l', 'a', 'o', 'l', 'a'])
        self.assertEqual(spa.tokenize('español'),  ['e', 's', 'p', 'a', 'ɲ', 'o', 'l'])

    def test_fra_epitran_tokenizer(self):
        fra = read_epitran_tokenizer('fra-Latn')
        self.assertEqual(fra.tokenize('français'), ['f', 'ʁ', 'ɑ̃', 's', 'ɛ'])

        fra = read_epitran_tokenizer('fra-Latn', use_lexicon=False)
        self.assertEqual(fra.tokenize('français'), ['f', 'ʁ', 'ɑ̃', 's', 'ɛ'])

    def test_deu_epitran_tokenizer(self):
        deu = read_epitran_tokenizer('deu-Latn')
        self.assertEqual(deu.tokenize('Deutsche'), ['d', 'o', 'i', 't͡ʃ', 'ə'])

        deu = read_epitran_tokenizer('deu-Latn', use_lexicon=False)
        self.assertEqual(deu.tokenize('Deutsche'),['d', 'o', 'i', 't͡ʃ', 'ə'])

    def test_ita_epitran_tokenizer(self):
        ita = read_epitran_tokenizer('ita-Latn')
        self.assertEqual(ita.tokenize('Italia'), ['i', 't', 'a', 'l', 'j', 'a'])

        # g2p is slightly different here
        ita = read_epitran_tokenizer('ita-Latn', use_lexicon=False)
        self.assertEqual(ita.tokenize('Italia'), ['i', 't', 'a', 'l', 'i', 'a'])

    def test_tur_epitran_tokenizer(self):

        tur = read_epitran_tokenizer('tur-Latn', use_lexicon=False)
        self.assertEqual(tur.tokenize('Türkçe'), ['t', 'y', 'ɾ', 'k', 't͡ʃ', 'e'])

        tur = read_epitran_tokenizer('tur-Latn')
        self.assertEqual(tur.tokenize('Türkçe'), ['t', 'y', 'ɾ', 'k', 't͡ʃ', 'e'])
