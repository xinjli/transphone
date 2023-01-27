import unittest
from transphone.tokenizer import read_tokenizer


class TestTokenizer(unittest.TestCase):

    def test_eng_tokenizer(self):
        eng = read_tokenizer('eng')

        self.assertEqual(eng.tokenize_words('hello world'), ['hello', 'world'])
        self.assertEqual(eng.tokenize_words('hello world!!  '), ['hello', 'world'])
        self.assertEqual(eng.tokenize('hello world'), ['h', 'ʌ', 'l', 'o', 'w', 'w', 'ɹ̩', 'l', 'd'])

    def test_spa_tokenizer(self):

        spa = read_tokenizer('spa')
        self.assertEqual(spa.tokenize('hola hola'), ['o', 'l', 'a', 'o', 'l', 'a'])
        self.assertEqual(spa.tokenize('español'),  ['e', 's', 'p', 'a', 'ɲ', 'o', 'l'])

        spa = read_tokenizer('spa', use_lexicon=False)
        self.assertEqual(spa.tokenize('hola hola'), ['o', 'l', 'a', 'o', 'l', 'a'])
        self.assertEqual(spa.tokenize('español'),  ['e', 's', 'p', 'a', 'ɲ', 'o', 'l'])

    def test_fra_tokenizer(self):

        fra = read_tokenizer('fra')
        self.assertEqual(fra.tokenize('français'), ['f', 'ʁ', 'ɑ̃', 's', 'ɛ'])

        fra = read_tokenizer('fra', use_lexicon=False)
        self.assertEqual(fra.tokenize('français'), ['f', 'ʁ', 'ɑ̃', 's', 'ɛ'])

    def test_deu_tokenizer(self):

        deu = read_tokenizer('deu')
        self.assertEqual(deu.tokenize('Deutsche'), ['d', 'o', 'i', 't͡ʃ', 'ə'])

        deu = read_tokenizer('deu', use_lexicon=False)
        self.assertEqual(deu.tokenize('Deutsche'),['d', 'o', 'i', 't͡ʃ', 'ə'])

    def test_ita_tokenizer(self):
        ita = read_tokenizer('ita')
        self.assertEqual(ita.tokenize('Italia'), ['i', 't', 'a', 'l', 'j', 'a'])

        # g2p is slightly different here
        ita = read_tokenizer('ita', use_lexicon=False)
        self.assertEqual(ita.tokenize('Italia'), ['i', 't', 'a', 'l', 'i', 'a'])

    def test_tur_tokenizer(self):

        tur = read_tokenizer('tur', use_lexicon=False)
        self.assertEqual(tur.tokenize('Türkçe'), ['t', 'y', 'ɾ', 'k', 't͡ʃ', 'e'])

        tur = read_tokenizer('tur')
        self.assertEqual(tur.tokenize('Türkçe'), ['t', 'y', 'ɾ', 'k', 't͡ʃ', 'e'])
