from transphone.utils import import_with_auto_install
from transphone.lang.jpn.kana2phoneme import Kana2Phoneme
from transphone.g2p import read_g2p
from phonepiece.inventory import read_inventory
from transphone.lang.base_tokenizer import BaseTokenizer


class JPNTokenizer(BaseTokenizer):

    def __init__(self, lang_id, g2p_model='latest'):

        super(JPNTokenizer, self).__init__(lang_id, g2p_model)

        # import mecab and its dict
        MeCab = import_with_auto_install('MeCab', 'mecab-python3')
        import_with_auto_install('unidic_lite', 'unidic-lite')

        self.tagger = MeCab.Tagger()
        self.kana2phoneme = Kana2Phoneme()

    def tokenize(self, text, use_g2p=True, verbose=False):

        raw_words = self.tagger.parse(text).split('\n')

        result = []

        # exclude the last EOS word
        for word in raw_words[:-2]:

            kana = word.split(',')[-1]
            raw = word.split('\t')[0]

            res = self.kana2phoneme.convert(kana)

            if verbose:
                print(kana, res)

            if res != ['*']:
                result.extend(res)
            elif use_g2p:
                if raw in self.cache:
                    result.extend(self.cache[raw])
                else:
                    phonemes = self.g2p.inference(raw)
                    remapped_phonemes = self.inventory.remap(phonemes)
                    if verbose:
                        print(f"g2p {raw} ->  {remapped_phonemes}")
                    self.cache[raw] = remapped_phonemes
                    result.extend(remapped_phonemes)

        return result