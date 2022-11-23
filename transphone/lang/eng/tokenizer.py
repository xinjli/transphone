from transphone.utils import import_with_auto_install
from transphone.lang.base_tokenizer import BaseTokenizer
from phonepiece.arpa import ArpaConverter

class ENGTokenizer(BaseTokenizer):

    def __init__(self, lang_id='eng', g2p_model='latest'):

        super().__init__(lang_id, g2p_model)

        # import jieba and pypinyin for segmentation
        cmudict_module = import_with_auto_install('cmudict', 'cmudict')
        self.cmudict = cmudict_module.dict()
        self.converter = ArpaConverter()
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

    def tokenize(self, text, use_g2p=True, verbose=False):

        text = text.translate(str.maketrans('', '', self.punctuation)).lower()
        ipa_lst = []

        for word in text.split():
            if len(word) >= 1:
                if word in self.cmudict:
                    arpa = self.cmudict[word][0]
                    ipa_lst.extend(self.converter.convert(arpa))
                elif use_g2p:
                    phonemes = self.g2p.inference(word)
                    remapped_phonemes = self.inventory.remap(phonemes)
                    if verbose:
                        print(f"g2p {word} ->  {remapped_phonemes}")
                    self.cache[word] = remapped_phonemes
                    ipa_lst.extend(remapped_phonemes)

        return ipa_lst