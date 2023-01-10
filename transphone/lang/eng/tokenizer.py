from transphone.utils import import_with_auto_install
from transphone.lang.base_tokenizer import BaseTokenizer
from phonepiece.arpa import ArpaConverter
from transphone.lang.eng.normalizer import ENGNormalizer

class ENGTokenizer(BaseTokenizer):

    def __init__(self, lang_id='eng', g2p_model='latest', device=None):

        super().__init__(lang_id, g2p_model, device)

        # import jieba and pypinyin for segmentation
        cmudict_module = import_with_auto_install('cmudict', 'cmudict')
        self.cmudict = cmudict_module.dict()
        self.converter = ArpaConverter()
        self.normalizer = ENGNormalizer()

    def tokenize(self, text, use_g2p=True, use_space=False, verbose=False):

        norm_text = self.normalizer(text)

        log = f"normalization: {text} -> {norm_text}"
        self.logger.info(log)
        if verbose:
            print(log)

        ipa_lst = []
        text = norm_text

        for word in text.split():
            if len(word) >= 1:
                if word in self.cache:
                    ipas = self.cache[word]
                    ipa_lst.extend(ipas)

                elif word in self.cmudict:
                    arpa = self.cmudict[word][0]
                    ipas = self.converter.convert(arpa)
                    ipa_lst.extend(ipas)

                    log = f"CMUdict: {word} -> {arpa} -> {ipas}"
                    self.logger.info(f"CMUdict: {word} -> {arpa} -> {ipas}")
                    if verbose:
                        print(log)

                    self.cache[word] = ipas

                elif use_g2p:
                    phonemes = self.g2p.inference(word)
                    remapped_phonemes = self.inventory.remap(phonemes)

                    log = f"g2p {word} ->  {remapped_phonemes}"
                    self.logger.info(log)
                    if verbose:
                        print(log)
                    self.cache[word] = remapped_phonemes
                    ipa_lst.extend(remapped_phonemes)
                if use_space:
                    ipa_lst.append('<SPACE>')

        return ipa_lst