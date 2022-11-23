from phonepiece.iso import normalize_lang_id
from transphone.lang.base_tokenizer import BaseTokenizer
from transphone.lang.eng.tokenizer import ENGTokenizer
from transphone.lang.cmn.tokenizer import CMNTokenizer
from transphone.lang.jpn.tokenizer import JPNTokenizer


def read_tokenizer(lang_id, g2p_model='latest'):

    lang_id = normalize_lang_id(lang_id)

    if lang_id == 'eng':
        tokenizer = ENGTokenizer(lang_id, g2p_model)
    elif lang_id == 'cmn':
        tokenizer = CMNTokenizer(lang_id, g2p_model)
    elif lang_id == 'jpn':
        tokenizer = JPNTokenizer(lang_id, g2p_model)
    else:
        tokenizer = G2PTokenizer(lang_id, g2p_model)

    return tokenizer


class G2PTokenizer(BaseTokenizer):

    def tokenize(self, text, use_g2p=True, verbose=False):

        text = text.translate(str.maketrans('', '', self.punctuation)).lower()
        result = []

        for word in text.split():
            if word in self.cache:
                result.extend(self.cache[word])
            else:
                phonemes = self.g2p.inference(word, self.lang_id)
                remapped_phonemes = self.inventory.remap(phonemes)
                if verbose:
                    print(f"g2p {word} ->  {remapped_phonemes}")
                self.cache[word] = remapped_phonemes
                result.extend(remapped_phonemes)

        return result
