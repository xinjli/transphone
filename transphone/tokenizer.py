from phonepiece.iso import normalize_lang_id
from phonepiece.lexicon import read_lexicon
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

    def __init__(self, lang_id, g2p_model='latest'):
        super().__init__(lang_id, g2p_model)

        self.lexicon = read_lexicon(lang_id)


    def tokenize(self, text, use_g2p=True, use_space=False, verbose=False):

        text = text.translate(str.maketrans('', '', self.punctuation)).lower()
        result = []

        for word in text.split():
            if word in self.cache:
                result.extend(self.cache[word])
            elif word in self.lexicon:
                phonemes = self.lexicon[word]
                result.extend(phonemes)
                self.cache[word] = phonemes
                if verbose:
                    print(f"lexicon {word} -> {phonemes}")
            else:
                phonemes = self.g2p.inference(word, self.lang_id)
                remapped_phonemes = self.inventory.remap(phonemes)
                if verbose:
                    print(f"g2p {word} ->  {remapped_phonemes}")
                self.cache[word] = remapped_phonemes
                result.extend(remapped_phonemes)
            if use_space:
                result.append('<SPACE>')

        return result
