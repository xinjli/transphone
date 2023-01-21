from phonepiece.iso import normalize_lang_id
from phonepiece.lexicon import read_lexicon
from transphone.lang.base_tokenizer import BaseTokenizer
from transphone.lang.eng.tokenizer import ENGTokenizer
from transphone.lang.cmn.tokenizer import CMNTokenizer
from transphone.lang.jpn.tokenizer import JPNTokenizer
from transphone.lang.g2p_tokenizer import G2PTokenizer
from transphone.lang.epitran_tokenizer import read_epitran_tokenizer

def read_tokenizer(lang_id, g2p_model='latest', device=None, use_lexicon=True):

    lang_id = normalize_lang_id(lang_id)

    if lang_id == 'eng':
        tokenizer = ENGTokenizer(lang_id, g2p_model, device)
    elif lang_id == 'cmn':
        tokenizer = CMNTokenizer(lang_id, g2p_model, device)
    elif lang_id == 'jpn':
        tokenizer = JPNTokenizer(lang_id, g2p_model, device)
    elif lang_id == 'spa':
        tokenizer = read_epitran_tokenizer('spa-Latn', use_lexicon=use_lexicon)
    elif lang_id == 'deu':
        tokenizer = read_epitran_tokenizer('deu-Latn', use_lexicon=use_lexicon)
    elif lang_id == 'fra':
        tokenizer = read_epitran_tokenizer('fra-Latn', use_lexicon=use_lexicon)
    elif lang_id == 'ita':
        tokenizer = read_epitran_tokenizer('ita-Latn', use_lexicon=use_lexicon)
    elif lang_id == 'rus':
        tokenizer = read_epitran_tokenizer('rus-Cyrl', use_lexicon=use_lexicon)
    elif lang_id == 'tur':
        tokenizer = read_epitran_tokenizer('tur-Latn', use_lexicon=use_lexicon)
    else:
        tokenizer = G2PTokenizer(lang_id, g2p_model, device)

    return tokenizer