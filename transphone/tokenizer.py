from phonepiece.lang import normalize_lang_id
from phonepiece.lexicon import read_lexicon
from transphone.lang.base_tokenizer import BaseTokenizer
from transphone.lang.eng.tokenizer import read_eng_tokenizer
from transphone.lang.cmn.tokenizer import read_cmn_tokenizer
from transphone.lang.jpn.tokenizer import read_jpn_tokenizer
from transphone.lang.g2p_tokenizer import read_g2p_tokenizer
from transphone.model.utils import resolve_model_name
from transphone.lang.epitran_tokenizer import read_epitran_tokenizer

lang2tokenizer = {
    'eng': read_eng_tokenizer,
    'cmn': read_cmn_tokenizer,
    'jpn': read_jpn_tokenizer,
    'spa': read_epitran_tokenizer,
    'deu': read_epitran_tokenizer,
    'ita': read_epitran_tokenizer,
    'rus': read_epitran_tokenizer,
    'fra': read_epitran_tokenizer,
    'vie': read_epitran_tokenizer,
    'tha': read_epitran_tokenizer,
    'swa': read_epitran_tokenizer,
    'ckb': read_epitran_tokenizer,
    'cat': read_epitran_tokenizer,
}

def read_tokenizer(lang_id, g2p_model='latest', device=None, use_lexicon=True):

    lang_id = normalize_lang_id(lang_id)

    if lang_id in lang2tokenizer:
        return lang2tokenizer[lang_id](lang_id=lang_id, g2p_model=g2p_model, device=device, use_lexicon=use_lexicon)
    else:
        return read_g2p_tokenizer(lang_id=lang_id, g2p_model=g2p_model, device=device)