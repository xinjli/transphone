from phonepiece.iso import normalize_lang_id
from phonepiece.lexicon import read_lexicon
from transphone.lang.base_tokenizer import BaseTokenizer
from transphone.lang.eng.tokenizer import ENGTokenizer
from transphone.lang.cmn.tokenizer import CMNTokenizer
from transphone.lang.jpn.tokenizer import JPNTokenizer
from transphone.lang.g2p_tokenizer import G2PTokenizer
from transphone.lang.epitran_tokenizer import read_raw_epitran_tokenizer, read_customized_epitran_tokenizer


customized_epitran_dict = {
    'spa': 'spa-Latn',
    'deu': 'deu-Latn',
    'ita': 'ita-Latn',
    'rus': 'rus-Cyrl',
    'fra': 'fra-Latn',
}

raw_epitran_dict = {
     'tur': 'tur-Latn',
     'vie': 'vie-Latn',
     'aar': 'aar-Latn',
     'got': 'got-Latn',
     'lsm': 'lsm-Latn',
     'swa': 'swa-Latn',
     'aii': 'aii-Syrc',
     'hak': 'hak-Latn',
     'ltc': 'ltc-Latn-bax',
     'swe': 'swe-Latn',
     'amh': 'amh-Ethi-red',
     'hat': 'hat-Latn-bab',
     'tam': 'tam-Taml-red',
     'hau': 'hau-Latn',
     'mal': 'mal-Mlym',
     'hin': 'hin-Deva',
     'mar': 'mar-Deva',
     'tel': 'tel-Telu',
     'ara': 'ara-Arab',
     'hmn': 'hmn-Latn',
     'mlt': 'mlt-Latn',
     'tgk': 'tgk-Cyrl',
     'ava': 'ava-Cyrl',
     'hrv': 'hrv-Latn',
     'mon': 'mon-Cyrl-bab',
     'tgl': 'tgl-Latn-red',
     'aze': 'aze-Latn',
     'hsn': 'hsn-Latn',
     'mri': 'mri-Latn',
     'hun': 'hun-Latn',
     'msa': 'msa-Latn',
     'tha': 'tha-Thai',
     'ben': 'ben-Beng-red',
     'ilo': 'ilo-Latn',
     'mya': 'mya-Mymr',
     'tir': 'tir-Ethi-red',
     'ind': 'ind-Latn',
     'nan': 'nan-Latn-tl',
     'ita': 'ita-Latn',
     'bxk': 'bxk-Latn',
     'jam': 'jam-Latn',
     'nld': 'nld-Latn',
     'tpi': 'tpi-Latn',
     'cat': 'cat-Latn',
     'jav': 'jav-Latn',
     'nya': 'nya-Latn',
     'tuk': 'tuk-Latn',
     'ceb': 'ceb-Latn',
     'jpn': 'jpn-Ktkn-red',
     'ood': 'ood-Latn-sax',
     'ces': 'ces-Latn',
     'cjy': 'cjy-Latn',
     'ori': 'ori-Orya',
     'ckb': 'ckb-Arab',
     'orm': 'orm-Latn',
     'cmn': 'cmn-Latn',
     'kat': 'kat-Geor',
     'pan': 'pan-Guru',
     'uew': 'uew',
     'kaz': 'kaz-Latn',
     'pii': 'pii-latn_Wiktionary',
     'uig': 'uig-Arab',
     'csb': 'csb-Latn',
     'ukr': 'ukr-Cyrl',
     'deu': 'deu-Latn-np',
     'pol': 'pol-Latn',
     'urd': 'urd-Arab',
     'kbd': 'kbd-Cyrl',
     'por': 'por-Latn',
     'uzb': 'uzb-Latn',
     'khm': 'khm-Khmr',
     'ron': 'ron-Latn',
     'fas': 'fas-Arab',
     'kin': 'kin-Latn',
     'run': 'run-Latn',
     'kir': 'kir-Latn',
     'rus': 'rus-Cyrl',
     'wuu': 'wuu-Latn',
     'sag': 'sag-Latn',
     'xho': 'xho-Latn',
     'sin': 'sin-Sinh',
     'yor': 'yor-Latn',
     'kmr': 'kmr-Latn',
     'sna': 'sna-Latn',
     'yue': 'yue-Latn',
     'som': 'som-Latn',
     'zha': 'zha-Latn',
     'ful': 'ful-Latn',
     'lao': 'lao-Laoo-prereform',
     'spa': 'spa-Latn-eu',
     'zul': 'zul-Latn',
     'gan': 'gan-Latn',
     'sqi': 'sqi-Latn',
     'lij': 'lij-Latn'
}


def read_tokenizer(lang_id, g2p_model='latest', device=None, use_lexicon=True):

    lang_id = normalize_lang_id(lang_id)

    if lang_id == 'eng':
        tokenizer = ENGTokenizer(lang_id, g2p_model, device)
    elif lang_id == 'cmn':
        tokenizer = CMNTokenizer(lang_id, g2p_model, device)
    elif lang_id == 'jpn':
        tokenizer = JPNTokenizer(lang_id, g2p_model, device)
    elif lang_id in customized_epitran_dict:
        tokenizer = read_customized_epitran_tokenizer(customized_epitran_dict[lang_id], use_lexicon=use_lexicon)
    elif lang_id in raw_epitran_dict:
        tokenizer = read_raw_epitran_tokenizer(raw_epitran_dict[lang_id], use_lexicon=use_lexicon)
    else:
        tokenizer = G2PTokenizer(lang_id, g2p_model, device)

    return tokenizer