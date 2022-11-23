from transphone.utils import import_with_auto_install
from transphone.lang.base_tokenizer import BaseTokenizer
from phonepiece.pinyin import PinyinConverter

class CMNTokenizer(BaseTokenizer):

    def __init__(self, lang_id='cmn', g2p_model='latest'):

        super().__init__(lang_id, g2p_model)

        # import jieba and pypinyin for segmentation
        self.jieba = import_with_auto_install('jieba', 'jieba')
        pypinyin = import_with_auto_install('pypinyin', 'pypinyin')

        self.pinyin = pypinyin.lazy_pinyin
        self.converter = PinyinConverter()

    def tokenize(self, text, use_g2p=True, verbose=False):

        words = list(self.jieba.cut(text, use_paddle=True))
        ipa_lst = []

        for word in words:
            pinyins = self.pinyin(word)
            for pinyin in pinyins:
                ipa_lst.extend(self.converter.convert(pinyin))
        return ipa_lst