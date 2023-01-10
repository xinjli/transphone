from transphone.utils import import_with_auto_install
from transphone.lang.base_tokenizer import BaseTokenizer
from phonepiece.pinyin import PinyinConverter
from transphone.lang.cmn.normalizer import CMNNormalizer

class CMNTokenizer(BaseTokenizer):

    def __init__(self, lang_id='cmn', g2p_model='latest', device=None):

        super().__init__(lang_id, g2p_model, device)

        # import jieba and pypinyin for segmentation
        self.jieba = import_with_auto_install('jieba', 'jieba')
        pypinyin = import_with_auto_install('pypinyin', 'pypinyin')

        self.pinyin = pypinyin.lazy_pinyin
        self.converter = PinyinConverter()
        self.normalizer = CMNNormalizer()

    def tokenize_words(self, text):

        text = self.normalizer(text)
        words = list(self.jieba.cut(text, use_paddle=False))
        return words


    def tokenize(self, text, use_g2p=True, use_space=False, verbose=False):

        norm_text = self.normalizer(text)

        log = f"normalization: {text} -> {norm_text}"
        self.logger.info(log)
        if verbose:
            print(log)

        text = norm_text

        words = list(self.jieba.cut(text, use_paddle=False))

        ipa_lst = []

        for word in words:
            pinyins = self.pinyin(word)

            self.logger.info(f"pinyin: {word} -> {pinyins}")
            if verbose:
                print(f"pinyin: {word} -> {pinyins}")

            for pinyin in pinyins:
                ipa_lst.extend(self.converter.convert(pinyin))
            if use_space:
                ipa_lst.append('<SPACE>')
        return ipa_lst