from phonepiece.inventory import read_inventory
from transphone.g2p import read_g2p
from transphone.config import TransphoneConfig

class BaseTokenizer:

    def __init__(self, lang_id, g2p_model='latest', device=None):
        self.lang_id = lang_id
        self.inventory = read_inventory(lang_id)

        if g2p_model is None:
            self.g2p = None
        else:
            self.g2p = read_g2p(g2p_model, device)

        # cache for g2p
        self.cache = {}

        # this will temporarily store new caches, which will be flashed to disk
        self.cache_log = {}

        if self.g2p is not None and self.g2p.cache_path is not None:
            lang_cache_path = self.g2p.cache_path / f"{lang_id}.txt"
            if lang_cache_path.exists():
                for line in open(lang_cache_path, 'r'):
                    fields = line.strip().split()
                    self.cache[fields[0]] = fields[1:]

        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
        self.logger = TransphoneConfig.logger

    def add_cache(self, word, phonemes):

        self.cache[word] = phonemes

        if self.g2p is None or self.g2p.cache_path is None:
            return

        # handle new cache
        self.cache_log[word] = phonemes

        # flash them to disk if the cache is large enough
        if len(self.cache_log) >= 100 and self.g2p.cache_path is not None and self.g2p.cache_path.exists():
            w = open(self.g2p.cache_path / f"{self.lang_id}.txt", 'a')
            for word, phonemes in self.cache_log.items():
                w.write(f"{word}\t{' '.join(phonemes)}\n")
            w.close()
            self.cache_log = {}

    def tokenize(self, text: str):
        raise NotImplementedError

    def tokenize_words(self, text:str):
        text = text.translate(str.maketrans('', '', self.punctuation)).lower()

        words = text.split()
        cleaned_words = [word for word in words if len(word) > 0]

        return text.split()

    def convert_tokens_to_ids(self, lst):
        lst = list(filter(lambda s: s!='<SPACE>', lst))

        return self.inventory.phoneme.atoi(lst)

    def convert_ids_to_tokens(self, lst):
        return self.inventory.phoneme.itoa(lst)
