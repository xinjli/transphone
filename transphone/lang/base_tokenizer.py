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

        self.cache = {}

        if self.g2p is not None and self.g2p.cache_path is not None:
            lang_cache_path = self.g2p.cache_path / f"{lang_id}.txt"
            if lang_cache_path.exists():
                for line in open(lang_cache_path, 'r'):
                    fields = line.strip().split()
                    self.cache[fields[0]] = fields[1:]

        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.logger = TransphoneConfig.logger

    def tokenize(self, text: str):
        raise NotImplementedError

    def tokenize_words(self, text:str):
        text = text.translate(str.maketrans('', '', self.punctuation)).lower()

        return text.split()

    def convert_tokens_to_ids(self, lst):
        lst = list(filter(lambda s: s!='<SPACE>', lst))

        return self.inventory.phoneme.atoi(lst)

    def convert_ids_to_tokens(self, lst):
        return self.inventory.phoneme.itoa(lst)
