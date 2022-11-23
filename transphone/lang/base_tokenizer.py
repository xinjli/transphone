from phonepiece.inventory import read_inventory
from transphone.g2p import read_g2p

class BaseTokenizer:

    def __init__(self, lang_id, g2p_model='latest'):
        self.lang_id = lang_id
        self.inventory = read_inventory(lang_id)
        self.g2p = read_g2p(g2p_model)
        self.cache = {}
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

    def tokenize(self, text: str):
        raise NotImplementedError

    def convert_tokens_to_ids(self, lst):
        return self.inventory.phoneme.atoi(lst)

    def convert_ids_to_tokens(self, lst):
        return self.inventory.phoneme.itoa(lst)
