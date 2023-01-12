from phonepiece.iso import normalize_lang_id
from transphone.model.checkpoint_utils import torch_load
from transphone.model.transformer import TransformerG2P
from transphone.model.ensemble import ensemble
from transphone.bin.download_model import download_model
from transphone.config import TransphoneConfig
from transphone.data.vocab import Vocab
from transphone.utils import Singleton
from phonepiece.tree import read_tree
from phonepiece.inventory import read_inventory
import torch
import unidecode
from itertools import groupby


def read_g2p(model_name='latest', device=None, alt_model_path=None):

    if device is not None:
        assert device in ['cpu', 'cuda']
        TransphoneConfig.device = device

    if alt_model_path:
        # check whether a customized path is used or not
        model_path = alt_model_path
    else:
        model_path = TransphoneConfig.data_path / 'model' / model_name

    # if not exists, we try to download the model
    if not model_path.exists():
        download_model(model_name)

    if not model_path.exists():
        raise ValueError(f"could not download or read {model_name} inventory")

    model = G2P(model_path, {})

    return model


class G2P(metaclass=Singleton):

    def __init__(self, model_path, inference_config):

        self.model_path = model_path
        self.grapheme_vocab = Vocab.read(model_path / 'grapheme.vocab')
        self.phoneme_vocab = Vocab.read(model_path / 'phoneme.vocab')
        self.inference_config = inference_config

        # setup available languages
        self.supervised_langs = []
        for word in self.grapheme_vocab.words[2:]:
            if len(word) == 5 and word[0] == '<' and word[-1] == '>':
                self.supervised_langs.append(word[1:-1])


        # cache to find proper supervised language
        self.lang_map = {}

        # inventory
        self.lang2inv = {}

        # tree to estimate language's similarity
        self.tree = read_tree()
        self.tree.setup_target_langs(self.supervised_langs)
        self.supervised_langs = set(self.supervised_langs)


        SRC_VOCAB_SIZE = len(self.grapheme_vocab)+1
        TGT_VOCAB_SIZE = len(self.phoneme_vocab)+1

        EMB_SIZE = 512
        NHEAD = 8
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 4
        NUM_DECODER_LAYERS = 4
        torch.manual_seed(0)


        self.model = TransformerG2P(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                            NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(TransphoneConfig.device)


        torch_load(self.model, model_path / "model.pt")


    def get_target_langs(self, lang_id, num_lang=10, debug=False, force_approximate=False):

        if lang_id in self.lang_map:
            target_langs = self.lang_map[lang_id]
        else:

            if force_approximate or lang_id not in self.supervised_langs:
                target_langs = self.tree.get_nearest_langs(lang_id, num_lang)
                if debug:
                    print("lang ", lang_id, " is not available directly, use ", target_langs, " instead")
                self.lang_map[lang_id] = target_langs
            else:
                self.lang_map[lang_id] = [lang_id]
                target_langs = [lang_id]

        return target_langs

    def inference_word(self, word, lang_id='eng', num_lang=10, debug=False, force_approximate=False):

        target_langs = self.get_target_langs(lang_id, num_lang, debug, force_approximate)

        phones_lst = []

        for target_lang_id in target_langs:
            lang_tag = '<' + target_lang_id + '>'

            graphemes = [lang_tag]+[w.lower() for w in list(word)]

            grapheme_ids = []
            for grapheme in graphemes:
                if grapheme not in self.grapheme_vocab:

                    # romanize chars not available in training languages
                    romans = list(unidecode.unidecode(grapheme))

                    if debug:
                        print("WARNING: not found grapheme ", grapheme, " in vocab. use ", romans, " instead")

                    for roman in romans:

                        # discard special chars such as $
                        if roman in self.grapheme_vocab:
                            grapheme_ids.append(self.grapheme_vocab.atoi(roman))
                    continue
                grapheme_ids.append(self.grapheme_vocab.atoi(grapheme))

            x = torch.LongTensor(grapheme_ids).unsqueeze(0).to(TransphoneConfig.device)

            phone_ids = self.model.inference(x)

            phones = [self.phoneme_vocab.itoa(phone) for phone in phone_ids]

            # ignore empty
            if len(phones) == 0:
                continue

            # if it is a mapped language, we need to map the inference_phone to the correct language inventory
            if lang_id not in self.lang2inv:
                inv = read_inventory(lang_id)
                self.lang2inv[lang_id] = inv

            inv = self.lang2inv[lang_id]
            phones = inv.remap(phones)

            if debug:
                print(target_lang_id, ' ', phones)

            phones_lst.append(phones)


        if len(phones_lst) == 0:
            phones = []
        else:
            phones = ensemble(phones_lst)

        return phones

    def inference(self, text, lang_id='eng', num_lang=10, debug=False, force_approximate=False):
        lang_id = normalize_lang_id(lang_id)

        phones_lst = []

        words = text.split()

        for word in words:
            phones = self.inference_word(word, lang_id, num_lang, debug, force_approximate)
            phones = [x[0] for x in groupby(phones)]
            phones_lst.extend(phones)

        return phones_lst