from transphone.model.checkpoint_utils import torch_load
from transphone.model.transformer import TransformerG2P
from transphone.bin.download_model import download_model
from transphone.model.utils import resolve_model_name, get_all_models
from transphone.config import TransphoneConfig
from transphone.data.vocab import Vocab
import torch
import torch.nn as nn
from pathlib import Path
from argparse import Namespace


def read_g2p(inference_config_or_name='g2p', alt_model_path=None):

    if alt_model_path:
        if not alt_model_path.exists():
            download_model(inference_config_or_name, alt_model_path)

    # download specified model automatically if no model exists
    if len(get_all_models()) == 0:
        download_model('latest', alt_model_path)

    # create default config if input is the model's name
    if isinstance(inference_config_or_name, str):
        model_name = resolve_model_name(inference_config_or_name, alt_model_path)
        inference_config = Namespace(model=model_name, device_id=-1, lang='ipa', approximate=False, prior=None)
    else:
        assert isinstance(inference_config_or_name, Namespace)
        inference_config = inference_config_or_name

    if alt_model_path:
        model_path = alt_model_path / inference_config.model
    else:
        model_path = Path(__file__).parent / 'pretrained' / inference_config.model

    if inference_config.model == 'latest' and not model_path.exists():
        download_model(inference_config, alt_model_path)

    assert model_path.exists(), f"{inference_config.model} is not a valid model"

    model = G2P(model_path, inference_config)

    return model

class G2P:

    def __init__(self, model_path, inference_config):

        self.model_path = model_path
        self.grapheme_vocab = Vocab.read(model_path / 'grapheme.vocab')
        self.phoneme_vocab = Vocab.read(model_path / 'phoneme.vocab')
        self.inference_config = inference_config

        SRC_VOCAB_SIZE = len(self.grapheme_vocab)+1
        TGT_VOCAB_SIZE = len(self.phoneme_vocab)+1

        EMB_SIZE = 512
        NHEAD = 8
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 4
        NUM_DECODER_LAYERS = 4
        torch.manual_seed(0)

        self.model = TransformerG2P(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                            NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).cuda()

        torch_load(self.model, model_path / "model.pt")

    def inference(self, word, lang_id='eng'):

        lang_tag = '<'+lang_id+'>'

        graphemes = [lang_tag]+[w.lower() for w in list(word)]

        grapheme_ids = [self.grapheme_vocab.atoi(grapheme) for grapheme in graphemes]

        x = torch.LongTensor(grapheme_ids).unsqueeze(0).cuda()
        phone_ids = self.model.inference(x)

        phones = [self.phoneme_vocab.itoa(phone) for phone in phone_ids]

        return phones