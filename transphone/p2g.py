from transphone.utils.checkpoint_utils import torch_load
from transphone.model.transformer import TransformerG2P
from transphone.config import TransphoneConfig
from transphone.data.vocab import Vocab
import torch
import torch.nn as nn

def read_p2g():

    model = P2G()
    return model


class P2G:

    def __init__(self):

        self.grapheme_vocab = Vocab.read('/home/xinjianl/Git/allospeech/data/wilderness/bfa/exp/grapheme.vocab')
        self.phoneme_vocab = Vocab.read('/home/xinjianl/Git/allospeech/data/wilderness/bfa/exp/phoneme.vocab')

        SRC_VOCAB_SIZE = len(self.phoneme_vocab)+1
        TGT_VOCAB_SIZE = len(self.grapheme_vocab)+1

        EMB_SIZE = 512
        NHEAD = 8
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 4
        NUM_DECODER_LAYERS = 4
        torch.manual_seed(0)

        self.model = TransformerG2P(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                            NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).cuda()

        torch_load(self.model, "/home/xinjianl/Git/allospeech/data/wilderness/bfa/exp/model_0.101436.pt")

    def inference(self, phonemes):

        #phone_ids = [self.phoneme_vocab.atoi(phone) for phone in phones]
        phoneme_ids = [self.phoneme_vocab.atoi(phoneme) for phoneme in phonemes]

        x = torch.LongTensor(phoneme_ids).unsqueeze(0).cuda()
        grapheme_ids = self.model.inference(x)

        graphemes = [self.grapheme_vocab.itoa(grapheme_id) for grapheme_id in grapheme_ids]
        return graphemes