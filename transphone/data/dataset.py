from transphone.config import TransphoneConfig
from transphone.data.vocab import Vocab
from tqdm import tqdm
import torch
import random

def read_dataset(lang_id='eng'):

    r = open(TransphoneConfig.lang_path / lang_id / 'new_input', 'r')

    phoneme_lst = []
    grapheme_lst = []

    phoneme_set = set()
    grapheme_set = set()

    for line in r:
        phoneme_str, grapheme_str = line.strip().split('\t')
        phonemes = phoneme_str.split()
        graphemes = grapheme_str.split()
        phoneme_lst.append(phonemes)
        grapheme_lst.append(graphemes)
        phoneme_set.update(phonemes)
        grapheme_set.update(graphemes)

    phoneme_vocab = Vocab(phoneme_set)
    grapheme_vocab = Vocab(grapheme_set)

    return Dataset(phoneme_lst, grapheme_lst, phoneme_vocab, grapheme_vocab)


def read_p2g_dataset():
    r = open('/home/xinjianl/Git/allospeech/data/wilderness/bfa/exp/train.txt', 'r')

    train_phoneme_lst = []
    train_grapheme_lst = []

    test_phoneme_lst = []
    test_grapheme_lst = []

    phoneme_set = set()
    grapheme_set = set()

    max_len = 128
    for line in r:
        phoneme_str, grapheme_str = line.strip().split('\t')
        phonemes = phoneme_str.split()
        graphemes = grapheme_str.split()

        if len(phonemes) > max_len or len(graphemes) > max_len:
            continue

        train_phoneme_lst.append(phonemes)
        train_grapheme_lst.append(graphemes)
        phoneme_set.update(phonemes)
        grapheme_set.update(graphemes)


    r.close()

    r = open('/home/xinjianl/Git/allospeech/data/wilderness/bfa/exp/test.txt', 'r')
    for line in r:
        phoneme_str, grapheme_str = line.strip().split('\t')
        phonemes = phoneme_str.split()
        graphemes = grapheme_str.split()

        if len(phonemes) > max_len or len(graphemes) > max_len:
            continue

        test_phoneme_lst.append(phonemes)
        test_grapheme_lst.append(graphemes)
        phoneme_set.update(phonemes)
        grapheme_set.update(graphemes)

    phoneme_vocab = Vocab(phoneme_set)
    grapheme_vocab = Vocab(grapheme_set)

    train_set = P2GDataset(train_phoneme_lst, train_grapheme_lst, phoneme_vocab, grapheme_vocab)
    test_set = P2GDataset(test_phoneme_lst, test_grapheme_lst, phoneme_vocab, grapheme_vocab)

    return train_set, test_set

def read_multilingual_dataset():

    phoneme_lst = []
    grapheme_lst = []

    test_phoneme_lst = []
    test_grapheme_lst = []

    phoneme_set = set()
    grapheme_set = set()

    print("loading dataset...")

    for lang_dir in tqdm(list(TransphoneConfig.lang_path.glob('*'))):
        lang_id = lang_dir.stem

        if not (lang_dir / 'new_input').exists():
            continue
        lang_tag = '<'+lang_id+'>'

        grapheme_set.add(lang_tag)

        r = open(lang_dir / 'new_input', 'r')

        lines = r.readlines()

        random.shuffle(lines)

        # ignore the last 25
        max_num = max(0, len(lines)-5)
        min_num = min(max_num, 10000)

        for line in lines[:min_num]:
            grapheme_str, phoneme_str = line.strip().split('\t')
            grapheme_str = lang_tag + ' '+grapheme_str
            phonemes = phoneme_str.split()
            graphemes = grapheme_str.split()
            phoneme_lst.append(phonemes)
            grapheme_lst.append(graphemes)
            phoneme_set.update(phonemes)
            grapheme_set.update(graphemes)

        # last 25 for test
        for line in lines[-5:]:
            grapheme_str, phoneme_str = line.strip().split('\t')
            grapheme_str = lang_tag + ' '+ grapheme_str
            phonemes = phoneme_str.split()
            graphemes = grapheme_str.split()
            test_phoneme_lst.append(phonemes)
            test_grapheme_lst.append(graphemes)
            phoneme_set.update(phonemes)
            grapheme_set.update(graphemes)

    phoneme_vocab = Vocab(phoneme_set)
    grapheme_vocab = Vocab(grapheme_set)

    train_dataset = Dataset(phoneme_lst, grapheme_lst, phoneme_vocab, grapheme_vocab)
    test_dataset = Dataset(test_phoneme_lst, test_grapheme_lst, phoneme_vocab, grapheme_vocab)

    return train_dataset, test_dataset


class Dataset:

    def __init__(self, phoneme_lst, grapheme_lst, phoneme_vocab, grapheme_vocab):

        self.phoneme_lst = phoneme_lst
        self.grapheme_lst = grapheme_lst
        self.phoneme_vocab = phoneme_vocab
        self.grapheme_vocab = grapheme_vocab

    def __getitem__(self, item):

        phones = self.phoneme_lst[item]
        graphemes = self.grapheme_lst[item]

        phone_ids = [self.phoneme_vocab.atoi(phone) for phone in phones]
        grapheme_ids = [self.grapheme_vocab.atoi(grapheme) for grapheme in graphemes]

        return (torch.LongTensor(grapheme_ids), torch.LongTensor(phone_ids))

    def __len__(self):
        return len(self.phoneme_lst)


class P2GDataset:

    def __init__(self, phoneme_lst, grapheme_lst, phoneme_vocab, grapheme_vocab):

        self.phoneme_lst = phoneme_lst
        self.grapheme_lst = grapheme_lst
        self.phoneme_vocab = phoneme_vocab
        self.grapheme_vocab = grapheme_vocab

    def __getitem__(self, item):

        phones = self.phoneme_lst[item]
        graphemes = self.grapheme_lst[item]

        phone_ids = [self.phoneme_vocab.atoi(phone) for phone in phones]
        grapheme_ids = [self.grapheme_vocab.atoi(grapheme) for grapheme in graphemes]

        return (torch.LongTensor(phone_ids), torch.LongTensor(grapheme_ids))

    def __len__(self):
        return len(self.phoneme_lst)