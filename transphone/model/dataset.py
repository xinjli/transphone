from phonepiece.lexicon import read_lexicon
from phonepiece.lang import read_all_langs
from transphone.model.vocab import Vocab
from transphone.config import TransphoneConfig
from tqdm import tqdm
import torch


def read_dataset(lang_id='eng'):

    lexicon = read_lexicon(lang_id)

    phoneme_lst = []
    grapheme_lst = []

    phoneme_set = set()
    grapheme_set = set()

    for grapheme_str, phonemes in lexicon.word2phoneme.items():
        graphemes = list(grapheme_str)
        phoneme_lst.append(phonemes)
        grapheme_lst.append(graphemes)
        phoneme_set.update(phonemes)
        grapheme_set.update(graphemes)

    phoneme_vocab = Vocab(phoneme_set)
    grapheme_vocab = Vocab(grapheme_set)

    return Dataset(phoneme_lst, grapheme_lst, phoneme_vocab, grapheme_vocab)


def read_multilingual_dataset():

    phoneme_lst = []
    grapheme_lst = []

    test_phoneme_lst = []
    test_grapheme_lst = []

    phoneme_set = set()
    grapheme_set = set()

    print("loading dataset...")

    for lang_id in tqdm(read_all_langs()):

        try:
            lexicon = read_lexicon(lang_id)
        except:
            print("skip ", lang_id)

        if len(lexicon.word2phoneme) < 50:
            continue

        lang_tag = '<'+lang_id+'>'

        grapheme_set.add(lang_tag)

        word2phoneme_lst = list(lexicon.word2phoneme.items())

        # train set
        for grapheme_str, phonemes in word2phoneme_lst[:-50]:
            graphemes = [lang_tag] + list(grapheme_str)
            phoneme_lst.append(phonemes)
            grapheme_lst.append(graphemes)
            phoneme_set.update(phonemes)
            grapheme_set.update(graphemes)

        # dev set
        for grapheme_str, phonemes in word2phoneme_lst[-50:-25]:
            graphemes = [lang_tag] + list(grapheme_str)
            test_phoneme_lst.append(phonemes)
            test_grapheme_lst.append(graphemes)
            phoneme_set.update(phonemes)
            grapheme_set.update(graphemes)

    phoneme_vocab = Vocab(phoneme_set)
    grapheme_vocab = Vocab(grapheme_set)

    train_dataset = Dataset(phoneme_lst, grapheme_lst, phoneme_vocab, grapheme_vocab)
    dev_dataset = Dataset(test_phoneme_lst, test_grapheme_lst, phoneme_vocab, grapheme_vocab)

    return train_dataset, dev_dataset


def read_test_dataset(model_name):

    model_path = TransphoneConfig.data_path / 'model' / model_name
    grapheme_vocab = Vocab.read(model_path / 'grapheme.vocab')
    phoneme_vocab = Vocab.read(model_path / 'phoneme.vocab')

    test_phoneme_lst = []
    test_grapheme_lst = []
    lang_lst = []

    for lang_id in tqdm(read_all_langs()):

        try:
            lexicon = read_lexicon(lang_id)
        except:
            print("skip ", lang_id)

        if len(lexicon.word2phoneme) <= 50:
            continue

        word2phoneme_lst = list(lexicon.word2phoneme.items())
        lang_phoneme_lst = []
        lang_grapheme_lst = []

        # last 25 for testing
        for grapheme_str, phonemes in word2phoneme_lst[-25:]:

            graphemes = list(grapheme_str)

            skip = False

            for phoneme in phonemes:
                if phoneme not in phoneme_vocab:
                    skip = True
                    break

            for grapheme in graphemes:
                if grapheme not in grapheme_vocab:
                    skip = True
                    break

            if skip:
                continue

            lang_phoneme_lst.append(phonemes)
            lang_grapheme_lst.append(grapheme_str)

        if len(lang_phoneme_lst) > 0:
            test_phoneme_lst.append(lang_phoneme_lst)
            test_grapheme_lst.append(lang_grapheme_lst)
            lang_lst.append(lang_id)

    return test_grapheme_lst, test_phoneme_lst, lang_lst


def read_zsl_dataset(model_name):

    model_path = TransphoneConfig.data_path / 'model' / model_name
    grapheme_vocab = Vocab.read(model_path / 'grapheme.vocab')
    phoneme_vocab = Vocab.read(model_path / 'phoneme.vocab')

    test_phoneme_lst = []
    test_grapheme_lst = []
    lang_lst = []

    for lang_id in tqdm(read_all_langs()):

        try:
            lexicon = read_lexicon(lang_id)
        except:
            print("skip ", lang_id)

        if len(lexicon.word2phoneme) >= 50 or len(lexicon.word2phoneme) == 0:
            continue

        word2phoneme_lst = list(lexicon.word2phoneme.items())

        lang_phoneme_lst = []
        lang_grapheme_lst = []

        # last 10 for testing
        for grapheme_str, phonemes in word2phoneme_lst[-25:]:
            graphemes = list(grapheme_str)

            skip = False

            for phoneme in phonemes:
                if phoneme not in phoneme_vocab:
                    skip = True
                    break

            for grapheme in graphemes:
                if grapheme not in grapheme_vocab:
                    skip = True
                    break

            if skip:
                continue

            lang_phoneme_lst.append(phonemes)
            lang_grapheme_lst.append(grapheme_str)

        test_phoneme_lst.append(lang_phoneme_lst)
        test_grapheme_lst.append(lang_grapheme_lst)
        lang_lst.append(lang_id)

    return test_grapheme_lst, test_phoneme_lst, lang_lst


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