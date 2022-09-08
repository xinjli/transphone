from pathlib import Path


class Vocab:

    def __init__(self, word_set):
        self.words = ['<pad>', '<eos>']+list(sorted(word_set))
        self.map = dict()

        for i, word in enumerate(self.words):
            self.map[word] = i


    @classmethod
    def read(cls, file_path):

        vocab = cls([])

        vocab.words = []
        vocab.map = dict()

        for i, line in enumerate(open(Path(file_path))):
            word = line.strip()
            vocab.words.append(word)
            vocab.map[word] = i

        return vocab

    def __len__(self):
        return len(self.words)

    def __contains__(self, item):
        return item in self.map

    def atoi(self, word):

        return self.map[word]

    def itoa(self, idx):
        word = self.words[idx]
        if word == '<space>':
            word = ' '

        return word

    def write(self, file_path):

        w = open(file_path, 'w', encoding='utf-8')

        for word in self.words:
            w.write(word+'\n')

        w.close()