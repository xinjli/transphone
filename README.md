# transphone

![CI Test](https://github.com/xinjli/transphone/actions/workflows/python.yml/badge.svg)

`transphone` is a multilingual grapheme-to-phoneme conversion toolkit derived from our paper: [Zero-shot Learning for Grapheme to Phoneme Conversion with Language Ensemble](https://aclanthology.org/2022.findings-acl.166/).

It provides approximiated phoneme tokenizers and G2P model for 7546 languages registered in the [Glottolog database](https://glottolog.org/glottolog/language).  You can see the full list of supported languages in [the language doc](./doc/language.md) 

## Install

transphone is available from pip

```bash
pip install transphone
```

You can clone this repository and install

```bash
python setup.py install
```

## Tokenizer Usage

The tokenizer converts a string into each languages' phonemes. By default, it combines a few approach to decide the pronunciation of a word for the target language:

- **lexicon-based**: it will first lookup lexicon dictionary for pronunciation (from Wikitionary, cmudict, and other sources), currently around 1k languages have at least some entries.
- **transducer-based**: it will use rule-based transducer from [epitran](https://github.com/dmort27/epitran) for several languages considering accuracy and speed. 
- **g2p-based**: use the G2P model as described in the next section.

### python interface

You can use it from python as follows:

```python
In [1]: from transphone import read_tokenizer                                                                                                  

# use 2-char or 3-char ISO id to specify your target language 
In [2]: eng = read_tokenizer('eng')                                                                                                            

# tokenize a string of text into a list of phonemes
In [3]: lst = eng.tokenize('hello world')                                                                                                      

In [4]: lst                                                                                                                                    
Out[4]: ['h', 'ʌ', 'l', 'o', 'w', 'w', 'ɹ̩', 'l', 'd']

In [5]: ids = eng.convert_tokens_to_ids(lst)                                                                                                   

In [6]: ids                                                                                                                                    
Out[6]: [7, 36, 11, 14, 21, 21, 33, 11, 3]

In [7]: eng.convert_ids_to_tokens(ids)                                                                                                         
Out[7]: ['h', 'ʌ', 'l', 'o', 'w', 'w', 'ɹ̩', 'l', 'd']

In [8]: jpn = read_tokenizer('jpn')                                                                                                            

In [9]: jpn.tokenize('こんにちは世界')                                                                                                         
Out[9]: ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', 's', 'e', 'k', 'a', 'i']

In [10]: cmn = read_tokenizer('cmn')                                                                                                           

In [11]: cmn.tokenize('你好世界')                                                                                                              
Out[11]: ['n', 'i', 'x', 'a', 'o', 'ʂ', 'ɻ̩', 't͡ɕ', 'i', 'e']

In [12]: deu = read_tokenizer('deu')                                    

In [13]: deu.tokenize('Hallo Welt')                                     
Out[13]: ['h', 'a', 'l', 'o', 'v', 'e', 'l', 't']

```

### command line interface

A command line tool is also available

```bash
# compute pronunciation for every word in input file
$ python -m transphone.run --lang eng --input sample.txt 
h ɛ l o ʊ
w ə l d
t ɹ æ n s f ə ʊ n

# by specifying combine flag, you can get word + pronunciation per line
$ python -m transphone.run --lang eng --input sample.txt --combine=True
hello h ɛ l o ʊ
world w ə l d
transphone t ɹ æ n s f ə ʊ n
```

## G2P Backend Usage

The tokenizer in the previous section uses the G2P as one of the backend option. You can also use the G2P model directly.

### python interface

A simple python usage is as follows:

```python
In [1]: from transphone.g2p import read_g2p                                                                                                     

# read a pretrained model. It will download the pretrained model automatically into repo_root/data/model
In [2]: model = read_g2p()                                                                                                                      

# to infer pronunciation for a word with ISO 639-3 id
# For any pretrained languages (~900 languages), it will use the pretrained model without approximation
In [3]: model.inference('transphone', 'eng')                                                                                                    
Out[3]: ['t', 'ɹ', 'æ', 'n', 's', 'f', 'ə', 'ʊ', 'n']

# If the specified language is not available, then it will approximate it using nearest languages
# in this case, aaa (Ghotuo language) is not one of the training languages, we fetch 10 nearest languages to approximate it 
In [4]: model.inference('transphone', 'aaa')                                                                                                    
lang  aaa  is not available directly, use  ['bin', 'bja', 'bkh', 'bvx', 'dua', 'eto', 'gwe', 'ibo', 'kam', 'kik']  instead
Out[4]: ['t', 'l', 'a', 'n', 's', 'f', 'o', 'n', 'e']

# To gain deeper insights, you can also specify debug flag to see output of each language
In [5]: model.inference('transphone', 'aaa', debug=True)                                                                                        
bin   ['s', 'l', 'a', 'n', 's', 'f', 'o', 'n', 'e']
bja   ['s', 'l', 'a', 'n', 's', 'f', 'o', 'n']
bkh   ['t', 'l', 'a', 'n', 's', 'f', 'o', 'n', 'e']
bvx   ['t', 'r', 'a', 'n', 's', 'f', 'o', 'n', 'e']
dua   ['t', 'r', 'n', 's', 'f', 'n']
eto   ['t', 'l', 'a', 'n', 's', 'f', 'o', 'n', 'e']
gwe   ['t', 'l', 'a', 'n', 's', 'f', 'o', 'n', 'e']
ibo   ['t', 'l', 'a', 'n', 's', 'p', 'o', 'n', 'e']
kam   ['t', 'l', 'a', 'n', 's', 'f', 'o', 'n', 'e']
kik   ['t', 'l', 'a', 'n', 's', 'f', 'ɔ', 'n', 'ɛ']
Out[5]: ['t', 'l', 'a', 'n', 's', 'f', 'o', 'n', 'e']
```

### Pretrained Models

This pretrained models roughly following our paper accepted at `Findings of ACL 2022`: [Zero-shot Learning for Grapheme to Phoneme Conversion with Language Ensemble](https://aclanthology.org/2022.findings-acl.166/). 

You can see the G2P evaluation over 1k languages on the [performance doc](./doc/performance/README.md)

Note this is the pure G2P evaluation on unseen words. The tokenizer combines other existing resources (i.e. lexicon) as well, so the tokenizer's performance is expected to be much better than this. 

|        model         | # supported languages | supervised language PER | zero-shot language PER |       description        |
|:--------------------:|:---------------------:|:-----------------------:|:----------------------:|:------------------------:|
| 042801_base (latest) |          ~8k          |           13%           |          31%           | based on our work at [1] |

### Training

We also provide the training code for G2P. You can reproduce the pretrained model using `transphone.bin.train_g2p` 

## Epitran backend

This repo also provides a wrapper of a customized version of [epitran](https://github.com/dmort27/epitran). For a few languages, it will use epitran as the backend considering accuracy and speed.

You can also use epitran directly as follows:

```python
In [1]: tokenizer = read_epitran_tokenizer('spa', use_lexicon=False)

In [2]: tokenizer.tokenize('hola')
Out[2]: ['o', 'l', 'a']
```

## Reference

- [1] Li, Xinjian, et al. "Zero-shot Learning for Grapheme to Phoneme Conversion with Language Ensemble." Findings of the Association for Computational Linguistics: ACL 2022. 2022.
- [2] Li, Xinjian, et al. "Phone Inventories and Recognition for Every Language" LREC 2022. 2022