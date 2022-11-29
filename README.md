# transphone

![CI Test](https://github.com/xinjli/transphone/actions/workflows/python.yml/badge.svg)

`transphone` is a grapheme-to-phoneme conversion toolkit. It provides phoneme tokenizers as well as approximation G2P model for 8000 languages.

This repo contains our code and pretrained models roughly following our paper accepted at `Findings of ACL 2022`

`Zero-shot Learning for Grapheme to Phoneme Conversion with Language Ensemble`

It is a multilingual G2P (grapheme-to-phoneme) model that can be applied to all 8k languages registered in the [Glottolog database](https://glottolog.org/glottolog/language). You can read our papers at [Open Review](https://openreview.net/pdf?id=dKTTArRu8G2)

Our approach:
- We first trained our supervised multilingual model with ~900 languages using lexicons from [Wikitionary](https://en.wiktionary.org/wiki/Wiktionary:Main_Page)
- if the target language (any language from the 8k languages) does not have any training set, we approximate its g2p model by using similar lanuguages from the supervised language sets and ensemble their inference results to obtain the target g2p.


## Install

transphone is available from pip

```bash
pip install transphone
```

You can clone this repository and install

```bash
python setup.py install
```

## Usage

### Tokenizer interface

The tokenizer has a similar interface as HuggingFace tokenizer, which converts a string into each languages' phonemes

The tokenizer will first lookup lexicon dictionary for pronunciation, it will fall back to the G2P engine if lexicon is not available.  Currently, more than 200 languages have lexicon available inside. Other languages will use G2P instead.

```python
In [1]: from transphone import read_tokenizer                                                                                                  

In [2]: eng = read_tokenizer('eng')                                                                                                            

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

### G2P Command line

A command line tool is available

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

### python G2P interface

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

## Models

| model | # supported languages |       description        |
| :----: |:---------------------:|:------------------------:|
| latest |          ~8k          | based on our work at [1] |

## Reference

- [1] Li, Xinjian, et al. "Zero-shot Learning for Grapheme to Phoneme Conversion with Language Ensemble." Findings of the Association for Computational Linguistics: ACL 2022. 2022.
- [2] Li, Xinjian, et al. "Phone Inventories and Recognition for Every Language" LREC 2022. 2022
