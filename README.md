# transphone

This repo will contain our code and pretrained model for our paper accepted at `Findings of ACL 2022`

`Zero-shot Learning for Grapheme to Phoneme Conversion with Language Ensemble`

It is a multilingual G2P (grapheme-to-phoneme) model that can be applied to all 8k languages registered in the [Glottolog database](https://glottolog.org/glottolog/language). You can read our papers at [Open Review](https://openreview.net/pdf?id=dKTTArRu8G2)

Our approach:
- We first trained our supervised multilingual model with ~300 languages using lexicons from Wikitionary
- if the target language (any language from the 8k languages) does not have any training set, we approximate its g2p model by using similar lanuguages from the supervised language sets and ensemble their inference results to obtain the target g2p.
