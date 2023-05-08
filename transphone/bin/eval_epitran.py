from transphone.tokenizer import raw_epitran_dict
from transphone.lang.epitran_tokenizer import read_raw_epitran_tokenizer
from phonepiece.lexicon import read_lexicon
from phonepiece.distance import phonological_distance
import editdistance
from transphone.model.checkpoint_utils import *
import argparse

def eval_epitran(langs=None, exclude_langs=None):

    exp_dir = TransphoneConfig.data_path / 'decode' / 'epitran'
    exp_dir.mkdir(exist_ok=True, parents=True)

    log_w = open(exp_dir / f'result.md', 'w')

    log_w.write('| language | phoneme error rate | phonological distance |\n')
    log_w.write('|----------|--------------------|-----------------------|\n')

    if langs is not None and len(langs) != 0:
        target_langs = langs
    elif exclude_langs is not None and len(exclude_langs) != 0:
        target_langs = []
        for lang in  list(raw_epitran_dict.keys()):
            if lang not in exclude_langs:
                target_langs.append(lang)
    else:
        target_langs = list(raw_epitran_dict.keys())

    tot_cer = 0
    tot_fer = 0
    tot_csum = 0

    for lang in target_langs:
        cer = 0
        fer = 0
        csum = 0

        print("processing ", lang)

        epitran_id = raw_epitran_dict[lang]

        try:
            lexicon = read_lexicon(lang)
        except:
            print("could not read lexicon of ", lang)
            continue

        if len(lexicon) < 50:
            print("skipping ", lang)
            continue

        w = open(exp_dir / f'{lang}.txt', 'w')

        # at most 10000 to prevent overfitting
        word2phoneme_lst = list(lexicon.word2phoneme.items())

        try:
            epitran_tokenizer = read_raw_epitran_tokenizer(epitran_id, use_lexicon=False)
        except:
            print('failed to read epitran', lang)

        # last 10 for validation
        for grapheme_str, phonemes in word2phoneme_lst[-25:]:
            hyp_phonemes = epitran_tokenizer.tokenize(grapheme_str)
            w.write(f'lang: {lang}\n')
            w.write(f'inp : {grapheme_str}\n')
            w.write(f'ref : {" ".join(phonemes)}\n')
            w.write(f'hyp : {" ".join(hyp_phonemes)}\n')

            cur_cer = editdistance.distance(phonemes, hyp_phonemes)
            cur_fer = phonological_distance(phonemes, hyp_phonemes)
            cur_sum = len(phonemes)
            w.write(f'cer: {cur_cer/cur_sum}\n')
            w.write(f'fer: {cur_fer/cur_sum}\n\n')

            cer += cur_cer
            fer += cur_fer
            csum += cur_sum

        w.close()

        tot_cer += cer
        tot_fer += fer
        tot_csum += csum

        log_w.write(f'| {lang} | {cer / csum:.3f} | {fer / csum:.3f} |\n')

        print('lang   :', lang)
        print("val cer: ", cer / csum)
        print("val fer: ", fer / csum)

    print('all')
    print("val cer: ", tot_cer / tot_csum)
    print("val fer: ", tot_fer / tot_csum)
    log_w.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval epitran g2p')
    parser.add_argument('--lang', type=str, help='language')
    parser.add_argument('--exclude_lang', type=str, help='excluded_language')

    args = parser.parse_args()

    lang = set()
    exclude_lang = set()

    if args.lang is not None:
        lang.udpate(args.lang.split(','))
    if args.exclude_lang is not None:
        exclude_lang.update(args.exclude_lang.split(','))

    eval_epitran(lang, exclude_lang)



