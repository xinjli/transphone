from transphone.g2p import read_g2p
from transphone.bin.download_model import download_model
from transphone.model.utils import get_all_models, resolve_model_name
from pathlib import Path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('running transphone g2p model')
    parser.add_argument('-m', '--model', type=str, default='latest', help='specify which model to use. default is to use the latest local model')
    parser.add_argument('-l', '--lang', type=str,  default='ipa',help='specify which language inventory to use for recognition. default is to use all phone inventory')
    parser.add_argument('-i', '--input', type=str, required=True, help='specify your input wav file/directory')
    parser.add_argument('-o', '--output', type=str, default='stdout', help='specify output file. the default will be stdout')
    parser.add_argument('-c', '--combine', type=bool, default=False, help='write outputs by including both grapheme inputs and phonemes in the same line, delimited by space')

    args = parser.parse_args()

    # download specified model automatically if no model exists
    if len(get_all_models()) == 0:
        download_model('latest')

    # resolve model's name
    model_name = resolve_model_name(args.model)
    if model_name == "none":
        print("Model ", model_name, " does not exist. Please download this model or use an existing model in list_model")
        exit(0)

    args.model = model_name

    # create model
    g2p = read_g2p(model_name)

    # output file descriptor
    output_fd = None
    if args.output != 'stdout':
        output_fd = open(args.output, 'w', encoding='utf-8')

    # cache infered words
    word_cache = {}

    # input file/path
    input_path = Path(args.input)

    for line in open(input_path, 'r'):
        words = line.strip().split()
        phoneme_lst = []

        for word in words:
            if word in word_cache:
                phonemes = word_cache[word]
            else:
                phonemes = g2p.inference(word, args.lang)

            phoneme_lst.extend(phonemes)

        if args.combine:
            line_output = ' '.join(words)+' '+' '.join(phoneme_lst)

        else:
            line_output = ' '.join(phoneme_lst)

        if output_fd:
            output_fd.write(line_output+'\n')
        else:
            print(line_output)


    if output_fd:
        output_fd.close()