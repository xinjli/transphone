from transphone.bin.download_model import download_model
from transphone.model.utils import resolve_model_name
from transphone.tokenizer import read_tokenizer
from pathlib import Path
import argparse
import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser('running transphone g2p model')
    parser.add_argument('-m', '--model', type=str, default='latest',
                        help='specify which model to use. default is to use the latest local model')
    parser.add_argument('-l', '--lang', type=str, default='eng',
                        help='specify which language inventory to use for recognition. default is to use all phone inventory')
    parser.add_argument('-i', '--input', type=str, required=True, help='specify your input wav file/directory')
    parser.add_argument('-o', '--output', type=str, default='stdout',
                        help='specify output file. the default will be stdout')
    parser.add_argument('-d', '--device', help='specify device to use, if not specified, it will try using gpu when applicable')
    parser.add_argument('-f', '--format', type=str, default='text', help='kaldi or text')
    parser.add_argument('-c', '--combine', type=bool, default=False,
                        help='write outputs by including both grapheme inputs and phonemes in the same line, delimited by space')

    args = parser.parse_args()

    # resolve model's name
    model_name = resolve_model_name(args.model)

    # format
    file_format = args.format

    if args.combine:
        assert file_format == 'text'

    # download specified model automatically if no model exists
    download_model(model_name)

    device = None
    if args.device is not None and isinstance(args.device, str) and str.isdigit(args.device):
        device = int(args.device)

    # create model
    tokenizer = read_tokenizer(args.lang, g2p_model=model_name, device=device)

    # output file descriptor
    output_fd = None
    if args.output != 'stdout':
        output_fd = open(args.output, 'w', encoding='utf-8')

    # input file/path
    input_path = Path(args.input)

    for line in tqdm.tqdm(open(input_path, 'r').readlines(), disable=args.output=='stdout'):
        fields = line.strip().split()

        utt_id = None

        if file_format == 'text':
            text = ' '.join(fields)
        else:
            text = ' '.join(fields[1:])
            utt_id = fields[0]

        phonemes = tokenizer.tokenize(text)
        line_output = ' '.join(phonemes)

        if args.combine:
            line_output = text + '\t' + line_output

        if utt_id is not None:
            line_output = utt_id + ' ' + line_output

        if output_fd:
            output_fd.write(line_output + '\n')
        else:
            print(line_output)

    if output_fd:
        output_fd.close()