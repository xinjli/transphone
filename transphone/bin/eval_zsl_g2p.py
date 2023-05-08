from transphone.model.dataset import read_zsl_dataset
from transphone.model.utils import read_model_config
from transphone.g2p import read_g2p
import editdistance
import tqdm
from transphone.model.checkpoint_utils import *
from phonepiece.distance import phonological_distance
import argparse


def get_decode_dir(exp_name, checkpoint, ensemble):

    decode_dir = TransphoneConfig.data_path / 'decode' / 'zsl'

    perf = str(checkpoint).split('/')[-1].split('.')[1]

    # create test directory
    exp_dir = decode_dir / f"{exp_name}-{perf}-{ensemble}"
    return exp_dir


def eval_zsl_test(exp, checkpoint, ensemble, device):

    exp_dir = get_decode_dir(exp, checkpoint, ensemble)
    exp_dir.mkdir(exist_ok=True, parents=True)

    test_grapheme_lst, test_phoneme_lst, test_lang_lst = read_zsl_dataset(exp)
    print("test size: ", len(test_lang_lst))

    model = read_g2p(exp, device, checkpoint=checkpoint)

    tot_err = 0
    tot_dst = 0
    tot_csum = 0

    exp_dir = get_decode_dir(exp, checkpoint, ensemble)
    exp_dir.mkdir(exist_ok=True, parents=True)
    log_w = open(exp_dir / 'result.md', 'w')
    log_w.write('| language | phoneme error rate | phonological distance |\n')
    log_w.write('|----------|--------------------|-----------------------|\n')

    for grapheme_lst, phonemes_lst, lang_id in tqdm.tqdm(zip(test_grapheme_lst, test_phoneme_lst, test_lang_lst)):

        err = 0
        dst = 0
        csum = 0

        w = open(exp_dir / f'{lang_id}.txt', 'w')

        for grapheme, phonemes in zip(grapheme_lst, phonemes_lst):

            predicted = model.inference_word_batch(grapheme, lang_id=lang_id, num_lang=ensemble, force_approximate=True)

            w.write(f'lang: {lang_id}\n')
            w.write(f'inp : {grapheme}\n')
            w.write(f'ref : {" ".join(phonemes)}\n')
            w.write(f'hyp : {" ".join(predicted)}\n')
            cur_err = editdistance.distance(phonemes, predicted)
            cur_dst = phonological_distance(phonemes, predicted)
            cur_sum = len(phonemes)
            w.write(f'err: {cur_err / cur_sum}\n')
            w.write(f'dst: {cur_dst / cur_sum}\n\n')
            err += cur_err
            dst += cur_dst
            csum += cur_sum

        w.close()

        tot_err += err
        tot_dst += dst
        tot_csum += csum

        if csum > 0:
            log_w.write(f'| {lang_id} | {err / csum:.3f} | {dst / csum:.3f} |\n')
            print('lang   :', lang_id)
            print("val cer: ", err / csum)
            print("val dst: ",  dst / csum)

    print('all')
    print("val err: ", tot_err / tot_csum)
    print("val dst: ", tot_dst / tot_csum)
    log_w.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='eval zero-shot learning g2p')
    parser.add_argument('--exp', type=str, help='exp')
    parser.add_argument('--checkpoint', type=str, help='checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ensemble', type=int, default=10)

    args = parser.parse_args()

    checkpoint = args.checkpoint
    exp = args.exp
    device = args.device
    ensemble = args.ensemble


    if exp is None:
        exp = Path(checkpoint).parent.stem

    if checkpoint is None:
        model_path = TransphoneConfig.data_path / 'model' / exp
        if (model_path / "model.pt").exists():
            checkpoint = model_path / "model.pt"
        else:
            target_model = find_topk_models(model_path)[0]
            print("using model ", target_model)
            checkpoint = target_model

    eval_zsl_test(exp, checkpoint, ensemble, device)



