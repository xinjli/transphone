from transphone.model.loader import read_loader
from transphone.model.dataset import read_multilingual_dataset
from transphone.model.transformer import TransformerG2P
from transphone.model.utils import read_model_config
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import editdistance
import tqdm
from transphone.model.checkpoint_utils import *
import argparse


def train(exp, checkpoint):

    config = read_model_config(exp)

    model_dir = TransphoneConfig.data_path / "model" / exp
    model_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = read_multilingual_dataset()
    SRC_VOCAB_SIZE = len(train_dataset.grapheme_vocab)+1
    TGT_VOCAB_SIZE = len(train_dataset.phoneme_vocab)+1

    print("src vocab size ", SRC_VOCAB_SIZE)
    print("tgt vocab size ", TGT_VOCAB_SIZE)

    train_dataset.grapheme_vocab.write(model_dir / 'grapheme.vocab')
    train_dataset.phoneme_vocab.write(model_dir / 'phoneme.vocab')

    EMB_SIZE = config.embed_size
    NHEAD = config.num_head
    FFN_HID_DIM = config.hidden_size
    NUM_ENCODER_LAYERS = config.num_encoder
    NUM_DECODER_LAYERS = config.num_decoder
    torch.manual_seed(0)

    model = TransformerG2P(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                            NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).cuda()


    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if checkpoint is not None:
        torch_load(model, checkpoint)

    opt = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    print('training size: ', len(train_dataset))
    print("testing size: ", len(test_dataset))

    train_loader = read_loader(train_dataset, batch_size=64)
    test_loader = read_loader(test_dataset, batch_size=1)

    epoch = 400
    writer = SummaryWriter()

    iteration = 0

    best_cer = 2.0

    for i in range(epoch):

        it = iter(train_loader)

        loss_sum = 0
        it_cnt = len(it)

        for batch in tqdm.tqdm(it):

            x,y = batch
            #print(x)
            #print(y)
            #print(torch.max(x).item(), torch.max(y).item())

            x = x.cuda()
            y = y.cuda()

            opt.zero_grad()
            loss = model.train_step(x, y)
            loss.backward()
            opt.step()

            loss_val = loss.item()
            iteration += 1
            #print(iteration, ' ', loss_val)
            writer.add_scalar('Loss/Train', loss_val, iteration)
            loss_sum += loss

        print("epoch ", i, " loss ", loss_sum/it_cnt)

        test_it = iter(test_loader)

        cer = 0
        csum = 0
        wer = 0
        wsum = 0

        ploted =False

        if i % 5 != 0:
            continue

        for x,y in test_it:
            x = x.cuda()
            y = y.cuda()
            predicted = model.inference(x)
            y = y.squeeze(0).tolist()

            dist = editdistance.eval(predicted, y)
            cer += dist
            csum += len(y)

            if dist != 0:
                wer += 1
            wsum += 1

        writer.add_scalar('CER/Test', cer/csum, i)
        writer.add_scalar('WER/Test', wer/wsum, i)

        print("val cer: ", cer/csum)
        print("val wer: ", wer/wsum)

        cer = cer/csum

        if cer <= best_cer:
            best_cer = cer
            model_path = model_dir / f"model_{best_cer:0.6f}.pt"
            torch_save(model, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train g2p')
    parser.add_argument('--exp', type=str, help='exp name')
    parser.add_argument('--checkpoint', type=str, help='checkpoint')

    args = parser.parse_args()

    exp = args.exp
    checkpoint=args.checkpoint

    train(exp, checkpoint)



