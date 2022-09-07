from transphone.data.loader import read_loader
from transphone.data.dataset import read_dataset, read_p2g_dataset
import torch.optim as optim
from transphone.model.transformer import TransformerG2P
import torch.nn as nn
from torch.utils.data import random_split
import torch
from transphone.data.utils import pad_sos_eos
from torch.utils.tensorboard import SummaryWriter
import editdistance
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from allospeech.am.utils.checkpoint_utils import *

if __name__ == '__main__':

    train_dataset, test_dataset = read_p2g_dataset()
    SRC_VOCAB_SIZE = len(train_dataset.phoneme_vocab)+1
    TGT_VOCAB_SIZE = len(train_dataset.grapheme_vocab)+1

    print(SRC_VOCAB_SIZE)
    print(TGT_VOCAB_SIZE)

    train_dataset.grapheme_vocab.write('/home/xinjianl/Git/allospeech/data/wilderness/bfa/exp/grapheme.vocab')
    train_dataset.phoneme_vocab.write('/home/xinjianl/Git/allospeech/data/wilderness/bfa/exp/phoneme.vocab')

    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 32
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    torch.manual_seed(0)

    model = TransformerG2P(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                            NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).cuda()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    opt = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    print('training size: ', len(train_dataset))
    print("testing size: ", len(test_dataset))

    train_loader = read_loader(train_dataset, batch_size=BATCH_SIZE)
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

        print("epoch ", epoch, " loss ", loss_sum/it_cnt)

        test_it = iter(test_loader)

        cer = 0
        csum = 0
        wer = 0
        wsum = 0

        ploted =False

        if i % 10 != 0:
            continue

        for x,y in test_it:
            x = x.cuda()
            y = y.cuda()
            predicted = model.inference(x)
            y = y.squeeze().tolist()
            dist = editdistance.eval(predicted, y)
            cer += dist
            csum += len(y)

            if dist != 0:
                wer += 1
            wsum += 1
            #print(y, predicted)

            #if csum >= 1000:
            #    break

        writer.add_scalar('CER/Test', cer/csum, i)
        writer.add_scalar('WER/Test', wer/wsum, i)

        print("val cer: ", cer/csum)
        print("val wer: ", wer/wsum)

        cer = cer/csum

        if cer <= best_cer:
            best_cer = cer
            model_path = "/home/xinjianl/Git/allospeech/data/wilderness/bfa/exp/model_{:0.6f}.pt".format(best_cer)
            torch_save(model, model_path)