from transphone.data.loader import read_loader
from transphone.data.dataset import read_dataset
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

if __name__ == '__main__':

    SRC_VOCAB_SIZE = 200
    TGT_VOCAB_SIZE = 200
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    torch.manual_seed(0)

    model = TransformerG2P(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                            NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).cuda()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    opt = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    dataset = read_dataset('eng')
    training_cnt = int(len(dataset)*0.95)
    test_cnt = len(dataset) - training_cnt

    train_dataset, test_dataset = random_split(dataset, [training_cnt, test_cnt], generator=torch.Generator().manual_seed(42))
    loader = read_loader(train_dataset, 64)
    test_loader = read_loader(test_dataset, batch_size=1)

    epoch = 100
    writer = SummaryWriter()

    iteration = 0

    #it = iter(loader)
    #batch = next(it)

    #batch = (torch.LongTensor([[2], [3]]), torch.LongTensor([[2], [3]]))

    for i in range(epoch):

        it = iter(loader)

        loss_sum = 0
        it_cnt = len(it)

        for batch in tqdm.tqdm(it):

            x,y = batch
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
            print(y, predicted)

            if csum >= 1000:
                break

        writer.add_scalar('CER/Test', cer/csum, i)
        writer.add_scalar('WER/Test', wer/wsum, i)

        print("val cer: ", cer/csum)
        print("val wer: ", wer/wsum)