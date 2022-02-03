import torch
import torch.nn as nn
from transphone.data.utils import pad_sos_eos
import torch.nn.functional as F
import math
import numpy as np

class Encoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, layer_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, layer_size, batch_first=True)
        #self.linear = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, input_tensor):
        embed = self.embed(input_tensor)
        output, (hidden, _) = self.rnn(embed)
        #output = self.linear(output)

        return output, hidden


class Decoder(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, input_tensor, prev_hidden, prev_cell):
        # input: [B]

        embed = self.embed(input_tensor)
        output, (hidden, cell) = self.rnn(embed, (prev_hidden, prev_cell))
        #print('after rnn', output)
        output = self.linear(output)
        #print('after linear:', output)
        output = self.logsoftmax(output)
        #print('after softmax', output)

        return output, hidden, cell


class AttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(2*hidden_size, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.linearQ = nn.Linear(2*hidden_size, hidden_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.hidden_size = hidden_size

    def forward(self, input_tensor, encoder_vector, encoder_mask, prev_hidden, prev_cell):
        # input: [B, 1]
        # encoder: [B,T,H]
        # encoder mask: (B,T)
        # prev_hidden: (1,B,H)
        # prev_cell: (1,B,H)

        # [B,1,H]
        embed = self.embed(input_tensor).squeeze(1)

        # [B,2H] -> [B,H,1]
        Q = self.linearQ(torch.cat([embed, prev_hidden.squeeze(0)], dim=1)).unsqueeze(-1)
        K = encoder_vector

        # (B,T)
        unnormed_weights = torch.bmm(K, Q).squeeze(2)/math.sqrt(self.hidden_size)

        # masking (B,T)
        masked_weights = unnormed_weights.masked_fill(~encoder_mask, -np.inf)

        # [B,T,H] [B,H,1] -> [B,T,1] -> [B,T]
        attn_weights = F.softmax(masked_weights, dim=1)

        # [B,H,T]x[B,T,1] -> [B,H,1] -> [B,H]
        attn_applied = torch.bmm(encoder_vector.transpose(1,2), attn_weights.unsqueeze(-1)).squeeze(2)

        # (B,2H) -> (B,1,2H)
        lstm_input = torch.cat([attn_applied, embed], dim=1).unsqueeze(1)

        # [B,1,2H]
        output, (hidden, cell) = self.rnn(lstm_input, (prev_hidden, prev_cell))

        #print('after rnn', output)
        output = self.linear(output)
        #print('after linear:', output)
        output = self.logsoftmax(output)
        #print('after softmax', output)

        return output, hidden, cell, attn_weights


class G2P(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_size = 256
        self.vocab_size = 200
        self.encoder = Encoder(self.vocab_size, self.hidden_size, 1)
        self.decoder = Decoder(self.vocab_size, self.hidden_size)

        self.criterion = nn.NLLLoss(ignore_index=0)

    def train_step(self, x, y):

        self.train()
        batch_size = x.shape[0]

        output, prev_hidden = self.encoder(x)
        prev_cell = x.new_zeros(1, batch_size, self.hidden_size, dtype=torch.float)

        ys_in, ys_out = pad_sos_eos(y, 1, 1)
        ys_in = ys_in.transpose(1,0)
        ys_out = ys_out.transpose(1,0)

        loss = 0

        for i in range(len(ys_in)):
            y_in = ys_in[i].view(batch_size, 1)
            y_out = ys_out[i]

            output, prev_hidden, prev_cell = self.decoder(y_in, prev_hidden, prev_cell)

            output = output.squeeze()
            #print('----')
            #print(y_out.shape)
            #print(output.shape)
            #print('y_out', y_out)
            #print('output', output)
            loss += self.criterion(output, y_out)

        return loss

    def inference(self, x):

        self.eval()

        #x = torch.LongTensor([[52, 74, 57, 72, 63,  0,  0,  0,  0,  0,  0,  0]])

        batch_size = 1

        output, prev_hidden = self.encoder(x)
        prev_cell = prev_hidden.new_zeros((1, batch_size, self.hidden_size), dtype=torch.float)

        #print(prev_hidden)

        #y_in = torch.LongTensor([[1]])
        y_out = []

        w = 1

        while(True):
            y_in = x.new([[w]])
            output, prev_hidden, prev_cell = self.decoder(y_in, prev_hidden, prev_cell)
            output = output.squeeze()
            #print(output)
            w = output.data.topk(1)[1].item()
            y_out.append(w)

            if w == 1 or len(y_out)>16:
                break

        return y_out


class AttnG2P(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_size = 512
        self.vocab_size = 200
        self.encoder = Encoder(self.vocab_size, self.hidden_size, 2)
        self.decoder = AttentionDecoder(self.vocab_size, self.hidden_size)

        self.criterion = nn.NLLLoss(ignore_index=0)

    def train_step(self, x, y):

        self.train()
        batch_size = x.shape[0]
        encoder_mask = (x != 0)

        encoder_output, _ = self.encoder(x)
        prev_cell = x.new_zeros(1, batch_size, self.hidden_size, dtype=torch.float)
        prev_hidden = x.new_zeros(1, batch_size, self.hidden_size, dtype=torch.float)

        ys_in, ys_out = pad_sos_eos(y, 1, 1)
        ys_in = ys_in.transpose(1,0)
        ys_out = ys_out.transpose(1,0)

        loss = 0

        for i in range(len(ys_in)):
            y_in = ys_in[i].view(batch_size, 1)
            y_out = ys_out[i]

            output, prev_hidden, prev_cell, _ = self.decoder(y_in, encoder_output, encoder_mask, prev_hidden, prev_cell)

            output = output.squeeze()
            #print('----')
            #print(y_out.shape)
            #print(output.shape)
            #print('y_out', y_out)
            #print('output', output)
            loss += self.criterion(output, y_out)

        return loss

    def inference_with_attention(self, x):

        self.eval()
        encoder_mask = (x != 0)

        #x = torch.LongTensor([[52, 74, 57, 72, 63,  0,  0,  0,  0,  0,  0,  0]])

        batch_size = 1

        encoder_output, _ = self.encoder(x)
        prev_cell = x.new_zeros((1, batch_size, self.hidden_size), dtype=torch.float)
        prev_hidden = x.new_zeros(1, batch_size, self.hidden_size, dtype=torch.float)

        #print(prev_hidden)

        #y_in = torch.LongTensor([[1]])
        y_out = []

        w = 1

        weights = []

        while(True):
            y_in = x.new([[w]])
            output, prev_hidden, prev_cell, attn_weights = self.decoder(y_in, encoder_output, encoder_mask, prev_hidden, prev_cell)
            output = output.squeeze()
            w = output.data.topk(1)[1].item()

            weights.append(attn_weights.unsqueeze(-1))
            if w == 1 or len(y_out)>16:
                break

            y_out.append(w)

        weights = torch.cat(weights, dim=2)

        return y_out, weights



    def inference(self, x):

        self.eval()

        #x = torch.LongTensor([[52, 74, 57, 72, 63,  0,  0,  0,  0,  0,  0,  0]])

        batch_size = 1

        encoder_output, prev_hidden = self.encoder(x)
        prev_cell = prev_hidden.new_zeros((1, batch_size, self.hidden_size), dtype=torch.float)

        #print(prev_hidden)

        #y_in = torch.LongTensor([[1]])
        y_out = []

        w = 1

        while(True):
            y_in = x.new([[w]])
            output, prev_hidden, prev_cell = self.decoder(y_in, encoder_output, prev_hidden, prev_cell)
            output = output.squeeze()
            #print(output)
            w = output.data.topk(1)[1].item()
            y_out.append(w)

            if w == 1 or len(y_out)>16:
                break

        return y_out