from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from transphone.data.utils import pad_sos_eos
from transphone.config import TransphoneConfig

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 0, 1, 1

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=TransphoneConfig.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=TransphoneConfig.device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class TransformerG2P(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(TransformerG2P, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory,tgt_mask)


    def train_step(self, x,y):

        self.train()
        batch_size = x.shape[0]

        ys_in, ys_out = pad_sos_eos(y, 1, 1)

        # T,B
        tgt_input = ys_in.transpose(1,0)
        tgt_out = ys_out.transpose(1,0)

        src = x.transpose(1,0)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = self.forward(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        return loss


    def inference(self, x):

        self.eval()
        src = x.view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(TransphoneConfig.device)
        max_len=num_tokens + 5

        memory = self.encode(src, src_mask)
        ys = x.new_ones(1, 1).fill_(BOS_IDX).type(torch.long)

        for i in range(max_len-1):
            memory = memory.to(TransphoneConfig.device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(TransphoneConfig.device)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break

        out = ys.squeeze(1).tolist()[1:]

        if out[-1] == 1:
            out = out[:-1]

        return out

    def inference_batch(self, x):

        self.eval()

        # (T,B)
        src = x.transpose(0, 1)

        num_tokens = src.shape[0]
        batch_size = src.shape[1]

        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(TransphoneConfig.device)
        max_len = num_tokens + 5

        memory = self.encode(src, src_mask)
        ys = x.new_ones(1, batch_size).fill_(BOS_IDX).type(torch.long)

        is_done = [False] * batch_size

        for i in range(max_len-1):
            memory = memory.to(TransphoneConfig.device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(TransphoneConfig.device)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])

            _, next_words = torch.max(prob, dim=1)

            for j in range(batch_size):
                if next_words[j].item() == EOS_IDX:
                    is_done[j] = True

                if is_done[j]:
                    next_words[j] = EOS_IDX

            ys = torch.cat([ys, next_words.unsqueeze(0)], dim=0)

            if all(is_done):
                break

        outs = []
        for y in ys.transpose(0,1).tolist():
            outs.append([i for i in y if i >= 2])

        return outs