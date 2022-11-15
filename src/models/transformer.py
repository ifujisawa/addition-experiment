#!/usr/bin/env python
# coding: utf-8


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1,
                                                                          float(0.0))
    return mask

def create_mask(src, tgt, pad_idx):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


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

    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 bos_idx,
                 max_ans_letters,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
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
        
        self.bos_idx = bos_idx
        self.max_ans_letters = max_ans_letters

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
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)
        batch_size = src.shape[1]

        memory = self.encode(src, src_mask)
        ys = torch.ones(1, batch_size).fill_(start_symbol).type(torch.long).to(DEVICE)
        for i in range(max_len - 1):
            memory = memory.to(DEVICE)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool).to(DEVICE))
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = torch.unsqueeze(next_word, 0)
            ys = torch.cat([ys, next_word], dim=0)

        return ys

    def predict(self, src):
        self.eval()
        with torch.no_grad():
            src = src.T
            num_tokens, batch_size = src.shape
            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
            tgt_tokens = self.greedy_decode(src, src_mask,
                                            max_len=self.max_ans_letters,
                                            start_symbol=self.bos_idx)
            tgt_tokens = tgt_tokens.flatten()
            preds = tgt_tokens.cpu().numpy().reshape((batch_size, -1), order='F')
        return preds


from trainer import BaseTrainer
class TrainerTransformer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, num_epoch, evaluator, metrics, 
                 train_batch_size, eval_batch_size, model_dir, save_all, pad_idx):
        super().__init__(model, optimizer, loss_fn,
                         num_epoch, evaluator, metrics,
                         train_batch_size, eval_batch_size, model_dir, save_all, DEVICE)
        self.pad_idx = pad_idx

    def train_batch(self, x_batch, y_batch):
        self.model.train()
        losses = 0
        for b_idx, (xb, yb) in enumerate(zip(x_batch, y_batch)):
            src = torch.tensor(xb.T, device=DEVICE)
            targets = torch.tensor(yb.T, device=DEVICE)
            tgt_input, tgt_out = targets[:-1], targets[1:]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.pad_idx)
            logits = self.model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            self.optimizer.zero_grad()
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1, norm_type=1.0)
            self.optimizer.step()
            losses += loss.item()
        return losses / (b_idx + 1)
    
    def eval_batch(self, x_batch, y_batch, batch_size):
        self.model.eval()
        with torch.no_grad():
            losses = 0
            self.evaluator.eval_init()
            for b_idx, (xb, yb) in enumerate(zip(x_batch, y_batch)):
                src = torch.tensor(xb.T, device=DEVICE)
                targets = torch.tensor(yb.T, device=DEVICE)
                tgt_input, tgt_out = targets[:-1], targets[1:]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.pad_idx)
                logits = self.model(src, tgt_input, src_mask, tgt_mask,
                                    src_padding_mask, tgt_padding_mask, src_padding_mask)

                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                losses += loss.item()
                
                # evaluation with metrics (accuracy, exactmatch)
                self.evaluator.eval_stack(xb, yb)

            acc, em = self.evaluator.eval_ret()
        return losses / (b_idx + 1), acc, em