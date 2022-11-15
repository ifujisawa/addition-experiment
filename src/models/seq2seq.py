#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, self.hidden_dim, batch_first=True)
        
    def forward(self, tokens):
        embedding = self.emb(tokens)
        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)
        _, state = self.gru(embedding,
                            torch.zeros(1, len(tokens), self.hidden_dim,
                                        device=DEVICE))
        return state

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, index, state):
        embedding = self.emb(index)
        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)
        gruout, state = self.gru(embedding, state)
        output = self.output(gruout)
        return output, state

    
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, enc_emb_dim, enc_hid_dim,
                 dec_emb_dim, dec_hid_dim, pad_idx, bos_idx, max_ans_letters):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, enc_emb_dim, enc_hid_dim, pad_idx)
        self.decoder = Decoder(tgt_vocab_size, dec_emb_dim, dec_hid_dim, pad_idx)
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.max_ans_letters = max_ans_letters
    
    def forward(self, src, tgt):
        state = self.encoder(src)
        dec_outs = []
        for t_idx in range(self.max_ans_letters - 1):
            dec_out, state = self.decoder(tgt[:, t_idx], state)
            dec_outs.append(dec_out)
        return torch.cat(dec_outs, axis=1)
        
    def predict(self, inputs):
        state = self.encoder(inputs)
        tokens = torch.tensor([self.bos_idx] * len(inputs), device=DEVICE)
        
        pred = torch.unsqueeze(tokens, dim=-1).cpu()
        for _ in range(self.max_ans_letters - 1):
            dec_out, state = self.decoder(tokens, state)
            tokens = torch.argmax(dec_out, axis=-1)
            pred = np.concatenate([pred, tokens.cpu()], axis=-1)
        return pred

from trainer import BaseTrainer
class TrainerSeq2Seq(BaseTrainer):
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
            inputs = torch.tensor(xb, device=DEVICE)
            outputs = torch.tensor(yb, device=DEVICE)
            source, target = outputs[:, :-1], outputs[:, 1:]

            self.optimizer.zero_grad()
            logits = self.model(inputs, source)
            loss = self.loss_fn(logits.reshape((-1, logits.shape[-1])), target.reshape(-1))
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
        return losses / (b_idx + 1)

    def eval_batch(self, x_batch, y_batch, batch_size):
        self.model.eval()
        with torch.no_grad():
            losses = 0
            self.evaluator.eval_init()
            for b_idx, (xb, yb) in enumerate(zip(x_batch, y_batch)):
                inputs = torch.tensor(xb, device=DEVICE)
                outputs = torch.tensor(yb, device=DEVICE)
                source, target = outputs[:, :-1], outputs[:, 1:]

                logits = self.model(inputs, source)
                loss = self.loss_fn(logits.reshape((-1, logits.shape[-1])), target.reshape(-1))
                losses += loss.item()
                self.evaluator.eval_stack(xb, yb)
            acc, em = self.evaluator.eval_ret()
        return losses / (b_idx + 1), acc, em