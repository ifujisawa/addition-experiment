#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from trainer import BaseTrainer


class MLP(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, output_dim,
                 pad_idx, tgt_vocab_size, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hid_dim)
        self.fc2_1 = nn.Linear(hid_dim, hid_dim)
        self.fc2_2 = nn.Linear(hid_dim, hid_dim)
        self.fc2_3 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear (hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.tgt_vocab_size = tgt_vocab_size
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2_1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2_2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2_3(x)
        x = self.dropout(x)
        x = self.relu(x)
        ret = self.fc3(x)
        return ret
    
    def predict(self, inputs):
        inputs = inputs / self.tgt_vocab_size
        logits = self(inputs)
        logits = logits.reshape((-1, self.tgt_vocab_size))
        preds = torch.argmax(logits, axis=-1).reshape((len(inputs), -1))
        preds = preds.cpu().numpy()
        return preds


class TrainerMLP(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, num_epoch, evaluator, metrics, 
                 train_batch_size, eval_batch_size, model_dir, save_all, device, tgt_vocab_size):
        super().__init__(model, optimizer, loss_fn,
                         num_epoch, evaluator, metrics,
                         train_batch_size, eval_batch_size, model_dir, save_all, device)
        self.tgt_vocab_size = tgt_vocab_size

    def train_batch(self, x_batch, y_batch):
        self.model.train()
        losses = 0
        for b_idx, (xb, yb) in enumerate(zip(x_batch, y_batch)):
            inputs = torch.tensor(xb, device=self.device)
            inputs = inputs / self.tgt_vocab_size
            targets = torch.tensor(yb, device=self.device)
            logits = self.model(inputs)
            logits = logits.reshape((-1, self.tgt_vocab_size))

            self.optimizer.zero_grad()
            loss = self.loss_fn(logits, targets.flatten())
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
                inputs = torch.tensor(xb, device=self.device)
                inputs = inputs / self.tgt_vocab_size
                targets = torch.tensor(yb, device=self.device)
                logits = self.model(inputs)
                logits = logits.reshape((-1, self.tgt_vocab_size))

                loss = self.loss_fn(logits, targets.flatten())
                losses += loss.item()
                
                # evaluation with metrics (accuracy, exactmatch)
                self.evaluator.eval_stack(xb, yb)

            acc, em = self.evaluator.eval_ret()
        return losses / (b_idx + 1), acc, em