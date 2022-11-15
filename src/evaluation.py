#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch

class Evaluator():
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.eos_idx = tokenizer.char2id['<eos>']
        
    def calc_hit_em(self, y_true, y_pred):
        eos_position_true = (y_true == self.eos_idx).argmax(axis=1)
        eos_position_pred = (y_pred == self.eos_idx).argmax(axis=1)
        len_match_mask = (eos_position_true == eos_position_pred)

        hits, ems = [], []
        for yt, yp, et, ep, lm in zip(y_true, y_pred,
                                      eos_position_true, eos_position_pred,
                                      len_match_mask):
            match_ = (yt[1:et] == yp[1:ep]) if lm else np.array([False])
            hits.append(match_.sum())
            ems.append(match_.all())

        hit = sum(hits)
        num_chars = sum(eos_position_true - 1)
        em = sum(ems)
        num_ex = len(y_true)
        return hit, num_chars, em, num_ex

    def eval_init(self):
        self.hits, self.num_chars = 0, 0
        self.ems, self.num_examples = 0, 0

    def eval_stack(self, xb, yb):
        inputs = torch.tensor(xb, device=self.device)
        pred = self.model.predict(inputs)

        h, nc, e, ne = self.calc_hit_em(yb, pred)
        self.hits += h
        self.num_chars += nc
        self.ems += e
        self.num_examples += ne
    
    def eval_ret(self):
        acc = self.hits / self.num_chars
        em = self.ems / self.num_examples
        return acc, em