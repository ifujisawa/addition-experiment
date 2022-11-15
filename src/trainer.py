#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import torch
from sklearn.utils import shuffle

def get_current_time():
    '''return current datetime'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def batch(x, batch_size, drop_last=False, enum=None):
    bs = batch_size
    length = len(x)
    if drop_last or length % bs == 0:
        len_batch = int(length / bs)
    else:
        len_batch = int(length / bs) + 1
    
    for idx in range(len_batch):
        if enum is None:
            yield x[idx * bs : (idx + 1) * bs]
        else:
            yield enum + idx, x[idx * bs : (idx + 1) * bs]

class BaseTrainer():
    def __init__(self, model, optimizer, loss_fn,
                 num_epoch, evaluator, metrics,
                 train_batch_size, eval_batch_size, model_dir, save_all, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epoch = num_epoch
        self.metrics = metrics
        self.evaluator = evaluator
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.model_dir = model_dir
        self.save_all = save_all
        self.device = device
    
    def train_batch(self, x_batch, y_batch):
        raise NotImplementedError
    
    def eval_batch(self, x_batch, y_batch, batch_size):
        raise NotImplementedError

    def train_epoch(self, x, y, batch_size):
        x_shuffled, y_shuffled = shuffle(x, y)
        x_train_batch = batch(x_shuffled, batch_size=batch_size)
        y_train_batch = batch(y_shuffled, batch_size=batch_size)
        loss = self.train_batch(x_train_batch, y_train_batch)
        return loss

    def eval_epoch(self, x, y, batch_size):
        x_batch = batch(x, batch_size=batch_size)
        y_batch = batch(y, batch_size=batch_size)
        loss, acc, em = self.eval_batch(x_batch, y_batch, batch_size)
        return loss, acc.item(), em.item()

    def print_metrics(self, epoch, loss, acc, em,
                      val_loss, val_acc, val_em,
                      test_loss, test_acc, test_em):
        print(get_current_time(), f'  Epoch:{epoch:>7}')
        print(f'     loss:{loss:>13.5f}')
        print(f' val_loss:{val_loss:>13.5f}')
        print(f'test_loss:{test_loss:>13.5f}')
        print(f'     acc:{acc:>7.4f}         EM:{em:>7.4f}')
        print(f' val_acc:{val_acc:>7.4f}     val_em:{val_em:>7.4f}')
        print(f'test_acc:{test_acc:>7.4f}    test_em:{test_em:>7.4f}')
        print('=' * 40)
    
    def main_loop(self, x_train, y_train, x_val, y_val, x_test, y_test):
        his = {met:[] for met in self.metrics}
        for epoch in range(self.num_epoch):
            ## training
            loss_ = self.train_epoch(x_train, y_train, self.train_batch_size)
            ## evaluation
            loss, acc, em = self.eval_epoch(x_train, y_train, self.eval_batch_size)
            val_loss, val_acc, val_em = self.eval_epoch(x_val, y_val,
                                                        self.eval_batch_size)
            test_loss, test_acc, test_em = self.eval_epoch(x_test, y_test,
                                                           self.eval_batch_size)
            ## save models
            if epoch == 0:
                best_metrics = {}
                for met in self.metrics:
                    if met == 'epoch':
                        continue
                    best_metrics[met] = eval(met)
                    torch.save(self.model.state_dict(),
                               self.model_dir / ('best_' + met + '_model.pt'))

            for met in best_metrics.keys():
                score = eval(met)
                if met.find('loss') != -1:
                    if score <= best_metrics[met]:
                        torch.save(self.model.state_dict(),
                                   self.model_dir / ('best_' + met + '_model.pt'))
                        best_metrics[met] = min(score, best_metrics[met])
                else:                    
                    if score >= best_metrics[met]:
                        torch.save(self.model.state_dict(),
                                   self.model_dir / ('best_' + met + '_model.pt'))
                        best_metrics[met] = max(score, best_metrics[met])
            
            if self.save_all:
                torch.save(self.model.state_dict(),
                           self.model_dir / 'params' / (str(epoch).zfill(6) + '.pt'))
            
            ## save the history
            for met in self.metrics:
                his[met].append(eval(met))
                
            self.print_metrics(epoch, loss, acc, em,
                               val_loss, val_acc, val_em,
                               test_loss, test_acc, test_em)
            torch.save(his, self.model_dir / 'his.dict')
        return his