#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path

import numpy as np
import yaml
import torch
from sklearn.model_selection import train_test_split

from data import Tokenizer, DataMaker
from evaluation import Evaluator
from lcurve import draw_history

def train_val_test_split(x, y, train_size, val_size):
    train_val_size = train_size + val_size
    x_trainval_, x_test, y_trainval_, y_test = train_test_split(x, y,
                                                                train_size=train_val_size)
    x_train, x_val, y_train, y_val = train_test_split(x_trainval_, y_trainval_,
                                                      train_size=train_size / train_val_size)
    return x_train, x_val, x_test, y_train, y_val, y_test    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    ## expand the configs
    model_type = cfg['train_params']['model_type']
    experiment_name = cfg['train_params']['experiment_name']
    num_epoch = cfg['train_params']['num_epoch']
    batch_size = cfg['train_params']['batch_size']
    eval_batch_size = cfg['train_params']['eval_batch_size']
    train_size = cfg['train_params']['train_size']
    val_size = cfg['train_params']['val_size']
    MAX_EQN_LETTERS = cfg['data_params']['max_eqn_letters']
    MAX_ANS_LETTERS = cfg['data_params']['max_ans_letters']
    ARITHMETIC = cfg['data_params']['arithmetic']
    MIN_NUM = cfg['data_params']['min_num']
    MAX_NUM = cfg['data_params']['max_num']
    NUM_EXAMPLES = cfg['data_params']['num_examples']
    save_all = cfg['others']['save_all']

    model_dir = Path(f'../model/{experiment_name}')
    model_dir.mkdir(parents=True, exist_ok=True)

    param_dir = Path(f'../model/{experiment_name}/params')
    param_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(f'../model/{experiment_name}/data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = ['epoch', 'loss', 'val_loss', 'acc', 'em',
               'val_acc', 'val_em', 'test_loss', 'test_acc', 'test_em']

    ## define tokenizer
    tokenizer = Tokenizer()
    PAD_IDX = tokenizer.char2id['<pad>']
    BOS_IDX = tokenizer.char2id['<sos>']
    EOS_IDX = tokenizer.char2id['<eos>']
    SRC_VOCAB_SIZE = len(tokenizer.char2id)
    TGT_VOCAB_SIZE = len(tokenizer.char2id)

    ## create datasets
    datamaker = DataMaker(tokenizer, MAX_EQN_LETTERS, MAX_ANS_LETTERS)
    x, y = datamaker.create_examples(NUM_EXAMPLES, MIN_NUM, MAX_NUM,
                                     arithmetic=ARITHMETIC)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y, train_size, val_size)
    torch.save(x_train, data_dir / 'x_train.pkl')
    torch.save(x_val, data_dir / 'x_val.pkl')
    torch.save(x_test, data_dir / 'x_test.pkl')
    torch.save(y_train, data_dir / 'y_train.pkl')
    torch.save(y_val, data_dir / 'y_val.pkl')
    torch.save(y_test, data_dir / 'y_test.pkl')
        
    if model_type == 'seq2seq':
        from models.seq2seq import Seq2Seq
        from models.seq2seq import TrainerSeq2Seq as Trainer

        ENCODER_EMBEDDING_DIM = 512
        ENCODER_HIDDEN_DIM = 512
        DECODER_EMBEDDING_DIM = 512
        DECODER_HIDDEN_DIM = 512

        model = Seq2Seq(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                        ENCODER_EMBEDDING_DIM, ENCODER_HIDDEN_DIM,
                        DECODER_EMBEDDING_DIM, DECODER_HIDDEN_DIM,
                        PAD_IDX, BOS_IDX, MAX_ANS_LETTERS).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        evaluator = Evaluator(tokenizer, model, device)
        trainer = Trainer(model, optimizer, loss_fn, num_epoch, evaluator, metrics,
                          batch_size, eval_batch_size, model_dir, save_all, PAD_IDX)
        his = trainer.main_loop(x_train, y_train, x_val, y_val, x_test, y_test)
        torch.save(his, model_dir / 'his.dict')
        draw_history(his, model_dir)

    elif model_type == 'mlp':
        from models.mlp import MLP
        from models.mlp import TrainerMLP as Trainer

        HIDDEN_DIM = 512
        model = MLP(len(tokenizer.char2id), MAX_EQN_LETTERS, 
                   HIDDEN_DIM, MAX_ANS_LETTERS * TGT_VOCAB_SIZE, PAD_IDX, TGT_VOCAB_SIZE).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        evaluator = Evaluator(tokenizer, model, device)
        trainer = Trainer(model, optimizer, loss_fn, num_epoch, evaluator, metrics,
                          batch_size, eval_batch_size, model_dir, save_all, device, TGT_VOCAB_SIZE)

        his = trainer.main_loop(x_train, y_train, x_val, y_val, x_test, y_test)
        torch.save(his, model_dir / 'his.dict')
        draw_history(his, model_dir)

    elif model_type == 'transformer':
        from models.transformer import Seq2SeqTransformer
        from models.transformer import TrainerTransformer as Trainer

        NHEAD = 8
        EMB_SIZE = 256
        FFN_HID_DIM = 256
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3

        model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                   NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                   BOS_IDX, MAX_ANS_LETTERS, FFN_HID_DIM).to(device)
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
   
        evaluator = Evaluator(tokenizer, model, device)
        trainer = Trainer(model, optimizer, loss_fn, num_epoch, evaluator, metrics,
                          batch_size, eval_batch_size, model_dir, save_all, PAD_IDX)
        
        his = trainer.main_loop(x_train, y_train, x_val, y_val, x_test, y_test)
        draw_history(his, model_dir)

