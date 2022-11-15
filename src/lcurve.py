#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

def draw_history(his, model_dir, loss=True, acc=True, em=True):
    if loss:
        learning_curve_loss(his, model_dir)
    if acc:
        learning_curve_acc(his, model_dir)
    if em:
        learning_curve_em(his, model_dir)

def learning_curve_loss(his, savepath=None, noshow=True):
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    plt.plot(his['loss'], label='train')
    plt.plot(his['val_loss'], label='val')
    plt.plot(his['test_loss'], label='test')
    plt.grid()
    plt.title('Loss', size=18)
    plt.legend(loc='best', fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('Epoch', size=15)
    if savepath is not None:
        plt.savefig(savepath / 'loss.png')
    if not noshow:
        plt.show()
    plt.close()

def learning_curve_acc(his, savepath=None, noshow=True):
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    plt.plot(his['acc'], label='train')
    plt.plot(his['val_acc'], label='val')
    plt.plot(his['test_acc'], label='test')
    plt.grid()
    plt.title('Accuracy', size=18)
    plt.legend(loc='best', fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('Epoch', size=15)
    if savepath is not None:
        plt.savefig(savepath / 'acc.png')
    if not noshow:
        plt.show()
    plt.close()

def learning_curve_em(his, savepath=None, noshow=True):
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    plt.plot(his['em'], label='train')
    plt.plot(his['val_em'], label='val')
    plt.plot(his['test_em'], label='test')
    plt.grid()
    plt.title('Exact Match', size=18)
    plt.legend(loc='best', fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel('Epoch', size=15)
    if savepath is not None:
        plt.savefig(savepath / 'em.png')
    if not noshow:
        plt.show()
    plt.close()