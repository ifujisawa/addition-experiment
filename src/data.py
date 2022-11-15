#!/usr/bin/env python
# coding: utf-8

import os
import string
from multiprocessing import Pool

import numpy as np
from sklearn.utils import shuffle

class Tokenizer():
    def __init__(self):
        self.char2id, self.id2char = self.token_dict()
    
    def token_dict(self):
        char2id = {str(i):i for i in range(10)}
        for i in string.ascii_letters:
            char2id[i] = len(char2id)
        for i in string.punctuation:
            char2id[i] = len(char2id)
        char2id['<sos>'] = len(char2id)
        char2id['<eos>'] = len(char2id)
        char2id['<pad>'] = len(char2id)
        id2char = {b:a for a, b in char2id.items()}
        return char2id, id2char
    
    def tokenize(self, xs, pad=True, fixed_len=-1):
        if pad:
            max_len = max(list(map(len, xs)))
            if fixed_len == -1:
                fixed_len = max_len
            elif fixed_len < max_len:
                raise ValueError('fixed_len is smaller than the length of some examples.')

        tokens = []
        for x in xs:
            token = [self.char2id[s] for s in x]
            if pad:
                token += (fixed_len - len(token)) * [self.char2id['<pad>']]
            tokens.append(token)

        if pad:
            tokens = np.array(tokens)
        return tokens
    
    def show(self, x):
        if len(x.shape) == 2:
            for elem in x:
                chars = np.array([self.id2char[e] for e in elem])
                chars = chars[(chars != '<pad>') & (chars != '<sos>') & (chars != '<eos>')]
                print(''.join(chars))
        elif len(x.shape) == 1:
            chars = np.array([self.id2char[e] for e in x])
            chars = chars[(chars != '<pad>') & (chars != '<sos>') & (chars != '<eos>')]
            print(''.join(chars))

    def detokenize(self, x, datatype='x'):
        if datatype == 'x':
            if len(x.shape) == 2:
                rets = []
                for elem in x:
                    chars = np.array([self.id2char[e] for e in elem])
                    chars = chars[(chars != '<pad>') & (chars != '<sos>') & (chars != '<eos>')]
                    rets.append(''.join(chars))
                return rets
            elif len(x.shape) == 1:
                chars = np.array([self.id2char[e] for e in x])
                chars = chars[(chars != '<pad>') & (chars != '<sos>') & (chars != '<eos>')]
                return ''.join(chars)
        elif datatype == 'y':
            if len(x.shape) == 2:
                rets = []
                for elem in x:
                    sos_idx = (elem == self.char2id['<sos>']).argmax()
                    eos_idx = (elem == self.char2id['<eos>']).argmax()
                    rets.append(''.join(map(str, elem[sos_idx+1:eos_idx])))
                return rets
            elif len(x.shape) == 1:
                sos_idx = (x == self.char2id['<sos>']).argmax()
                eos_idx = (x == self.char2id['<eos>']).argmax()
                return ''.join(map(str, x[sos_idx+1:eos_idx]))
            
class DataMaker():
    def __init__(self, tokenizer, max_eqn_letters, max_ans_letters):
        self.tokenizer = tokenizer
        self.max_eqn_letters = max_eqn_letters
        self.max_ans_letters = max_ans_letters
    
    def concat_numbers(self, numbers, arithmetic='add', equal=True):
        '''create an equation as a problem'''
        if arithmetic == 'add':
            ret_str = '+'.join(list(map(str, numbers)))
        elif arithmetic == 'sub':
            ret_str = '-'.join(list(map(str, numbers)))
        elif arithmetic == 'mul':
            ret_str = '*'.join(list(map(str, numbers)))
        else:
            raise ValueError('')

        if equal:
            ret_str += '='
        return list(ret_str)

    def calc_answer(self, numbers, arithmetic='add', sos_eos=True):
        '''calculate an answer from numbers'''
        if arithmetic == 'add':
            ret_str = str(sum(numbers))
        elif arithmetic == 'sub':
            ret_str = str(numbers[0] - sum(numbers[1:]))
        elif arithmetic == 'mul':
            ret_str = str(np.prod(numbers))

        if sos_eos:
            ret_str = ['<sos>'] + list(ret_str) + ['<eos>']
        return list(ret_str)

    def generate_random_integers(self, num_examples,
                                 min_num, max_num,
                                 max_iter_generate=3):
        if num_examples >= (max_num - min_num) ** 2:
            raise ValueError('num_examples is too large.')
            
        rand_nums = np.random.randint(min_num, max_num,
                                      size=num_examples * 10).reshape(-1, 2)

        iter_num = 0
        unique_nums = np.unique(rand_nums, axis=0)
        while (len(unique_nums) < num_examples) and (iter_num < max_iter_generate):
            iter_num += 1
            tmp_rand_nums = np.random.randint(min_num, max_num,
                                              size=num_examples * 10).reshape(-1, 2)
            rand_nums = np.concatenate([rand_nums, tmp_rand_nums])
            unique_nums = np.unique(rand_nums, axis=0)
        if not (iter_num < max_iter_generate):
            print(f'The number of unique generated examples is only {len(arr)}')
        ret_rand_nums = shuffle(unique_nums)[:num_examples]
        return ret_rand_nums
    
    def create_examples(self, num_examples, min_num, max_num,
                        arithmetic='add',
                        tokenize=True):
        rand_nums = self.generate_random_integers(num_examples, min_num, max_num)
        x = [self.concat_numbers(nums, arithmetic=arithmetic) for nums in rand_nums]
        y = [self.calc_answer(nums, arithmetic=arithmetic) for nums in rand_nums]

        if tokenize:
            x = self.tokenizer.tokenize(x, fixed_len=self.max_eqn_letters)
            y = self.tokenizer.tokenize(y, fixed_len=self.max_ans_letters)
        return x, y
