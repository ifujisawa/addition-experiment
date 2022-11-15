#!/usr/bin/env python
# coding: utf-8

import re
from pathlib import Path
import random
import numpy as np
import pandas as pd

from sklearn.utils import shuffle

def left_numchar(char, text):
    try:
        return re.match(f'{char}+', text).span()[-1]
    except:
        return 0

seed = 1234
random.seed(seed)
np.random.seed(seed)

LOWEST_DIGIT = '0'
max_digit = 100
num_samples = 200000
savedir = Path('../data/gptneox_prompt')
savedir.mkdir(exist_ok=True, parents=True)

lengths = np.random.randint(1, max_digit, num_samples)
long_str = ''.join(list(map(str, np.random.randint(0, 10, max_digit * num_samples))))
long_str = long_str[left_numchar('0', long_str):]
rands = []
for leng in lengths:
    rands.append(long_str[:leng])
    long_str = long_str[leng:]
    long_str = long_str[left_numchar(LOWEST_DIGIT, long_str):]
assert sum(lengths) == len(''.join(rands))

rands = shuffle(rands, random_state=1234567)
questions = ['What is ' + ' + '.join([a, b]) + '?' for a, b in zip(rands[::2], rands[1::2])]
len_ques = list(map(len, questions))
sorted_questions = np.array(questions)[np.argsort(len_ques)]

for cnt in range(20):
    df = pd.DataFrame(sorted_questions[5000*cnt:5000*(cnt+1)])
    df.to_csv(savedir / f'prompt_{cnt:>02d}.csv', index=False, header=None)