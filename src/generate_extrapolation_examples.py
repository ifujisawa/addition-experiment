#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd

from data import Tokenizer, DataMaker

out_prefix = Path('../data/test_examples/')
out_prefix.mkdir(exist_ok=True, parents=True)
min_num, max_num = 500, 1500
min_test, max_test = 0, 2500

tk = Tokenizer()
datamaker = DataMaker(tk, 10, 10)
x, y = datamaker.create_examples(60000, min_test, max_test, arithmetic='add')

aa = [x.split('+') for x in tk.detokenize(x)]
a = [[int(a[0]), int(a[1][:-1])] for a in aa]
x_2d = np.array(a)
y_true = np.array(tk.detokenize(y, 'y'))

interp = (min_num <= x_2d[:, 0]) & (x_2d[:, 0] < max_num) & (min_num <= x_2d[:, 1]) & (x_2d[:, 1] < max_num)
x_test = x[~interp][:50000]
x_test_2d = x_2d[~interp][:50000]
y_test = y[~interp][:50000]
y_test_true = y_true[~interp][:50000]

pd.to_pickle(x_test, out_prefix.parent / 'x_test.pkl')
pd.to_pickle(x_test_2d, out_prefix.parent / 'x_test_2d.pkl')
pd.to_pickle(y_test, out_prefix.parent / 'y_test.pkl')
pd.to_pickle(y_test_true.astype(int), out_prefix.parent / 'y_true.pkl')
print('finished to generate test examples in the extrapolation regime.')