import numpy as np
import json

import Config

def get_tensor(filename, config : Config.Config, word_dict, word2vec, model_type='CNN', order='train'):
    file = open(filename, "r", encoding='utf-8')    # open *.train
    x0 = []
    y0 = []
    max_sentence_len = -1
    for line in file.readlines():
        line = line[:-1].split()
        x0.append(line[10:])
        max_sentence_len = max(max_sentence_len, len(x0[-1]))
        label = []
        zero_label = [0. for i in range(config.label_num)]
        emotionParameters_sum = int(line[1][6:])
        if emotionParameters_sum == 0:
            label = zero_label
        else:
            for i in range(2, 2+config.label_num):
                label.append(float(int(line[i][3:]) / emotionParameters_sum))
        # y0已经归一化
        y0.append(label)

    x = np.zeros(shape=[len(x0), max_sentence_len], dtype=int)
    for i in range(len(x0)):
        for j in range(len(x0[i])):
            if x0[i][j] in word_dict:
                x[i][j] = word_dict[x0[i][j]]

    if model_type=='RNN':
        y1 = []
        for i in range(len(y0)):
            y1.append(y0[i].index(max(y0[i])))
        y = np.array(y1)
        if order=='test':
            return x, y, np.array(y0)
        elif order=='train':
            return x, y
    elif model_type=='CNN':
        y = np.array(y0)
        if order=='test':
            return x, y, 0
        elif order=='train':
            return x, y