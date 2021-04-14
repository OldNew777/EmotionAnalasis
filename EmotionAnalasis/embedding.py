import torch
import numpy as np
import json

import Config

class Embedding():
    def __init__(self, config):
        all_words = set()
        # 统计train中出现的所有词语
        with open(config.train_file, 'r', encoding='utf-8') as file_t:
            for line in file_t:
                line = line.split()[10:]
                line[-1] = line[-1].strip()
                for word in line:
                    all_words.add(word)

        # 载入真正需要的词向量
        with open(config.raw_embedding_file, 'r', encoding='utf-8') as file_m:
            info = file_m.readline().split()
            info[1] = info[1][:-1]
            self.words_num = int(info[0])
            self.embed_dim = int(info[1])


            word_dict = {}
            word2vec = [[0 for i in range(config.embed_dim)]]       # for padding
            text = file_m.readlines()
            count = 1
            for i in range(self.words_num):
                text[i] = text[i].split()
                text[i][-1] = text[i][-1][:-1]
                word = text[i][0]
                if word in all_words:
                    word_dict[word] = count
                    count += 1
                    for j in range(1, len(text[i])):
                        text[i][j] = float(text[i][j])
                    word2vec.append(text[i][1:])
            del text

        with open(config.vocab_dict_file, 'w', encoding='utf-8') as file_out:
            json.dump(word_dict, file_out, indent=2)

        np.save(config.embedding_file, np.array(word2vec))
            
def build_embed():
    config = Config.Config()
    embedding = Embedding(config)