import torch.nn as nn
import torch.nn.functional as F
import torch

import Config

class TextRNN(nn.Module):
    def __init__(self, config : Config.Config, word2vec_embed):
        super(TextRNN, self).__init__()
        self.embed = nn.Embedding(config.vocab_num, 
                                  config.embed_dim, 
                                  padding_idx=0)
        self.embed.weight.data.copy_(torch.from_numpy(word2vec_embed))
        self.embed.requires_grad_(False)
        
        self.lstm = nn.LSTM(config.embed_dim, 
                            config.hidden_size, 
                            config.num_layers,
                            bidirectional=True, 
                            batch_first=True, 
                            dropout=config.hidden_dropout)
        # self.fc_dropout = nn.Dropout(config.fc_dropout)
        self.linear = nn.Linear(config.hidden_size * 2, 
                                config.label_num)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input, is_training=True):
        # originally, inputs = (batch_size, sentence_len, embed_dim) word2vec
        input = self.embed(input)     # inputs = (batch_size, sentence_len, embed_dim) word2vec

        out, _ = self.lstm(input)
        out = self.linear(out[:, -1, :])

        return out