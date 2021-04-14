import torch.nn as nn
import torch.nn.functional as F
import torch

import Config

class TextCNN(nn.Module):
    def __init__(self, config : Config.Config, word2vec_embed):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(config.vocab_num, 
                                  config.embed_dim, 
                                  padding_idx=0)
        self.embed.weight.data.copy_(torch.from_numpy(word2vec_embed))
        self.embed.requires_grad_(False)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, 
                                              out_channels = config.kernel_dim, 
                                              kernel_size = (K, config.embed_dim)) for K in config.kernel_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(len(config.kernel_sizes) * config.kernel_dim, 
                            out_features=config.label_num)

        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, input, is_training=True):
        # originally, inputs = (batch_size, sentence_len, embed_dim) word2vec
        input = self.embed(input)     # inputs = (batch_size, sentence_len, embed_dim) word2vec
        input = input.unsqueeze(1)    # get (batch_size, 1, sentence_len, embed_dim)

        input = [F.relu(conv(input)).squeeze(3) for conv in self.convs]

        input = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in input]
        concated = torch.cat(input, 1)
        concated = concated.view(-1, concated.size(1))
        
        if is_training:
            concated = self.dropout(concated)
        out = self.fc(concated) 

        return F.softmax(out,1)