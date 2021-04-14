import torch
import numpy as np
import json
import torch.nn.functional as F
from sklearn import metrics
from tensorboardX import SummaryWriter
from scipy.stats import pearsonr

from embedding import Embedding
from textCNN import TextCNN
from textRNN import TextRNN
import process
import Config


def train():
    config = Config.Config()
    print("device = ", config.device)
    torch.set_default_tensor_type(torch.DoubleTensor)

    word_dict = json.load(open(config.vocab_dict_file, 'r', encoding='utf-8'))
    word2vec_embed = np.load(config.embedding_file)

    model_name = input('choose CNN/RNN\n')
    x_right, y_right = process.get_tensor(config.train_file, config, 
                                          word_dict, 
                                          word2vec_embed, 
                                          model_name, 
                                          'train')
    # x_right为 [all_sentence_num, max_sentence_len] 的二维array，要继续经过Word2vec的embedding映射才能得到词向量
    # y_right为 [all_sentence_num, label_size] 的情感标签array
    

    # TextCNN   ->   train
    if model_name=='CNN':
        model = TextCNN(config, word2vec_embed).to(config.device)
    elif model_name=='RNN':
        model = TextRNN(config, word2vec_embed).to(config.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    batch_sum = len(x_right) // config.batch_size
    if len(x_right) % config.batch_size != 0:
        batch_sum += 1

    batch_num = 0
    for k in range(config.epochs):
        for i in range(batch_sum):
            x = torch.LongTensor(x_right[i * config.batch_size : (i+1) * config.batch_size]).to(config.device)
            if model_name=='CNN':
                y = torch.tensor(y_right[i * config.batch_size : (i+1) * config.batch_size]).to(config.device)
            elif model_name=='RNN':
                y = torch.LongTensor(y_right[i * config.batch_size : (i+1) * config.batch_size]).to(config.device)
        
            y_pred = model.forward(x)
            loss = model.loss_fn(y_pred, y)

            batch_num += 1
            if batch_num % config.print_epochs_num == 0 or batch_num==batch_sum * config.epochs:
                print('Batch [{}/{}]\tLoss = {}'.format(batch_num, batch_sum * config.epochs, loss.item()))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save after each epoch
        if model_name=='CNN':
            torch.save(model, config.CNN_model_file)
        elif model_name=='RNN':
            torch.save(model, config.RNN_model_file)

    if model_name=='CNN':
        torch.save(model, config.CNN_model_file)
    elif model_name=='RNN':
        torch.save(model, config.RNN_model_file)
    


def test():
    config = Config.Config()
    print("device = ", config.device)
    torch.set_default_tensor_type(torch.DoubleTensor)

    word_dict = json.load(open(config.vocab_dict_file, 'r', encoding='utf-8'))
    word2vec_embed = np.load(config.embedding_file)

    model_name = input('choose CNN/RNN\n')
    if model_name=='CNN':
        model = torch.load(config.CNN_model_file)
    elif model_name=='RNN':
        model = torch.load(config.RNN_model_file)

    x_right, y_right, y_original_RNN = process.get_tensor(config.test_file, 
                                                          config, word_dict, 
                                                          word2vec_embed, 
                                                          model_name, 
                                                          'test')
    # x_right为 [all_sentence_num, max_sentence_len] 的二维array，要继续经过Word2vec的embedding映射才能得到词向量
    # y_right为 [all_sentence_num, label_size] 的情感标签array
    
    model.to(config.device)
    model.eval()
    sentence_num = len(x_right)
    batch_sum = sentence_num // config.batch_size
    if sentence_num % config.batch_size != 0:
        batch_sum += 1
        
    y_predict_list = []
    for i in range(batch_sum):
        x = torch.LongTensor(x_right[i * config.batch_size : (i+1) * config.batch_size]).to(config.device)
        y = torch.tensor(y_right[i * config.batch_size : (i+1) * config.batch_size]).to(config.device)
    
        y_pred = model.forward(x)

        # 归一化
        y_pred = y_pred.to(config.cpu).detach().numpy()
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
    
        y_predict_list += y_pred.tolist()
        print('Batch [{}/{}]'.format(i + 1, batch_sum))

    if model_name=='CNN':
        y_right_label = []
        y_predict_label = []
        y_right = y_right.tolist()
        accuracy = 0
        corr_x1, corr_x2 = 0., 0.
        for j in range(sentence_num):
            y_right_label.append(y_right[j].index(max(y_right[j])))
            y_predict_label.append(y_predict_list[j].index(max(y_predict_list[j])))
            if y_right_label[j]==y_predict_label[j]:
                accuracy += 1
            (corr_x1_tmp, corr_x2_tmp) = pearsonr(y_right[j], y_predict_list[j])
            corr_x1 += corr_x1_tmp
            corr_x2 += corr_x2_tmp

    elif model_name=='RNN':
        y_right_label = y_right.tolist()
        y_original_RNN = y_original_RNN.tolist()
        y_predict_label = []
        accuracy = 0
        corr_x1, corr_x2 = 0., 0.
        for j in range(sentence_num):
            y_predict_label.append(y_predict_list[j].index(max(y_predict_list[j])))
            if y_right_label[j]==y_predict_label[j]:
                accuracy += 1
            (corr_x1_tmp, corr_x2_tmp) = pearsonr(y_original_RNN[j], y_predict_list[j])
            corr_x1 += corr_x1_tmp
            corr_x2 += corr_x2_tmp

    accuracy = accuracy / sentence_num
    f1_score = metrics.f1_score(y_right_label, y_predict_label, average='weighted')
    corr = (corr_x1 / sentence_num, corr_x2 / sentence_num)
    print('Accuracy = {}'.format(accuracy))
    print('F1_score = {}'.format(f1_score))
    print('Corr = {}'.format(corr))
