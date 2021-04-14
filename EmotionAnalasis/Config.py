import torch

class Config():
    def __init__(self):
        self.train_file = './data/sinanews.train'
        self.demo_file = './data/sinanews.demo'
        self.test_file = './data/sinanews.test'
        self.raw_embedding_file = './data/sgns.renmin.word'    # renmin
        #self.raw_embedding_file = './data/sgns.merge.word'      # merge

        self.embedding_file = './data/word2vec_embedding.npy'
        self.word2vec_list_file = './data/word2vec_list.txt'
        self.vocab_dict_file = './data/vocabulary_dictionary.json'
        self.CNN_model_file = './data/CNN_model.pth'
        self.RNN_model_file = './data/RNN_model.pth'

        # 算上全0的padding
        self.vocab_num = 36813     # renmin
        #self.vocab_num = 38995      # merge

        USE_GPU = True
        self.device = torch.device('cuda:0' if (torch.cuda.is_available() and USE_GPU) else 'cpu')
        self.cpu = torch.device('cpu')
        
        self.label_num = 8
        self.embed_dim = 300

        self.learning_rate = 4e-3
        self.batch_size = 64        # CNN 64 // RNN 32
        self.epochs = 20
        self.print_epochs_num = 10

        # CNN
        self.kernel_dim=100
        self.kernel_sizes=(3,4,5)
        self.dropout = 0.2

        # RNN
        self.hidden_size = 64      # LSTM隐藏层
        self.num_layers = 2         # LSTM层数
        self.hidden_dropout = 0
        self.embed_dropout = 0.1
        self.fc_dropout = 0.1