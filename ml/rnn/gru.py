import numpy as np

import gensim

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from copy import deepcopy

class GRUGrader(nn.Module):
    def __init__(self, gensim_emb_weights=None, freeze_emb=False, output_size=1, gru_size=64, drop_prob=0.1, 
            gru_num_layer=8, bidirectional=False, decoder_depth=3, decoder_size = [256, 512, 64],
            word_embedding='glove-wiki-gigaword-200'):
        super().__init__()
        if decoder_depth != len(decoder_size):
            raise ValueError("decoder_depth must be equal to len(decoder_size)")

        if gensim_emb_weights is None:
            gensim_model = gensim.downloader.load(word_embedding)
            gensim_emb_weights = gensim_model.vectors

        self.emb_dim = gensim_emb_weights.shape[1]

        wv_emb = np.vstack((gensim_emb_weights, 
                np.random.uniform(-0.25, 0.25, (1, self.emb_dim)),
                np.zeros(shape=(1, self.emb_dim))
            )).copy()
        weights = torch.FloatTensor(wv_emb)
        self.embeddings = nn.Embedding.from_pretrained(weights)
        if freeze_emb:
            self.embeddings.weight.requires_grad = False
        self.gru_size = gru_size
        self.gru_num_layer = gru_num_layer
        self.decoder_depth = decoder_depth
        self.decoder_size = decoder_size
        self.output_size = output_size
        self.gru = nn.GRU(
            input_size = self.emb_dim, 
            hidden_size = gru_size, 
            num_layers = gru_num_layer, 
            bias = True,
            batch_first=True, 
            dropout=drop_prob,
            bidirectional=bidirectional)
        
        self.num_channel_hidden_out = gru_num_layer
        if bidirectional:
            self.num_channel_hidden_out *= 2
        
        self.size_hidden_out = self.num_channel_hidden_out * gru_size

        if decoder_depth == 2:
            decoder_mlp = nn.Sequential(
                nn.Linear(self.size_hidden_out, decoder_size[0]),
                nn.ReLU(),
                nn.Linear(decoder_size[0], decoder_size[1]),
                nn.ReLU(),
                nn.Linear(decoder_size[1], output_size)
            )
        elif decoder_depth == 3:
            decoder_mlp = nn.Sequential(
                nn.Linear(self.size_hidden_out, decoder_size[0]),
                nn.ReLU(),
                nn.Linear(decoder_size[0], decoder_size[1]),
                nn.ReLU(),
                nn.Linear(decoder_size[1], decoder_size[2]),
                nn.ReLU(),
                nn.Linear(decoder_size[2], output_size)
            )
        elif decoder_depth == 4:
            decoder_mlp = nn.Sequential(
                nn.Linear(self.size_hidden_out, decoder_size[0]),
                nn.ReLU(),
                nn.Linear(decoder_size[0], decoder_size[1]),
                nn.ReLU(),
                nn.Linear(decoder_size[1], decoder_size[2]),
                nn.ReLU(),
                nn.Linear(decoder_size[2], decoder_size[3]),
                nn.ReLU(),
                nn.Linear(decoder_size[3], output_size)
            )

        self.task1_mlp = deepcopy(decoder_mlp)
        self.task2_mlp = deepcopy(decoder_mlp)
        self.task3_mlp = deepcopy(decoder_mlp)
        self.task4_mlp = deepcopy(decoder_mlp)
        self.task5_mlp = deepcopy(decoder_mlp)
        self.task6_mlp = deepcopy(decoder_mlp)


        
    
    def forward(self, input_data):
        x = input_data[0]
        s = input_data[1].cpu()
        # print("before embedding", x.shape, s.shape)
        x = self.embeddings(x)
        # print("after embedding", x.shape)
        batch_size = x.shape[0]
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        # print("after pack", x_pack.data.shape)
        _, ht = self.gru(x_pack)
        ht = ht.permute((1,0,2))
        ht = ht.reshape(batch_size, self.size_hidden_out)
        out1 = self.task1_mlp(torch.relu(ht))
        out2 = self.task2_mlp(torch.relu(ht))
        out3 = self.task3_mlp(torch.relu(ht))
        out4 = self.task4_mlp(torch.relu(ht))
        out5 = self.task5_mlp(torch.relu(ht))
        out6 = self.task6_mlp(torch.relu(ht))
        out_array = torch.hstack((out1, out2, out3, out4, out5, out6)).type(torch.float64)
        # print(out_array.shape)
        # print(out_array.dtype)
        return out_array