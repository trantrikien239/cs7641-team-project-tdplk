import numpy as np

import gensim

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from copy import deepcopy

# def load_embedding(embedding_pth="/home/jovyan/code/onemount/semantics-layer/src/consumer_ml/demographic_inference/sequence/data/vcm_embeddings/products_v5.emb"):    
#     wv_emb = KeyedVectors.load(embedding_pth)
#     emb_size = wv_emb.vectors.shape[1]
#     emb_vector_mod = np.vstack([np.array([[0]*emb_size]), wv_emb.vectors])
#     weights = torch.FloatTensor(emb_vector_mod)
#     return wv_emb, weights

# wv_emb, weights = load_embedding()

class GRUEncoder(nn.Module):
    def __init__(self, weights=None, output_size=64, gru_size=128, drop_prob=0.0, 
            gru_num_layer=8, bidirectional=False, word_embedding='glove-wiki-gigaword-200'):
        super().__init__()
        if weights is None:
            gensim_model = gensim.downloader.load(word_embedding)

        self.k2i = gensim_model.key_to_index
        self.i2k = gensim_model.index_to_key
        wv_emb = np.vstack(gensim_model.vectors, np.array([[0]*gensim_model.vectors.shape[1]]))
        weights = torch.FloatTensor(wv_emb)
        self.embeddings = nn.Embedding.from_pretrained(weights)
        self.gru_size = gru_size
        self.gru_num_layer = gru_num_layer
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
        self.fc1 = nn.Linear(self.size_hidden_out, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        
    
    def forward(self, input_data):
        x = input_data[0]
        s = input_data[1]
        x = self.embeddings(x)
        batch_size = x.shape[0]
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        _, ht = self.gru(x_pack)
        ht = ht.permute((1,0,2))
        ht = ht.reshape(batch_size, self.size_hidden_out)
        out = self.fc1(torch.relu(ht))
        out = self.fc2(torch.relu(out))
        return out
    
class LSTMEncoder(nn.Module):
    def __init__(self, weights=None, output_size=128, gru_size=128, drop_prob=0.0,
            gru_num_layer=4, bidirectional=False) -> None:
        super().__init__()

class GRUGrader(nn.Module):
    def __init__(self, gensim_emb_weights=None, output_size=1, gru_size=64, drop_prob=0.1, 
            gru_num_layer=8, bidirectional=False, decoder_depth=3, decoder_size = [128, 512, 64],
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
            ))
        weights = torch.FloatTensor(wv_emb)
        self.embeddings = nn.Embedding.from_pretrained(weights)
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

        self.task1_mlp = deepcopy(decoder_mlp)
        self.task2_mlp = deepcopy(decoder_mlp)
        self.task3_mlp = deepcopy(decoder_mlp)
        self.task4_mlp = deepcopy(decoder_mlp)
        self.task5_mlp = deepcopy(decoder_mlp)
        self.task6_mlp = deepcopy(decoder_mlp)


        
    
    def forward(self, input_data):
        x = input_data[0]
        s = input_data[1]
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