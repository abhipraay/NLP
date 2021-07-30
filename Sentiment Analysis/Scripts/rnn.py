import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, pretrained_embeddings):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.embedding.load_state_dict({'weight': pretrained_embeddings})
        self.embedding.weight.requires_grad = False

        self.rnn = torch.nn.RNN(embedding_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        
        embedded = self.embedding(text)
        
        output, hidden = self.rnn(embedded)
        output = torch.mean(output, dim = 0)
        out = self.linear(output)
        return out

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, hidden_dim, pretrained_embeddings, dropout = 0.2, bidirectional = False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.dropout = torch.nn.Dropout(p=dropout)

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.embedding.load_state_dict({'weight': pretrained_embeddings})
        self.embedding.weight.requires_grad = False

        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, 
                                         num_layers=num_layers,
                                         bidirectional=bidirectional,
                                         dropout = dropout)
        if bidirectional:
            self.linear = torch.nn.Linear(hidden_dim*2, 2)
        else:
            self.linear = torch.nn.Linear(hidden_dim*num_layers, 2)
    def forward(self, text):
        embedded = self.embedding(text)
        #embedded = torch.transpose(embedded, dim0=1, dim1=0)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        lstm_out = torch.mean(lstm_out, 0)
        out = self.linear(self.dropout(lstm_out))
        return out