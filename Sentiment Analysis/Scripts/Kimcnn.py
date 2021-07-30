import torch
import torch.nn as nn
import torch.nn.functional as F

class KimCNN(nn.Module):
    def __init__(self, input_dim, embed_dim, n_filters, filter_sizes, output_dim, pretrained_embeddings):
        super().__init__()
        
        self.embedding_dim = embed_dim
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.embedding.load_state_dict({'weight': pretrained_embeddings})
        self.embedding.weight.requires_grad = False
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, self.embedding_dim)) 
                                    for fs in filter_sizes
                                    ])

        self.fc = nn.Linear(n_filters*len(filter_sizes), output_dim)
        
    
    def forward(self,x):
        x = x.permute(1,0)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        convs_x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in convs_x]
        cat_x = torch.cat(pooled_x, dim = 1)
        x = self.fc(cat_x)
        return x