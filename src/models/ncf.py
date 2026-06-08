import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class NetflixDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, layers=[64, 32, 16], dropout=0.2):
        super(NCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        
        # NCF MLP architecture
        self.fc_layers = nn.ModuleList()
        
        # First layer input is concatenated user and item embeddings
        input_size = embedding_dim * 2
        
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_size, layer_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            input_size = layer_size
            
        # Final prediction layer
        self.output_layer = nn.Linear(layers[-1], 1)
        
    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        for layer in self.fc_layers:
            x = layer(x)
            
        prediction = self.output_layer(x)
        return prediction.squeeze()
