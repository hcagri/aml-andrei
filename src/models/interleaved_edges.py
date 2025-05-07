# import transformer
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
from src.models.mpnn import GnnHelper
from torch_geometric.nn import Linear

class Interleaved_Edges(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, 
                 edge_updates=False,edge_dim=None, final_dropout=0.5, 
                deg=None, config=None,
                no_heads_transformer=4, num_layers_transformer=2, dropout_rate_transformer=0.1,
                ):
        super().__init__()
        self.config = config
        self.n_hidden = n_hidden
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

    
        self.gnn1 = GnnHelper(num_gnn_layers=num_gnn_layers, n_hidden=n_hidden, edge_updates=edge_updates, final_dropout=final_dropout,
                            deg=deg, config=config)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_hidden,
                nhead=no_heads_transformer,
                dim_feedforward=2*n_hidden,
                dropout=dropout_rate_transformer,
                activation='relu'
            ),
            num_layers=num_layers_transformer
        )

        self.gnn2 = GnnHelper(num_gnn_layers=num_gnn_layers, n_hidden=n_hidden, edge_updates=edge_updates, final_dropout=final_dropout,
                            deg=deg, config=config)



        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                            Linear(25, n_classes))

    def forward(self, data):
        # Initial Embedding Layers
        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr) 

        # First GNN Layer
        x, edge_attr = self.gnn1(x, data.edge_index, edge_attr)
        
        # Transformer Layer
        edge_attr = self.transformer(edge_attr)

        # Second GNN Layer
        x, edge_attr = self.gnn2(x, data.edge_index, edge_attr)
        
        
        # Prediction Head
        x = x[data.edge_index.T].reshape(-1, 2*self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = self.mlp(x)
        
        return out
