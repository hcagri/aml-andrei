# import transformer
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
from src.models.mpnn import GnnHelper
from torch_geometric.nn import Linear
from src.util import unpack_dict_ns
from src.models.fusion_l import Fusion_Layer
class Full_Fusion(torch.nn.Module):
    def __init__(self, num_features, n_classes=2, n_hidden=100, 
                 edge_dim=None, final_dropout=0.5, 
                deg=None, config=None,
                ):
        super().__init__()
        #print(config)
        self.config = config
        self.n_hidden = n_hidden
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        gnn1_config = unpack_dict_ns(config, 0)
        transformer1_config = unpack_dict_ns(config, 1)
        fusion1_config = unpack_dict_ns(config, 2)

        gnn2_config = unpack_dict_ns(config, 3)
        transformer2_config = unpack_dict_ns(config, 4)
        fusion2_config = unpack_dict_ns(config, 5)

        #print(fcpy)   
        self.gnn1 = GnnHelper(num_gnn_layers=gnn1_config.n_gnn_layers, n_hidden=gnn1_config.n_hidden, edge_updates=config.emlps, final_dropout=gnn1_config.final_dropout,
                            deg=deg, config=gnn1_config)
        
        print(transformer1_config)
        #print(scpy)
        self.transformer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer1_config.n_hidden,
                nhead=transformer1_config.no_heads,
                dim_feedforward=2*transformer1_config.n_hidden,
                dropout=transformer1_config.dropout,
                activation=transformer1_config.activation,
                batch_first=True
            ),
            num_layers=transformer1_config.n_layers,
        )

        self.fusion1 = Fusion_Layer(input_dim_a=gnn1_config.n_hidden, input_dim_b=transformer1_config.n_hidden,
                                    n_hidden=fusion1_config.n_hidden, config=fusion1_config)


        self.gnn2 = GnnHelper(num_gnn_layers=gnn2_config.n_gnn_layers, n_hidden=gnn2_config.n_hidden, edge_updates=config.emlps, final_dropout=gnn2_config.final_dropout,
                            deg=deg, config=gnn2_config)

        self.transformer2 = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=transformer2_config.n_hidden,
                        nhead=transformer2_config.no_heads,
                        dim_feedforward=2*transformer2_config.n_hidden,
                        dropout=transformer2_config.dropout,
                        activation=transformer2_config.activation,
                        batch_first=True
                    ),
                    num_layers=transformer2_config.n_layers,
        )

        self.fusion2 = Fusion_Layer(input_dim_a=gnn2_config.n_hidden, input_dim_b=transformer2_config.n_hidden,
                                     n_hidden=fusion2_config.n_hidden, config=fusion2_config)

        input_sz = 3*gnn2_config.n_hidden + transformer2_config.n_hidden
        self.mlp = nn.Sequential(Linear(input_sz, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                            Linear(25, n_classes))

    def forward(self, data):
        # Initial Embedding Layers
        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr) 

        # First GNN Layer
        x_gnn, edge_a_gnn = self.gnn1(x, data.edge_index, edge_attr)
        
        # Transformer Layer
        edge_a_t = edge_attr.unsqueeze(0)
        edge_a_t = self.transformer1(edge_a_t)
        edge_a_t = edge_a_t.squeeze(0)
        

        #print(edge_a_gnn.shape, edge_a_t.shape)
        #print((torch.cat((edge_a_gnn, edge_a_t), dim=1)).shape)
        # Second GNN Layer
        edge_a_gnn, edge_a_t = self.fusion1(torch.cat((edge_a_gnn, edge_a_t), dim=1))

        #print(edge_a_gnn.shape, edge_a_t.shape)
                

        x_gnn, edge_a_gnn = self.gnn2(x_gnn, data.edge_index, edge_a_gnn)
        
        edge_a_t = self.transformer2(edge_a_t.unsqueeze(0))
        edge_a_t = edge_a_t.squeeze(0)

        edge_a_gnn, edge_a_t = self.fusion2(torch.cat((edge_a_gnn, edge_a_t), dim=1))

        edge_attr = torch.cat((edge_a_gnn, edge_a_t), dim=1)
        x = x_gnn
        #print(edge_attr.shape, x.shape)
        # Prediction Head
        x = x[data.edge_index.T].reshape(-1, 2*self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = self.mlp(x)
        
        return out
