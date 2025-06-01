# import transformer
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
from src.models.mpnn import GnnHelper
from torch_geometric.nn import Linear
from src.util import unpack_dict_ns
from src.models.fusion_l import Fusion_Layer
# from src.models.pos_enc import EdgePositionalEncoder
class Full_Fusion(torch.nn.Module):
    def __init__(self, num_features, n_classes=2, n_hidden=100, 
                 edge_dim=None, final_dropout=0.5, 
                deg=None, config=None,
                ):
        """
        Instantiates the Full-Fusion model.
        The full-fusion model is a combination of two GNNs and two transformers.
        The flow of the model is as follows:

        
        Node & Edge Embedding -> GNN1  -> |               | ->     GNN2     -> | 
                                          | -> Fusion1 -> |                    | -> Fusion2 -> MLP
        Edge Embedding -> Transformer1 -> |               | -> Transformer2 -> | 

        Args:
        - num_features (int): Number of features in the input graph.
        - n_classes (int): Number of classes in the output graph.
        - n_hidden (int): Number of hidden units in the model.
        - edge_dim (int): Number of features in the edge embedding.
        - final_dropout (float): Dropout rate for the final layer.
        - deg (int): Degree of the graph.
        - config (dict): Configuration dictionary containing the model parameters.
        """
        super().__init__()
        #print(config)
        self.config = config
        self.n_hidden = n_hidden
        self.final_dropout = final_dropout


        gnn1_config = unpack_dict_ns(config, 0)
        transformer1_config = unpack_dict_ns(config, 1)
        fusion1_config = unpack_dict_ns(config, 2)

        gnn2_config = unpack_dict_ns(config, 3)
        transformer2_config = unpack_dict_ns(config, 4)
        fusion2_config = unpack_dict_ns(config, 5)


        self.node_emb_gnn = nn.Linear(num_features, gnn1_config.n_hidden)
        self.edge_emb_gnn = nn.Linear(edge_dim, gnn1_config.n_hidden)

        # if config.use_pe:
        #     self.edge_pos_enc = EdgePositionalEncoder(
        #         edge_feat_dim=edge_dim,
        #         out_dim=transformer1_config.n_hidden,
        #     )
        # else :
        #     self.edge_pos_enc = None

        self.edge_emb_tr = nn.Linear(edge_dim, transformer1_config.n_hidden)
           
        self.gnn1 = GnnHelper(num_gnn_layers=gnn1_config.n_gnn_layers, n_hidden=gnn1_config.n_hidden, edge_updates=config.emlps, final_dropout=gnn1_config.final_dropout,
                            deg=deg, config=gnn1_config)
        
        
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
        """
        Forward pass of the Full-Fusion model.
        """
        # Initial Embedding Layers
        x_gnn = self.node_emb_gnn(data.x)
        edge_a_gnn = self.edge_emb_gnn(data.edge_attr)

        

        # if self.edge_pos_enc is not None:
        #     edge_a_t = self.edge_pos_enc(data) 
        # else: 
        edge_a_t = self.edge_emb_tr(data.edge_attr)
        
        # First GNN Layer
        x_gnn, edge_a_gnn = self.gnn1(x_gnn, data.edge_index, edge_a_gnn)
        
        # Transformer Layer
        edge_a_t = edge_a_t.unsqueeze(0)
        edge_a_t = self.transformer1(edge_a_t)
        edge_a_t = edge_a_t.squeeze(0)
        

        # Fusion Layer
        edge_a_gnn, edge_a_t = self.fusion1(torch.cat((edge_a_gnn, edge_a_t), dim=1))


        # Second GNN Layer
        x_gnn, edge_a_gnn = self.gnn2(x_gnn, data.edge_index, edge_a_gnn)

        # Second Transformer Layer
        edge_a_t = self.transformer2(edge_a_t.unsqueeze(0))
        edge_a_t = edge_a_t.squeeze(0)

        # Second Fusion Layer
        edge_a_gnn, edge_a_t = self.fusion2(torch.cat((edge_a_gnn, edge_a_t), dim=1))

        # Concatenate the edge embeddings of the GNN and Transformer layers
        edge_attr = torch.cat((edge_a_gnn, edge_a_t), dim=1)
        
        # Keep the node features from the GNN 
        x = x_gnn
        
        # Prediction Head
        x = x[data.edge_index.T].reshape(-1, 2*self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = self.mlp(x)
        
        return out
