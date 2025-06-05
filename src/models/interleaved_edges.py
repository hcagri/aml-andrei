# import transformer
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
from src.models.mpnn import GnnHelper
from torch_geometric.nn import Linear
from src.util import unpack_dict_ns
from src.models.pos_enc.pearl import get_PEARL_wrapper


class Interleaved_Edges(torch.nn.Module):
    def __init__(
        self,
        num_features,
        n_classes=2,
        n_hidden=20,
        n_hidden_edge=20,
        edge_dim=None,
        final_dropout=0.5,
        deg=None,
        config=None,
    ):
        super().__init__()
        # print(config)
        self.config = config
        self.n_hidden = n_hidden
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden_edge)

        if config.use_pe:
            pecpy = unpack_dict_ns(config, -1)
            self.posenc = get_PEARL_wrapper(pecpy)

        fcpy = unpack_dict_ns(config, 0)
        # print(fcpy)
        self.gnn1 = GnnHelper(
            num_gnn_layers=fcpy.n_gnn_layers,
            n_hidden=fcpy.n_hidden,
            edge_updates=config.emlps,
            final_dropout=fcpy.final_dropout,
            deg=deg,
            config=fcpy,
        )

        scpy = unpack_dict_ns(config, 1)
        # print(scpy)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=scpy.n_hidden,
                nhead=scpy.no_heads,
                dim_feedforward=2 * scpy.n_hidden,
                dropout=scpy.dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=scpy.n_layers,
        )

        tcpy = unpack_dict_ns(config, 2)
        self.gnn2 = GnnHelper(
            num_gnn_layers=tcpy.n_gnn_layers,
            n_hidden=tcpy.n_hidden,
            edge_updates=config.emlps,
            final_dropout=tcpy.final_dropout,
            deg=deg,
            config=tcpy,
        )

        self.mlp = nn.Sequential(
            Linear(n_hidden * 3, 50),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            Linear(25, n_classes),
        )

    def forward(self, data):
        # Initial Embedding Layers
        # print("after")
        # print("data.x", data.x.shape, "data.edge_attr", data.edge_attr.shape)

        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr)

        # print("x", x.shape, "edge_attr", edge_attr.shape)

        # print("data.x", data.x.shape, "data.edge_attr", data.edge_attr.shape)

        if self.config.use_pe:
            data.x = x
            data.edge_attr = edge_attr
            data = self.posenc(data)

            x = data.x
            edge_attr = data.edge_attr

        # print("data.x", data.x.shape, "data.edge_attr", data.edge_attr.shape)

        # print("x", x.shape, "edge_attr", edge_attr.shape)

        # First GNN Layer
        x, edge_attr = self.gnn1(x, data.edge_index, edge_attr)

        # Transformer Layer
        edge_attr = edge_attr.unsqueeze(0)
        edge_attr = self.transformer(edge_attr)
        edge_attr = edge_attr.squeeze(0)

        # Second GNN Layer
        x, edge_attr = self.gnn2(x, data.edge_index, edge_attr)

        # print("x", x.shape, "edge_attr", edge_attr.shape)
        # Prediction Head
        x = x[data.edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = self.mlp(x)

        return out
