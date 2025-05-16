# import transformer
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
from src.models.mpnn import GnnHelper
from torch_geometric.nn import Linear
from src.util import unpack_dict_ns
class Fusion_Layer(torch.nn.Module):
    def __init__(self, input_dim_a=20, input_dim_b=20,
                  n_hidden=100, config=None
                ):
        super().__init__()
        #print(config)
        self.config = config
        self.n_hidden = n_hidden

        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        
        if config.model == "fmlp":
            activation = None
            if config.activation == "relu":
                activation = nn.ReLU()
            elif config.activation == "gelu":
                activation = nn.GELU()


            self.fusion = nn.Sequential(Linear(input_dim_a+input_dim_b, config.n_hidden),
                                        activation,nn.Dropout(config.dropout),
                                        Linear(config.n_hidden, input_dim_a+input_dim_b),
                                        activation,nn.Dropout(config.dropout),
            )
        elif config.model == "gmu":
            self.fusion = None
            raise NotImplementedError("GMU fusion not implemented")



    def forward(self, data):
        # Initial Embedding Layers
        out = self.fusion(data)

        out1 = out[:, :self.input_dim_a]
        out2 = out[:, self.input_dim_a:]
        
        return out1, out2
