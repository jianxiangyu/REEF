import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch import Tensor
from typing import Optional
from torch_geometric.utils import index_sort, one_hot, scatter, spmm

class MyGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MyGCNConv, self).__init__(aggr='mean') 
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Learnable mappers
        self.weight_mapper = nn.Sequential(
            nn.Linear(384, in_channels * out_channels),
            nn.ReLU(),
            nn.Linear(in_channels * out_channels, in_channels * out_channels)
        )
        self.root_mapper = nn.Sequential(
            nn.Linear(384, in_channels * out_channels),
            nn.ReLU(),
            nn.Linear(in_channels * out_channels, in_channels * out_channels)
        )
        self.bias_mapper = nn.Sequential(
            nn.Linear(384, out_channels)
        )
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.root)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type, edge_embeds, dataset_embeds):
        # Generate weights for each edge type
        self.weights = self.weight_mapper(edge_embeds).view(-1, self.in_channels, self.out_channels)
        self.num_relations = self.weights.size(0)

        # Apply message passing
        out = self.propagate(edge_index, x=x, edge_type=edge_type, size=(x.size(0), x.size(0)))
        # Add root and bias terms
        self.root = self.root_mapper(dataset_embeds).view(self.in_channels, self.out_channels)
        self.bias = self.bias_mapper(dataset_embeds)
        out = out + torch.matmul(x, self.root) + self.bias
        return out

    def message(self, x_j: Tensor, edge_type: Tensor) -> Tensor:
        out = torch.bmm(x_j.unsqueeze(-2), self.weights[edge_type]).squeeze(-2)
        return out

    def aggregate(self, inputs: Tensor, edge_type: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        # Compute normalization in separation for each `edge_type`.
        if self.aggr == 'mean':
            norm = one_hot(edge_type, self.num_relations, dtype=inputs.dtype)
            norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
            norm = torch.gather(norm, 1, edge_type.view(-1, 1))
            norm = 1. / norm.clamp_(1.)
            inputs = norm * inputs
        out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)
        return out