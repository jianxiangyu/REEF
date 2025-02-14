import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import MyGCNConv

class GFM(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, dropout, if_test=False):
        super(GFM, self).__init__()
        self.gc1 = MyGCNConv(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gc2 = MyGCNConv(hidden_dim, hidden_dim)

        self.adapt_classifier = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        
        self.feature_mlp = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
        

    def set_dataset_embeds(self, dataset_embeds):
        if isinstance(dataset_embeds, torch.Tensor):
            self.dataset_embeds = nn.Parameter(dataset_embeds, requires_grad=True)
        else:
            raise ValueError("dataset_embeds is not a tensor")
        
    def subgraph_conv(self, x, edge_index, edge_type, dataset_embeds):
        self.feature_prompt = F.normalize(self.feature_prompt, dim=1)
        x = x + self.feature_prompt
        x = self.gc1(x, edge_index, edge_type.to(torch.long), self.relation_embeds, dataset_embeds)
        x = self.dropout(x)
        x = self.gc2(x, edge_index, edge_type.to(torch.long), self.relation_embeds, dataset_embeds)
        return x

    def forward(self, subgraphs1, subgraphs2, dataset, dataset_id, relation_embeds, task_embeds):
        self.relation_embeds = relation_embeds
        self.feature_prompt = self.feature_mlp(self.dataset_embeds[dataset_id]).unsqueeze(0)
        x1 = self.subgraph_conv(subgraphs1.x, subgraphs1.edge_index, subgraphs1.edge_type, self.dataset_embeds[dataset_id])
        x2 = self.subgraph_conv(subgraphs2.x, subgraphs2.edge_index, subgraphs2.edge_type, self.dataset_embeds[dataset_id])
        x1 = x1[subgraphs1.node_idx]
        x2 = x2[subgraphs2.node_idx]
        x = torch.mul(x1,x2)
        self.classifier = self.adapt_classifier(task_embeds)
        batch_size = x.size(0)
        predictions = torch.sigmoid(torch.mm(x, self.classifier.T)).expand(batch_size, -1) 
        return predictions