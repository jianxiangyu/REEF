from copy import deepcopy
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork
from torch_geometric.data import Data
import os
from ogb.nodeproppred import PygNodePropPredDataset
from dataset import GFMDataset
from .utils import relation_description
import pandas as pd
import torch
import numpy as np

def load_cite(data_name, cache_dir):
    if data_name in ['cora', 'citeseer', 'pubmed']:
        data = Planetoid(root=cache_dir, name=data_name.capitalize())._data
        # 622
        num_nodes = len(data.y)
        node_id = np.arange(num_nodes)
        np.random.shuffle(node_id)

        train_id = np.sort(node_id[:int(num_nodes * 0.6)])
        val_id = np.sort(
            node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
        test_id = np.sort(node_id[int(num_nodes * 0.8):])

        data.train_mask = torch.tensor(
            [x in train_id for x in range(num_nodes)])
        data.val_mask = torch.tensor(
            [x in val_id for x in range(num_nodes)])
        data.test_mask = torch.tensor(
            [x in test_id for x in range(num_nodes)])
    else:
        raise ValueError(f'Unknown dataset: {data_name}')
    
    # assert isinstance(data, (Data, dict)), f'Unknown data type: {type(data)}'
    # yield data if isinstance(data, Data) else Data(**data)
    return data

class CitationDataset(GFMDataset):
    def load_data(self):
        cur_path = os.path.dirname(__file__)
        text_data_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(cur_path)), 'data'), self.name)
        data = load_cite(self.name, text_data_path)
        edge_text = [relation_description('citation')]
        data.edge_text = edge_text
        data.edge_type = torch.zeros([data.edge_index.shape[1],]).int()
        return data