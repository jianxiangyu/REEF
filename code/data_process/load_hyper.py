from copy import deepcopy
from torch_geometric.datasets import WebKB
from torch_geometric.data import Data
import os
from dataset import GFMDataset
from .utils import relation_description
import torch

def load_hyper(data_name, cache_dir):
    if data_name in ["wisconsin", "texas", "cornell"]:
        data = WebKB(root=cache_dir, name=data_name.capitalize())._data
    else:
        raise ValueError(f'Unknown dataset: {data_name}')
    
    return data

class HyperDataset(GFMDataset):
    def load_data(self):
        cur_path = os.path.dirname(__file__)
        text_data_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(cur_path)), 'data'), self.name)
        data = load_hyper(self.name, text_data_path)
        edge_text = [relation_description('hyperlinks')]
        data.edge_text = edge_text
        data.edge_type = torch.zeros([data.edge_index.shape[1],]).int()
        return data