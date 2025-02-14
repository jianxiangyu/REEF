import torch
from torch_geometric.data import Data
import os
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset
from llm_model import SentenceEncoder
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

class GFMDataset(InMemoryDataset, ABC):
    def __init__(self, name: str, k: int, svd_dim: int, encoder: Optional[SentenceEncoder] = None,
                 root: str = "../cache_data/dataset/", transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        self.root = root
        self.encoder = encoder
        self.k = k
        self.svd_dim = svd_dim
        self.data_dir = os.path.join(self.root, self.name)
        super().__init__(self.data_dir, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.edge_embeds = torch.load(self.processed_paths[1])
        self.subgraphs = torch.load(self.processed_paths[2])

    def data2vec(self, data: list[str]) -> torch.Tensor:
        r"""
        Encode a list of string to a len(data)-by-d matrix, where d is the output dimension of the LLM.
        """
        if self.encoder is None:
            raise NotImplementedError("LLM encoder is not defined")
        if data is None:
            return None
        embeddings = self.encoder.encode(data)
        return embeddings

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["pyg_data_processed.pt", "edge_embeds.pt", "subgraph.pt"]

    def text2feature(self, texts):
        if isinstance(texts[0], str):
            return self.data2vec(texts)
        return [self.text2feature(t) for t in texts]
    
    def SVD_decomposition(self, x, dim):
        if x.size(0) < dim:
            N = x.size(0)
            x_padded = torch.cat([x, torch.zeros(x.size(1) - N, x.size(1))], dim=0)
            svd = TruncatedSVD(n_components=dim)
            x_svd = torch.tensor(svd.fit_transform(x_padded.numpy()), dtype=torch.float)[:N]  # 取前 3 行
        elif x.size(0) == dim:
            x_svd = x
        else:
            svd = TruncatedSVD(n_components=dim)
            x_svd = torch.tensor(svd.fit_transform(x.numpy()), dtype=torch.float)
        return x_svd

    @abstractmethod
    def load_data(self) -> tuple[list[pyg.data.Data], list[list[str]], Any]:
        pass
    
    def process(self):
        new_data = self.load_data()

        if not os.path.exists(self.processed_paths[1]):
            edge_embeds = self.text2feature(new_data.edge_text)
            torch.save(edge_embeds, self.processed_paths[1])
        
        if not os.path.exists(self.processed_paths[0]):
            data, slices = self.collate([new_data])
            torch.save((data, slices), self.processed_paths[0], pickle_protocol=4)

        if not os.path.exists(self.processed_paths[2]):
            torch.save(self.construct_subgraph(new_data), self.processed_paths[2])

    def construct_subgraph(self, data, max_nodes=500):
        print('Constructing Subgraphs...')
        subgraphs = []
        x = data.x
        for node_idx in tqdm(range(data.x.size(0))):
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx, self.k, data.edge_index, relabel_nodes=True
            )
            edge_type = data.edge_type[edge_mask]
            if subset.size(0) > max_nodes:
                sampled_indices = torch.randperm(subset.size(0))[:max_nodes - 1]
                sampled_indices = torch.unique(torch.cat([mapping, sampled_indices]))
                mask = torch.isin(edge_index[0], sampled_indices) & torch.isin(edge_index[1], sampled_indices)
                edge_index = edge_index[:, mask]
                edge_type = edge_type[mask]
            sub_data = Data(subset=subset, edge_index=edge_index, edge_type=edge_type)
            sub_data.node_idx = torch.zeros(subset.size(0), dtype=torch.bool)
            sub_data.node_idx[mapping] = True
            if data.y is not None:
                sub_data.y = data.y[node_idx]
            subgraphs.append(sub_data)
        return subgraphs