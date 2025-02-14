import os
import torch
import json
from dataset import *
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def pth_safe_save(obj, path):
    if obj is not None:
        torch.save(obj, path)

def pth_safe_load(path):
    if os.path.exists(path):
        return torch.load(path)
    return None

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def relation_description():
    with open('./relation_info.json', 'r') as f:
        relation_dict = json.load(f)
        return relation_dict

def read_dataset(d, svd_dim, k, sentence_model):
    if d in ['FB15K237', 'WN18RR']:
        from data_process.load_kg import KGDataset
        data = KGDataset(d, k, svd_dim, sentence_model)
    elif d in ["cora", "pubmed", "citeseer"]:
        from data_process.load_cite import CitationDataset
        data = CitationDataset(d, k, svd_dim, sentence_model)
    elif d in ["wisconsin", "texas", "cornell"]:
        from data_process.load_hyper import HyperDataset
        data = HyperDataset(d, k, svd_dim, sentence_model)
    elif d in ["computers", "photo"]:
        from data_process.load_amazon import AmazonDataset
        data = AmazonDataset(d, k, svd_dim, sentence_model)
    else:
        raise ValueError("Unseen Dataset.")
    return data


def get_label_embeds(data, svd_dim=128):
    train_indices = torch.nonzero(data.train_mask).squeeze()
    label_num = data.y.max() + 1
    label_embeds = torch.zeros(label_num, svd_dim)
    sample_label_num = [0 for _ in range(label_num)]
    for i in range(train_indices.size(0)):
        label_embeds[data.y[train_indices[i]]] += data.x[train_indices[i]]
        sample_label_num[data.y[train_indices[i]]] += 1
    for i in range(label_num):
        if sample_label_num[i] == 0:
            label_embeds[i] = torch.randn(svd_dim)
        else:
            label_embeds[i] = label_embeds[i] / sample_label_num[i]
    return label_embeds

def create_train_triples(dataset, data, edge_texts2id, svd_dim=0, if_finetune=False):
    subgraphs1 = []
    subgraphs2 = []
    edge_type_list = []
    if dataset in ["cora", "pubmed", "wisconsin", "texas", "citeseer", "photo", "cornell", "computers"]:
        edge_index = data.edge_index
        edge_type = data.edge_type

        for i in range(len(data.subgraphs)):
            if hasattr(data.subgraphs[i], 'y'):
                del data.subgraphs[i].y

        if dataset in ["wisconsin", "texas", "cornell"] and not if_finetune:
            data.train_mask = data.train_mask[:,0]
            data.val_mask = data.val_mask[:,0]
            data.test_mask = data.test_mask[:,0]

        data.x = SVD_decomposition(data.x, svd_dim)
        train_indices = torch.nonzero(data.train_mask).squeeze()
        label_num = data.y.max() + 1
        data.label_embeds = get_label_embeds(data, svd_dim)

        label_graphs = []
        for i in range(label_num):
            label_graphs.append(Data(
                        x=data.label_embeds[i].unsqueeze(0),
                        edge_index=torch.tensor([], dtype=torch.long).reshape(2, 0),
                        edge_type=torch.tensor([], dtype=torch.long),
                        node_idx=torch.ones(1, dtype=torch.bool)
                    ))

        for i in tqdm(train_indices.tolist()):
            label = data.y[i]
            s1 = data.subgraphs[i]
            del s1.y
            s1.x = data.x[s1.subset]
            for j in range(len(label_graphs)):
                if j != label:
                    subgraphs1.append(s1)
                    edge_type_list.append(0)
                    subgraphs2.append(label_graphs[j])
                else:
                    subgraphs1.append(s1)
                    edge_type_list.append(1)
                    subgraphs2.append(label_graphs[j])

        if dataset in ["cora", "pubmed", "citeseer"]:
            task_id = [edge_texts2id[relation_description()["downstream"]["paper_classification"]]]
        elif dataset in ["wisconsin", "texas", "cornell"]:
            task_id = [edge_texts2id[relation_description()["downstream"]["webpage_classification"]]]
        elif dataset in ["photo", "computers"]:
            task_id = [edge_texts2id[relation_description()["downstream"]["product_classification"]]]
        train_data = {
            'subgraphs1': subgraphs1,
            'subgraphs2': subgraphs2,
            'edge_type_list': edge_type_list,
            'relation_id': [edge_texts2id[text] for text in data.edge_text],
            'task_id': task_id
        }
        return train_data
    
    elif dataset in ['FB15K237', 'WN18RR']:
        subgraphs1 = []
        subgraphs2 = []
        edge_type_list = []
        edge_index = data.edge_index[:, data.train_mask]
        edge_type = data.edge_type[data.train_mask]

        data.x = SVD_decomposition(data.x, svd_dim)
        for i in range(len(data.subgraphs)):
            if hasattr(data.subgraphs[i], 'y'):
                del data.subgraphs[i].y

        for i in tqdm(range(len(edge_type))):
            s1 = data.subgraphs[edge_index[0,i]]
            s2 = data.subgraphs[edge_index[1,i]]
            s1.x = data.x[s1.subset]
            s2.x = data.x[s2.subset]
            subgraphs1.append(s1)
            edge_type_list.append(edge_type[i])
            subgraphs2.append(s2)

        train_data = {
            'subgraphs1': subgraphs1,
            'subgraphs2': subgraphs2,
            'edge_type_list': edge_type_list,
            'relation_id': [edge_texts2id[text] for text in data.edge_text],
            'task_id': [edge_texts2id[text] for text in data.edge_text]
        }
        return train_data


def cal_acc(y_pred, y_true):
    correct = (y_pred == y_true).sum().item()
    accuracy = correct / y_true.size(0)
    return accuracy


def get_shuffled_indices(num_samples):
    indices = torch.randperm(num_samples)
    return indices

def call_Sentence_LM(sentence_model_name, device, relation_text_list):
    global sentence_model
    if 'sentence_model' not in globals():
        sentence_model = SentenceEncoder(sentence_model_name, device)
    relation_embeds = sentence_model.encode(relation_text_list)
    return relation_embeds

def SVD_decomposition(x, dim):
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


def pertub_relation(data, pertub_rate=0.2):
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > pertub_rate
    data.edge_index = edge_index[:, mask]
    data.edge_type = data.edge_type[mask]
    return data