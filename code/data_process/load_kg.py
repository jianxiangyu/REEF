import json
import numpy as np
import torch
import os 
from dataset import GFMDataset
from torch_geometric.data import Data

def gen_entities(name, data_path):
    if name == "WN18RR":
        entity2id = {}
        entity_lst = []
        text_lst = []
        with open(os.path.join(data_path, name, "entity2text.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip().split("\t")
                entity_lst.append(tmp[0])
                text_lst.append(tmp[1])

        entity2id = {entity: i for i, entity in enumerate(entity_lst)}
    elif name == "FB15K237":
        entity_lst = []
        text_lst = []
        with open(os.path.join(data_path, name, "entity2wikidata.json"), "r") as f:
            data = json.load(f)

        for k in data:
            entity_lst.append(k)
            text_lst.append("Entity names: " + data[k]["label"] + ", Entity alternatives: " + ", ".join(
                data[k]["alternatives"]) + ". Entity descriptions:" + data[k]["description"] + '.' if data[k][ "description"] is
                                                                                                not None else "None")

        entity2id = {entity: i for i, entity in enumerate(entity_lst)}
    else:
        raise NotImplementedError("Dataset " + name + " is not implemented.")
    return entity_lst, text_lst, entity2id


def read_knowledge_graph(files, name, data_path):
    entity_lst, text_lst, entity2id = gen_entities(name, data_path)
    relation2id = {}

    converted_triplets = {}
    rel_list = []
    rel = len(relation2id)

    for file_type, file_path in files.items():
        edges = []
        edge_types = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split("\n")[:-1]]
        unknown_entity = 0
        for triplet in file_data:
            if triplet[0] not in entity2id:
                text_lst.append("entity names: Unknown")
                entity_lst.append(triplet[0])
                entity2id[triplet[0]] = len(entity2id)
                unknown_entity += 1
            if triplet[2] not in entity2id:
                text_lst.append("entity names: Unknown")
                entity_lst.append(triplet[2])
                entity2id[triplet[2]] = len(entity2id)
                unknown_entity += 1
            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel_list.append(triplet[1])
                rel += 1

            edges.append([entity2id[triplet[0]], entity2id[triplet[2]], ])
            edge_types.append(relation2id[triplet[1]])
        print('Unknow_entity:', unknown_entity)
        converted_triplets[file_type] = [edges, edge_types]

    num_edges = len(converted_triplets["train"][0]) + len(converted_triplets["valid"][0]) + len(converted_triplets["test"][0])
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    valid_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    train_num = len(converted_triplets["train"][0])
    valid_num = len(converted_triplets["valid"][0])

    train_mask[:train_num] = True
    valid_mask[train_num: train_num + valid_num] = True
    test_mask[train_num + valid_num:] = True

    edge_text = ["Feature edge. Relation between two entities. " + relation for relation in rel_list] 

    data = Data(
        x=torch.load(os.path.join(data_path, name, "ent_embeddings.pt")),
        edge_index=torch.cat([
            torch.tensor(converted_triplets["train"][0]).T,
            torch.tensor(converted_triplets["valid"][0]).T,
            torch.tensor(converted_triplets["test"][0]).T,
        ], dim=1),
        edge_type=torch.cat([
            torch.tensor(converted_triplets["train"][1]),
            torch.tensor(converted_triplets["valid"][1]),
            torch.tensor(converted_triplets["test"][1]),
        ]),
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask,
        edge_text=edge_text,
        y=None
    )
    return data


class KGDataset(GFMDataset):
    def load_data(self):
        cur_path = os.path.dirname(__file__)
        data_path = os.path.join(os.path.dirname(os.path.dirname(cur_path)), 'data')
        names = ["train", "valid", "test"]
        name_dict = {n: os.path.join(data_path, self.name, n + ".txt") for n in names}
        new_data = read_knowledge_graph(name_dict, self.name, data_path)
        new_data.label_text = None
        return new_data