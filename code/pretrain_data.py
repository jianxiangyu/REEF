import os
import torch
from llm_model import SentenceEncoder
from utils import *


class PreTrainDataset:
    def __init__(self, args):
        self.args = args
        self.dataset_list = {}

        self.node_embeds = []
        self.relation_embeds = []

        self.train_data = {}
        self.relation_info = read_json("relation_info.json")

    def load_pre_train_dataset(self):
        save_path = './pretrained_data/' +  '&'.join(self.args.pre_train_datasets)
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.sentence_model = SentenceEncoder(self.args.sentence_model, self.args.device)
        if os.path.exists(os.path.join(save_path, 'info.pt')):
            pretrain_info = torch.load(os.path.join(save_path, 'info.pt'))

            self.edge_embeds = pretrain_info["edge_embeds"]
            self.edge_texts2id = pretrain_info["edge_texts2id"]
            if pretrain_info["datasets"] != self.args.pre_train_datasets:
                raise ValueError("Datasets error")
        else:
            edge_texts = []
            edge_embeds = []
            print('Constructing Pretrained dataset...')

            for d in self.args.pre_train_datasets:
                print(d)
                data = read_dataset(d, self.args.svd_dim, self.args.k, self.sentence_model)
                self.dataset_list[d] = data
                edge_texts += data.edge_text
                edge_embeds += data.edge_embeds

                if d in ["pubmed", "citeseer"]:
                    edge_texts += [self.relation_info["downstream"]["paper_classification"]]
                    edge_embeds += call_Sentence_LM(self.args.sentence_model, self.args.device, [self.relation_info["downstream"]["paper_classification"]])
                elif d in ["wisconsin", "texas"]:
                    edge_texts += [self.relation_info["downstream"]["webpage_classification"]]
                    edge_embeds += call_Sentence_LM(self.args.sentence_model, self.args.device, [self.relation_info["downstream"]["webpage_classification"]])
                elif d in ["photo"]:
                    edge_texts += [self.relation_info["downstream"]["product_classification"]]
                    edge_embeds += call_Sentence_LM(self.args.sentence_model, self.args.device, [self.relation_info["downstream"]["product_classification"]])
                else:
                    pass

            unique_edge_texts = set(edge_texts)
            u_edge_texts_lst = list(unique_edge_texts)

            self.edge_texts2id = {v: i for i, v in enumerate(u_edge_texts_lst)}
            edge_embeds = [edge_embeds[edge_texts.index(text)] for text in u_edge_texts_lst]
            self.edge_embeds = torch.stack(edge_embeds)

            pretrain_edge_info = {
                'datasets': self.args.pre_train_datasets,
                'edge_embeds': self.edge_embeds,
                'edge_texts2id': self.edge_texts2id,
            }
            torch.save(pretrain_edge_info, os.path.join(save_path, 'info.pt'))

        for d in self.args.pre_train_datasets:
            print(d)
            if os.path.exists(os.path.join(save_path, d + '.pt')) or os.path.exists(os.path.join(save_path, d + '_subgraph2.pt')):
                print("[%s] have been created, load it" % d)
                train_data_dataset = torch.load(os.path.join(save_path, d + '.pt'))
            else:
                if d not in self.dataset_list:
                    data = read_dataset(d, self.args.svd_dim, self.args.k, self.sentence_model)
                else:
                    data = self.dataset_list[d]
                train_data_dataset = create_train_triples(d, data, self.edge_texts2id, self.args.svd_dim)
                print("Create [%s] pretrain dataset finished" % d)
                torch.save(train_data_dataset, os.path.join(save_path, d + '.pt'))

            self.train_data[d] = train_data_dataset

        return self.train_data, \
                    self.edge_embeds, \
                    self.edge_texts2id