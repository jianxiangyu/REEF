import numpy
import torch
import warnings
import random
from args import params
from utils import *
from model import *
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from downstream import *
import pandas as pd
import copy
import numpy as np

warnings.filterwarnings('ignore')

def set_seed(seed):
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

def get_gpu(gpu):
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
        torch.cuda.set_device(gpu)
    else:
        device = torch.device("cpu")
    return device

def train_mask(data, k=1):
    num_nodes = data.x.size(0)
    node_id = [[] for _ in range(data.y.max()+1)]
    for i in range(num_nodes):
        node_id[data.y[i]].append(i)
    train_id = []
    remain_id = []
    for i in range(data.y.max()+1):
        np.random.shuffle(node_id[i])
        train_id = train_id + node_id[i][:k]
        remain_id = remain_id + node_id[i][k:]
    data.train_mask = torch.tensor(
        [x in train_id for x in range(num_nodes)])
    np.random.shuffle(remain_id)
    split_point = len(remain_id) // 10
    val_id = remain_id[:split_point]
    test_id = remain_id[split_point:]
    data.valid_mask = torch.tensor(
        [x in val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor(
        [x in test_id for x in range(num_nodes)])
    return data


def load_task_dataset(dataset, task, data, type='test', b_size=64, if_finetune=False):
    s1_list, r_list, s2_list = create_triplets_test(dataset, data, task, type, if_finetune)
    if task == 'link_prediction':
        test_s1 = [s.cuda() for s in s1_list]
        test_s2 = [s.cuda() for s in s2_list]
        r_list = torch.tensor(r_list).cuda()
        test_s1 = DataLoader(test_s1, batch_size=b_size, shuffle=False)
        test_s2 = DataLoader(test_s2, batch_size=b_size, shuffle=False)
        r_list = DataLoader(r_list, batch_size=b_size, shuffle=False)
        return test_s1, test_s2, r_list
    elif task == 'node_classification':
        test_s1 = [s.cuda() for s in s1_list]
        test_s1 = DataLoader(test_s1, batch_size=b_size, shuffle=False)
        r_list = torch.tensor(r_list).cuda()
        r_list = DataLoader(r_list, batch_size=b_size, shuffle=False)
        return test_s1, s2_list, r_list

def out_domain_test(args, starttime, epoch_times):
    save_path = './pretrained_result/' + starttime
    if os.path.exists(os.path.join(save_path, 'edge_texts2id.json')):
        edge_texts2id = json.load(open(os.path.join(save_path, 'edge_texts2id.json'), 'r'))
    else:
        raise FileNotFoundError("edge_texts2id.json File not found")
    model_path = os.path.join(save_path, 'Epoch_' + epoch_times + '.pth')
    pretrain_info = torch.load(os.path.join('./pretrained_data/' +  '&'.join(args.pre_train_datasets), 'info.pt'))
    edge_embeds = pretrain_info['edge_embeds']

    sentence_model = SentenceEncoder(args.sentence_model, args.device)

    for pair in args.transfer_tasks:
        print('lr: ', args.test_lr, 'wd: ', args.test_wd, 'dropout: ', args.dropout)
        dataset, task = pair
        print("==================" + dataset + "==================")
        raw_data = read_dataset(dataset, args.svd_dim, args.k, sentence_model)
        dataset_description = read_json('./relation_info.json')['dataset_description']
        dataset_init_embeds = call_Sentence_LM(args.sentence_model, args.device, [dataset_description[dataset]])
        test_acc_list = []
        test_auc_list = []
        test_f1_list = []
        for time in range(5):
            best_acc = 0
            best_epoch = 0
            best_model = None
            best_dataset_embeds = None

            data = train_mask(copy.deepcopy(raw_data), k=1)
            ft_data = create_train_triples(dataset, data, edge_texts2id, args.svd_dim, if_finetune=True)
            dataset_embeds = dataset_init_embeds
            task_id_dict = {dataset: ft_data['task_id']}
            relation_id_dict = {dataset: ft_data['relation_id']}
            s1 = [ft_data['subgraphs1'][i].cuda() for i in range(len(ft_data['subgraphs1']))]
            s1 = Batch.from_data_list(s1)
            s2 = [ft_data['subgraphs2'][i].cuda() for i in range(len(ft_data['subgraphs2']))]
            s2 = Batch.from_data_list(s2)
            r = torch.tensor(ft_data['edge_type_list']).cuda()

            edge_embeds = edge_embeds.cuda()
            dataset_embeds = dataset_embeds.cuda()
            model = GFM(args.svd_dim,
                        args.hidden_dim,
                        edge_embeds.size(1),
                        args.dropout,
                        )
            state_dict = torch.load(model_path, map_location=args.device)
            if 'dataset_embeds' in state_dict:
                del state_dict['dataset_embeds']
            model.load_state_dict(state_dict)
            model.set_dataset_embeds(dataset_embeds)
            
            model = model.cuda()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.test_lr, weight_decay=args.test_wd)

            model.eval()
            criterion = nn.BCELoss(reduction='none')

            val_accbatch_size = data.valid_mask.sum().item() if dataset != 'computers' else 128
            test_accbatch_size = data.test_mask.sum().item() if dataset != 'computers' else 128
            val_s1, val_s2, val_r_list = load_task_dataset(dataset, task, data, type='valid', b_size=val_accbatch_size, if_finetune=True)
            test_s1, test_s2, test_r_list = load_task_dataset(dataset, task, data, type='test', b_size=test_accbatch_size, if_finetune=True)
            for epoch in tqdm(range(500)):
                model.train()
                optimizer.zero_grad()
                out = model(s1.cuda(), s2.cuda(), dataset, 0, edge_embeds[relation_id_dict[dataset]], edge_embeds[task_id_dict[dataset]])
                loss = criterion(out.squeeze(1), r.cuda().float()).mean()
                loss.backward()
                optimizer.step()
                model.eval()
                val_acc, val_auc, val_f1 = downstream_test(model, dataset, val_s1, val_s2, val_r_list, task, edge_embeds[relation_id_dict[dataset]], edge_embeds[task_id_dict[dataset]], 0, task_type="Val")
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    best_model = copy.deepcopy(model.state_dict())
                    best_dataset_embeds = copy.deepcopy(model.dataset_embeds)
            model.load_state_dict(best_model)
            model.set_dataset_embeds(best_dataset_embeds)
            model.eval()
            test_acc, test_auc, test_f1 = downstream_test(model, dataset, test_s1, test_s2, test_r_list, task, edge_embeds[relation_id_dict[dataset]], edge_embeds[task_id_dict[dataset]], 0, task_type="Test")
            print('Time: {:d}'.format(time), dataset, 'Epoch: {:d}'.format(best_epoch), 'Best Test acc: {:.4f}'.format(test_acc), 'Best Test auc: {:.4f}'.format(test_auc), 'Best Test f1: {:.4f}'.format(test_f1))
            test_acc_list.append(test_acc)
            test_auc_list.append(test_auc)
            test_f1_list.append(test_f1)
        test_acc_list = np.array(test_acc_list)
        test_auc_list = np.array(test_auc_list)
        test_f1_list = np.array(test_f1_list)
        print('Acc: ', test_acc_list)
        print('Auc: ', test_auc_list)
        print('F1: ', test_f1_list)
        print(f'Avg Best Test Acc: {sum(test_acc_list) / len(test_acc_list):.4f} +/- {np.std(test_acc_list):.4f}')
        print(f'Avg Best Test auc: {sum(test_auc_list) / len(test_auc_list):.4f} +/- {np.std(test_auc_list):.4f}')
        print(f'Avg Best Test f1: {sum(test_f1_list) / len(test_f1_list):.4f} +/- {np.std(test_f1_list):.4f}')

if __name__ == '__main__':
    args = params()
    args.device = get_gpu(args.gpu)
    set_seed(args.seed)

    starttime = '2025-02-14 22:43:34.927159'
    epoch_times = '0'
    out_domain_test(args, starttime, epoch_times)