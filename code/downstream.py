import torch
import warnings
from args import params
from utils import *
from model import *
from torch_geometric.data import Data, DataLoader, Batch
from tqdm import tqdm
import torch.nn.functional as F
from llm_model import SentenceEncoder
from  torchmetrics import Accuracy, F1Score, AUROC

warnings.filterwarnings('ignore')


def create_triplets_test(dataset, data, task_type, data_type, if_finetune=False):
    s1_list = []
    s2_list = []
    r_list = []
    subgraph = data.subgraphs
    if task_type == 'link_prediction':
        edge_index = data.edge_index[:,data.test_mask]
        edge_types = data.edge_type[data.test_mask]
        
        subgraph1 = [subgraph[i] for i in edge_index[0,:]]
        subgraph2 = [subgraph[i] for i in edge_index[1,:]]
        data.x = SVD_decomposition(data.x, 128)
        for s1,r,s2 in zip(subgraph1, edge_types, subgraph2):
            s1.x = data.x[s1.subset]
            s2.x = data.x[s2.subset]
            s1_list.append(s1)
            r_list.append(r.item())
            s2_list.append(s2)
        return s1_list, r_list, s2_list
    elif task_type == 'node_classification':
        if dataset in ["wisconsin", "texas", "cornell", "chameleon", "squirrel"] and not if_finetune:
            data.train_mask = data.train_mask[:,0]
            data.val_mask = data.val_mask[:,0]
            data.test_mask = data.test_mask[:,0]
        data.x = SVD_decomposition(data.x, 128)
        label_embeds = get_label_embeds(data)
        
        s1_list = []
        label_num = data.y.max() + 1
        s2_list = []
        for i in range(label_num):
            s2_list.append(Data(
                        x=label_embeds[i].unsqueeze(0),
                        edge_index=torch.tensor([], dtype=torch.long).reshape(2, 0),
                        edge_type=torch.tensor([], dtype=torch.long),
                        node_idx=torch.ones(1, dtype=torch.bool)
                    ))
        
        if data_type == 'test':
            indices = torch.nonzero(data.test_mask).squeeze()
        elif data_type == 'valid':
            if hasattr(data, 'valid_mask'):
                indices = torch.nonzero(data.valid_mask).squeeze()
            else:
                indices = torch.nonzero(data.val_mask).squeeze()
        elif data_type == 'train':
            indices = torch.nonzero(data.train_mask).squeeze()
        for i in tqdm(indices.tolist()):
            r_list.append(data.y[i])
            s1 = data.subgraphs[i]
            if s1.subset.device.type == 'cuda':
                s1.subset = s1.subset.cpu()
            s1.x = data.x[s1.subset]
            s1_list.append(s1)
        return s1_list, r_list, s2_list

def load_test_dataset(args, type):
    sentence_model = SentenceEncoder(args.sentence_model, args.device)
    test_tasks = {}
    for pair in args.downstream_tasks:
        dataset, task = pair
        data = read_dataset(dataset, args.svd_dim, args.k, sentence_model)
        s1_list, r_list, s2_list = create_triplets_test(dataset, data, task, type)
       
        if task == 'link_prediction':
            b_size = 64
            test_s1 = [s.cuda() for s in s1_list]
            test_s2 = [s.cuda() for s in s2_list]
            r_list = torch.tensor(r_list).cuda()
            test_s1 = DataLoader(test_s1, batch_size=b_size, shuffle=False)
            test_s2 = DataLoader(test_s2, batch_size=b_size, shuffle=False)
            r_list = DataLoader(r_list, batch_size=b_size, shuffle=False)
            test_tasks[pair] = (test_s1, test_s2, r_list, task)
        elif task == 'node_classification':
            b_size = 64
            test_s1 = [s.cuda() for s in s1_list]
            test_s1 = DataLoader(test_s1, batch_size=b_size, shuffle=False)
            r_list = torch.tensor(r_list).cuda()
            r_list = DataLoader(r_list, batch_size=b_size, shuffle=False)
            test_tasks[pair] = (test_s1, s2_list, r_list, task)
    return test_tasks

def downstream_test(model, dataset, test_s1, test_s2, r_list, task, relation_id, task_id, dataset_embeds, task_type="Train"):
    if task == 'link_prediction':
        y_true = []
        y_predict = []
        for i, (r, s1, s2) in tqdm(enumerate(zip(r_list, test_s1, test_s2))):
            with torch.no_grad():
                out = model(s1.cuda(), s2.cuda(), dataset, dataset_embeds, relation_id, task_id)
                y_true.append(r.cuda())
                y_predict.append(out.argmax(dim=1))
        y_true = torch.cat(y_true, dim=0)
        y_predict = torch.cat(y_predict, dim=0)
        return cal_acc(y_predict, y_true)
    elif task == 'node_classification':
        test_s2_cuda = [t.cuda() for t in test_s2]
        accuracy_metric = Accuracy(task='multiclass', num_classes=len(test_s2))
        auc_score_metric = AUROC(task='multiclass', num_classes=len(test_s2))
        f1_score_metric = F1Score(task='multiclass', num_classes=len(test_s2), average='macro')
        y_predict = []
        y_true = []
        num = 0
        for j, (s1, r) in enumerate(zip(test_s1, r_list)):
            predict = []
            num += len(s1)
            for i in range(len(test_s2)):
                label_subgraphs = [test_s2_cuda[i] for _ in range(len(s1))]
                label_subgraphs = Batch.from_data_list(label_subgraphs)
                with torch.no_grad():
                    out = model(s1, label_subgraphs, dataset, dataset_embeds, relation_id, task_id)
                    predict.append(out.squeeze(1))
            predict = torch.stack(predict, dim=1)
            y_predict.append(predict)
            y_true.append(r.unsqueeze(1))
        y_predict = torch.cat(y_predict, dim=0)
        y_true = torch.cat(y_true, dim=0)
        y_true = y_true.squeeze(1)
        predict_softmax = F.softmax(y_predict, dim=1)

        
        acc_metric = accuracy_metric(y_predict.argmax(dim=1).cpu(), y_true.cpu())
        f1_metric = f1_score_metric(y_predict.argmax(dim=1).cpu(), y_true.cpu())
        auroc_metric = auc_score_metric(predict_softmax.cpu(), y_true.cpu())

        if task_type == "Train":
            return acc_metric.cpu()
        else:
            return acc_metric.cpu(), auroc_metric.cpu(), f1_metric.cpu()