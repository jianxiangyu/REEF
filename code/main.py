import numpy
import torch
import warnings
import datetime
import random
from args import params
from utils import *
from model import *
from downstream import downstream_test, load_test_dataset
from pretrain_data import PreTrainDataset

warnings.filterwarnings('ignore')


def set_seed(seed):
    ## random seed ##
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_gpu(gpu):
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
        torch.cuda.set_device(gpu)
    else:
        device = torch.device("cpu")
    return device


def check_train2testtask(args):
    for pair in args.downstream_tasks:
        dataset, task = pair
        if dataset not in args.pre_train_datasets:
            raise ValueError("Test task not in training dataset")
        

def create_training_sequence(train_data, training_sequence_dataset, batch_size_list):
    indices_list = []
    for i, dataset in enumerate(training_sequence_dataset):
        num_samples = len(train_data[dataset]['subgraphs1'])
        dataset_indices = list(range(num_samples))
        random.shuffle(dataset_indices)
        batch_size = batch_size_list[i]
        batched_indices = [dataset_indices[j:j + batch_size] for j in range(0, len(dataset_indices), batch_size)]
        for batch in batched_indices:
            indices_list.extend([(dataset, batch)])
    random.shuffle(indices_list)
    return indices_list


def pre_train(args):
    check_train2testtask(args)
    starttime = datetime.datetime.now()
    train_data, edge_embeds, edge_texts2id = PreTrainDataset(args).load_pre_train_dataset()
    if not os.path.exists('./pretrained_result/' + str(starttime)):
        os.makedirs('./pretrained_result/' + str(starttime))
    save_json('./pretrained_result/' + str(starttime) + '/edge_texts2id.json', edge_texts2id)
    task_id_dict = {}
    relation_id_dict = {}
    best_val_result = {}
    best_test_result = {}
    for d in args.downstream_tasks:
        dataset, task = d
        task_id_dict[dataset] = train_data[dataset]['task_id']
        relation_id_dict[dataset] = train_data[dataset]['relation_id']
        best_val_result[dataset] = 0
        best_test_result[dataset] = 0

    val_tasks = load_test_dataset(args, "valid")
    test_tasks = load_test_dataset(args, "test")

    print("Pre-training Dataset Ready.")
    
    training_sequence_dataset =  args.pre_train_datasets
    batch_size_list = [args.batch_size * len(args.pre_train_datasets)]

    dataset_description = read_json('./relation_info.json')['dataset_description']
    dataset_embeds = []
    dataset_id_dict = {}
    for i, dataset in enumerate(training_sequence_dataset):
        dataset_embeds.append(call_Sentence_LM(args.sentence_model, args.device, [dataset_description[dataset]]))
        dataset_id_dict[dataset] = i
    dataset_embeds = torch.stack(dataset_embeds)
    dataset_embeds = dataset_embeds.squeeze(1).cuda()
    edge_embeds = edge_embeds.cuda()
    
    indices_list = create_training_sequence(
        train_data, training_sequence_dataset, batch_size_list)
    print("Random training sequence Constructed.")
    
    print("Relation size:", edge_embeds.size(0))
    model = GFM(args.svd_dim,
                args.hidden_dim,
                edge_embeds.size(1),
                args.dropout)
    model.set_dataset_embeds(dataset_embeds)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()

    criterion = nn.BCELoss(reduction='none')
    
    print('Epoch times:', args.epoch_times)

    for epoch in range(args.epoch_times):
        model.train()
        print("---------------------------------------------------")
        print("Epoch:",epoch)
        with tqdm(total=len(indices_list)) as pbar:
            for i, (dataset, idx) in enumerate(indices_list):
                # Create batch using indices
                s1_list = [train_data[dataset]['subgraphs1'][i] for i in idx]
                s2_list = [train_data[dataset]['subgraphs2'][i] for i in idx]
                r = [train_data[dataset]['edge_type_list'][i] for i in idx]
                s1 = Batch.from_data_list(s1_list).cuda()
                s2 = Batch.from_data_list(s2_list).cuda()
                r = torch.tensor(r).cuda()
                s1 = pertub_relation(s1)
                s2 = pertub_relation(s2)
                optimizer.zero_grad()
                relation_id = relation_id_dict[dataset]
                task_id = task_id_dict[dataset]
                out = model(s1, s2, dataset, dataset_id_dict[dataset], edge_embeds[relation_id], edge_embeds[task_id])
                if dataset == 'FB15K237' or dataset == 'WN18RR':
                    target = torch.zeros(r.size(0), len(task_id)).cuda()
                    target.scatter_(1, r.unsqueeze(1).long(), 1)
                    loss = criterion(out, target.float()).mean()
                elif dataset in ['pubmed', 'wisconsin', 'texas', 'citeseer', 'computers']:
                    loss = criterion(out.squeeze(1), r.float()).mean()
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Dataset: {dataset} Loss: {loss.item():.4f}")
                pbar.update(1)
                del s1,s2,r,out,loss

            torch.cuda.empty_cache()
                
        model.eval()
        for k,v_val in val_tasks.items():
            val_s1, val_s2, r_list, task = v_val
            dataset = k[0]
            val_result = downstream_test(model, dataset, val_s1, val_s2, r_list, task, edge_embeds[relation_id_dict[dataset]], edge_embeds[task_id_dict[dataset]], dataset_id_dict[dataset])
            print(k, ' valid: ', val_result)
            del val_s1, val_s2, r_list
            torch.cuda.empty_cache()
            
            test_s1, test_s2, r_list, task = test_tasks[k]
            dataset = k[0]
            test_result = downstream_test(model, dataset, test_s1, test_s2, r_list, task, edge_embeds[relation_id_dict[dataset]], edge_embeds[task_id_dict[dataset]], dataset_id_dict[dataset])
            print(k, ' test: ', test_result)
            del test_s1, test_s2, r_list
            torch.cuda.empty_cache()

            if val_result > best_val_result[dataset]:
                best_val_result[dataset] = val_result
                best_test_result[dataset] = test_result
        if not os.path.exists('./pretrained_result/' + str(starttime) + '/best_result.txt'):
            with open('./pretrained_result/' + str(starttime) + '/best_result.txt', 'w') as f:
                f.write('Epoch: ' + str(epoch) + '\n')
                for dataset in best_val_result:
                    f.write(str(dataset) + ': Valid: {:.4f}'.format(best_val_result[dataset]*100) + ' Test: {:.4f}'.format(best_test_result[dataset]*100) + '\n')
        else:
            with open('./pretrained_result/' + str(starttime) + '/best_result.txt', 'a') as f:
                f.write('Epoch: ' + str(epoch) + '\n')
                for dataset in best_val_result:
                    f.write(str(dataset) + ': Valid: {:.4f}'.format(best_val_result[dataset]*100) + ' Test: {:.4f}'.format(best_test_result[dataset]*100) + '\n')
        if epoch % 10 == 0:
            torch.save(model.state_dict(), './pretrained_result/' + str(starttime) + '/' + 'Epoch_' + str(epoch) + '.pth')

    torch.save(model.state_dict(), './pretrained_result/' + str(starttime) + '/' + 'Epoch_' + str(epoch) + '.pth')
    print('Finished Pre-training.')


if __name__ == '__main__':
    args = params()
    set_seed(args.seed)
    args.device = get_gpu(args.gpu)
    pre_train(args)