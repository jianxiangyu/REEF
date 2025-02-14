import argparse

def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    
    # Pre-traing
    ## Dataset
    parser.add_argument('--svd_dim', type=int, default=128)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='../cache_data/dataset/')
    parser.add_argument('--pre_train_datasets', type=list, default=['FB15K237','WN18RR', 'citeseer','pubmed', 'wisconsin', 'texas', 'photo'])

    ## The parameters of pre-training
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--epoch_times', type=int, default=100)

    # llm_model
    parser.add_argument('--sentence_model', type=str, default='Sentence-Bert', choices=['Salesforce/SFR-Embedding-Mistral','Sentence-Bert'])

    # Downstream
    parser.add_argument('--downstream_tasks', type=list, default=[('FB15K237','link_prediction'), ('WN18RR','link_prediction'), ('wisconsin', 'node_classification'), ('texas', 'node_classification'), ('citeseer', 'node_classification'), ('pubmed', 'node_classification'), ('photo', 'node_classification')])
    parser.add_argument('--transfer_tasks', type=list, default=[('cora','node_classification')])

    ## Target dataset
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test_lr', type=float, default=5e-3)
    parser.add_argument('--test_wd', type=float, default=5e-4)

    args, _ = parser.parse_known_args()
    return args