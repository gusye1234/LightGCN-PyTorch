'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
import torch
from enum import Enum
from parse import parse_args
import multiprocessing
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()
print(args)
config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain

if args.pretrain == 1:
    config['user_emb'] = np.load('data/' + args.dataset + '/user_emb.npy')
    config['item_emb'] = np.load('data/' + args.dataset + '/item_emb.npy')

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

#device = torch.device("cpu")

dataset = args.dataset
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")

if dataset in ['gowalla', 'yelp']:
    config['A_split'] = False
    config['bigdata'] = False
else:
    config['A_split'] = False
    config['bigdata'] = False



TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
# topks = eval(args.topks)
topks = [20]
top_k = 20
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
print(logo)