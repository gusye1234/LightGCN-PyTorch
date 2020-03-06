'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import os
import torch
from enum import Enum
import parser

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parser.parser()

config = {}
all_dataset = ['lastfm', 'gowalla']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
#device = torch.device("cpu")

dataset = args.dataset
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")

if dataset in ['gowalla']:
    config['A_split'] = True
    config['bigdata'] = True
else:
    config['A_split'] = False



TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
topks = eval(args.topks)
top_k = 5
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