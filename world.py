import os
import torch
from enum import Enum

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config = {}
config['batch_size'] = 4096
config['bpr_batch_size'] = 4096
config['latent_dim_rec'] = 32
config['lightGCN_n_layers']=2
config['dropout'] = False
config['keep_prob']  = 0.6

GPU = torch.cuda.is_available()

TRAIN_epochs = 1000
LOAD = False
PATH = './checkpoints'
top_k = 5
tensorboard = True
comment = "lgn"
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)




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