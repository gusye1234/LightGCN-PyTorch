import os
import world
import dataloader
import utils
from world import cprint
import torch
from model import LightGCN
from pprint import pprint
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
import Procedure

if world.dataset == 'gowalla':
    dataset = dataloader.Gowalla()
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
    
world.config['num_users'] = dataset.n_users
world.config['num_items'] = dataset.m_items

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("using bpr loss")
print('===========end===================')

Recmodel = LightGCN(world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel)

Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter("./runs/"+time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
else:
    w = None
    
try:
    for epoch in range(world.TRAIN_epochs):
        print('======================')
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
        # output_information = Procedure.BPR_train(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        start = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f"[TOTAL TIME] {time.time() - start}")
        print(f'[saved][{output_information}]')
        torch.save(Recmodel.state_dict(), os.path.join(world.PATH,"Rec-lgn.pth.tar"))
        if epoch %5 == 0 and epoch != 0:
            cprint("[TEST]")
            testDict = dataset.getTestDict()
            Procedure.Test(dataset, Recmodel, world.top_k, epoch, w, world.config['multicore'])
finally:
    if world.tensorboard:
        w.close()