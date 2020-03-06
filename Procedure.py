'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from time import time
from tqdm import tqdm
import model
import multiprocessing



def BPR_train(dataset,recommend_model, loss_class, epoch, neg_k = 4,w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr : utils.BPRLoss = loss_class  
    allusers = list(range(dataset.n_users))        
    S, sam_time = utils.UniformSample_allpos_largeDataset(allusers, dataset, neg_k)
    print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:,0]).long()
    posItems = torch.Tensor(S[:,1]).long()
    negItems = torch.Tensor(S[:,2]).long()
    
    users    = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users)//world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i, 
        (batch_users, 
         batch_pos, 
         batch_neg)) in tqdm(enumerate(utils.minibatch(users, 
                                                  posItems, 
                                                  negItems, 
                                                  batch_size=world.config['bpr_batch_size'])),total=total_batch):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch*int(len(users)/world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss/total_batch
    return f"[BPR[{aver_loss:.3e}][{sam_time[0]:.1f}={sam_time[1]:.1f}+{sam_time[1]:.1f}]]"
    
    

def Test(dataset, Recmodel, top_k, epoch, w=None):
    dataset : utils.BasicDataset
    testDict : dict = dataset.getTestDict()
    Recmodel : model.RecMF
    with torch.no_grad():
        Recmodel.eval()
        users = torch.Tensor(list(testDict.keys()))
        users = users.to(world.device)
        GroundTrue = [testDict[user] for user in users.cpu().numpy()]
        rating = Recmodel.getUsersRating(users)
        rating = rating.cpu()
        # exclude positive train data
        allPos = dataset.getUserPosItems(users)
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i]*len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = 0.
        # assert torch.all(rating >= 0.)
        # assert torch.all(rating <= 1.)
        # end excluding
        _, top_items = torch.topk(rating, top_k)
        top_items = top_items.cpu().numpy()
        metrics = utils.recall_precisionATk(GroundTrue, top_items, top_k)
        metrics['mrr'] = utils.MRRatK(GroundTrue, top_items, top_k)
        metrics['ndcg'] = utils.NDCGatK(GroundTrue, top_items, top_k)
        # pprint(metrics)
        if world.tensorboard:
            w.add_scalar(f'Test/Recall@{top_k}', metrics['recall'], epoch)
            w.add_scalar(f'Test/Precision@{top_k}', metrics['precision'], epoch)
            w.add_scalar(f'Test/MRR@{top_k}', metrics['mrr'], epoch)
            w.add_scalar(f'Test/NDCG@{top_k}', metrics['ndcg'], epoch)
            
            
            
def test_large(dataset, Recmodel, top_k, epoch, w=None):
    u_batch_size = world.config['test_u_batch_size']