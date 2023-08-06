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
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import wandb

CORES = multiprocessing.cpu_count() // 2
np.set_printoptions(suppress=True)


def train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    get_loss: utils.BPRLoss = loss_class
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset, neg_ratio=neg_k)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2:]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    #import pdb;pdb.set_trace()
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = get_loss.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    if world.config['weight_type'] == 'scalar':
        wandb.log({
            'weight': Recmodel.weight.item(),
            'bias': Recmodel.bias.item(),
            })
    if world.config['is_affine']:
        wandb.log({
            'scale_u': Recmodel.scale_u.item(),
            'shift_u': Recmodel.shift_u.item(),
            'scale_i': Recmodel.scale_i.item(),
            'shift_i': Recmodel.shift_i.item()
            })

    wandb.log({
        'train_loss': round(aver_loss, 3)
    })
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
def test_one_batch_grp(X, user_grp_batch, item_grp, n_u_class, n_i_class):
    sorted_items = X[0].numpy() # BX20
    groundTrue = X[1]
    top_K_items_grp = item_grp[sorted_items.flatten()].reshape(sorted_items.shape) # item groups that are recommended as top K
    r = utils.getLabel(groundTrue, sorted_items)
    recall_lst_ui, ndcg_lst_ui = np.zeros((n_u_class, n_i_class)), np.zeros((n_u_class, n_i_class))
    recall_lst_u, ndcg_lst_u, recall_lst_i, ndcg_lst_i = np.zeros(n_u_class), np.zeros(n_u_class), np.zeros(n_i_class), np.zeros(n_i_class)
    k=20
    for i in user_grp_batch.unique().int():
        for j in item_grp.unique().int():
            test_user_pos_items_list_grp = list(np.array(groundTrue, dtype=object)[user_grp_batch == i])
            r_grp = r[user_grp_batch == i]
            r_grp[top_K_items_grp[user_grp_batch == i] != j] = 0
            recall_lst_ui[i][j] += utils.RecallPrecision_ATk(test_user_pos_items_list_grp, r_grp, k)['recall']
            ndcg_lst_ui[i][j] += utils.NDCGatK_r(test_user_pos_items_list_grp, r_grp, k)
    # get recall and ndcg in each user conformity group
    for i in user_grp_batch.unique().int():
        test_user_pos_items_list_grp = list(np.array(groundTrue, dtype=object)[user_grp_batch == i])
        r_grp = r[user_grp_batch == i]
        recall_lst_u[i] += utils.RecallPrecision_ATk(test_user_pos_items_list_grp, r_grp, k)['recall']
        ndcg_lst_u[i] += utils.NDCGatK_r(test_user_pos_items_list_grp, r_grp, k)

    # get recall and ndcg in each item popularity group
    for j in item_grp.unique().int():
        test_user_pos_items_list_grp = list(np.array(groundTrue, dtype=object))
        r_grp = r.copy()
        r_grp[top_K_items_grp != j] = 0
        recall_lst_i[j] += utils.RecallPrecision_ATk(test_user_pos_items_list_grp, r_grp, k)['recall']
        ndcg_lst_i[j] += utils.NDCGatK_r(test_user_pos_items_list_grp, r_grp, k)
    return {
               'recall_u': recall_lst_u,
               'ndcg_u': ndcg_lst_u,
               'recall_i': recall_lst_i,
               'ndcg_i': ndcg_lst_i,
               'recall_ui': recall_lst_ui,
               'ndcg_ui': ndcg_lst_ui
    }

def Test(dataset, Recmodel, epoch, w=None, multicore=0, use_orig=False):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    user_grp, item_grp = dataset.user_grp.squeeze(), dataset.item_grp.squeeze()
    n_u_class, n_i_class = dataset.n_u_class, dataset.n_i_class

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    grp_results = {
               'recall_u': np.zeros(n_u_class),
               'ndcg_u': np.zeros(n_u_class),
               'recall_i': np.zeros(n_i_class),
               'ndcg_i': np.zeros(n_i_class),
               'recall_ui': np.zeros((n_u_class, n_i_class)),
               'ndcg_ui': np.zeros((n_u_class, n_i_class))}

    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        user_grp_lst = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users) # batchuser의 모든 pos item
            groundTrue = [testDict[u] for u in batch_users] # testset groundtruth
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating, rating_orig = Recmodel.getUsersRating(batch_users_gpu) # B X I. batch user에 대한 prediction
            if use_orig:
                rating = rating_orig
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K) # B X 20
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users) # list of list
            rating_list.append(rating_K.cpu()) # list of list
            groundTrue_list.append(groundTrue) # list of list
            user_grp_lst.append(user_grp[batch_users])
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
            pre_results_grp = pool.map(test_one_batch_grp, (X, user_grp_lst, item_grp, n_u_class, n_i_class))
        else:
            pre_results, pre_results_grp = [], []
            for (i,x) in enumerate(X):
                pre_results.append(test_one_batch(x))
                pre_results_grp.append(test_one_batch_grp(x, user_grp_lst[i], item_grp, n_u_class, n_i_class))

        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        for result in pre_results_grp:
            grp_results['recall_u'] += result['recall_u']
            grp_results['recall_i'] += result['recall_i']
            grp_results['ndcg_u'] += result['ndcg_u']
            grp_results['ndcg_i'] += result['ndcg_i']
            grp_results['recall_ui'] += result['recall_ui']#.reshape(n_u_class, n_i_class)
            grp_results['ndcg_ui'] += result['ndcg_ui']#.reshape(n_u_class, n_i_class)
        grp_results['recall_u'] /= float(len(users))
        grp_results['recall_i'] /= float(len(users))
        grp_results['ndcg_u'] /= float(len(users))
        grp_results['ndcg_i'] /= float(len(users))
        grp_results['recall_ui'] /= float(len(users))
        grp_results['ndcg_ui'] /= float(len(users))
        top_K_items_grp = dataset.item_grp[np.concatenate(rating_list)].squeeze() # item groups that are recommended as top K
        c_ratio = utils.C_Ratio(top_K_items_grp)

        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        print('c_ratio:', round(c_ratio, 5))
        wandb.log({
            'recall': np.round(results['recall'], 5),
            'precision': np.round(results['precision'], 5),
            'ndcg': np.round(results['ndcg'], 5),
            'c_ratio': round(c_ratio, 5)
        })

        print(grp_results)
        wandb.log({
            'user_recall_diff': utils.get_group_diff(grp_results['recall_u']),
            'user_ndcg_diff': utils.get_group_diff(grp_results['ndcg_u']),
            'item_recall_diff': utils.get_group_diff(grp_results['recall_i']),
            'item_ndcg_diff': utils.get_group_diff(grp_results['recall_i'])
        })

        return results
