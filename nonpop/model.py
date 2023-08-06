"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.user_conf_zdist = self.dataset.user_conf_zdist.to(world.device).to(torch.float32)
        self.item_pop_zdist = self.dataset.item_pop_zdist.to(world.device).to(torch.float32)
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user_conf = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim//2)
        self.embedding_user_nonconf = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim//2)

        self.embedding_item_pop = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim//2)
        self.embedding_item_nonpop = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim//2)

        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user_conf.weight, std=0.1)
            nn.init.normal_(self.embedding_user_nonconf.weight, std=0.1)
            nn.init.normal_(self.embedding_item_pop.weight, std=0.1)
            nn.init.normal_(self.embedding_item_nonpop.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user_conf.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_user_nonconf.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item_pop.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            self.embedding_item_nonpop.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretrained data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        if self.config['weight_type'] == 'scalar':
            self.weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device='cuda'))
            self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device='cuda'))
        elif self.config['weight_type'] == 'vector1':
            self.weight = nn.Parameter(torch.full([self.num_users], 0.0, dtype=torch.float32, device='cuda'))
            self.bias = nn.Parameter(torch.full([self.num_users], 0.0, dtype=torch.float32, device='cuda'))
        elif self.config['weight_type'] == 'vector2':
            self.weight = nn.Parameter(torch.full([self.num_items], 0.0, dtype=torch.float32, device='cuda'))
            self.bias = nn.Parameter(torch.full([self.num_items], 0.0, dtype=torch.float32, device='cuda'))

        if self.config['is_affine']:
            self.scale_u = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'))
            self.shift_u = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device='cuda'))
            self.scale_i = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'))
            self.shift_i = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device='cuda'))

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def dcor(self, x, y):

        a = torch.norm(x[:,None] - x, p = 2, dim = 2)
        b = torch.norm(y[:,None] - y, p = 2, dim = 2)

        A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
        B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean() 

        n = x.size(0)

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def computer(self, users_emb, items_emb):
        """
        propagate methods for lightGCN
        """       
        #users_emb = self.embedding_user_conf.weight
        #items_emb = self.embedding_item_pop.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users_conf, all_items_pop = self.computer(self.embedding_user_conf.weight,self.embedding_item_pop.weight)
        users_conf = all_users_conf[users.long()]
        items_pop = all_items_pop

        all_users_nonconf, all_items_nonpop = self.computer(self.embedding_user_nonconf.weight, self.embedding_item_nonpop.weight)
        users_nonconf = all_users_nonconf[users.long()]
        items_nonpop = all_items_nonpop

        rating = self.f(torch.matmul(users_conf, items_pop.t())) + self.f(torch.matmul(users_nonconf, items_nonpop.t()))
        rating_orig = self.f(torch.matmul(users_nonconf, items_nonpop.t()))
        return rating, rating_orig
    
    def getEmbedding(self, users, pos_items, neg_items):
        item_all = torch.unique(torch.cat((pos_items, neg_items)))

        all_users_conf, all_items_pop = self.computer(self.embedding_user_conf.weight,self.embedding_item_pop.weight)
        users_emb_conf = all_users_conf[users]
        emb_pop = all_items_pop[item_all]
        pos_emb_pop = all_items_pop[pos_items]
        neg_emb_pop = all_items_pop[neg_items]

        all_users_nonconf, all_items_nonpop = self.computer(self.embedding_user_nonconf.weight, self.embedding_item_nonpop.weight)
        users_emb_nonconf = all_users_nonconf[users]
        emb_nonpop = all_items_pop[item_all]
        pos_emb_nonpop = all_items_nonpop[pos_items]
        neg_emb_nonpop = all_items_nonpop[neg_items]

        # E0s
        users_emb_ego_conf = self.embedding_user_conf(users)
        users_emb_ego_nonconf = self.embedding_user_nonconf(users)
        pos_emb_ego_pop = self.embedding_item_pop(pos_items)
        neg_emb_ego_pop = self.embedding_item_pop(neg_items)
        pos_emb_ego_nonpop = self.embedding_item_nonpop(pos_items)
        neg_emb_ego_nonpop = self.embedding_item_nonpop(neg_items)

        return users_emb_conf, emb_pop, pos_emb_pop, neg_emb_pop, users_emb_nonconf, emb_nonpop, pos_emb_nonpop, neg_emb_nonpop, users_emb_ego_conf, pos_emb_ego_pop, neg_emb_ego_pop, users_emb_ego_nonconf, pos_emb_ego_nonpop, neg_emb_ego_nonpop
    
    def bpr_loss(self, users, pos, neg):
        neg = neg.squeeze()
        (user_conf, item_pop, pos_emb_pop, neg_emb_pop,
         user_nonconf, item_nonpop, pos_emb_nonpop, neg_emb_nonpop,
         users_emb_conf0, pos_emb_pop0, neg_emb_pop0,
         users_emb_nonconf0, pos_emb_nonpop0, neg_emb_nonpop0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(users_emb_conf0.norm(2).pow(2) + 
                         pos_emb_pop0.norm(2).pow(2)  +
                         neg_emb_pop0.norm(2).pow(2))/float(len(users))
        reg_loss = (1/2)*(users_emb_nonconf0.norm(2).pow(2) + 
                         pos_emb_nonpop0.norm(2).pow(2)  +
                         neg_emb_nonpop0.norm(2).pow(2))/float(len(users))
        # f(conf,pop)
        pos_scores_pop = torch.sum(torch.mul(user_conf, pos_emb_pop), dim=1)
        neg_scores_pop = torch.sum(torch.mul(user_conf, neg_emb_pop), dim=1)
        #f(nonconf,nonpop)
        pos_scores_nonpop = torch.sum(torch.mul(user_conf, pos_emb_nonpop), dim=1)
        neg_scores_nonpop = torch.sum(torch.mul(user_conf, neg_emb_nonpop), dim=1)
        #f(conf,pop)+f(nonconf,nonpop)
        pos_scores = pos_scores_pop + pos_scores_nonpop
        neg_scores = neg_scores_pop + neg_scores_nonpop
        
        loss_int = torch.mean(torch.nn.functional.softplus(neg_scores_nonpop - pos_scores_nonpop))
        loss_total = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        
        sign = torch.sign(self.user_conf_zdist[users]*(self.item_pop_zdist[pos]-self.item_pop_zdist[neg]))
        loss_pop = torch.mean(sign*torch.nn.functional.softplus(neg_scores_pop - pos_scores_pop))

        discrepency_loss = self.dcor(item_pop, item_nonpop) + self.dcor(user_conf, user_nonconf)
        self.int_weight = 0.1
        self.pop_weight = 0.1
        self.dis_pen = 0.1
        loss = self.int_weight*loss_int + self.pop_weight*loss_pop + loss_total - self.dis_pen*discrepency_loss

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
