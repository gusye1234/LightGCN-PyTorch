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
import utils

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
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
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
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
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
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        user_conf = self.user_conf_zdist[users]
        item_pop = self.item_pop_zdist.squeeze()
        rating_orig = self.f(torch.matmul(users_emb, items_emb.t()))

        if self.config['is_affine']:
            user_conf = user_conf * self.scale_u + self.shift_u
            item_pop = item_pop * self.scale_i + self.shift_i
        if self.config['weight_type']:
            adj = user_conf * item_pop
            if self.config['weight_type'] == 'vector1':
                adj = adj * self.weight[users].unsqueeze(dim=1) + self.bias[users].unsqueeze(dim=1)
            elif self.config['weight_type'] == 'vector2':
                adj = adj * self.weight + self.bias
            else:
                adj = adj * self.weight + self.bias
            rating = self.f(torch.matmul(users_emb, items_emb.t()) + adj)
        else:
            rating = rating_orig
        return rating, rating_orig
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        #import pdb;pdb.set_trace()
        neg = neg.squeeze()
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        
        if self.config['weight_type']:
            user_conf = self.user_conf_zdist[users] # B
            pos_pop = self.item_pop_zdist[pos] # B
            neg_pop = self.item_pop_zdist[neg] # B
            if self.config['is_affine']:
                user_conf = user_conf * self.scale_u + self.shift_u
                pos_pop = pos_pop * self.scale_i + self.shift_i
                neg_pop = neg_pop * self.scale_i + self.shift_i
            if self.config['weight_type'] == 'vector1':
                pos_adj = (user_conf*pos_pop) * self.weight[users] + self.bias[users]
                neg_adj = (user_conf*neg_pop) * self.weight[users] + self.bias[users]
            elif self.config['weight_type'] == 'vector2':
                pos_adj = (user_conf*pos_pop) * self.weight[pos] + self.bias[pos]
                neg_adj = (user_conf*neg_pop) * self.weight[neg] + self.bias[neg]
            else:
                pos_adj = (user_conf*pos_pop) * self.weight + self.bias
                neg_adj = (user_conf*neg_pop) * self.weight + self.bias
            pos_scores = torch.mul(users_emb, pos_emb) 
            pos_scores = torch.sum(pos_scores, dim=1) 
            neg_scores = torch.mul(users_emb, neg_emb) 
            neg_scores = torch.sum(neg_scores, dim=1)

            # 또는 softmax
            pos_adj =  F.binary_cross_entropy(self.f(pos_adj), torch.ones(pos_adj.shape).cuda())
            neg_adj =  F.binary_cross_entropy(self.f(neg_adj), torch.zeros(neg_adj.shape).cuda())
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores)) + pos_adj + neg_adj
            #loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores)+torch.nn.functional.softplus(neg_adj-pos_adj))

        else:
            pos_scores = torch.mul(users_emb, pos_emb)
            pos_scores = torch.sum(pos_scores, dim=1)
            neg_scores = torch.mul(users_emb, neg_emb)
            neg_scores = torch.sum(neg_scores, dim=1)
            
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

    def bce_loss(self, users, pos, neg):
        all_users, all_items = self.computer()
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))

        temp_u = torch.mm(all_users.t(),all_users) - torch.eye(all_users.shape[1]).cuda()
        temp_i = torch.mm(all_items.t(),all_items) - torch.eye(all_items.shape[1]).cuda()
        L_ortho = torch.pow(torch.linalg.matrix_norm(temp_u),2) + torch.pow(torch.linalg.matrix_norm(temp_i),2)

        user_declf_losses = [torch.pow(utils.pearsonr(self.user_conf_zdist.squeeze(), user), 2)
                        for user in all_users.t()]
        user_declf_loss = torch.stack(user_declf_losses).sum()

        item_declf_losses = [torch.pow(utils.pearsonr(self.item_pop_zdist.squeeze(), item), 2)
                        for item in all_items.t()]
        item_declf_loss = torch.stack(item_declf_losses).sum()

        if self.config['weight_type']:
            user_conf = self.user_conf_zdist[users].to(torch.float32) # B
            pos_pop = self.item_pop_zdist[pos].to(torch.float32) # B
            neg_pop = self.item_pop_zdist[neg].to(torch.float32) # B
            if self.config['is_affine']:
                user_conf = user_conf * self.scale_u + self.shift_u
                pos_pop = pos_pop * self.scale_i + self.shift_i
                neg_pop = neg_pop * self.scale_i + self.shift_i
            if self.config['weight_type'] == 'vector1':
                pos_adj = (user_conf*pos_pop) * self.weight[users] + self.bias[users]
                neg_adj = (user_conf*neg_pop) * self.weight[users] + self.bias[users]
            elif self.config['weight_type'] == 'vector2':
                pos_adj = (user_conf*pos_pop) * self.weight[pos] + self.bias[pos]
                neg_adj = (user_conf.unsqueeze(1)*neg_pop) * self.weight[neg] + self.bias[neg]
            else:
                pos_adj = (user_conf*pos_pop) * self.weight + self.bias
                neg_adj = (user_conf.unsqueeze(1)*neg_pop) * self.weight + self.bias
                neg_adj = neg_adj.squeeze().reshape(-1,1)

            pos_scores = torch.mul(users_emb, pos_emb)
            pos_scores = torch.sum(pos_scores, dim=1) 
            neg_scores = torch.mul(users_emb.unsqueeze(1), neg_emb)
            neg_scores = torch.sum(neg_scores, dim=2).reshape(-1,1).squeeze()
            
            loss = F.binary_cross_entropy(torch.sigmoid(pos_scores+pos_adj.squeeze()), torch.ones(pos_scores.shape).cuda()) + F.binary_cross_entropy(torch.sigmoid(neg_scores + neg_adj.squeeze()), torch.zeros(neg_scores.shape).cuda())
            #adj_loss =  F.binary_cross_entropy(self.f(pos_adj), torch.ones(pos_adj.shape).cuda()) + F.binary_cross_entropy(self.f(neg_adj), torch.ones(neg_adj.shape).cuda())
            #loss = F.binary_cross_entropy(torch.sigmoid(pos_scores), torch.ones(pos_scores.shape).cuda()) + F.binary_cross_entropy(torch.sigmoid(neg_scores), torch.zeros(neg_scores.shape).cuda())
            #loss =loss + adj_loss

        else:
            pos_scores = torch.mul(users_emb, pos_emb)
            pos_scores = torch.sum(pos_scores, dim=1)
            neg_scores = torch.mul(users_emb.unsqueeze(1), neg_emb)
            neg_scores = torch.sum(neg_scores, dim=2).reshape(-1,1).squeeze()
            
            loss = F.binary_cross_entropy(torch.sigmoid(pos_scores), torch.ones(pos_scores.shape).cuda()) + F.binary_cross_entropy(torch.sigmoid(neg_scores), torch.zeros(neg_scores.shape).cuda())
        
        loss = loss + 0.02*(user_declf_loss + item_declf_loss) + 0.02 * (L_ortho)
        return loss, reg_loss

    def softmax_loss(self, users, pos, neg):
        #import pdb;pdb.set_trace()
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        """
        if self.config['weight_type']:
            user_conf = self.user_conf_zdist[users] # B
            pos_pop = self.item_pop_zdist[pos] # B
            neg_pop = self.item_pop_zdist[neg] # B
            if self.config['is_affine']:
                user_conf = user_conf * self.scale_u + self.shift_u
                pos_pop = pos_pop * self.scale_i + self.shift_i
                neg_pop = neg_pop * self.scale_i + self.shift_i
            if self.config['weight_type'] == 'vector1':
                pos_adj = (user_conf*pos_pop) * self.weight[users] + self.bias[users]
                neg_adj = (user_conf*neg_pop) * self.weight[users] + self.bias[users]
            elif self.config['weight_type'] == 'vector2':
                pos_adj = (user_conf*pos_pop) * self.weight[pos] + self.bias[pos]
                neg_adj = (user_conf*neg_pop) * self.weight[neg] + self.bias[neg]
            else:
                pos_adj = (user_conf*pos_pop) * self.weight + self.bias
                neg_adj = (user_conf*neg_pop) * self.weight + self.bias
            pos_scores = torch.mul(users_emb, pos_emb) 
            pos_scores = torch.sum(pos_scores, dim=1) + pos_adj.squeeze()
            neg_scores = torch.mul(users_emb, neg_emb) 
            neg_scores = torch.sum(neg_scores, dim=1) + neg_adj.squeeze()
            #pos_adj = pos_adj.squeeze()
            #neg_adj = neg_adj.squeeze()
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
            #loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores)+torch.nn.functional.softplus(neg_adj-pos_adj))
"""
        #else:
        pos_scores = F.cosine_similarity(users_emb.unsqueeze(1), pos_emb, dim=-1) # user X item
        log_softmax_var = F.log_softmax(pos_scores, 0).diag()
        loss = - torch.mean(log_softmax_var)
                
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
