#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from kgreasoning.dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from kgreasoning.util import dcg_at_k, ndcg_at_k
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os
import json
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
#from kgreasoning.finch import FINCH

def Identity(x):
    return x

def ConvertRelations(x, user_cluster, nrelation, nclusters):
    if x == 0:
        x = nrelation+user_cluster
    elif x == 1:
        x = nrelation+nclusters+user_cluster
    return x
convert_relations = np.vectorize(ConvertRelations)

class BoxOffsetIntersection(nn.Module):
    
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0) 
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate

class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings)) # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding

class VecAggregate3p(nn.Module):

    def __init__(self, dim):
        super(VecAggregate3p, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        #nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):  
        #all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        #layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        #attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)
        attention = F.relu(self.layer1(embeddings))
        
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding
    
class BetaAggregate3p(nn.Module):

    def __init__(self, dim):
        super(BetaAggregate3p, self).__init__()
        self.dim = dim
        #self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        #self.layer2 = nn.Linear(2 * self.dim, self.dim)
        self.layer1 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        #nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):  
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        #layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        #attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)
        attention = F.relu(self.layer1(all_embeddings))
        
        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding
    
class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):  
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)
        
        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding

class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim) # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim) # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, nitems, hidden_dim, gamma, 
                 geo, pretrain_embedding_dir, cluster_embedding_dir, test_batch_size=1,
                 box_mode=None, use_cuda=False, use_clustering=0, use_pretraining=0, nclusters=0,  
                 query_name_dict=None, beta_mode=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.nitems = nitems
        self.nusers = nentity - nitems
        self.best_recall = 0
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.use_clustering = use_clustering 
        self.use_pretraining = use_pretraining # pretrain
        self.nclusters = nclusters
        self.pretrain_embedding_dir = pretrain_embedding_dir
        self.cluster_embedding_dir = cluster_embedding_dir
        self.batch_entity_range = torch.arange(nitems).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nitems).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        
        if self.geo == 'box':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # centor for entities
            activation, cen = box_mode
            self.cen = cen # hyperparameter that balances the in-box distance and the out-box distance
            if activation == 'none':
                self.func = Identity
            elif activation == 'relu':
                self.func = F.relu
            elif activation == 'softplus':
                self.func = F.softplus
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        elif self.geo == 'vec':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # center for entities
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        elif self.geo == 'beta':
            
            if self.use_pretraining == 1:
                
                pretrain_path = self.pretrain_embedding_dir
                pretrain_data = np.load(pretrain_path)
                self.item_pre_embed = pretrain_data['item_embed']
                self.user_pre_embed = pretrain_data['user_embed']
                
                #assert self.user_pre_embed.shape[0] == self.n_users
                assert self.item_pre_embed.shape[0] == self.nitems
                #assert self.user_pre_embed.shape[1] == self.entity_dim
                #assert self.item_pre_embed.shape[1] == self.entity_dim

                # TODO: clustering
                #######################################################
                #clustering = AffinityPropagation(random_state=5).fit(self.user_pre_embed)
                #clustering = KMeans(n_clusters=self.nclusters, random_state=5).fit(self.user_pre_embed)
                #self.user_clusters = clustering.labels_
                #self.nclusters = clustering.cluster_centers_indices_.shape[0]
                #print('num cluster:',self.nclusters)
                #c, num_clust, req_c = FINCH(self.user_pre_embed)
                #print('c:',c)
                #print('num_clust:',num_clust)
                #print('req_c:',req_c)
                #######################################################
                
                """
                #concat = np.concatenate((self.item_pre_embed, self.user_pre_embed), axis=0)
                self.item_pre_embed = (self.item_pre_embed - self.item_pre_embed.min()) / (self.item_pre_embed.max() - self.item_pre_embed.min())
                self.user_pre_embed = (self.user_pre_embed - self.user_pre_embed.min()) / (self.user_pre_embed.max() - self.user_pre_embed.min())

                # alpha + beta = 10
                alpha = self.item_pre_embed*10
                beta = 10 - alpha
                self.item_pre_embed = np.concatenate((alpha, beta), axis=1)
                alpha = self.user_pre_embed*10
                beta = 10 - alpha
                self.user_pre_embed = np.concatenate((alpha, beta), axis=1)
                """
                
                self.entity_embedding = torch.zeros(nentity, self.entity_dim * 2) # alpha and beta
                #self.relation_embedding = torch.zeros(self.nrelation+self.nclusters*2, self.relation_dim)
                self.entity_embedding[:self.nitems,:] = torch.from_numpy(self.item_pre_embed) 
                self.entity_embedding[self.nitems:,:] = torch.from_numpy(self.user_pre_embed) 
                #self.relation_embedding = nn.Parameter(self.relation_embedding)
                self.entity_embedding = nn.Parameter(self.entity_embedding)
                #print('relation embedding shape', self.relation_embedding.shape)
                #print('entity_embedding embedding shape', self.entity_embedding.shape)
                
                if self.use_clustering:

                    pretrain_path = self.cluster_embedding_dir
                    pretrain_data = np.load(pretrain_path)
                    #self.user_pre_embed = pretrain_data['user_embed']
                    #self.item_pre_embed = pretrain_data['item_embed']
                    self.rel_pre_embed = pretrain_data['rel_embed']

                    assert self.rel_pre_embed.shape[0] == self.nrelation
                    assert self.rel_pre_embed.shape[1] == self.entity_dim
                    #assert self.user_pre_embed.shape[0] == self.nusers
                    #assert self.user_pre_embed.shape[1] == self.entity_dim*2
                    #assert self.item_pre_embed.shape[0] == self.nitems
                    #assert self.item_pre_embed.shape[1] == self.entity_dim*2
                    
                    # clustering
                    #user_alpha_embedding, user_beta_embedding = np.split(self.user_pre_embed, 2, axis=-1)
                    #user_embedding = user_alpha_embedding/(user_alpha_embedding+user_beta_embedding)
                    """
                    clustering = KMeans(n_clusters=1000, random_state=5).fit(self.user_pre_embed)
                    self.user_clusters = clustering.labels_
                    self.nclusters = clustering.cluster_centers_.shape[0]
                    """
                    clustering = AffinityPropagation(random_state=5).fit(self.user_pre_embed)
                    self.user_clusters = clustering.labels_
                    self.nclusters = clustering.cluster_centers_indices_.shape[0]
                    
                    
                    print('num cluster:',self.nclusters)
                    
                    #self.entity_embedding = torch.zeros(nentity, self.entity_dim * 2) # alpha and beta
                    self.relation_embedding = torch.zeros(self.nrelation+self.nclusters*2, self.relation_dim)
                    #self.entity_embedding[:self.nitems,:] = torch.from_numpy(self.item_pre_embed) 
                    #self.entity_embedding[self.nitems:,:] = torch.from_numpy(self.user_pre_embed) 
                    self.relation_embedding[:self.nrelation,:] = torch.from_numpy(self.rel_pre_embed)
                    self.relation_embedding[self.nrelation:self.nrelation+self.nclusters,:] = torch.from_numpy(self.rel_pre_embed[0]).repeat(self.nclusters,1)
                    self.relation_embedding[self.nrelation+self.nclusters:self.nrelation+self.nclusters*2,:] = torch.from_numpy(self.rel_pre_embed[1]).repeat(self.nclusters,1)

                    self.relation_embedding = nn.Parameter(self.relation_embedding)
                    #self.entity_embedding = nn.Parameter(self.entity_embedding)

                    print('relation embedding shape', self.relation_embedding.shape)
                    #print('entity_embedding embedding shape', self.entity_embedding.shape)
                else:
                    self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.relation_dim))
                    nn.init.uniform_(
                        tensor=self.relation_embedding, 
                        a=-self.embedding_range.item(), 
                        b=self.embedding_range.item()
                    )

            else:
                self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2)) # alpha and beta
                nn.init.uniform_(
                    tensor=self.entity_embedding, 
                    a=-self.embedding_range.item(), 
                    b=self.embedding_range.item()
                )
                self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.relation_dim))
                nn.init.uniform_(
                    tensor=self.relation_embedding, 
                    a=-self.embedding_range.item(), 
                    b=self.embedding_range.item()
                )
                
            self.entity_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings after relation projection are positive


        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding, 
                a=0., 
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)
            self.aggregate3p_net = VecAggregate3p(self.entity_dim)
        elif self.geo == 'vec':
            self.center_net = CenterIntersection(self.entity_dim)
            self.aggregate3p_net = VecAggregate3p(self.entity_dim)
        elif self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2, 
                                             self.relation_dim, 
                                             hidden_dim, 
                                             self.projection_regularizer, 
                                             num_layers)
            self.aggregate3p_net = BetaAggregate3p(self.entity_dim)

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        if self.geo == 'box':
            return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'vec':
            return self.forward_vec(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        
    def embed_query_box(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                if self.use_cuda:
                    offset_embedding = torch.zeros_like(embedding).cuda()
                else:
                    offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "box cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                    offset_embedding += self.func(r_offset_embedding)
                idx += 1
        else:
            
            embedding_3p_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[i], idx)
                if self.query_name_dict[query_structure[i]] == '1p':
                    embedding_1p = embedding
                else:
                    embedding_3p_list.append(embedding)
                embedding_3p_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            if len(embedding_3p_list) == 1:
                embedding_3p = embedding_3p_list[0]
            else:
                embedding_3p = self.aggregate3p_net(torch.stack(embedding_3p_list))
            embedding = self.center_net(torch.stack([embedding_1p, embedding_3p]))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))
            
            '''
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))
            '''

        return embedding, offset_embedding, idx

    def embed_query_vec(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_vec(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                idx += 1
        else:
            embedding_3p_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(queries, query_structure[i], idx)
                if self.query_name_dict[query_structure[i]] == '1p':
                    embedding_1p = embedding
                else:
                    embedding_3p_list.append(embedding)
                embedding_3p_list.append(embedding)
            if len(embedding_3p_list) == 1:
                embedding_3p = embedding_3p_list[0]
            else:
                embedding_3p = self.aggregate3p_net(torch.stack(embedding_3p_list))
            embedding = self.center_net(torch.stack([embedding_1p, embedding_3p]))

        return embedding, idx

    def embed_query_beta(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        # query_structure[-1] : e.g. ('r','r','r')
        # query_structure[0] : e.g. e
        all_relation_flag = True  # ('r','r','r')와 같이 모두 'r'로 이루어져 있으면 True, 아니면 False
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            #print('batch query: ',queries)
            if query_structure[0] == 'e':
                #print('user: ',queries[:, idx])
                users = queries[:, idx]
                embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n': # if: 'n'인 경우
                    assert (queries[:, idx] == -2).all()
                    embedding = 1./embedding  # 알파와 베타를 역수를 취함으로써 negation 표현
                else: # else: 'r'인 경우
                    if self.use_clustering:
                        device = queries[:, idx].device
                        converted_indexes = convert_relations(queries[:, idx].detach().cpu(), self.user_clusters[users.detach().cpu()-self.nitems], self.nrelation, self.nclusters)
                        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=torch.tensor(converted_indexes, device=device))

                    else:
                        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                        
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            #print(idx)
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else: # intersection 부분
            
            alpha_embedding_3p_list = []
            beta_embedding_3p_list = []
            #print('--------------------------intersection starts---------------------------')
            #print('length:',len(query_structure))
            for i in range(len(query_structure)):
                #print('@@@@@@@',query_structure[i],'@@@@@@@')
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[i], idx)
                if self.query_name_dict[query_structure[i]] == '1p':
                    alpha_embedding_1p = alpha_embedding
                    beta_embedding_1p = beta_embedding
                else:
                    alpha_embedding_3p_list.append(alpha_embedding) # alpha_embedding: (batch, dim)
                    beta_embedding_3p_list.append(beta_embedding)  
            #print('--------------------------intersection ends---------------------------')    
            if len(alpha_embedding_3p_list) == 1:
                alpha_embedding_3p = alpha_embedding_3p_list[0]
                beta_embedding_3p = beta_embedding_3p_list[0]
            else:
                alpha_embedding_3p, beta_embedding_3p = self.aggregate3p_net(torch.stack(alpha_embedding_3p_list), torch.stack(beta_embedding_3p_list))
                
            alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_1p, alpha_embedding_3p]), torch.stack([beta_embedding_1p, beta_embedding_3p]))

            """
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))
            """

        return alpha_embedding, beta_embedding, idx

    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding) # 모든 entitiy들의 distribution
        #print('entity_dist: ',entity_dist.sample().shape)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        # test step에는 positive sample 이 없고 모든 entity를 negative sample로 취급
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []  
        for query_structure in batch_queries_dict:
            alpha_embedding, beta_embedding, _ = self.embed_query_beta(batch_queries_dict[query_structure], 
                                                                           query_structure, 
                                                                           0)
            all_idxs.extend(batch_idxs_dict[query_structure])
            all_alpha_embeddings.append(alpha_embedding)
            all_beta_embeddings.append(beta_embedding)
            """
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query_beta(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                     query_structure), 
                                          self.transform_union_structure(query_structure), 
                                          0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_query_beta(batch_queries_dict[query_structure], 
                                                                           query_structure, 
                                                                           0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)
            """
        # positive embedding, negative embedding -> positive sample embedding, negative sample embedding
        # alpha embedding + beta embedding = all_dists -> query embedding
        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            #print('all_alpha_embeddings',all_alpha_embeddings.shape) [128,1, 64]
            #print('all_beta_embeddings',all_beta_embeddings.shape) [128,1, 64]
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)
            #print('all_dists',all_dists.batch_shape)
        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0]//2, 2, 1, -1)
            all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0]//2, 2, 1, -1)
            all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs] # positive samples for non-union queries in this batch
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_beta(positive_embedding, all_dists)
                #print('positive_logit',positive_logit.shape)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs] # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1))
                positive_union_logit = self.cal_logit_beta(positive_embedding, all_union_dists)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))
                negative_logit = self.cal_logit_beta(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)
            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1))
                negative_union_logit = self.cal_logit_beta(negative_embedding, all_union_dists)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs, all_alpha_embeddings, all_beta_embeddings, negative_embedding

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_logit_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure], 
                                                                             query_structure, 
                                                                             0)
            all_center_embeddings.append(center_embedding)
            all_offset_embeddings.append(offset_embedding)
            all_idxs.extend(batch_idxs_dict[query_structure])
            '''
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                    query_structure), 
                                         self.transform_union_structure(query_structure), 
                                         0)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure], 
                                                                             query_structure, 
                                                                             0)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])
            '''

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_box(positive_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(positive_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_box(negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_box(negative_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs, _, _, _
    
    def cal_logit_vec(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            center_embedding, _ = self.embed_query_vec(batch_queries_dict[query_structure], query_structure, 0)
            all_center_embeddings.append(center_embedding)
            all_idxs.extend(batch_idxs_dict[query_structure])
            '''
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_vec(self.transform_union_query(batch_queries_dict[query_structure], 
                                                                    query_structure), 
                                                                self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_vec(batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])
            '''

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_vec(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_vec(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_vec(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_vec(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs, _, _, _

    @staticmethod
    def train_step(model, optimizer, train_iterator_list, args, step):
        model.train()
        optimizer.zero_grad()
        
        train_iterator = random.choice(train_iterator_list)
        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        #print('positive_sample:  ',len(positive_sample))   #batch 400개씩 
        #print('negative_sample:  ',len(negative_sample))
        #print('batch_queries:  ',len(batch_queries))
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries): # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _, _, _, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

        # loss 계산
        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        l2_reg = torch.tensor(0.).to('cuda:0')
        for p in model.parameters():
            if p.requires_grad:
                l2_reg += torch.norm(p)
                
        l2_reg = l2_reg*args.l2_lambda
        loss = (positive_sample_loss + negative_sample_loss)/2 + l2_reg
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss,
            'negative_sample_loss': negative_sample_loss,
            'loss': loss.item(),
        }
        
        """
        self.center_net = BetaIntersection(self.entity_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2, 
                                             self.relation_dim, 
                                             hidden_dim, 
                                             self.projection_regularizer, 
                                             num_layers)
        self.aggregate3p_net = BetaAggregate3p(self.entity_dim)
        """
            
        
        return log

    @staticmethod
    def test_step(model, answers, total_train_interactions, args, test_dataloader, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)
        
        with torch.no_grad():
            # test query 수 만큼 iterate
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                # 모든 entity는 negative sample임. -> negative sample을 바꾸면 원하는 entity에 대해서만 evaluate 가능할듯..??
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs, alpha_embeddings, beta_embeddings, entity_embeddings = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                # entity_embeddings : beta embedding of entities
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]

                query = queries_unflatten[0]
                if isinstance(query[0],int): # path queries
                    user = query[0]
                else: # intersection queries
                    user = query[0][0]
                
                if user not in answers:
                    print('user',user)
                    continue
                answer = list(answers[user])
                
                # 나중에 이 코드 고치자!!!!
                for ans in answer:
                    ans -= args.start_item     

                num_answer = len(answer)
                answer = set(answer)
                
                ######################################
                negative_logit = negative_logit[:,args.start_item:args.start_item+args.nitems]
                # cold start user
                if user in total_train_interactions:
                    negative_logit[:,total_train_interactions[user]] = float("-inf")
                
                ######################################

                argsort = torch.argsort(negative_logit, dim=1, descending=True)[0].detach().cpu().tolist()
                """
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                #########################################

                if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range

                    if args.cuda: 
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nitems).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1).cuda()
                                                   ) # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nitems).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1)
                                                   ) # achieve the ranking of all entities
                ########################################
                                                   
                # 나중에 for문 없애기
                #for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):


                #################################
                cur_ranking = ranking[0, list(answer)]
                cur_ranking, indices = torch.sort(cur_ranking)
                #######################################
                """
                #K = args.K 
                #cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                #mrr = torch.mean(1./cur_ranking).item()
                K_list = [5,10,20,50,100]
                hits = []
                precision = []
                recall = []
                ndcg = []
                
                r = []
                max_K = max(K_list)
                for i in argsort[:max_K]:
                    if i in answer:
                        r.append(1)
                    else:
                        r.append(0)

                for K in K_list:
                    ndcg.append(ndcg_at_k(r, K))
                    hit_num = len(set(argsort[:K]) & answer)
                    if hit_num > 0: 
                        hits.append(1) 
                    else: 
                        hits.append(0)
                    precision.append(hit_num / K)
                    recall.append(hit_num / num_answer)
                
                """
                    if torch.sum((cur_ranking < K).to(torch.float)).item() == 0:
                        #hits = 0
                        hits.append(0)
                    else:
                        #hits = 1
                        hits.append(1)
                    precision.append(torch.sum((cur_ranking < K).to(torch.float)).item() / K)
                    recall.append(torch.mean((cur_ranking < K).to(torch.float)).item())
                    dcg = torch.sum( ((cur_ranking < K).to(torch.float))/(torch.log2(cur_ranking+2).to(torch.float)) )
                    #print('dcg',((cur_ranking < K).to(torch.float))/(torch.log2(cur_ranking+2).to(torch.float)))
                    if K <= num_answer:
                        if args.cuda:
                            answer_list = torch.arange(K).to(torch.float).cuda() 
                        else:
                            answer_list = torch.arange(K).to(torch.float)
                        idcg = torch.sum( (torch.ones(K, dtype=torch.float).to(cur_ranking.device)) / (torch.log2(answer_list+2).to(torch.float)) )
                        #print('idcg:',(torch.ones(K, dtype=torch.float).to(cur_ranking.device)) / (torch.log2(answer_list+2).to(torch.float)))
                    else:
                        if args.cuda:
                            answer_list = torch.arange(num_answer).to(torch.float).cuda() 
                        else:
                            answer_list = torch.arange(num_answer).to(torch.float)
                        idcg = torch.sum( (torch.ones(num_answer, dtype=torch.float).to(cur_ranking.device)) / (torch.log2(answer_list+2).to(torch.float)) )
                        #print('idcg:',(torch.ones(num_answer, dtype=torch.float).to(cur_ranking.device)) / (torch.log2(answer_list+2).to(torch.float)))
                    ndcg.append(float(dcg/idcg))
                    #precision = torch.sum((cur_ranking < K).to(torch.float)).item() / K
                    #recall = torch.mean((cur_ranking < K).to(torch.float)).item()
                    #dcg = torch.sum( ((cur_ranking < K).to(torch.float))/(torch.log2(cur_ranking+2).to(torch.float)) )
                    #idcg = torch.sum( (torch.ones_like(cur_ranking)) / (torch.log2(answer_list+2).to(torch.float)) )
                    #ndcg = float(dcg/idcg)
                    """
                """
                logs[user].append({
                #    'MRR': mrr,
                    'HITS_'+str(K): hits,
                    'PRECISION_'+str(K): precision,
                    'RECALL_'+str(K): recall,
                    'nDCG_'+str(K): ndcg,
                })
                """
                logs[user].append({
                #    'MRR': mrr,
                    'HITS': hits,
                    'PRECISION': precision,
                    'RECALL': recall,
                    'nDCG': ndcg,
                })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for user in logs:
            for metric in logs[user][0].keys():
                #metrics[user][metric] = sum([log[metric] for log in logs[user]])/len(logs[user])
                metrics[user][metric] = np.sum([[at_K for at_K in log[metric]] for log in logs[user]], axis=0) / len(logs[user])
            #metrics[user]['num_queries'] = len(logs[user])
        return metrics
