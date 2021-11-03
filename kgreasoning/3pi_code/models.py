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
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os
import json

def Identity(x):
    return x

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
                 geo, pretrain_embedding_dir, test_batch_size=1,
                 box_mode=None, use_cuda=False, use_pretrain=0, 
                 query_name_dict=None, beta_mode=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.nitems = nitems
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.user_pretrain = use_pretrain # pretrain
        self.pretrain_embedding_dir = pretrain_embedding_dir
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
        elif self.geo == 'vec':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # center for entities
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        elif self.geo == 'beta':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2)) # alpha and beta
            self.entity_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings after relation projection are positive
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
            
            # transr 
            if self.user_pretrain == 1:
                pretrain_path = self.pretrain_embedding_dir
                with open(pretrain_path, 'r') as f:
                    pretrain_data = json.load(f)
                self.entity_pre_embed = np.array(pretrain_data['ent_embeddings.weight'])
                self.relation_pre_embed = np.array(pretrain_data['rel_embeddings.weight'])
                
                assert len(self.entity_pre_embed) == self.nentity
                assert len(self.entity_pre_embed[0]) == self.entity_dim
                assert len(self.relation_pre_embed) == self.nrelation
                assert len(self.relation_pre_embed[0]) == self.entity_dim
                
                entity_pre_embed_scale = (self.entity_pre_embed - self.entity_pre_embed.min()) / (self.entity_pre_embed.max() - self.entity_pre_embed.min())
                # alpha + beta = 10
                alpha = entity_pre_embed_scale*10
                beta = 10 - alpha
                self.entity_embedding = nn.Parameter(torch.from_numpy(np.concatenate((alpha, beta), axis=1)).float())
                print('entity shape:',self.entity_embedding.shape)
                self.relation_embedding = nn.Parameter(torch.from_numpy(self.relation_pre_embed).float())
                print('relation shape',self.relation_embedding.shape)

                
            """
            # mf.npz
            if self.user_pretrain == 1:
                pretrain_path = self.pretrain_embedding_dir
                pretrain_data = np.load(pretrain_path)
                self.item_pre_embed = pretrain_data['item_embed']
                self.user_pre_embed = pretrain_data['user_embed']
                
                #assert self.user_pre_embed.shape[0] == self.n_users
                assert self.item_pre_embed.shape[0] == self.nitems
                assert self.user_pre_embed.shape[1] == self.entity_dim
                assert self.item_pre_embed.shape[1] == self.entity_dim
                
                num_items = self.item_pre_embed.shape[0]
                num_users = self.user_pre_embed.shape[0] 
                self.nusers = num_users # 나중에 nusers는 stats.txt에서 받아오는걸로 수정하자
                dim = self.user_pre_embed.shape[1]
                concat = np.concatenate((self.item_pre_embed, self.user_pre_embed), axis=0)
                concat_scale = (concat - concat.min()) / (concat.max() - concat.min())

                # alpha + beta = 10
                alpha = concat_scale*10
                beta = 10 - alpha
                self.item_pre_embed = np.concatenate((alpha[:num_items:,:], beta[:num_items,:]), axis=1)
                self.user_pre_embed = np.concatenate((alpha[-num_users:,:], beta[-num_users:, :]), axis=1)
                
                # item이랑 user 합쳐서 0~1 사이로 normalize -> a,b 할당
                self.entity_embedding[:num_items,:] = torch.from_numpy(self.item_pre_embed) 
                self.entity_embedding[-num_users:,:] = torch.from_numpy(self.user_pre_embed) 
                self.entity_embedding = nn.Parameter(self.entity_embedding)
            """
                
        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding, 
                a=0., 
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)
        elif self.geo == 'vec':
            self.center_net = CenterIntersection(self.entity_dim)
        elif self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2, 
                                             self.relation_dim, 
                                             hidden_dim, 
                                             self.projection_regularizer, 
                                             num_layers)

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
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))

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
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

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
            #print('user: ',queries[:, idx])
            if query_structure[0] == 'e':
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
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    #print('r_embedding: ',r_embedding.shape)
                    #print('embedding: ',embedding.shape)
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            #print(idx)
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else: # intersection 부분
            alpha_embedding_list = []
            beta_embedding_list = []
            #print('--------------------------intersection starts---------------------------')
            for i in range(len(query_structure)):
                #print('@@@@@@@',query_structure[i],'@@@@@@@')
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            #print('--------------------------intersection ends---------------------------')

            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))
            
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
        # positive embedding, negative embedding -> positive sample embedding, negative sample embedding
        # alpha embedding + beta embedding = all_dists -> query embedding
        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)
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

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs
    
    def cal_logit_vec(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
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

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()
        
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

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }

        return log

    @staticmethod
    def test_step(model, answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
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
                negative_logit = negative_logit[:,args.start_item:args.start_item+args.nitems]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.ntiems).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1).cuda()
                                                   ) # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nitems).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1)
                                                   ) # achieve the ranking of all entities

                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    
                    if isinstance(query[0],int): # path queries
                        user = query[0]
                    else: # intersection queries
                        user = query[0][0]
                    if user not in answers:
                        continue
                    answer = list(answers[user])
                    
                    # 나중에 이 코드 고치자!!!!
                    for ans in answer:
                        ans -= args.start_item
                    
                    num_answer = len(answer)
                    answer = set(answer)
                    cur_ranking = ranking[idx, list(answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)

                    if args.cuda:
                        answer_list = torch.arange(num_answer).to(torch.float).cuda() 
                    else:
                        answer_list = torch.arange(num_answer).to(torch.float) 
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting

                    mrr = torch.mean(1./cur_ranking).item()
                    if torch.sum((cur_ranking <= 20).to(torch.float)).item() == 0:
                        h20 = 0
                    else:
                        h20 = 1
                    pre20 = torch.sum((cur_ranking <= 20).to(torch.float)).item() / 20.0
                    re20 = torch.mean((cur_ranking <= 20).to(torch.float)).item()
                    dcg20 = torch.sum( ((cur_ranking <= 20).to(torch.float))/(torch.log2(cur_ranking+2).to(torch.float)) )
                    idcg20 = torch.sum( (torch.ones_like(cur_ranking)) / (torch.log2(answer_list+2).to(torch.float)) )
                    ndcg20 = float(dcg20/idcg20)
                    
                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS20': h20,
                        'PRECISION20': pre20,
                        'RECALL20': re20,
                        'nDCG20': ndcg20,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1
                

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])
            
        return metrics