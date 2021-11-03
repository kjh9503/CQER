#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset, IterableDataset
import random
from kgreasoning.util import list2tuple, tuple2list, flatten

class TrainDataset(Dataset):
    def __init__(self, queries, nentity, nrelation, negative_sample_size, answer):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        """
        if len(self.answer[query]) == 0:
            print('wrong!!!!')
            print(query)
        """
        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))
        subsampling_weight = self.count[query] 
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample, 
                self.answer[query], 
                assume_unique=True, 
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        #print('negative_sample: ',negative_sample)
        positive_sample = torch.LongTensor([tail])
        #print('positive sample: ',positive_sample)
        #print('subsampling_weight: ',subsampling_weight)
        #print('flatten(query): ',flatten(query))
        #print('')
    
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        return positive_sample, negative_sample, subsample_weight, query, query_structure
    
    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count
"""
class TestDataset(Dataset):
    def __init__(self, queries, nentity, nrelation):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        negative_sample = torch.LongTensor(range(self.nentity))
        return negative_sample, flatten(query), query, query_structure
    
    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]
        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        return negative_sample, query, query_unflatten, query_structure
"""
        
class TestDataset(IterableDataset):
    def __init__(self, test_queries, nentity, nrelation):
        self.path_samples_num = 100
        self.test_queries = test_queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.query_dict = {
            '1p' : ('e',('r',)),
            '2p' : ('e', ('r', 'r')),
            '3p' : ('e', ('r', 'r', 'r')),
        }
        
    def __len__(self):
        return len(self.test_queries)
        
    def __iter__(self):
        for user in self.test_queries:
            if len(self.test_queries[user]) > 100:
                batch = random.sample(self.test_queries[user], self.path_samples_num)
                query = tuple(batch)
                query_structure = tuple([('e',('r',))]*len(batch))
            elif len(self.test_queries[user]) == 1:
                query = self.test_queries[user][0]
                query_structure = ('e',('r',))
            else:
                batch = self.test_queries[user]
                query = tuple(batch)
                query_structure = tuple([('e',('r',))]*len(batch))
            negative_sample = torch.LongTensor(range(self.nentity))
            yield negative_sample, flatten(query), query, query_structure
    
    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]
        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        return negative_sample, query, query_unflatten, query_structure
    
    @staticmethod
    def worker_init_fn(_):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        worker_id = worker_info.id
        split_size = len(dataset.test_queries) // worker_info.num_workers
        dataset.test_queries = dict(list(dataset.test_queries.items())[worker_id * split_size: (worker_id + 1) * split_size])    
    
class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data