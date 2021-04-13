"""
The overall entity set is divided into three mutually disjoint subsets.
FewNERD(INTRA):
    Assign all the fine-grained entity types belonging to People, MISC, Art, 
    Product in training set, all the fine-grained entity types belonging to
    Event, Building in development set, and all the fine-grained entity types
    belonging to ORG, LOC in test set, respectively.

FewNERD(INTER):
    Although all the fine-grained entity types are mutually disjoint, the
    coarse-grained types are shared.
    Assign 60% fine-grained types of all the 9 coarse-grained types in 
    training set, 20% in development set, and 20% in test set, respectively.
"""
import os
import numpy as np
import copy
from batch_encoder import BertEncoder
import torch
import torch.utils.data as data
from transformers import BertTokenizer, BertModel
import pandas as pd
import random


def read_data(path):
    data = []
    with open(path) as f:
        lines = [line.rstrip() for line in f]
    sent = []
    for line in lines:
        if line == "":
            data.append(sent)
            sent = []
        else:
            word, label = line.split("\t")
            sent.append((word, label))
    return data

def update(D, N, K):
    return len(D) <= N and all(val <= 2*K for val in D.values())


    

class Sample:
    def __init__(self, raw_sent):
        self.words = [t[0] for t in raw_sent]
        self.labels = [t[1] for t in raw_sent]
        self.entity_count = {}
        
    def __count_entities__(self):
        cur_label = self.labels[0]
        for next_label in self.labels[1:]:
            if next_label == cur_label: continue
            else:
                if cur_label != 'O':
                    if cur_label in self.entity_count:
                        self.entity_count[cur_label] += 1
                    else:
                        self.entity_count[cur_label] = 1
                cur_label = next_label
        if cur_label != 'O':
            if cur_label in self.entity_count:
                self.entity_count[cur_label] += 1
            else:
                self.entity_count[cur_label] = 1
                
    def get_entity_count(self):
        if self.entity_count:
            return self.entity_count
        else:
            self.__count_entities__()
            return self.entity_count
            
        
            


class FewNERDataset(data.Dataset):
    
    def __init__(self, filepath, encoder, N, K, Q):
        self.path = filepath
        if not os.path.exists(self.path):
            print("[ERROR] Data file does not exist!")
        self.data = read_data(self.path)
        self.samples = []
        all_classes = []
        for sent in self.data:
            cur_sample = Sample(sent)
            self.samples.append(cur_sample)
            all_classes += cur_sample.labels
        self.classes = list(set(all_classes))
        self.classes.remove('O')
        
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder
        self.max_len = encoder.max_len
    
    def __len__(self):
        return 1000000000 
    
    def __additem__(self, d, tokens_ts, labels_lst, 
                    atn_masks_ts, true_masks_ts, entities_lst):
        #token_label_tup = [(tokens_list[i], labels[i]) 
        #                   for i in range(len(labels))]
        d['tokens'] += tokens_ts
        d['labels'] += labels_lst
        d['atn_masks'] += atn_masks_ts
        d['true_masks'] += true_masks_ts
        d['entities'] += entities_lst
    
    
    def update(D, N, K):
        return len(D) <= N and all(val <= 2*K for val in D.values())
    
    def get_candidates(self, target_classes):
        candidates = []
        for idx, sample in enumerate(self.samples):
            labels = set(sample.get_entity_count().keys())
            if len(labels) != 0 and labels.issubset(set(target_classes)):
                candidates.append(idx)
                
        return candidates
            
    
    def __getitem__(self, idx):
        support_set = {'tokens': [], 'labels': [], 'atn_masks': [], 
                       'true_masks': [], 'entities':[]}
        query_set = {'tokens': [], 'labels': [], 'atn_masks': [], 
                     'true_masks': [], 'entities': []}
        
        
        # sample N classes
        target_classes = np.random.choice(self.classes, self.N, replace=False).tolist()
        #print(target_classes)
        candidates = self.get_candidates(target_classes)
        # sample support set
        support = []
        support_ind = []
        selected = {cls:0 for cls in target_classes}

        while np.any(np.array(list(selected.values())) < self.K):
            idx = np.random.choice(candidates, 1)[0]
            if idx not in support_ind:
                sample = self.samples[idx]
                entity_count = sample.get_entity_count()
                
                C = copy.deepcopy(selected)
                for cls, count in entity_count.items():
                    if cls in C:
                        C[cls] += count
                    else:
                        print('warning: candidates contain non-target class')
                if update(C, self.N, self.K):
                    support.append(sample)
                    support_ind.append(idx)
                    selected = copy.deepcopy(C)
        #print(selected)
        # sample query set
        query = []
        query_ind = []
        selected = {cls:0 for cls in target_classes}
        # take examples in support out of candidates for query
        available_ind = list(set(candidates)-set(support_ind))
        while np.any(np.array(list(selected.values())) < self.Q):
            idx = np.random.choice(available_ind, 1)[0]
            if idx not in support_ind:
                sample = self.samples[idx]
                entity_count = sample.get_entity_count()
                
                C = copy.deepcopy(selected)
                for cls, count in entity_count.items():
                    if cls in C:
                        C[cls] += count
                    else:
                        print('warning: candidates contain non-target class')
                if update(C, self.N, self.Q):
                    query.append(sample)
                    query_ind.append(idx)
                    selected = copy.deepcopy(C)        
        #print(selected)
        # 'O' is not counted in nway, so needs to be added manually
        nway = ['O'] + target_classes
        # match classes with indices
        int2cls = {ind: cls for ind,cls in enumerate(nway)}
        cls2int = {cls: ind for ind,cls in enumerate(nway)}
        
        #print(support_ind)
        #print(query_ind)
        for sent in support:
            raw_tokens = sent.words
            raw_labels = [cls2int[lab] for lab in sent.labels]
            #print(raw_tokens)
            #print(raw_labels)
            tokens_lst, labels_lst, atn_masks_lst, \
            true_masks_lst, entities_lst = \
            self.encoder.tokenize(raw_tokens, raw_labels)
            tokens_ts = [torch.tensor(sublist).long() for sublist in tokens_lst]
            atn_masks_ts = [torch.tensor(sublist).long() for sublist in atn_masks_lst]
            true_masks_ts = [torch.tensor(sublist).long() for sublist in true_masks_lst]
            self.__additem__(support_set, tokens_ts, labels_lst,
                             atn_masks_ts, true_masks_ts, entities_lst)
            #print(support_set)
        
        #support_clusters = get_clusters(support_set, self.N)
        
        for sent in query:
            raw_tokens = sent.words
            raw_labels = [cls2int[lab] for lab in sent.labels]
            tokens_lst, labels_lst, atn_masks_lst, \
            true_masks_lst, entities_lst = \
            self.encoder.tokenize(raw_tokens, raw_labels)
            tokens_ts = torch.tensor(tokens_lst).long()
            atn_masks_ts = torch.tensor(atn_masks_lst).long()
            true_masks_ts = torch.tensor(true_masks_lst).long()
            self.__additem__(query_set, tokens_ts, labels_lst,
                             atn_masks_ts, true_masks_ts, entities_lst)
        
        # convert lists into Pytorch tensors
        support_set['tokens'] = torch.stack(support_set['tokens'], dim=0)
        support_set['atn_masks'] = torch.stack(support_set['atn_masks'], dim=0)
        support_set['true_masks'] = torch.stack(support_set['true_masks'], dim=0)
        query_set['tokens'] = torch.stack(query_set['tokens'], dim=0)
        query_set['atn_masks'] = torch.stack(query_set['atn_masks'], dim=0)
        query_set['true_masks'] = torch.stack(query_set['true_masks'], dim=0)
        query_set['int2cls'] = int2cls
        
        support_set['size'] = [len(support_set['labels'])]
        query_set['size'] = [len(query_set['labels'])]
        
        return support_set, query_set
        
    def sample_for_plot(self, K):
        # sample N classes
        target_classes = self.classes
        #print(target_classes)
        N = len(self.classes)
        candidates = [i for i in range(len(self.samples))]
        # sample
        mysamples = []
        mysamples_ind = []
        selected = {cls:0 for cls in target_classes}

        i = 0
        while i < 10000 and np.any(np.array(list(selected.values())) < K):
            idx = np.random.choice(candidates, 1)[0]
            if idx not in mysamples_ind:
                sample = self.samples[idx]
                entity_count = sample.get_entity_count()
                
                C = copy.deepcopy(selected)
                for cls, count in entity_count.items():
                    if cls in C:
                        C[cls] += count
                    else:
                        print('warning: candidates contain non-target class')
                if update(C, N, K):
                    mysamples.append(sample)
                    mysamples_ind.append(idx)
                    selected = copy.deepcopy(C)
                i += 1
        
        print(len(mysamples))
        print('i: '+str(i))
        print(selected)
        data_lst = []
        for sample in mysamples:
            data_lst.append(list(zip(sample.words, sample.labels)))
        assert(data_lst != [])
        return data_lst

def collate_fn(data):
    batch_support = {'tokens': [], 'labels': [], 'atn_masks': [], 
                     'true_masks': [], 'size': [], 'entities': []}
    batch_query = {'tokens': [], 'labels': [], 'atn_masks': [], 
                   'true_masks': [], 'size': [], 'int2cls': [], 'entities': []}
    support_sets, query_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in batch_support:
            batch_support[k] += support_sets[i][k]
        for k in batch_query:
            if k == 'int2cls':
                batch_query[k].append(query_sets[i][k])
            else: 
                batch_query[k] += query_sets[i][k]
    no_ts = ['labels', 'size', 'entities']
    for k in batch_support:
        if k not in no_ts:
            batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        if k not in no_ts and k!= 'int2cls':
            batch_query[k] = torch.stack(batch_query[k], 0)
    batch_support['labels'] = [torch.tensor(lab_lst).long() for lab_lst in batch_support['labels']]
    batch_support['entities'] = [ent_lst for ent_lst in batch_support['entities']]
    batch_query['labels'] = [torch.tensor(lab_lst).long() for lab_lst in batch_query['labels']]
    batch_query['entities'] = [ent_lst for ent_lst in batch_support['entities']]
    return batch_support, batch_query


'''     
def get_loader(filepath, encoder, N, K, Q, batch_size, max_len, num_workers=8, 
               collate_fn=collate_fn):
    dataset = FewNERDataset(filepath, encoder, N, K, Q)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn = collate_fn)
    return iter(data_loader)

'''
    
def get_loader(filepath, encoder, N, K, Q, batch_size, max_len, num_workers=8, 
               collate_fn=collate_fn):
    dataset = FewNERDataset(filepath, encoder, N, K, Q)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            collate_fn = collate_fn)
    return iter(data_loader)

