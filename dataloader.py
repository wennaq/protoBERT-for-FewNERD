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
os.chdir('/Users/qinwenna/Desktop/THU-Research/protoBERT')
import numpy as np
import copy
from batch_encoder import BertEncoder
import torch
import torch.utils.data as data
from transformers import BertTokenizer, BertModel
import pandas as pd


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

# select a sample: N-way K~2K examples
def sample_support(mydata, N, K):
    '''
    Parameters
    ----------
    mydata : list
        dataset of sentences (list of tuples (word, label)).
    N : int
        number of classes per episode.
    K : int
        number of examples per class in the support set.

    Returns
    -------
    sample : list
        sample of sentences.
    sample_ind: list
        indices of sentences in the selected sample.
    selected: dictionary
        stores the counts of examples in each class selected.

    '''
    sample = []
    sample_ind = []
    selected = dict()
    size = len(mydata)
    perm = np.random.permutation(size)
    
    i = 0
    while len(selected) < N or np.all(np.array(list(selected.values())) < K):
        #print(i)
        cur_idx = perm[i]
        sent = mydata[cur_idx]
        i += 1
        C = copy.deepcopy(selected)
        
        for tup in sent:
            word, label = tup
            # “O" is not counted as one of the N-way
            if label == "O": continue
            # obtain the counts after this update
            if label in C.keys():
                C[label] += 1
            else:
                C[label] = 1
        # decide whether to update or not   
        if update(C, N, K):
            sample.append(sent)
            sample_ind.append(cur_idx)
            selected = copy.deepcopy(C)            
        else:
            continue
    
    return sample, sample_ind, selected


def sample_query(mydata, support_ind, support_dict, N, Q):
    '''
    Sample query set.

    Parameters
    ----------
    mydata : list
        dataset of sentences.
    support_ind: list
        indices of examples selected into the support set.
    support_dict: dict
          stores counts of classes in the support set.
    N : int
        number of classes for each episode.
    Q : int
        number of examples per class in the query set.

    Returns
    -------
    sample : list
        sample of sentences.
    sample_ind: list
        indices of sentences selected into the query set.
    selected: dictionary
        stores the counts of examples in each class selected.

    '''
    size = len(mydata)
    all_ind = set(np.arange(size))
    available_ind = np.array(list(all_ind - set(support_ind)))
    perm = np.random.permutation(available_ind)
    assert(len(perm) == size - len(support_ind))
    
    sample = []
    sample_ind = []
    selected = dict()
    
    i = 0
    while len(selected) < N or np.all(np.array(list(selected.values())) < Q):
        #print(i)
        cur_idx = perm[i]
        sent = mydata[cur_idx]
        i += 1
        C = copy.deepcopy(selected)
        seen = True
        
        for tup in sent:
            word, label = tup
            # “O" is not counted as one of the N-way
            if label == "O": continue
            # avoid examples with word labels unseen in support set
            elif label not in support_dict.keys(): 
                seen = False
                break
            else:
                # obtain the counts after this update
                if label in C.keys():
                    C[label] += 1
                else:
                    C[label] = 1
                    
        # decide whether to update or not   
        if seen and update(C, N, Q):
            sample.append(sent)
            sample_ind.append(cur_idx)
            selected = copy.deepcopy(C)            
        else:
            continue
        
        # check if the query set contains only entity types seen in support
        assert(set(selected.keys()).issubset(set(support_dict.keys())))
        
    return sample, sample_ind, selected

def get_clusters(d, N):
    labels = d['labels']
    batch_size = len(labels)
    # initialize clusters (need to include 'O', which is excluded from nway)
    clusters = {i: [[] for j in range(batch_size)] for i in range(N+1)}
    # iterate over sentences
    for i in range(batch_size):
        # iterate over the label for each token in the sentence
        cur_sent = labels[i]
        for j in range(len(cur_sent)):
            cur_label = cur_sent[j]
            # '[CLS]' OR '[SEP]'
            if cur_label == -1:
                continue
            # paddings, end of the sentence
            elif cur_label == -2:
                break
            else:
                clusters[cur_label][i].append(j)
                
    return clusters
    

class FewNERD(data.Dataset):
    
    def __init__(self, name, encoder, N, K, Q, root):
        self.root = root
        path = os.path.join(root, name+'.txt')
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
        self.data = read_data(path)
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder
    
    def __len__(self):
        return len(self.data) # number of sentence
    
    def __additem__(self, d, tokens_tensor, masks_tensor, 
                    labels, endpos, true):
        #token_label_tup = [(tokens_list[i], labels[i]) 
        #                   for i in range(len(labels))]
        d['tokens'].append(tokens_tensor)
        d['masks'].append(masks_tensor)
        d['labels'].append(labels)
        d['end'].append(endpos)
        d['true'] += true
        
    def __getitem__(self, idx):
        support_set = {'tokens': [], 'masks': [], 'labels':[], 
                       'end': [], 'true':[]}
        query_set = {'tokens': [], 'masks': [], 'labels':[], 
                     'end': [], 'true':[]}
        
        support_spl, support_spl_ind, support_dict = sample_support(self.data,
                                                                    self.N,
                                                                    self.K)
        query_spl, query_spl_ind, query_dict = sample_query(self.data,
                                                            support_spl_ind,
                                                            support_dict,
                                                            self.N,
                                                            self.Q)
        nway = list(support_dict.keys())
        # 'O' is not counted in nway, so needs to be added manually
        nway.append('O')
        # match classes with indices
        int2cls = {v: k for v,k in enumerate(nway)}
        cls2int = {k: v for v,k in enumerate(nway)}
        #print(cls2int)

        
        for sent in support_spl:
            raw_tokens = [t[0] for t in sent]
            raw_labels = [cls2int[t[1]] for t in sent]
            print(raw_tokens)
            print(raw_labels)
            tokens_tensor, masks_tensor, labels, endpos, true = \
            self.encoder.tokenize(raw_tokens, raw_labels)
            self.__additem__(support_set, tokens_tensor, masks_tensor, 
                             labels, endpos, true)
        
        support_clusters = get_clusters(support_set, self.N)
        
        for sent in query_spl:
            raw_tokens = [t[0] for t in sent]
            raw_labels = [cls2int[t[1]] for t in sent]
            tokens_tensor, masks_tensor, labels, endpos, true = \
            self.encoder.tokenize(raw_tokens,raw_labels)
            self.__additem__(query_set, tokens_tensor, masks_tensor, 
                             labels, endpos, true)
        
        # convert lists into Pytorch tensors
        support_set['tokens'] = torch.stack(support_set['tokens'], dim=0)
        support_set['masks'] = torch.stack(support_set['masks'], dim=0)
        support_set['true'] = torch.tensor(support_set['true'])
        query_set['tokens'] = torch.stack(query_set['tokens'], dim=0)
        query_set['masks'] = torch.stack(query_set['masks'], dim=0)
        query_set['true'] = torch.tensor(query_set['true'])

            
        return support_set, query_set, int2cls, support_clusters
    
            
        
        
    

