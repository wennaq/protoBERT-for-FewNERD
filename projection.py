#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:43:52 2021

@author: qinwenna
"""
import os
import torch
import numpy as np
import argparse
import sys
import pickle
from sklearn.manifold import TSNE
from framework import FewShotNERFramework
from proto import Proto
from batch_encoder import BertEncoder
from dataloader import FewNERDataset
from dataloader import read_data
from transformers import BertTokenizer, BertModel


def get_all_classes(data):
    all_classes = []
    for sent in data:
        for (word, label) in sent:
            if label == 'O':
                continue
            else:
                all_classes.append(label)
    # make sure class 'O' is at the beginning, i.e. index 0
    all_classes = ['O']+list(set(all_classes))
    return all_classes

def coarsify(data):
    all_coarse_classes = []
    coarse_label_data = []
    for sent in data:
        coarse_sent = []
        for (word, label) in sent:
            if label == 'O': 
                coarse_sent.append((word, label))
            else:
                coarse, fine = label.split('-')
                all_coarse_classes.append(coarse)
                coarse_sent.append((word, coarse))
        coarse_label_data.append(coarse_sent)
    # make sure class 'O' is at the beginning, i.e. index 0
    coarse_cls = ['O']+list(set(all_coarse_classes))
    return coarse_label_data, coarse_cls

def tokenize(tokenizer,raw_tokens, raw_labels, max_len):
    '''
    Parameters
    ----------
    sent : list
        list of (token, label).
    tokenizer : obj
        tokenizer object to convert text 
        into BERT-readable tokens and ids.

    Returns
    -------
    tokenized_text: list of tokens
    tokens_tensor: torch tensor with token ids.
    segments_tensors: torch tensor with segments_ids

    '''
    
    raw_tokens = [token.lower() for token in raw_tokens]
    tokens = []
    labels = []
    for i in range(len(raw_tokens)):
        tok = raw_tokens[i]
        tokenized = tokenizer.tokenize(tok)
        tokens += tokenized
        
        lab = raw_labels[i]
        labels += [lab] * len(tokenized)
        
    assert(len(tokens) == len(labels))
    
    tokens_lst = []
    labels_lst = []
    # split the sentence if exceeding max_len
    while len(tokens) > (max_len - 2):
        split_pos = max_len - 2
        # make sure that the same word is in a single sentence
        while tokens[split_pos].startswith('#'):
            split_pos -= 1
            assert(split_pos > 0)

        tokens_lst.append(tokens[:split_pos])
        tokens = tokens[split_pos:]
        labels_lst.append(labels[:split_pos])
        labels = labels[split_pos:]

    if tokens:
        tokens_lst.append(tokens)
    if labels:
        labels_lst.append(labels)
    
    indexed_tokens_lst =[]
    attention_masks_lst = []
    true_masks_lst = []
    # add special tokens to each sentence
    for i in range(len(tokens_lst)):
        sent = ['[CLS]'] + tokens_lst[i] + ['[SEP]']
        indexed_tokens = tokenizer.convert_tokens_to_ids(sent)
        # padding
        while len(indexed_tokens) < max_len:
            indexed_tokens.append(0)
            
        attention_masks = np.zeros((max_len), dtype=np.int32)
        attention_masks[:len(sent)] = 1
        true_masks = np.zeros((max_len), dtype=np.int32)
        true_masks[1:(len(sent)-1)] = 1
        
        indexed_tokens_lst.append(indexed_tokens)
        attention_masks_lst.append(attention_masks)
        true_masks_lst.append(true_masks)

    
    return indexed_tokens_lst, labels_lst, \
           attention_masks_lst, true_masks_lst

def get_data_dict(data, cls2int, tokenizer, max_len):
    d = {'tokens': [], 'labels': [], 'atn_masks': [], 'true_masks': []}
    for sent in data:
        raw_tokens, raw_labels = list(zip(*sent))
        rawint_labels = [cls2int[lab] for lab in raw_labels]
        tokens_lst, labels_lst, atn_masks_lst, true_masks_lst = \
        tokenize(tokenizer, raw_tokens, rawint_labels, max_len)
        tokens_ts = [torch.tensor(sublist).long() for sublist in tokens_lst]
        atn_masks_ts = [torch.tensor(sublist).long() for sublist in atn_masks_lst]
        true_masks_ts = [torch.tensor(sublist).long() for sublist in true_masks_lst]
        d['tokens'] += tokens_ts
        d['labels'] += labels_lst
        d['atn_masks'] += atn_masks_ts
        d['true_masks'] += true_masks_ts
    # convert lists into Pytorch tensors
    d['tokens'] = torch.stack(d['tokens'], dim=0)
    d['atn_masks'] = torch.stack(d['atn_masks'], dim=0)
    d['true_masks'] = torch.stack(d['true_masks'], dim=0)
    
    return d
    

def wrapper(data, tokenizer, max_len, coarse):
    '''

    Parameters
    ----------
    data : 2d list
        a list of lists(sentences) of (word, label) pairs

    Returns
    -------
    None.

    '''
    if coarse:
        data, classes = coarsify(data)
    else:
        classes = get_all_classes(data)
    # note that 'O' is at the 0th position, thus assigned to 0
    cls2int = {cls: ind for ind,cls in enumerate(classes)}
    int2cls = {ind: cls for ind,cls in enumerate(classes)}

    d = get_data_dict(data, cls2int, tokenizer, max_len)

    d['int2cls'] = int2cls
    
    return d
        

def load_ckpt(ckpt_path):
    '''
    ckpt: Path of the checkpoint
    return: Checkpoint dict
    '''
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        print("Successfully loaded checkpoint '%s'" % ckpt_path)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt_path)

def get_projections(embeddings, true_masks):
    assert embeddings.size()[:2] == true_masks.size()
    emb = embeddings[true_masks==1].view(-1, embeddings.size(-1))
    emb = emb.detach().numpy()
    emb_2d = TSNE(n_components=2).fit_transform(emb)
    return emb_2d
    



def project(model, mydata, ckpt_path, tokenizer, max_len, coarse=False):
    '''

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    mydata : dict
        contain tokens and attention masks
    ckpt_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # load the trained model
    state_dict = load_ckpt(ckpt_path)['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('ignore {}'.format(name))
        else:
            print('load {} from {}'.format(name, ckpt_path))
            own_state[name].copy_(param)
            
    model.eval()
    
    data_dict = wrapper(mydata, tokenizer, max_len, coarse)
    
    embeddings = model.encoder(data_dict['tokens'], data_dict['atn_masks'])
    
    proj = get_projections(embeddings, data_dict['true_masks'])
    
    res = {'data':data_dict, 'emb':embeddings, 'proj': proj}
    
    return res
    
# assume that there exists a directory called 'obj' in cwd
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/data/private/qinwenna/FewNERD/train-intra.txt',
            help='data file')
    parser.add_argument('--ckpt_dir', default='/home/qinwenna/protoBERT-batch/checkpoint', 
            help='checkpoint directory')
    parser.add_argument('--ckpt_file', default='/proto-bert-train-intra.txt-val-intra.txt-5-1.pth.tar',
            help='ckpt file name')
    parser.add_argument('--enc_ckpt', default='bert-base-uncased',
            help='encoder pretrain path')
    parser.add_argument('--max_len', default=100, type=int,
           help='max length')
    parser.add_argument('--K', default=50, type=int,
            help='number of samples for each class')
    parser.add_argument('--coarse', action='store_true',
            help='use coarse labels')
    opt = parser.parse_args()
    myencoder = BertEncoder(opt.enc_ckpt, opt.max_len)
    tokenizer = BertTokenizer.from_pretrained(opt.enc_ckpt)
    model = Proto(myencoder)
    K = opt.K
    coarse = opt.coarse
    dataset = FewNERDataset(opt.data_path, myencoder, 5, 1, 1)
    data = dataset.sample_for_plot(K)
    ckpt_path = opt.ckpt_dir + opt.ckpt_file
    res = project(model, data, ckpt_path, tokenizer, 
                         opt.max_len, coarse=coarse)
    if coarse:
        lab_type = 'coarse'
    else:
        lab_type = 'fine'
    data_type = opt.data_path.split('/')[-1].split('-')[-1].split('.')[0]
    save_obj(res, data_type+lab_type+'_res_'+str(K))
    #save_obj(fine_res, 'fine_res')
                        
if __name__ == "__main__":
    main()
    
