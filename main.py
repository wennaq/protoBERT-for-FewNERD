#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 10:25:22 2021

@author: qinwenna
"""
from batch_encoder import BertEncoder
from proto import Proto
import dataloader
from dataloader import FewNERD

max_len = 128
N = 5
K = 1
Q = 1
root = '/Users/qinwenna/Desktop/THU-Research/FewNERD/'
name = 'train-intra'
myencoder = BertEncoder('bert-base-uncased', 128)
model = Proto(myencoder)
mydata = FewNERD(name, myencoder, N, K, Q, root)
support, query, int2cls, support_clusters = mydata.__getitem__(0)
logits, y_hat = model(support, query, support_clusters, N, K)
y_true = query['true']