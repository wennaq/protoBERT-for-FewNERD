import sys
import os
import framework
import torch
import random
import numpy as np
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Proto(framework.FewShotNERModel):
    
    def __init__(self, encoder, dot=False):
        framework.FewShotNERModel.__init__(self, encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [N+1, emb_size], Q [num_of_sent, max_len, emb_size]
        if Q.size()[:2] != q_mask.size():
            sys.stdout.write('unequal size!! \n')
            sys.stdout.write('Q.size() is {} \n'.format(Q.size()))
            sys.stdout.write('q_mask.size() is {} \n'.format(q_mask.size()))
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1))
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)
    
    def get_raw_class(self, labels, int2cls):
        'labels: 2d list'
        raw = [int2cls[torch.IntTensor.item(lab)] for sent in labels for lab in sent]
        return raw
    
    def get_protos(self, embeddings, labels, true_masks, N):
        protos = []
        embeddings = embeddings[true_masks==1].view(-1, embeddings.size(-1))
        labels_ts = torch.cat(labels, 0)
        
        for cls in range(N+1):
            proto = torch.mean(embeddings[labels_ts==cls], 0) 
            protos.append(proto)
            
        protos = torch.stack(protos)
        return protos
    
    def get_entities_dict(self, entities, N):
        d = {i: [] for i in range(N+1)} # 'O' is not one of the N classes
        # fill in the index of the sentence that contains each entity
        for i, sent in enumerate(entities):
            for ent in sent:
                ent.add_sent_num(i)
        # group the entities based on their labels
        for sent in entities:
            for ent in sent:
                if ent.label in d.keys():
                    d[ent.label] += ent
                else:
                    raise Exception('label not in keys')
        return d
    
    def contrastive(self, embeddings, labels, entities, N, K, max_len, t):
        '''
        

        Parameters
        ----------
        support : TYPE
            DESCRIPTION.
        entities : 2d list
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # loop over each entity
        # select a positive example
        # select K negative examples
        # compute the contrastive loss
        sent_lengths = [sent_labels.size()[0] for sent_labels in labels]
        d = self.get_entities_dict(entities, N) 
        # sent_num modified after this
        entities_flattened = [ent for sent in entities for ent in sent]
        contrastive_loss = 0
        
        for ent in entities_flattened:
            ent_emb = embeddings[ent.sent_num][ent.start:ent.end+1]
            # select a positive example
            if not (set(d[ent.label])-set([ent])): # empty
                print('no positive example exists')
                continue
            else:
                pos_ex = random.sample(set(d[ent.label])-set([ent]), 1)
                pos_emb = embeddings[pos_ex.sent_num]\
                          [pos_ex.start:(pos_ex.end+1)]
                # take the average of all token embeddings for the entity
                pos_emb_avg = torch.mean(pos_emb, 0)
                
            neg_emb_all = []
            # select negative examples, half of which from 'O'
            for cls, ent_lst in enumerate(d):
                if cls == ent.label:
                    continue
                elif cls == 0:
                    
                    o_masks = np.ones(max_len, dtype=np.int32)
                    # '[CLS]' should be ignored
                    o_masks[0] = 0
                    # named entities should be ignored
                    sent_entities = entities[ent.sent_num]
                    for x in sent_entities:
                        o_masks[x.start:(x.end+1)] = 0
                    # '[SEP]' and paddings should be ignored
                    o_masks[(sent_lengths[ent.sent_num]+1):] = 0
                    
                    o_ex_emb = embeddings[ent.sent_num][o_masks==1]
                    neg_emb_all.append(o_ex_emb)
                    
                    
                        
                else:
                    neg_ex = random.sample(d[cls], 1)
                    neg_ex_emb = embeddings[neg_ex.sent_num]\
                                 [neg_ex.start:neg_ex.end+1]
                    neg_ex_emb_avg = torch.mean(neg_ex_emb, 0)
                    neg_emb_all.append(neg_ex_emb_avg.unsqueeze(0))
            neg_emb = torch.cat(neg_emb_all, dim=0)
            
            # compute the loss on the current entity
            # divide over t to prevent overflow
            pos_dist = torch.pow(pos_emb - ent_emb, 2).sum(dim=0)
            numerator = torch.exp(pos_dist/t)
            neg_dist = torch.pow(neg_emb - ent_emb, 2).sum(dim=1)
            denom = torch.exp(neg_dist/t).sum(0)
            contrastive_loss += -torch.log(numerator/denom)
            
        
        return contrastive_loss

    def forward(self, support, query, N, K, total_Q, t):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        '''
        max_len = support['tokens'].size()[1]
        support_emb = self.encoder(support['tokens'],
                                   support['atn_masks']) #(num_samples, max_len, emb_size)
        query_emb = self.encoder(query['tokens'], query['atn_masks']) 
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)
        
        logits = []
        curs_pos = 0
        curq_pos = 0
        
        for idx, size in enumerate(support['size']):
            # compute protos with support set

            cur_batch_emb = support_emb[curs_pos:(curs_pos+size)]
            cur_batch_lab = support['labels'][curs_pos:(curs_pos+size)]
            cur_batch_mask = support['true_masks'][curs_pos:(curs_pos+size)]
            cur_proto = self.get_protos(cur_batch_emb, cur_batch_lab,
                                        cur_batch_mask, N)
            curs_pos += size
            # make predictions for query set
            q_size = query['size'][idx]
            Q = query_emb[curq_pos:(curq_pos+q_size)]
            q_mask = query['true_masks'][curq_pos:(curq_pos+q_size)]
            cur_logits = self.__batch_dist__(cur_proto, Q, q_mask)
            logits.append(cur_logits)
            curq_pos += q_size
        
        logits_ts = torch.cat(logits, 0)    
        _, y_hat = torch.max(logits_ts, dim=-1)
        contrastive_loss = self.contrastive(support_emb, support['labels'], 
                                            support['entities'], N, K, 
                                            max_len, t)
        
        return logits_ts, y_hat, contrastive_loss
