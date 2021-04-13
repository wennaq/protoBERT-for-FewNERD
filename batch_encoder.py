import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel


class Entity:
    def __init__(self, start, end, label):
        self.start = start # index at which the entity starts
        self.end = end # index at which the entity ends
        self.label = label
        self.sent_num = None
    
    def add_sent_num(self, sent_num):
        if self.sent_num == None:
            self.sent_num = sent_num


class BertEncoder(nn.Module):
    
    def __init__(self, pretrain_path, max_len):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path, 
                                  output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.max_len = max_len
        
    def forward(self, tokens, atn_masks):
        outputs = self.bert(tokens, atn_masks)
        hidden_states = outputs[2]
        # hidden_states is a list
        # concatenate the tensors for all layers
        # use 'stack' to create a new dimension in the tensor
        all_embeddings = torch.stack(hidden_states, dim=0)
        # (layers, batch size, max_length, hidden size)
        assert(all_embeddings.size()[1] == tokens.size()[0])
        word_embeddings = torch.sum(all_embeddings[-4:], 0)
        
        return word_embeddings
    
    
    def get_entity(self, labels):
        entities = []
        inside = False
        for i, lab in enumerate(labels):
            if not inside:
                if lab == 0: # outside
                    continue
                else: # start of a new entity
                    inside = True
                    start = i
                    if i == len(labels)-1: # end of the list
                        cur_entity = Entity(start, start, lab)
                        entities.append(cur_entity)
            else: # inside
                if lab == labels[i-1]: # same entity
                    if i == len(labels)-1: # end of the list
                        end = i
                        cur_entity = Entity(start, end, labels[i-1])
                        entities.append(cur_entity)
                else: # different label
                    # add the current entity to the list 
                    end = i-1
                    cur_entity = Entity(start, end, labels[i-1])
                    entities.append(cur_entity)
                    if lab == 0:
                        inside = False
                    else: # start of a new entity
                        inside = True
                        start = i
        return entities
                       
    
    def tokenize(self,raw_tokens, raw_labels):
        
        raw_tokens = [token.lower() for token in raw_tokens]
        tokens = []
        labels = []
        for i in range(len(raw_tokens)):
            tok = raw_tokens[i]
            tokenized = self.tokenizer.tokenize(tok)
            tokens += tokenized
            
            lab = raw_labels[i]
            labels += [lab] * len(tokenized)
            
        assert(len(tokens) == len(labels))
        
        tokens_lst = []
        labels_lst = []
        # split the sentence if exceeding max_len
        while len(tokens) > (self.max_len - 2):
            split_pos = self.max_len - 2
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
        entities_lst = []
        # add special tokens to each sentence
        for i in range(len(tokens_lst)):
            sent = ['[CLS]'] + tokens_lst[i] + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            sent_labels = [0] + labels_lst[i] + [0]
            # padding
            while len(indexed_tokens) < self.max_len:
                indexed_tokens.append(0)
                sent_labels.append(0)
                
            attention_masks = np.zeros((self.max_len), dtype=np.int32)
            attention_masks[:len(sent)] = 1
            true_masks = np.zeros((self.max_len), dtype=np.int32)
            true_masks[1:(len(sent)-1)] = 1
            entities = self.get_entity(sent_labels)
            
            indexed_tokens_lst.append(indexed_tokens)
            attention_masks_lst.append(attention_masks)
            true_masks_lst.append(true_masks)
            entities_lst.append(entities)

        
        return indexed_tokens_lst, labels_lst, \
               attention_masks_lst, true_masks_lst, entities_lst
    
    
    
'''

    def forward(self, inputs):
        outputs = self.bert(inputs['tokens'], inputs['atn_masks'])
        hidden_states = outputs[2]
        # hidden_states is a list
        # concatenate the tensors for all layers
        # use 'stack' to create a new dimension in the tensor
        sent_embeddings = torch.stack(hidden_states, dim=0)
        # (layers, batch size, max_length, hidden size)
        assert(sent_embeddings.size()[1] == inputs['tokens'].size()[0])
        batch_size = sent_embeddings.size()[1]
        embeddings = []
        for i in range(batch_size):
            cur_sent_emb = sent_embeddings[:,i,:,:]
            # swap dimensions 0(layers) and 1(max_length)
            cur_sent_emb = cur_sent_emb.permute(1,0,2)
            # 'cur_token_emb' is of size [max_length x 12 x 768]
        
            # create word vectors by summing the last four layers
            token_vecs_sum = []
            for token in cur_sent_emb:
                # 'token' is a [12 x 768] tensor
            
                # sum the vectors from the last four layers
                # each layer vector contains 768 values, 
                sum_vec = torch.sum(token[-4:], dim=0)
                token_vecs_sum.append(sum_vec)
            
            embeddings.append(torch.stack(token_vecs_sum, dim=0))
        
        return torch.stack(embeddings, dim=0)s
        
        # convert tokens to ids
        tokens = ['[CLS]']
        labels = [-1]
        true = []
        for i in range(len(raw_tokens)):
            tok = raw_tokens[i]
            tokenized = self.tokenizer.tokenize(tok)
            tokens += tokenized
            
            label = raw_labels[i]
            labels += [label] * len(tokenized)
            
            true += [label] * len(tokenized)
            
        tokens += ['[SEP]']
        labels += [-1]
        assert(len(tokens) == len(labels))
        endpos = len(tokens)-1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
            labels.append(-2)
        # indexed_tokens = indexed_tokens[:self.max_length]
        
        # attention_masks
        attention_masks = np.zeros((self.max_length), dtype=np.int32)
        attention_masks[:len(tokens)] = 1
        
        # convert inputs into Pytorch tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        masks_tensor = torch.tensor(attention_masks)
        '''
