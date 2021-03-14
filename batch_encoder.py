import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel


class BertEncoder(nn.Module):
    
    def __init__(self, pretrain_path, max_length):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path, 
                                  output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.max_length = max_length
        
    def forward(self, inputs):
        outputs = self.bert(inputs['tokens'], inputs['masks'])
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
        
        return torch.stack(embeddings, dim=0)
    
    def tokenize(self,raw_tokens, raw_labels):
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
        
        return tokens_tensor, masks_tensor, labels, endpos, true
    
    
    
    