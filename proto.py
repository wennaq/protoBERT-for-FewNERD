import os
os.chdir('/Users/qinwenna/Desktop/THU-Research/protoBERT')
import framework
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Proto(framework.FewShotNERModel):
    
    def __init__(self, encoder):
        framework.FewShotNERModel.__init__(self, encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        #self.drop = nn.Dropout()

    def __dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, clusters, N, K):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        '''
        support_emb = self.encoder(support) 
        query_emb = self.encoder(query) 
        ns, max_length, hidden_size = support_emb.size()
        nq, _, _ = query_emb.size()
        #support = self.drop(support_emb)
        #query = self.drop(query_emb)
        
        protos = []
        # iterate over classes in dictionary
        for cls in range(N+1):
            indices = clusters[cls]
            cur_emb = []
            # iterate over each sentence
            for b in range(ns):
                assert(ns == len(indices))
                cur_ind = indices[b]
                cur_emb.append(support_emb[b][cur_ind])
                
            cur_emb_tensor = torch.cat(cur_emb, dim=0)
            protos.append(torch.mean(cur_emb_tensor, dim=0))
            
        protos = torch.stack(protos, dim=0)
        # compute euclidean distances between each token and prototypes
        query_emb_prime = query_emb.unsqueeze(2)
        logits = -(torch.pow(query_emb_prime-protos, 2)).sum(3)
        _, y_hat = torch.max(logits, dim=-1)
        
        endpos = query['end'] # idx of '[SEP]' in each sentence
        logits = torch.cat([logits[i][1:endpos[i]] for i in range(nq)], dim=0)
        y_hat = torch.cat([y_hat[i][1:endpos[i]] for i in range(nq)], dim=0)
        
        
        return logits, y_hat