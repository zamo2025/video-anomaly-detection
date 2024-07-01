# This file holds all the DPU logic. The DPU breaks the encodere vectors from to encoder
# into prototypes to create a prototype pool. The encoding vectors are then used to
# query the prototype pool to reconstruct a normalcy encoding

import torch
import torch.nn.functional as func

def mean_distance(a, b, weight=None, training=True):
    distance = ((a - b) ** 2).sum(-1)

    if weight is not None:
        distance *= weight 

    if not training:
        return distance
    else:
        return distance.mean().unsqueeze(0)

def distance(a, b):
    return ((a - b) ** 2).sum(-1)

class Dynamic_Prototype(torch.nn.Module):
    def __init__(self, prototypeSize, featureDim, keyDim, tempUpdate, tempGather, shrinkThreshold = 0):
        super(Dynamic_Prototype, self).__init__()

        self.protoSize = prototypeSize
        self.featureDim = featureDim
        self.keyDim = keyDim
        self.tempUpdate = tempUpdate
        self.tempGather = tempGather
        self.mHeads = torch.nn.Linear(keyDim, prototypeSize, bias=False)
        self.shrinkThreshold = shrinkThreshold
    
    def get_score(self, pro, query):
        bs, n, _ = query.size()
        bs, m, _ = pro.size()
        score = torch.bmm(query, pro.permute(0, 2, 1))
        score = score.view(bs, n, m)
        
        scoreQuery = func.softmax(score, dim = 1)
        scoreProto = func.softmax(score, dim = 2)
        
        return scoreQuery, scoreProto
    
    def forward(self, key, query, weights, train = True):

        batchSize, dims, height, width = key.size()
        key = key.permute(0,2,3,1)
        _, _, h, w = query.size()
        query = query.permute(0, 2, 3, 1)
        query = query.reshape((batchSize, -1, self.featureDim))

        if train:
            if weights == None:
                multiHeadWeights = self.mHeads(key)
            else:
                multiHeadWeights = func.linear(key, weights['prototype.Mheads.weight'])


            multiHeadWeights = multiHeadWeights.view((batchSize, height * width, self.protoSize, 1))

            multiHeadWeights = func.softmax(multiHeadWeights, dim = 1)

            key = key.reshape((batchSize, width * height, dims))
            protos = multiHeadWeights * key.unsqueeze(-2)
            protos = protos.sum(1)
            
            updatedQuery, featureLoss, compactnessLoss, diversityLoss = self.query_loss(query, protos, weights, train)

            updatedQuery = updatedQuery + query

            updatedQuery = updatedQuery.permute(0,2,1)
            updatedQuery = updatedQuery.view((batchSize, self.featureDim, h, w))
            return updatedQuery, protos, featureLoss, compactnessLoss, diversityLoss
        
        else:
            if weights == None:
                multiHeadWeights = self.mHeads(key)
            else:
                multiHeadWeights = func.linear(key, weights['prototype.Mheads.weight'])
                
            multiHeadWeights = multiHeadWeights.view((batchSize, height * width, self.protoSize, 1))

            multiHeadWeights = func.softmax(multiHeadWeights, dim = 1)

            key = key.reshape((batchSize, width * height, dims))
            protos = multiHeadWeights * key.unsqueeze(-2)
            protos = protos.sum(1)

            updatedQuery, featureLoss, query = self.query_loss(query, protos, weights, train)

            updatedQuery = updatedQuery + query

            updatedQuery = updatedQuery.permute(0, 2, 1)
            updatedQuery = updatedQuery.view((batchSize, self.featureDim, h, w))

            return updatedQuery, protos, query, featureLoss
        
    def query_loss(self, query, keys, weights, train):
        _, n, dims = query.size()
        if train:
            
            # Diversity constrain
            k = func.normalize(keys, dim = -1)
            dist = 1 - distance(k.unsqueeze(1), k.unsqueeze(2))
            
            mask = dist > 0
            dist *= mask.float()
            dist = torch.triu(dist, diagonal = 1)
            diversityLoss = dist.sum(1).sum(1) * 2 / (self.protoSize * (self.protoSize - 1))
            diversityLoss = diversityLoss.mean()

            # Maintain the compactness of same attribute vector
            compactnessLoss = mean_distance(k[1:], k[:-1])

            mseLoss = torch.nn.MSELoss()

            keys = func.normalize(keys, dim = -1)
            _, softScorePrototype = self.get_score(keys, query)

            newQuery = softScorePrototype.unsqueeze(-1) * keys.unsqueeze(1)
            newQuery = newQuery.sum(2)
            newQuery = func.normalize(newQuery, dim = -1)

            # Maintain the distinct attributes
            _, gathering_indices = torch.topk(softScorePrototype, 2, dim = -1)
        
            pos = torch.gather(keys, 1, gathering_indices[:, :, :1].repeat((1, 1, dims)))

            featureLoss = mseLoss(query, pos)

            return newQuery, featureLoss, compactnessLoss, diversityLoss
        
            
        else:
            mseLoss = torch.nn.MSELoss(reduction='none')

            keys = func.normalize(keys, dim = -1)
            _, softScorePrototype = self.get_score(keys, query)

            newQuery = softScorePrototype.unsqueeze(-1) * keys.unsqueeze(1)
            newQuery = newQuery.sum(2)
            newQuery = func.normalize(newQuery, dim = -1)

            _, gatheringInds = torch.topk(softScorePrototype, 2, dim = -1)
        
            pos = torch.gather(keys, 1, gatheringInds[:, :, :1].repeat((1, 1, dims)))

            featureLoss = mseLoss(query, pos)

            return newQuery, featureLoss, query