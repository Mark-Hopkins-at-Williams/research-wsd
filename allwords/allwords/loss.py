import torch

class LossWithZones:
    
    def __call__(self, predicted, gold, zones):
        revised_pred = torch.zeros(predicted.shape)
        for i, (zone_start, zone_stop) in enumerate(zones):
            normalizer = sum(predicted[i, zone_start:zone_stop])
            revised_pred[i,zone_start:zone_stop] = predicted[i, zone_start:zone_stop] / normalizer
        neglog_pred = -revised_pred 
        result = neglog_pred[list(range(len(neglog_pred))), gold]
        return torch.mean(result)
    
class NLLLossWithZones:
        
    def __call__(self, predicted, gold, zones):
        revised_pred = torch.zeros(predicted.shape)
        for i, (zone_start, zone_stop) in enumerate(zones):
            normalizer = sum(predicted[i, zone_start:zone_stop])
            revised_pred[i,zone_start:zone_stop] = predicted[i, zone_start:zone_stop] / normalizer
        neglog_pred = -torch.log(revised_pred)
        result = neglog_pred[list(range(len(neglog_pred))), gold]
        return torch.mean(result)
    
    
    