import torch

LARGE_NEGATIVE = -10000000

def predict(distribution):
    return distribution.argmax()

def accuracy(predicted_labels, gold_labels):
    assert(len(predicted_labels) == len(gold_labels))
    correct = [l1 for (l1, l2) in zip(predicted_labels, gold_labels)
                  if l1 == l2]
    return len(correct), len(predicted_labels)

def apply_zone_masks(outputs, zones):
    revised = torch.empty(outputs.shape)
    revised = revised.fill_(LARGE_NEGATIVE)
    for row in range(len(zones)):
        (start, stop) = zones[row]
        revised[row][start:stop] = outputs[row][start:stop]
    return revised

def decode(net, data):
    """
    Runs a trained neural network classifier on validation data, and iterates
    through the top prediction for each datum.
    
    """        
    net.eval()
    val_loader = data.batch_iter()
    for inst_ids, targets, evidence, response, zones in val_loader:
        val_outputs = net(evidence)
        revised = apply_zone_masks(val_outputs, zones)
        revised = revised.argmax(dim=-1)
        for element in zip(inst_ids, 
                           zip(targets,
                               zip(revised, response.tolist()))):                    
            (inst_id, (target, (prediction, gold))) = element
            yield inst_id, target, prediction, gold
    net.train()

def evaluate(net, data):
    """
    The accuracy (i.e. percentage of correct classifications) is returned.
    
    """
    decoded = list(decode(net, data))
    predictions = [pred for (_, _, pred, _) in decoded]
    gold = [g for (_, _, _, g) in decoded]
    correct, total = accuracy(predictions, gold)
    return correct/total
    