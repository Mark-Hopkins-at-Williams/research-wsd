def evaluate(net, data):
    """
    Evaluates a trained neural network classifier on validation data.

    The accuracy (i.e. percentage of correct classifications) is returned.
    
    """    
    correct = 0
    total = 0
    val_loader = data.batch_iter()
    for evidence, response in val_loader:
        val_outputs = net(evidence)
        correct_inc, total_inc = accuracy(val_outputs, response)
        correct += correct_inc
        total += total_inc
    return correct/total    

def predict(distribution):
    return distribution.argmax()

def accuracy(predicted_labels, gold_labels):
    assert(len(predicted_labels) == len(gold_labels))
    correct = [l1 for (l1, l2) in zip(predicted_labels, gold_labels)
                  if l1 == l2]
    return len(correct), len(predicted_labels)

