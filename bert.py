import torch

from pytorch_transformers import BertModel, BertConfig, BertTokenizer

config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel(config)    

def vectorize_instance(instance):
    """
    Converts a SenseInstance into a tensor.
    
    """
    tokens = instance.tokens
    position = instance.pos
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    outputs = bert(input_ids)[0].squeeze(0)
    return outputs[position].detach()

