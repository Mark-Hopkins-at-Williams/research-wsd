import torch
from util import cudaify
from pytorch_transformers import BertModel, BertConfig, BertTokenizer

config = BertConfig.from_pretrained('bert-base-uncased')
config.output_hidden_states = True
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel(config)
bert.eval()
bert = cudaify(bert)


def embed_bert(instance):
    """
    Converts a SenseInstance into a tensor.
    """
    tokens = instance.tokens
    position = instance.pos
    input_ids = torch.tensor(tokenizer.encode(tokenizer.convert_tokens_to_string(tokens),
                             add_special_tokens = True)).unsqueeze(0)
    results = bert(input_ids)
    embedding = results[0][position + 1]
    return embedding.detach()


def embed_bert_avg_both(instance):
        position = instance.pos + 1
        tokens = instance.tokens
        tokens = ["CLS"] + tokens + ["SEP"]
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
        results = bert(input_ids)
        final_hidden_layers = results[0].squeeze(0)
        prev_token_vec = final_hidden_layers[position - 1]
        token_vec = final_hidden_layers[position]
        next_token_vec = final_hidden_layers[position + 1]
        avged = torch.mean(torch.stack([prev_token_vec, token_vec, next_token_vec]), dim=0)
        return avged.detach()
    
def embed_bert_avg_left(instance):
    tokens = instance.tokens
    tokens = ["CLS"] + tokens + ["SEP"]
    position = instance.pos + 1
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    results = bert(input_ids)
    final_hidden_layers = results[0].squeeze(0)
    prev_token_vec = final_hidden_layers[position - 1]
    token_vec = final_hidden_layers[position]
    avged = torch.mean(torch.stack([prev_token_vec, token_vec]), dim=0)
    return avged.detach()

def embed_bert_avg_right(instance):
    tokens = instance.tokens
    position = instance.pos + 1
    tokens = ["CLS"] + tokens + ["SEP"]
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    results = bert(input_ids)
    final_hidden_layers = results[0].squeeze(0)
    token_vec = final_hidden_layers[position]
    next_token_vec = final_hidden_layers[position + 1]
    avged = torch.mean(torch.stack([token_vec, next_token_vec]), dim=0)
    return avged.detach()

def embed_bert_concat_both(instance):
    position = instance.pos + 1
    tokens = instance.tokens
    tokens = ["CLS"] + tokens + ["SEP"]
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    results = bert(input_ids)
    final_hidden_layers = results[0].squeeze(0)
    prev_token_vec = final_hidden_layers[position - 1]
    token_vec = final_hidden_layers[position]
    next_token_vec = final_hidden_layers[position + 1]
    concat = torch.cat([prev_token_vec, token_vec, next_token_vec])
    return concat.detach()

def embed_bert_concat_left(instance):
    position = instance.pos + 1
    tokens = instance.tokens
    tokens = ["CLS"] + tokens + ["SEP"]
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    results = bert(input_ids)
    final_hidden_layers = results[0].squeeze(0)
    prev_token_vec = final_hidden_layers[position - 1]
    token_vec = final_hidden_layers[position]
    concat = torch.cat([prev_token_vec, token_vec])
    return concat.detach()

def embed_bert_concat_right(instance):
    position = instance.pos + 1
    tokens = instance.tokens
    tokens = ["CLS"] + tokens + ["SEP"]
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
    results = bert(input_ids)
    final_hidden_layers = results[0].squeeze(0)
    token_vec = final_hidden_layers[position]
    next_token_vec = final_hidden_layers[position + 1]
    concat = torch.cat([token_vec, next_token_vec])
    return concat.detach()



